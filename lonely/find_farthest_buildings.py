#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Findet Gebäude mit Hausnummern, die den größten Abstand zu ihrem nächsten
Nachbarn haben. Der Bereich kann optional über eine Boundary-Relation aus dem
PBF ausgeschnitten werden. Der Ansatz basiert auf run_iso_pipeline.py.

Beispiel:
  python3 find_farthest_buildings.py \
    --pbf pbf/bremen-latest.osm.pbf \
    --out out/bremen_farthest \
    --limit 5

Optional:
  --relation 12345       # OSM-Relations-ID für eine Boundary
  --limit N              # Anzahl der ausgegebenen Gebäude
  --keep-intermediate    # behält Boundary/Clip/Filter-Zwischendateien
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.parse

from shapely.geometry import MultiPolygon, Polygon, shape, Point
from shapely.strtree import STRtree


def run(cmd, **kw):
    """Execute a subprocess command and raise on failure."""
    p = subprocess.run(cmd, **kw)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def haversine_m(lon1, lat1, lon2, lat2):
    """Entfernung zweier Punkte auf der Erde in Metern."""
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def compute_lonely_buildings(geojson_path, limit=1):
    """Lese Gebäude und bestimme die mit dem größten Abstand zum nächsten Nachbarn."""
    with open(geojson_path, "r", encoding="utf-8") as f:
        gj = json.load(f)

    items = []  # (lon, lat, osm_id)
    points = []  # shapely Points für räumlichen Index
    for i, feat in enumerate(gj.get("features", [])):
        props = feat.get("properties") or {}
        if "addr:housenumber" not in props:
            continue
        geom = feat.get("geometry")
        if not geom:
            continue
        g = shape(geom)
        if not isinstance(g, (Polygon, MultiPolygon)):
            continue
        c = g.centroid
        osm_id = props.get("osm_id") or props.get("id") or props.get("@id") or f"feat_{i}"
        lon = float(c.x)
        lat = float(c.y)
        items.append((lon, lat, osm_id))
        points.append(Point(lon, lat))

    if len(items) < 2:
        print("Zu wenige Gebäude mit addr:housenumber.", file=sys.stderr)
        return []

    tree = STRtree(points)
    results = []  # (osm_id, nn_osm_id, distance_m, lon, lat)
    for idx, (lon, lat, osm_id) in enumerate(items):
        nn_idx_arr, _ = tree.query_nearest(points[idx], return_distance=True, exclusive=True, all_matches=False)
        if len(nn_idx_arr) == 0:
            continue
        nn_idx = int(nn_idx_arr[0])
        nn_osm_id = items[nn_idx][2]
        lon1, lat1, _ = items[nn_idx]
        d = haversine_m(lon, lat, lon1, lat1)
        results.append((osm_id, nn_osm_id, d, lon, lat))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:limit]


def main():
    ap = argparse.ArgumentParser(description="Finde Gebäude mit dem größten Abstand zum nächsten Nachbarn in einer PBF-Datei.")
    ap.add_argument("--pbf", required=True, help="Eingabe-PBF")
    ap.add_argument("--out", required=True, help="Ausgabedatei-Präfix")
    ap.add_argument("--relation", help="OSM-Relations-ID der Boundary (optional)")
    ap.add_argument("--limit", type=int, default=1, help="Anzahl der ausgegebenen Ergebnisse")
    ap.add_argument("--keep-intermediate", action="store_true", help="Zwischendateien behalten")
    args = ap.parse_args()

    if shutil.which("osmium") is None:
        print("Fehlt: 'osmium' (osmium-tool). Installation erforderlich.", file=sys.stderr)
        sys.exit(2)

    tmpdir = tempfile.mkdtemp(prefix="osm_far_")
    try:
        pbf = os.path.abspath(args.pbf)

        if args.relation:
            rel = str(args.relation).lstrip("r")
            b_pbf = os.path.join(tmpdir, "boundary.osm.pbf")
            b_geo = os.path.join(tmpdir, "boundary.geojson")
            clip_pbf = os.path.join(tmpdir, "clip.pbf")
            run(["osmium", "getid", "-r", pbf, f"r{rel}", "-o", b_pbf, "-O"])
            run(["osmium", "export", b_pbf, "-o", b_geo, "-O"])
            run(["osmium", "extract", "-p", b_geo, pbf, "-o", clip_pbf, "-O"])
            source_pbf = clip_pbf
        else:
            source_pbf = pbf

        addr_pbf = os.path.join(tmpdir, "buildings_addr.pbf")
        addr_geo = os.path.join(tmpdir, "buildings_addr.geojson")

        run(["osmium", "tags-filter", source_pbf, "w/building", "w/addr:housenumber", "-o", addr_pbf, "-O"])
        run(["osmium", "export", addr_pbf, "-a", "id", "-o", addr_geo, "-O"])

        results = compute_lonely_buildings(addr_geo, args.limit)
        if not results:
            sys.exit(1)

        outprefix = args.out
        os.makedirs(os.path.dirname(outprefix), exist_ok=True) if os.path.dirname(outprefix) else None
        csv_path = outprefix + "_lonely.csv"
        html_path = outprefix + "_lonely.html"

        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("osm_id,nn_osm_id,distance_m,lon,lat\n")
            for osm_id, nn_osm_id, d, lon, lat in results:
                f.write(f"{osm_id},{nn_osm_id},{d:.2f},{lon},{lat}\n")

        with open(html_path, "w", encoding="utf-8") as f:
            f.write("<!DOCTYPE html><html><head><meta charset='utf-8'><title>Loneliest Buildings</title></head><body>\n")
            f.write("<h1>Top loneliest buildings</h1>\n<ul>\n")
            for osm_id, nn_osm_id, d, lon, lat in results:
                query = f"[out:json];(way({osm_id});way({nn_osm_id}););out geom;"
                link = "https://overpass-turbo.eu/?Q=" + urllib.parse.quote(query)
                f.write(f"<li><a href='{link}'>{osm_id} vs {nn_osm_id} ({d:.2f} m)</a></li>\n")
            f.write("</ul>\n</body></html>\n")

        print(f"Top {len(results)} loneliest buildings:")
        for osm_id, nn_osm_id, d, lon, lat in results:
            print(f"  {osm_id} vs {nn_osm_id}: {d:.2f} m @ ({lon},{lat})")
        print(f"CSV: {csv_path}")
        print(f"HTML: {html_path}")
    finally:
        if not args.keep_intermediate:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
