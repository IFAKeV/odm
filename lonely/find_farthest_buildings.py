#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Findet die Gebäudepaare mit Hausnummern, die innerhalb eines Bereichs am
weitesten voneinander entfernt liegen. Der Bereich kann optional über eine
Boundary-Relation aus dem PBF ausgeschnitten werden. Der Ansatz basiert auf
run_iso_pipeline.py.

Beispiel:
  python3 find_farthest_buildings.py \
    --pbf pbf/bremen-latest.osm.pbf \
    --out out/bremen_farthest \
    --limit 5

Optional:
  --relation 12345       # OSM-Relations-ID für eine Boundary
  --limit N              # Anzahl der ausgegebenen Paare
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

from shapely.geometry import MultiPolygon, Polygon, shape


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


def compute_farthest_pairs(geojson_path, limit=1):
    """Lese Gebäude aus GeoJSON und bestimme die am weitesten entfernten Paare."""
    with open(geojson_path, "r", encoding="utf-8") as f:
        gj = json.load(f)

    items = []  # (lon, lat, osm_id)
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
        items.append((float(c.x), float(c.y), osm_id))

    if len(items) < 2:
        print("Zu wenige Gebäude mit addr:housenumber.", file=sys.stderr)
        return []

    # Farthest pair liegt auf der konvexen Hülle (Monotone Chain)
    pts = sorted([(lon, lat, idx) for idx, (lon, lat, _id) in enumerate(items)])
    if len(pts) <= 1:
        return []

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    hull_indices = [idx for _, _, idx in hull]

    dists = []
    for h_idx, i in enumerate(hull_indices):
        lon0, lat0, _ = items[i]
        for j in hull_indices[h_idx + 1:]:
            lon1, lat1, _ = items[j]
            d = haversine_m(lon0, lat0, lon1, lat1)
            dists.append((d, i, j))

    dists.sort(reverse=True)
    res = []
    for d, i, j in dists[:limit]:
        res.append((items[i], items[j], d))
    return res


def main():
    ap = argparse.ArgumentParser(description="Finde die am weitesten entfernten Gebäudepaare mit Hausnummern in einer PBF-Datei.")
    ap.add_argument("--pbf", required=True, help="Eingabe-PBF")
    ap.add_argument("--out", required=True, help="Ausgabedatei-Präfix")
    ap.add_argument("--relation", help="OSM-Relations-ID der Boundary (optional)")
    ap.add_argument("--limit", type=int, default=1, help="Anzahl der ausgegebenen entferntesten Paare")
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

        results = compute_farthest_pairs(addr_geo, args.limit)
        if not results:
            sys.exit(1)

        outprefix = args.out
        os.makedirs(os.path.dirname(outprefix), exist_ok=True) if os.path.dirname(outprefix) else None
        csv_path = outprefix + "_farthest.csv"

        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("osm_id_1,osm_id_2,distance_m,lon1,lat1,lon2,lat2\n")
            for (lon0, lat0, osm0), (lon1, lat1, osm1), d in results:
                f.write(f"{osm0},{osm1},{d:.2f},{lon0},{lat0},{lon1},{lat1}\n")

        print(f"Top {len(results)} farthest distances:")
        for (lon0, lat0, osm0), (lon1, lat1, osm1), d in results:
            print(f"  {d:.2f} m zwischen {osm0} und {osm1}")
        print(f"CSV: {csv_path}")
    finally:
        if not args.keep_intermediate:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
