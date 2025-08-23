#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Dieses Script ist ein nicht lauffähiger Entwurf von GPT5 und soll folgendes machen:
- extrahiere einen Bereich aus einer .pbf Datei und finde in diesem Bereich die am weitesten voneinander entfernten Gebäude

run_iso_pipeline.py
Ein-Befehl-Pipeline:
NRW-PBF -> Boundary (Relation) aus PBF -> Clip -> building + addr:housenumber -> GeoJSON -> Hitliste isolierter Gebäude

Voraussetzungen:
  - osmium-tool (CLI)
  - Python: shapely, numpy

Beispiel:
  python3 run_iso_pipeline.py \
    --pbf nordrhein-westfalen-latest.osm.pbf \
    --relation 62644 \
    --out out/roesrath \
    --limit 200

Optional:
  --min-distance 50    # nur Einträge mit >= 50 m kleinstem Nachbarabstand
  --keep-intermediate  # behält Boundary/Clip/Filter-Zwischendateien
"""

import argparse, os, sys, subprocess, json, math, tempfile, shutil, csv
import numpy as np
from shapely.geometry import shape, Polygon, MultiPolygon, Point
from shapely.strtree import STRtree

def run(cmd, **kw):
    p = subprocess.run(cmd, **kw)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")

def haversine_m(lon1, lat1, lon2, lat2):
    R = 6371000.0
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2-lat1); dlmb = math.radians(lon2-lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlmb/2)**2
    return 2*R*math.asin(math.sqrt(a))

def compute_isolation(geojson_path, out_prefix, min_distance=0.0, limit=0):
    with open(geojson_path, "r", encoding="utf-8") as f:
        gj = json.load(f)
    feats = gj.get("features", [])

    # nur Gebäude mit addr:housenumber
    items = []
    for i, feat in enumerate(feats):
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
        osm_id = props.get("osm_id", f"feat_{i}")
        items.append((float(c.x), float(c.y), osm_id))

    if len(items) < 2:
        print("Zu wenige Gebäude mit addr:housenumber.", file=sys.stderr)
        return 0, None, None

    points = [Point(lon, lat) for lon, lat, _ in items]
    tree = STRtree(points)
    idx_map = {id(pt): i for i, pt in enumerate(points)}

    results = []
    for i, pt in enumerate(points):
        lon0, lat0, osm0 = items[i]
        nearest = None
        # progressive BBox-Radien (m)
        for r in (100, 250, 500, 1000, 2000, 5000, 10000):
            ddeg = r / 111000.0
            env = pt.buffer(ddeg).envelope
            cand = [idx_map[id(p)] for p in tree.query(env)]
            # exakte Haversine
            for j in cand:
                if j == i: 
                    continue
                lon1, lat1, osm1 = items[j]
                d = haversine_m(lon0, lat0, lon1, lat1)
                if nearest is None or d < nearest[0]:
                    nearest = (d, j)
            if nearest is not None:
                break
        if nearest is None:
            continue
        d, j = nearest
        results.append({
            "osm_id": osm0,
            "lon": lon0, "lat": lat0,
            "nearest_osm_id": items[j][2],
            "nearest_dist_m": float(d)
        })

    # sortiert (größte Distanz zuerst) = Hitliste
    results.sort(key=lambda r: r["nearest_dist_m"], reverse=True)
    if min_distance > 0:
        results = [r for r in results if r["nearest_dist_m"] >= min_distance]
    if limit and limit > 0:
        results = results[:limit]

    csv_path = out_prefix + "_isolated.csv"
    html_path = out_prefix + "_isolated.html"

    os.makedirs(os.path.dirname(out_prefix), exist_ok=True) if os.path.dirname(out_prefix) else None

    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["osm_id","nearest_osm_id","nearest_dist_m","osm_url","nearest_osm_url","lon","lat"])
        for r in results:
            url0 = f"https://www.openstreetmap.org/way/{r['osm_id']}" if str(r["osm_id"]).isdigit() else ""
            url1 = f"https://www.openstreetmap.org/way/{r['nearest_osm_id']}" if str(r["nearest_osm_id"]).isdigit() else ""
            w.writerow([r["osm_id"], r["nearest_osm_id"], f"{r['nearest_dist_m']:.2f}", url0, url1, r["lon"], r["lat"]])

    with open(html_path, "w", encoding="utf-8") as f:
        f.write("<!doctype html><meta charset='utf-8'><title>Isolated buildings</title>")
        f.write("<style>body{font-family:system-ui,Arial;margin:20px} a{display:block;margin:4px 0}</style>")
        f.write(f"<h1>Alleinstehendste Gebäude</h1><p>n={len(results)}</p>")
        for r in results:
            url0 = f"https://www.openstreetmap.org/way/{r['osm_id']}" if str(r['osm_id']).isdigit() else ""
            f.write(f"<a href='{url0}' target='_blank' rel='noreferrer noopener'>{url0}</a> – Abstand: {r['nearest_dist_m']:.1f} m<br/>\n")

    return len(results), csv_path, html_path

def main():
    ap = argparse.ArgumentParser(description="Ein-Befehl-Pipeline zur Hitliste isolierter Gebäude aus einer PBF + Boundary-Relation.")
    ap.add_argument("--pbf", required=True, help="Eingabe-PBF (z. B. nordrhein-westfalen-latest.osm.pbf)")
    ap.add_argument("--relation", required=True, help="OSM Relations-ID der Boundary (z. B. 62644)")
    ap.add_argument("--out", required=True, help="Ausgabepräfix (z. B. out/roesrath)")
    ap.add_argument("--limit", type=int, default=0, help="Top-K Ergebnisse (0 = alle)")
    ap.add_argument("--min-distance", type=float, default=0.0, help="nur Ergebnisse mit >= dieser Distanz (m)")
    ap.add_argument("--keep-intermediate", action="store_true", help="Zwischendateien behalten")
    args = ap.parse_args()

    # Checks
    for binname in ("osmium",):
        if shutil.which(binname) is None:
            print("Fehlt: 'osmium' (osmium-tool). Installation erforderlich.", file=sys.stderr)
            sys.exit(2)

    tmpdir = tempfile.mkdtemp(prefix="osm_iso_")
    try:
        pbf = os.path.abspath(args.pbf)
        rel = str(args.relation).lstrip("r")

        b_pbf = os.path.join(tmpdir, "boundary.osm.pbf")
        b_geo = os.path.join(tmpdir, "boundary.geojson")
        clip_pbf = os.path.join(tmpdir, "clip.pbf")
        bld_pbf = os.path.join(tmpdir, "buildings.pbf")
        addr_pbf = os.path.join(tmpdir, "buildings_addr.pbf")
        addr_geo = os.path.join(tmpdir, "buildings_addr.geojson")

        # 1) Boundary aus derselben PBF
        run(["osmium", "getid", "-r", pbf, f"r{rel}", "-o", b_pbf, "-O"])
        run(["osmium", "export", b_pbf, "-o", b_geo, "-O"])

        # 2) Clip
        run(["osmium", "extract", "-p", b_geo, pbf, "-o", clip_pbf, "-O"])

        # 3) building + addr:housenumber filtern (Streaming)
        p1 = subprocess.Popen(["osmium", "tags-filter", clip_pbf, "w/building", "-O", "-o", "-"], stdout=subprocess.PIPE)
        p2 = subprocess.Popen(["osmium", "tags-filter", "-", "w/addr:housenumber", "-O", "-o", addr_pbf], stdin=p1.stdout)
        p1.stdout.close()
        rc2 = p2.wait()
        if rc2 != 0:
            raise RuntimeError("tags-filter pipeline failed")

        # 4) Export GeoJSON
        run(["osmium", "export", addr_pbf, "-o", addr_geo, "-O"])

        # 5) Hitliste berechnen
        n, csv_path, html_path = compute_isolation(addr_geo, args.out, args.min_distance, args.limit)
        print(f"Treffer: {n}")
        print(f"CSV:  {csv_path}")
        print(f"HTML: {html_path}")

        if args.keep_intermediate:
            outdir = os.path.dirname(os.path.abspath(args.out)) or "."
            for src in (b_pbf, b_geo, clip_pbf, bld_pbf, addr_pbf, addr_geo):
                if os.path.exists(src):
                    shutil.copy(src, outdir)
            print(f"Zwischendateien in {outdir} kopiert.")
    finally:
        if not args.keep_intermediate:
            shutil.rmtree(tmpdir, ignore_errors=True)

if __name__ == "__main__":
    main()