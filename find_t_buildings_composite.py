#!/usr/bin/env python3
"""
find_t_buildings_composite.py

Erkennt T-förmige Gebäude in OSM-Daten (GeoJSON / Shapefile / GPKG).
Unterstützt das Zusammenführen benachbarter Footprints (z. B. Haus + Garage).

Nutzung:
    python3 find_t_buildings_composite.py buildings.geojson \
        --out out/koeln_composite \
        --ref-osm-id 129725545 \
        --merge-distance 2.0 \
        --gap-close 0.5
"""

import argparse, math, json, csv, os
import numpy as np
from shapely.geometry import shape, Polygon, MultiPolygon, LineString, MultiLineString
from shapely.affinity import rotate
from shapely.ops import unary_union
import fiona


def edge_angles(poly: Polygon):
    coords = list(poly.exterior.coords)
    angs = []
    for i in range(len(coords)-1):
        x1,y1 = coords[i]
        x2,y2 = coords[i+1]
        ang = math.degrees(math.atan2(y2-y1, x2-x1)) % 180
        angs.append(ang)
    return angs

def is_orthogonal(poly: Polygon, tol=12):
    angs = edge_angles(poly)
    if not angs:
        return False
    base = sorted(angs)[len(angs)//2]
    def near_right(a,b):
        d = abs((a-b+90) % 180 - 90)
        return (d < tol) or (abs(d-90) < tol)
    return all(near_right(a, base) for a in angs)

def reflex_count(poly: Polygon):
    coords = list(poly.exterior.coords)
    cnt = 0
    for i in range(1, len(coords)-1):
        x1,y1 = coords[i-1]
        x2,y2 = coords[i]
        x3,y3 = coords[i+1]
        ang1 = math.atan2(y1-y2, x1-x2)
        ang2 = math.atan2(y3-y2, x3-x2)
        diff = (ang2-ang1) % (2*math.pi)
        if diff > math.pi:  # Reflexwinkel
            cnt += 1
    return cnt

def slice_width(poly: Polygon, y_abs: float):
    minx, miny, maxx, maxy = poly.bounds
    seg = LineString([(minx-1, y_abs), (maxx+1, y_abs)])
    inter = poly.intersection(seg)
    total = 0.0
    if inter.is_empty:
        return 0.0
    if isinstance(inter, LineString):
        xs = [p[0] for p in inter.coords]
        total = abs(xs[-1] - xs[0])
    elif isinstance(inter, MultiLineString):
        for ln in inter.geoms:
            xs = [p[0] for p in ln.coords]
            total += abs(xs[-1] - xs[0])
    return float(total)

def width_profile(poly: Polygon, n=25):
    angs = edge_angles(poly)
    base = sorted(angs)[len(angs)//2] if angs else 0.0
    pR = rotate(poly, -base, origin='centroid', use_radians=False)
    minx, miny, maxx, maxy = pR.bounds
    ys = np.linspace(miny+0.07*(maxy-miny), maxy-0.07*(maxy-miny), n)
    widths = np.array([slice_width(pR, y) for y in ys])
    if widths.max() <= 0:
        return None
    return widths / widths.max()

def t_score(poly: Polygon, ref_prof=None,
            min_top_ratio=1.8, mid_stem_max=1.4,
            center_max=0.25, bar_frac_max=0.45,
            reflex_min=2, reflex_max=4,
            ortho_tol=12):
    if poly.area < 1e-9:
        return None

    if not is_orthogonal(poly, tol=ortho_tol):
        return None
    rcount = reflex_count(poly)
    if not (reflex_min <= rcount <= reflex_max):
        return None

    prof = width_profile(poly, n=25)
    if prof is None:
        return None

    bottom = prof[:len(prof)//3].mean()
    mid    = prof[len(prof)//3:2*len(prof)//3].mean()
    top    = prof[2*len(prof)//3:].mean()

    if bottom <= 0: 
        return None
    if top < bottom:  # Querbalken muss oben sein
        prof = prof[::-1]
        bottom = prof[:len(prof)//3].mean()
        mid    = prof[len(prof)//3:2*len(prof)//3].mean()
        top    = prof[2*len(prof)//3:].mean()
    ratio = top / bottom
    if ratio < min_top_ratio: 
        return None
    if max(mid, bottom)/min(mid, bottom) > mid_stem_max:
        return None

    # Querbalken-Dicke
    bar_frac = (prof > 1.6*bottom).mean()
    if bar_frac > bar_frac_max:
        return None

    # Symmetrie: Stamm unter Querbalken
    angs = edge_angles(poly)
    base = sorted(angs)[len(angs)//2] if angs else 0.0
    pR = rotate(poly, -base, origin='centroid', use_radians=False)
    minx, miny, maxx, maxy = pR.bounds
    stem_seg = LineString([(pR.centroid.x, miny-1), (pR.centroid.x, miny+(maxy-miny)/3)])
    inter = pR.intersection(stem_seg)
    if inter.is_empty:
        return None
    center_dev = abs(pR.centroid.x - (minx+maxx)/2) / (maxx-minx)
    if center_dev > center_max:
        return None

    corr = None
    if ref_prof is not None:
        corr = float(np.corrcoef(ref_prof, prof)[0,1])
        if corr < 0.8:
            return None

    return {"ratio": ratio, "bar_frac": bar_frac, "center_dev": center_dev, "corr": corr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile", help="Eingabedatei (GeoJSON, Shapefile, GPKG)")
    ap.add_argument("--out", required=True, help="Output-Basename")
    ap.add_argument("--ref-osm-id", type=int, help="Referenz-OSM-ID zur Profilableitung")
    ap.add_argument("--merge-distance", type=float, default=0.0, help="Cluster-Mergeschwelle in Metern")
    ap.add_argument("--gap-close", type=float, default=0.0, help="Morphologisches Closing")
    ap.add_argument("--min-part-area", type=float, default=0.0, help="Kleinflächen ignorieren (m²)")
    args = ap.parse_args()

    # Input laden
    feats = []
    with fiona.open(args.infile, "r") as src:
        for feat in src:
            geom = shape(feat["geometry"])
            props = dict(feat["properties"])
            feats.append((geom, props))

    # Referenzprofil
    ref_prof = None
    if args.ref_osm_id:
        for geom, props in feats:
            if props.get("osm_id") == args.ref_osm_id:
                if geom.geom_type == "Polygon":
                    ref_prof = width_profile(geom)
                elif geom.geom_type == "MultiPolygon":
                    ref_prof = width_profile(max(geom.geoms, key=lambda p: p.area))
                break

    results = []

    for geom, props in feats:
        polys = []
        if geom.geom_type == "Polygon":
            polys = [geom]
        elif geom.geom_type == "MultiPolygon":
            polys = list(geom.geoms)

        # Filter kleine Teile
        if args.min_part_area > 0:
            polys = [p for p in polys if p.area >= args.min_part_area]

        if not polys:
            continue

        # Composite
        if args.merge_distance > 0:
            buffered = [p.buffer(args.merge_distance/2) for p in polys]
            merged = unary_union(buffered).buffer(-args.merge_distance/2)
        else:
            merged = unary_union(polys)

        if args.gap_close > 0:
            merged = merged.buffer(args.gap_close).buffer(-args.gap_close)

        if merged.geom_type == "Polygon":
            candidates = [merged]
        elif merged.geom_type == "MultiPolygon":
            candidates = list(merged.geoms)
        else:
            continue

        for p in candidates:
            score = t_score(p, ref_prof=ref_prof)
            if score:
                results.append({
                    "osm_id": props.get("osm_id"),
                    "score": score
                })

    # Ausgabe
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    csvfile = args.out + "_t_matches.csv"
    with open(csvfile, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["osm_id","ratio","bar_frac","center_dev","corr","url"])
        for r in results:
            url = f"https://www.openstreetmap.org/way/{r['osm_id']}" if r["osm_id"] else ""
            sc = r["score"]
            w.writerow([r["osm_id"], sc["ratio"], sc["bar_frac"], sc["center_dev"], sc["corr"], url])

    htmlfile = args.out + "_t_links.html"
    with open(htmlfile, "w", encoding="utf-8") as f:
        f.write("<!doctype html><meta charset='utf-8'><title>T-Gebäude Links</title>")
        f.write("<style>body{font-family:system-ui,Arial;margin:20px} a{display:block;margin:4px 0}</style>")
        f.write(f"<h1>T-Gebäude</h1><p>Anzahl: {len(results)}</p>")
        for r in results:
            url = f"https://www.openstreetmap.org/way/{r['osm_id']}" if r["osm_id"] else ""
            f.write(f"<a href='{url}' target='_blank'>{url}</a><br/>\n")

    print(f"Fertig. Treffer: {len(results)}")
    print(f"CSV:  {csvfile}")
    print(f"HTML: {htmlfile}")


if __name__ == "__main__":
    main()
