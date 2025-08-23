#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
find_straight_ways_visual_v01g.py

Rein visueller Ansatz: Render aller OSM-highways aus einer .osm.pbf in PNGs mit transparentem Hintergrund.
- pro highway-Typ eine PNG
- zusätzlich eine Gesamt-PNG
- optional Bounding Box (WGS84) und Vereinfachung (Meter)
- Ausgabegröße (Pixel) und DPI steuerbar
- Linien über LineCollection (schnell), Hintergrund vollständig transparent

Beispiel:
python find_straight_ways_visual_v01g.py \
  --pbf zielgebiet.pbf \
  --outdir out_vis \
  --width 8000 --dpi 300 --lw 0.6 \
  --simplify 1.5 \
  --bbox 7.2,52.0,8.8,52.5
"""

import argparse
import os
import sys
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
from pyrosm import OSM
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def parse_bbox(bbox_str: str):
    """minlon,minlat,maxlon,maxlat (WGS84) -> tuple or None"""
    if not bbox_str:
        return None
    parts = [p.strip() for p in bbox_str.split(",")]
    if len(parts) != 4:
        raise SystemExit("--bbox erwartet 'minlon,minlat,maxlon,maxlat' (WGS84)")
    try:
        vals = list(map(float, parts))
    except Exception:
        raise SystemExit("--bbox Konnte nicht in floats konvertieren")
    minlon, minlat, maxlon, maxlat = vals
    if not (minlon < maxlon and minlat < maxlat):
        raise SystemExit("--bbox ungültig (min >= max)")
    return (minlon, minlat, maxlon, maxlat)


def load_highways_gdf(pbf_path: str, bbox=None) -> gpd.GeoDataFrame:
    """
    Lädt highways aus der PBF, projiziert nach EPSG:3857, explodiert MultiLines.
    Gibt GeoDataFrame mit Spalten ['geometry','highway'] zurück.
    """
    osm = OSM(pbf_path, bounding_box=bbox)
    gdf = osm.get_data_by_custom_criteria(
        custom_filter={"highway": True},
        filter_type="keep",
        keep_nodes=False
    )
    if gdf is None or gdf.empty:
        raise SystemExit("Keine highway-Geometrien gefunden.")
    # Nur Geometrie + highway-Spalte weiterführen
    cols = list(gdf.columns)
    if "highway" not in cols:
        # pyrosm sollte 'highway' liefern – zur Sicherheit fallback
        gdf["highway"] = gdf.get("highway", None)
    gdf = gdf[["geometry", "highway"]].copy()

    # Auf metrisch
    gdf = gdf.set_crs(4326, allow_override=True).to_crs(3857)

    # explode
    gdf = gdf.explode(index_parts=False, ignore_index=True)
    # Nur Linien
    gdf = gdf[gdf.geometry.type.isin(["LineString", "MultiLineString"])].copy()

    # MultiLineStrings in einzelne Lines zerteilen
    gdf = gdf.explode(index_parts=False, ignore_index=True)
    gdf = gdf[gdf.geometry.type == "LineString"].copy()

    # Null-/leere Geometrien raus
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()].copy()
    return gdf


def extent_from_gdf(gdf: gpd.GeoDataFrame, margin_frac=0.02):
    minx, miny, maxx, maxy = gdf.total_bounds
    dx, dy = maxx - minx, maxy - miny
    mx, my = dx * margin_frac, dy * margin_frac
    return (minx - mx, maxx + mx, miny - my, maxy + my)


def gdf_to_line_arrays(gdf: gpd.GeoDataFrame):
    """Konvertiert LineStrings → Nx2 float-Arrays für LineCollection."""
    lines = []
    for geom in gdf.geometry.values:
        try:
            arr = np.asarray(geom.coords, dtype=float)
            if arr.shape[0] >= 2:
                lines.append(arr)
        except Exception:
            continue
    return lines


def render_lines_png(lines, extent, out_png, width_px, height_px, dpi, lw, color, alpha):
    """
    Rendert eine Liste von Koordinaten-Arrays (EPSG:3857) in ein transparentes PNG.
    """
    # Figure-Größe aus Pixel und DPI
    fig_w = width_px / dpi
    fig_h = height_px / dpi

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    # Vollflächige Achse ohne Ränder
    ax = plt.axes([0, 0, 1, 1])
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    # Transparenter Hintergrund
    ax.set_facecolor((0, 0, 0, 0))

    if lines:
        lc = LineCollection(lines, colors=color, linewidths=lw, alpha=alpha, antialiased=True)
        ax.add_collection(lc)

    # Speichern – transparent
    fig.savefig(out_png, dpi=dpi, transparent=True, facecolor=(0, 0, 0, 0), edgecolor="none", pad_inches=0)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Render aller OSM-highways aus .pbf in PNGs (pro highway-Typ + Gesamt), transparenter Hintergrund.")
    ap.add_argument("--pbf", required=True, help="Pfad zur .osm.pbf")
    ap.add_argument("--outdir", required=True, help="Zielverzeichnis für PNGs und index.txt")
    ap.add_argument("--bbox", type=str, default=None, help="minlon,minlat,maxlon,maxlat (WGS84), optional")
    ap.add_argument("--simplify", type=float, default=0.0, help="Geometrievereinfachung in Metern (EPSG:3857); 0 = aus")
    ap.add_argument("--min-length", type=float, default=0.0, help="Minimale Linienlänge in Metern (filtert sehr kurze Wege)")
    ap.add_argument("--width", type=int, default=6000, help="Bildbreite in Pixel")
    ap.add_argument("--height", type=int, default=0, help="Bildhöhe in Pixel (0 = automatisch gemäß Extent-Seitenverhältnis)")
    ap.add_argument("--dpi", type=int, default=300, help="DPI fürs Rendering")
    ap.add_argument("--lw", type=float, default=0.6, help="Linienstärke in Pixel")
    ap.add_argument("--color", default="#0a0a0a", help="Linienfarbe (Hex), wirkt auf alle Typen identisch")
    ap.add_argument("--alpha", type=float, default=0.95, help="Linien-Deckkraft")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    bbox = parse_bbox(args.bbox) if args.bbox else None

    print(f"[INFO] Lade highways aus: {args.pbf}")
    gdf = load_highways_gdf(args.pbf, bbox=bbox)

    if args.simplify and args.simplify > 0:
        print(f"[INFO] Vereinfachung: {args.simplify} m")
        gdf["geometry"] = gdf.geometry.simplify(args.simplify, preserve_topology=True)

    if args.min_length and args.min_length > 0:
        gdf["len_m"] = gdf.length
        gdf = gdf[gdf["len_m"] >= args.min_length].copy()
        gdf.drop(columns=["len_m"], inplace=True, errors="ignore")

    if gdf.empty:
        print("[WARN] Keine Linien nach Filtern.")
        sys.exit(0)

    # Extent und Bildgröße
    xmin, ymin, xmax, ymax = gdf.total_bounds
    extent = xmin, xmax, ymin, ymax
    dx = xmax - xmin
    dy = ymax - ymin
    if args.height and args.height > 0:
        width_px = int(args.width)
        height_px = int(args.height)
    else:
        # Höhe aus Aspect ableiten (lange Seite = width)
        aspect = dy / dx if dx > 0 else 1.0
        width_px = int(args.width)
        height_px = max(1, int(round(width_px * aspect)))
    print(f"[INFO] Extent 3857: [{xmin:.1f},{ymin:.1f}] – [{xmax:.1f},{ymax:.1f}]  "
          f"→ Canvas: {width_px}×{height_px}px @ {args.dpi} DPI")

    # highway-Typen sammeln
    types = sorted([t for t in gdf["highway"].dropna().unique().tolist()])
    if not types:
        print("[WARN] Keine highway-Typen gefunden (leere 'highway'-Tags).")
        sys.exit(0)

    # Index für Übersicht
    idx_lines = []
    total_features = len(gdf)

    # Gesamtbild (alle highways)
    print("[INFO] Render Gesamtbild …")
    all_lines = gdf_to_line_arrays(gdf)
    out_all = os.path.join(args.outdir, "highways_all.png")
    render_lines_png(
        lines=all_lines,
        extent=extent,
        out_png=out_all,
        width_px=width_px,
        height_px=height_px,
        dpi=args.dpi,
        lw=args.lw,
        color=args.color,
        alpha=args.alpha
    )
    idx_lines.append(f"all\t{total_features}\t{out_all}")

    # Pro Typ rendern
    for t in types:
        sub = gdf[gdf["highway"] == t]
        if sub.empty:
            continue
        out_png = os.path.join(args.outdir, f"highway_{t}.png")
        print(f"[INFO] Render Typ '{t}'  (n={len(sub)}) → {out_png}")
        lines = gdf_to_line_arrays(sub)
        render_lines_png(
            lines=lines,
            extent=extent,
            out_png=out_png,
            width_px=width_px,
            height_px=height_px,
            dpi=args.dpi,
            lw=args.lw,
            color=args.color,
            alpha=args.alpha
        )
        idx_lines.append(f"{t}\t{len(sub)}\t{out_png}")

    # Index schreiben
    idx_path = os.path.join(args.outdir, "index.txt")
    with open(idx_path, "w", encoding="utf-8") as f:
        f.write("# highway_type\tfeature_count\tpng_path\n")
        f.write("\n".join(idx_lines))
    print(f"[OK] Fertig. Index: {idx_path}")
    print(f"[HINWEIS] Transparenter Hintergrund: PNGs lassen sich verlustfrei übereinanderlegen.")
    print(f"[HINWEIS] Mit --width/--height/--dpi/--lw/--simplify & ggf. --bbox kannst du Maßstab und Detail steuern.")
    

if __name__ == "__main__":
    main()