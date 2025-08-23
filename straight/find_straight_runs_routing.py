#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Finde die längsten zusammenhängenden geraden Straßenstücke in einer .osm.pbf-Region.
Routing-Ansatz:
- baue Kanten-Graph aus OSM (nur ländliche Straßentypen; Tracks nur wenn befestigt)
- "Strokes": an Knoten jeweils die Fortsetzung mit minimaler Winkelabweichung (≤ theta_tol) wählen
- Bewertung pro Stroke: Länge L, Straightness s=Chord/L, max_dev (max. Querdeviation zur Best-Fit-Chord)
- Ausschluss urbaner Flächen: place (city/town/village) + landuse (residential/industrial/commercial)
- Ausgabe: GeoJSON der Top-N Strokes + HTML-Karte

Beispiel:
python find_straight_runs_routing.py --pbf pbf/bremen-latest.osm.pbf \
  --out-prefix bremen_runs --theta-tol 1.2 --min-len 800 --s-min 0.985 --max-dev 6 --rural-quot 0.9 --topn 50
"""

import argparse, math, sys
from collections import defaultdict, deque
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, Point, Polygon, MultiPolygon
from shapely.ops import unary_union
from pyproj import Transformer
from pyrosm import OSM

def bearing_unoriented(p0: Point, p1: Point) -> float:
    dx, dy = (p1.x - p0.x), (p1.y - p0.y)
    ang = math.degrees(math.atan2(dy, dx))
    if ang < 0: ang += 360.0
    return ang if ang < 180.0 else ang - 180.0

def ang_diff(a: float, b: float) -> float:
    d = abs(a - b)
    return d if d <= 90.0 else 180.0 - d

def max_perp_dev_xy(coords_xy: np.ndarray) -> float:
    """Max. senkrechte Abweichung aller Punkte von der bestmöglichen Chord (Start-Ende).
       coords_xy: Nx2 in Meter (projiziert)"""
    if len(coords_xy) < 2:
        return 0.0
    a, b = coords_xy[0], coords_xy[-1]
    ab = b - a
    nrm = math.hypot(ab[0], ab[1])
    if nrm == 0.0:
        return 0.0
    ap = coords_xy - a
    cross = np.abs(ap[:, 0] * ab[1] - ap[:, 1] * ab[0])  # |x1*y2 - y1*x2|
    return float(np.max(cross / nrm))

def stroke_metrics(ls: LineString) -> tuple[float, float, float]:
    """Return (L, straightness, max_dev) – Metrik in EPSG:3857."""
    coords = np.asarray(ls.coords, dtype=float)
    if coords.shape[0] < 2:
        return 0.0, 0.0, 0.0
    # Länge
    seg = coords[1:] - coords[:-1]
    L = float(np.sum(np.hypot(seg[:,0], seg[:,1])))
    # Chord
    chord = float(np.hypot(*(coords[-1] - coords[0])))
    s = chord / L if L > 0 else 0.0
    # Abweichung
    dev = max_perp_dev_xy(coords)
    return L, s, dev

def build_graph(edges_gdf: gpd.GeoDataFrame, theta_tol: float) -> tuple[nx.Graph, pd.DataFrame]:
    """Baue Graph und berechne Bearings + Endpunkte (EPSG:3857)."""
    g = nx.Graph()
    edges = edges_gdf.copy()
    # vorbereiten: Endpunkte + bearing
    def endpts(ls: LineString):
        c = np.asarray(ls.coords, dtype=float)
        return Point(c[0,0], c[0,1]), Point(c[-1,0], c[-1,1])
    P0, P1, BR = [], [], []
    for geom in edges.geometry.values:
        p0, p1 = endpts(geom)
        P0.append(p0); P1.append(p1); BR.append(bearing_unoriented(p0, p1))
    edges["p0"] = P0; edges["p1"] = P1; edges["bearing"] = BR
    # Knoten-Key (leichtes Snapping: 0.5 m)
    def key(pt: Point): return (round(pt.x, 1), round(pt.y, 1))
    # Graph-Kanten anlegen
    for i, row in edges.iterrows():
        u = key(row.p0); v = key(row.p1)
        g.add_node(u, pt=row.p0); g.add_node(v, pt=row.p1)
        g.add_edge(u, v, idx=i, bearing=row.bearing, geom=row.geometry)
    # Adjazenz "candidate next edges" je Kante (per Knoten: min Winkelabweichung)
    # Wir bereiten ein Mapping von Knoten->Liste(Kantenindex) vor.
    node_edges: dict[tuple,float] = defaultdict(list)
    for u, v, data in g.edges(data=True):
        idx = data["idx"]
        node_edges[u].append(idx); node_edges[v].append(idx)
    # Für schnellen Zugriff: edges df by idx
    return g, edges

def follow_stroke(g: nx.Graph, edges: pd.DataFrame, start_idx: int, theta_tol: float, used: set) -> list[int]:
    """Baue Stroke ausgehend von einer Kante, greedy in beide Richtungen. Benutze 'used' zur Duplikatsvermeidung."""
    if start_idx in used:
        return []
    # Hilfsindizes: Karte (u,v)->idx und umgekehrt
    # Kante im Graph finden
    start_edge = None
    for u, v, d in g.edges(data=True):
        if d["idx"] == start_idx:
            start_edge = (u, v, d)
            break
    if start_edge is None:
        return []
    def step(from_node, current_node, prev_bearing, used_local):
        """Wähle am current_node die Fortsetzung mit minimaler Δθ, sofern <= theta_tol und nicht benutzt."""
        best = None; best_dtheta = None
        for nbr in g.neighbors(current_node):
            data = g.get_edge_data(current_node, nbr)
            idx = data["idx"]
            if idx in used_local:  # bereits im Stroke verwendet
                continue
            # Fortsetzung nur, wenn wir *vom current_node weg* gehen
            br = edges.loc[idx, "bearing"]
            dtheta = ang_diff(prev_bearing, br)
            if best is None or dtheta < best_dtheta:
                best = (nbr, idx, br)
                best_dtheta = dtheta
        if best is None or best_dtheta is None or best_dtheta > theta_tol:
            return None
        return best  # (next_node, edge_idx, bearing)

    # initial
    u, v, d = start_edge
    stroke_idxs = [d["idx"]]
    used_local = set([d["idx"]])
    br0 = edges.loc[d["idx"], "bearing"]

    # Richtung 1: von u -> v weiter
    cur_node = v; prev_node = u; prev_bearing = br0
    while True:
        nxt = step(prev_node, cur_node, prev_bearing, used_local)
        if nxt is None: break
        next_node, eidx, br = nxt
        stroke_idxs.append(eidx); used_local.add(eidx)
        prev_node, cur_node, prev_bearing = cur_node, next_node, br

    # Richtung 2: von v -> u weiter
    cur_node = u; prev_node = v; prev_bearing = br0
    while True:
        nxt = step(prev_node, cur_node, prev_bearing, used_local)
        if nxt is None: break
        next_node, eidx, br = nxt
        stroke_idxs.insert(0, eidx)  # vorn anhängen
        used_local.add(eidx)
        prev_node, cur_node, prev_bearing = cur_node, next_node, br

    # global used erst nach erfolgreichem Build markieren
    for i in used_local: used.add(i)
    return stroke_idxs

def edges_to_linestring(edges: pd.DataFrame, idxs: list[int]) -> LineString:
    """Kanten-Geometrien zu einer durchgehenden LineString mergen (in EPSG:3857)."""
    if not idxs:
        return LineString()
    parts = [edges.loc[i, "geometry"] for i in idxs]
    # robustes Mergen (Geometrien sind bereits topologisch verbunden)
    merged = parts[0]
    for seg in parts[1:]:
        if merged.coords[-1] != seg.coords[0]:
            # not perfectly connected – versuchen wir zu verbinden:
            merged = LineString(list(merged.coords) + list(seg.coords))
        else:
            merged = LineString(list(merged.coords) + list(seg.coords)[1:])
    return merged

def rural_mask(osm: OSM) -> gpd.GeoDataFrame:
    """Union der urbanen Flächen (place + landuse), WGS84 -> 3857."""
    polys = []
    # place-Polygone
    places = osm.get_pois(custom_filter={"place": ["city","town","village"]}, filter_type="keep", points=False)
    if places is not None and not places.empty:
        polys.append(places.set_crs(4326, allow_override=True).to_crs(3857).geometry)
    # landuse-Polygone
    land = osm.get_landuse()
    if land is not None and not land.empty:
        land = land[land["landuse"].isin(["residential","industrial","commercial"])]
        if not land.empty:
            polys.append(land.set_crs(4326, allow_override=True).to_crs(3857).geometry)
    if not polys:
        return gpd.GeoDataFrame({"geometry":[]}, geometry="geometry", crs=3857)
    union = gpd.GeoSeries(pd.concat(polys, ignore_index=True), crs=3857)
    return gpd.GeoDataFrame(geometry=[unary_union(union)], crs=3857)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pbf", required=True, help=".osm.pbf Datei")
    ap.add_argument("--out-prefix", default="straight_runs")
    ap.add_argument("--include-tracks", action="store_true", help="track nur wenn befestigt (grade1/surface=paved)")
    ap.add_argument("--theta-tol", type=float, default=1.2, help="max. Winkelabweichung [°] pro Fortsetzung")
    ap.add_argument("--min-len", type=float, default=1000.0, help="Mindestlänge Stroke [m]")
    ap.add_argument("--s-min", type=float, default=0.985, help="min. Straightness (Chord/L)")
    ap.add_argument("--max-dev", type=float, default=6.0, help="max. Querabweichung [m]")
    ap.add_argument("--rural-quot", type=float, default=0.9, help="Mindestanteil Länge außerhalb urbaner Flächen")
    ap.add_argument("--topn", type=int, default=100)
    args = ap.parse_args()

    print(f"[INFO] Lade PBF: {args.pbf}")
    osm = OSM(args.pbf)

    keep = {"unclassified","residential","service","tertiary"}
    if args.include_tracks:
        keep.add("track")

    # 1) Kanten einlesen (nur relevante highway)
    gdf = osm.get_data_by_custom_criteria(
        custom_filter={"highway": list(keep)},
        filter_type="keep",
        keep_nodes=True,
        extra_attributes=["highway","tracktype","surface","name"]
    )
    if gdf is None or gdf.empty:
        print("[WARN] Keine passenden Kanten."); sys.exit(0)

    # Tracks: nur befestigt
    if args.include_tracks:
        mask = gdf["highway"].eq("track")
        paved = (
            gdf.get("tracktype", pd.Series(index=gdf.index)).fillna("").str.lower().eq("grade1") |
            gdf.get("surface", pd.Series(index=gdf.index)).fillna("").str.lower().isin(
                ["asphalt","concrete","paving_stones","sett","cobblestone","paved"])
        )
        gdf = gdf[~mask | paved]

    # Linien isolieren, projizieren
    gdf = gdf.set_crs(4326, allow_override=True).to_crs(3857)
    gdf = gdf.explode(index_parts=False, ignore_index=True)
    gdf = gdf[gdf.geometry.type == "LineString"].copy()
    if gdf.empty:
        print("[WARN] Keine LineStrings."); sys.exit(0)

    # 2) Graph und Kanten-Bearing
    G, edges = build_graph(gdf[["geometry","highway"]].copy(), args.theta_tol)

    # 3) Strokes erzeugen
    used = set()
    strokes = []
    for idx in edges.index:
        if idx in used: continue
        s_idxs = follow_stroke(G, edges, idx, args.theta_tol, used)
        if not s_idxs: continue
        ls = edges_to_linestring(edges, s_idxs)
        L, s, dev = stroke_metrics(ls)
        if L >= args.min_len and s >= args.s_min and dev <= args.max_dev:
            strokes.append((ls, L, s, dev))

    if not strokes:
        print("[INFO] Keine Strokes nach Kriterien."); sys.exit(0)

    # 4) Rural-Filter
    rural = rural_mask(osm)  # 0..1 polys
    if not rural.empty and not rural.iloc[0].geometry.is_empty:
        urban_union = rural.iloc[0].geometry  # Union urbaner Flächen (in 3857)
        filtered = []
        for (ls, L, s, dev) in strokes:
            try:
                urb_len = (ls.intersection(urban_union)).length
            except Exception:
                urb_len = 0.0
            rural_len = max(L - urb_len, 0.0)
            quot = rural_len / L if L > 0 else 0.0
            if quot >= args.rural_quot:
                filtered.append((ls, L, s, dev, quot))
        strokes = filtered
    else:
        strokes = [(ls, L, s, dev, 1.0) for (ls, L, s, dev) in strokes]

    if not strokes:
        print("[INFO] Nach Rural-Filter keine Strokes."); sys.exit(0)

    # 5) Ranking + Ausgabe
    strokes.sort(key=lambda t: (t[1], t[2]), reverse=True)
    strokes = strokes[:args.topn]
    rows = []
    for i, (ls, L, s, dev, rq) in enumerate(strokes, start=1):
        rows.append({
            "rank": i,
            "length_m": float(L),
            "straight": float(s),
            "max_dev_m": float(dev),
            "rural_quot": float(rq),
            "geometry": ls
        })
    out = gpd.GeoDataFrame(rows, crs=3857).to_crs(4326)

    gj = f"{args.out_prefix}_strokes.geojson"
    out.to_file(gj, driver="GeoJSON")
    print(f"[OK] GeoJSON: {gj}  (Top {len(out)})")

    # einfache Karte
    try:
        import folium
        center = [out.geometry.union_all().centroid.y, out.geometry.union_all().centroid.x]
        m = folium.Map(location=center, zoom_start=11, control_scale=True)
        mx = max(out["length_m"])
        for _, r in out.iterrows():
            w = 3 + 7 * (r["length_m"]/mx)
            folium.GeoJson(
                data=r.geometry.__geo_interface__,
                tooltip=f"#{int(r['rank'])} • L={r['length_m']:.0f} m • s={r['straight']:.4f} • dev={r['max_dev_m']:.1f} m • rural={r['rural_quot']:.2f}",
                style_function=(lambda _w=w: {"weight": _w, "opacity": 0.95})
            ).add_to(m)
        folium.LayerControl().add_to(m)
        html = f"{args.out_prefix}_map.html"
        m.save(html)
        print(f"[OK] Karte: {html}")
    except Exception as e:
        print(f"[WARN] Karte nicht erzeugt: {e}")

if __name__ == "__main__":
    main()