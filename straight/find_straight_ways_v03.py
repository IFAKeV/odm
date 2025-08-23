#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, math
from collections import defaultdict, deque
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, Point
from shapely.ops import unary_union
from pyrosm import OSM

# ------------------ Geometrie-Helfer ------------------

def bearing_unoriented(p0: Point, p1: Point) -> float:
    dx, dy = (p1.x - p0.x), (p1.y - p0.y)
    ang = math.degrees(math.atan2(dy, dx))
    if ang < 0: ang += 360.0
    return ang if ang < 180.0 else ang - 180.0

def ang_diff(a: float, b: float) -> float:
    d = abs(a - b)
    return d if d <= 90.0 else 180.0 - d

def max_perp_dev_xy(coords: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    nrm = math.hypot(ab[0], ab[1])
    if nrm == 0.0:
        return 0.0
    ap = coords - a
    # |x1*y2 - y1*x2| / |ab|
    cross = np.abs(ap[:,0]*ab[1] - ap[:,1]*ab[0])
    return float(np.max(cross / nrm))

def stroke_metrics(ls: LineString) -> tuple[float,float,float]:
    c = np.asarray(ls.coords, dtype=float)
    if c.shape[0] < 2: return 0.0, 0.0, 0.0
    seg = c[1:] - c[:-1]
    L = float(np.sum(np.hypot(seg[:,0], seg[:,1])))
    chord = float(np.hypot(*(c[-1] - c[0])))
    s = chord / L if L > 0 else 0.0
    dev = max_perp_dev_xy(c, c[0], c[-1])
    return L, s, dev

def longest_straight_prefix(ls: LineString, ref_bearing: float,
                            max_dev_m: float, angle_tol: float, min_keep: float) -> tuple[LineString|None,float,float|None,bool]:
    """
    Schneidet von ls (EPSG:3857) den längsten 'geraden' Präfix ab,
    gemessen gegen ref_bearing. Rückgabe:
    (prefix_geom|None, prefix_len_m, new_bearing|None, reached_end: bool)
    """
    coords = np.asarray(ls.coords, dtype=float)
    if len(coords) < 2:
        return None, 0.0, None, True
    best_j = None
    for j in range(1, len(coords)):
        a, b = coords[0], coords[j]
        br = bearing_unoriented(Point(a[0], a[1]), Point(b[0], b[1]))
        if ang_diff(ref_bearing, br) > angle_tol:
            break
        if max_perp_dev_xy(coords[:j+1], a, b) > max_dev_m:
            break
        best_j = j
    if best_j is None:
        return None, 0.0, None, False
    prefix = LineString(coords[:best_j+1])
    Lp = prefix.length
    if Lp < min_keep:
        return None, 0.0, None, False
    new_bearing = bearing_unoriented(Point(prefix.coords[0]), Point(prefix.coords[-1]))
    reached_end = (best_j == len(coords)-1)
    return prefix, Lp, new_bearing, reached_end

# ------------------ Daten & Graph ------------------

def load_edges(pbf_path: str, include_tracks: bool) -> gpd.GeoDataFrame:
    osm = OSM(pbf_path)
    keep = {"unclassified","residential","service","tertiary"}
    if include_tracks:
        keep.add("track")
    gdf = osm.get_data_by_custom_criteria(
        custom_filter={"highway": list(keep)},
        filter_type="keep",
        keep_nodes=True,
        extra_attributes=["highway","surface","tracktype","name"]
    )
    if gdf is None or gdf.empty:
        return gpd.GeoDataFrame(columns=["geometry","highway"], geometry="geometry", crs=4326)
    if include_tracks:
        mask = gdf["highway"].eq("track")
        paved = (
            gdf.get("tracktype", pd.Series(index=gdf.index)).fillna("").str.lower().eq("grade1") |
            gdf.get("surface", pd.Series(index=gdf.index)).fillna("").str.lower().isin(
                ["asphalt","concrete","paving_stones","sett","cobblestone","paved"])
        )
        gdf = gdf[~mask | paved]
    gdf = gdf.set_crs(4326, allow_override=True).to_crs(3857)
    gdf = gdf.explode(index_parts=False, ignore_index=True)
    gdf = gdf[gdf.geometry.type == "LineString"][["geometry","highway"]].copy()
    return gdf

def build_graph(edges: gpd.GeoDataFrame):
    """Graph mit 0.5 m Node-Snap; speichert Geom & Bearing pro Kante."""
    G = nx.Graph()
    def key(pt: Point): return (round(pt.x, 1), round(pt.y, 1))
    bearings, p0s, p1s = [], [], []
    for geom in edges.geometry.values:
        p0 = Point(geom.coords[0]); p1 = Point(geom.coords[-1])
        p0s.append(p0); p1s.append(p1)
        bearings.append(bearing_unoriented(p0, p1))
    edges = edges.copy()
    edges["p0"] = p0s; edges["p1"] = p1s; edges["bearing"] = bearings
    for i, row in edges.iterrows():
        u = key(row.p0); v = key(row.p1)
        G.add_node(u, pt=row.p0); G.add_node(v, pt=row.p1)
        G.add_edge(u, v, idx=i, geom=row.geometry, bearing=row.bearing, hwy=row.highway)
    return G, edges

# ------------------ Stroke-Bildung mit Prefix-Cut ------------------

def best_next_edge(G: nx.Graph, edges: pd.DataFrame, current_node, prev_bearing,
                   used_local: set, angle_tol: float):
    """Wähle Kandidatenkante mit minimaler Δθ; ignoriere bereits genutzte."""
    best = None; best_d = None
    for nbr in G.neighbors(current_node):
        data = G.get_edge_data(current_node, nbr)
        idx = data["idx"]
        if idx in used_local:
            continue
        dtheta = ang_diff(prev_bearing, float(edges.loc[idx, "bearing"]))
        if best is None or dtheta < best_d:
            best = (nbr, idx)
            best_d = dtheta
    if best is None or best_d is None or best_d > angle_tol:
        return None
    return best  # (next_node, edge_idx)

def follow_stroke_with_prefix(G: nx.Graph, edges: pd.DataFrame, start_idx: int,
                              angle_tol: float, max_dev_m: float, min_keep: float,
                              used_global: set) -> LineString|None:
    """Stroke ausgehend von start_idx; pro Schritt nur geraden Präfix anfügen."""
    if start_idx in used_global:
        return None
    # Edge finden
    uv = None; data0 = None
    for u, v, d in G.edges(data=True):
        if d["idx"] == start_idx:
            uv = (u, v); data0 = d; break
    if uv is None:
        return None

    parts = []  # LineString-Teile (in Reihenfolge)
    used_local = set([start_idx])

    # Startkante: gesamte Kante als erster Teil
    first_geom = data0["geom"]
    parts.append(first_geom)
    br0 = float(edges.loc[start_idx, "bearing"])

    # Richtung A: uv[1] weiter
    prev_node, cur_node, prev_bearing = uv[0], uv[1], br0
    while True:
        nxt = best_next_edge(G, edges, cur_node, prev_bearing, used_local, angle_tol)
        if nxt is None: break
        next_node, eidx = nxt
        geom = G.get_edge_data(cur_node, next_node)["geom"]
        # sicherstellen, dass der Linienanfang am aktuellen Knoten liegt
        if Point(geom.coords[0]).distance(G.nodes[cur_node]["pt"]) > Point(geom.coords[-1]).distance(G.nodes[cur_node]["pt"]):
            geom = LineString(list(geom.coords)[::-1])
        prefix, Lp, new_bearing, reached_end = longest_straight_prefix(
            geom, prev_bearing, max_dev_m=max_dev_m, angle_tol=angle_tol, min_keep=min_keep
        )
        if prefix is None:
            # diese Kante taugt nicht → als benutzt markieren, aber Stroke beenden
            used_local.add(eidx)
            break
        parts.append(prefix)
        used_local.add(eidx)
        prev_bearing = new_bearing
        if not reached_end:
            # Kante wurde im Präfix geschnitten → an beginnender Kurve stoppen
            break
        prev_node, cur_node = cur_node, next_node

    # Richtung B: uv[0] zurück
    prev_node, cur_node, prev_bearing = uv[1], uv[0], br0
    left_parts = []
    while True:
        nxt = best_next_edge(G, edges, cur_node, prev_bearing, used_local, angle_tol)
        if nxt is None: break
        next_node, eidx = nxt
        geom = G.get_edge_data(cur_node, next_node)["geom"]
        if Point(geom.coords[0]).distance(G.nodes[cur_node]["pt"]) > Point(geom.coords[-1]).distance(G.nodes[cur_node]["pt"]):
            geom = LineString(list(geom.coords)[::-1])
        prefix, Lp, new_bearing, reached_end = longest_straight_prefix(
            geom, prev_bearing, max_dev_m=max_dev_m, angle_tol=angle_tol, min_keep=min_keep
        )
        if prefix is None:
            used_local.add(eidx)
            break
        left_parts.append(prefix)  # wird später vorn angefügt
        used_local.add(eidx)
        prev_bearing = new_bearing
        if not reached_end:
            break
        prev_node, cur_node = cur_node, next_node

    # zusammensetzen
    if left_parts:
        parts = left_parts[::-1] + parts
    # „Stitch“: einfache Koord-Konkatenation
    coords = []
    for i, seg in enumerate(parts):
        cs = list(seg.coords)
        if i > 0 and coords and coords[-1] == cs[0]:
            coords.extend(cs[1:])
        else:
            coords.extend(cs)
    if len(coords) < 2:
        return None
    stroke = LineString(coords)

    # global markieren (nur vollständige Kanten; Präfix-Cuts werden nicht als eigene Kante geführt)
    for e in used_local:
        used_global.add(e)
    return stroke

# ------------------ Rural-Maske (optional) ------------------

def build_urban_union(osm: OSM) -> gpd.GeoDataFrame:
    polys = []
    places = osm.get_pois(custom_filter={"place": ["city","town","village"]}, filter_type="keep", points=False)
    if places is not None and not places.empty:
        polys.append(places.set_crs(4326, allow_override=True).to_crs(3857).geometry)
    land = osm.get_landuse()
    if land is not None and not land.empty:
        land = land[land["landuse"].isin(["residential","industrial","commercial"])]
        if not land.empty:
            polys.append(land.set_crs(4326, allow_override=True).to_crs(3857).geometry)
    if not polys:
        return gpd.GeoDataFrame({"geometry":[]}, geometry="geometry", crs=3857)
    union = unary_union(pd.concat(polys, ignore_index=True))
    return gpd.GeoDataFrame(geometry=[union], crs=3857)

# ------------------ Main ------------------

def main():
    ap = argparse.ArgumentParser(description="Längste zusammenhängende gerade Straßen (Stroke + Prefix-Cut).")
    ap.add_argument("--pbf", required=True)
    ap.add_argument("--out-prefix", default="straight_v03")
    ap.add_argument("--include-tracks", action="store_true", help="track nur wenn befestigt (grade1/surface=paved)")
    # Stroke-Parameter
    ap.add_argument("--theta-tol", type=float, default=1.2, help="max. Winkelabweichung pro Schritt [°]")
    ap.add_argument("--max-dev", type=float, default=6.0, help="max. Querabweichung zum Chord [m]")
    ap.add_argument("--min-keep", type=float, default=80.0, help="min. Präfixlänge je Schritt [m]")
    ap.add_argument("--min-len", type=float, default=1000.0, help="min. Gesamtlänge Stroke [m]")
    ap.add_argument("--s-min", type=float, default=0.985, help="min. Straightness (Chord/L) des Strokes")
    # Rural
    ap.add_argument("--rural-quot", type=float, default=0.9, help="Mindestanteil außerhalb urbaner Flächen")
    ap.add_argument("--no-rural-filter", action="store_true", help="Rural-Filter deaktivieren")
    ap.add_argument("--topn", type=int, default=100)
    args = ap.parse_args()

    print(f"[INFO] Lade: {args.pbf}")
    edges = load_edges(args.pbf, include_tracks=args.include_tracks)
    if edges.empty:
        print("[WARN] Keine Kanten gefunden."); return

    G, E = build_graph(edges)

    used = set()
    strokes = []
    for idx in E.index:
        if idx in used: continue
        s = follow_stroke_with_prefix(G, E, idx,
                                      angle_tol=args.theta-tol if hasattr(args, 'theta-tol') else args.theta_tol,
                                      max_dev_m=args.max_dev, min_keep=args.min_keep,
                                      used_global=used)
        if s is None: continue
        L, S, D = stroke_metrics(s)
        if L >= args.min_len and S >= args.s_min and D <= args.max_dev:
            strokes.append((s, L, S, D))

    if not strokes:
        print("[INFO] Keine Strokes nach Kriterien."); return

    # Rural-Filter
    if not args.no_rural_filter:
        osm = OSM(args.pbf)
        urb = build_urban_union(osm)
        if not urb.empty and not urb.iloc[0].geometry.is_empty:
            U = urb.iloc[0].geometry
            filtered = []
            for (s, L, S, D) in strokes:
                try:
                    urb_len = s.intersection(U).length
                except Exception:
                    urb_len = 0.0
                rq = max(L - urb_len, 0.0) / L if L > 0 else 0.0
                if rq >= args.rural_quot:
                    filtered.append((s, L, S, D, rq))
            strokes = filtered
        else:
            strokes = [(s,L,S,D,1.0) for (s,L,S,D) in strokes]
    else:
        strokes = [(s,L,S,D,1.0) for (s,L,S,D) in strokes]

    if not strokes:
        print("[INFO] Nach Rural-Filter keine Strokes."); return

    # Ranking & Ausgabe
    strokes.sort(key=lambda t: (t[1], t[2]), reverse=True)
    strokes = strokes[:args.topn]
    out_rows = []
    for i,(s,L,S,D,RQ) in enumerate(strokes, start=1):
        out_rows.append({"rank":i, "length_m":float(L), "straight":float(S),
                         "max_dev_m":float(D), "rural_quot":float(RQ), "geometry":s})
    out = gpd.GeoDataFrame(out_rows, crs=3857).to_crs(4326)

    gj = f"{args.out_prefix}_strokes.geojson"
    out.to_file(gj, driver="GeoJSON")
    print(f"[OK] GeoJSON: {gj}  (Top {len(out)})")

    # Karte
    try:
        import folium
        center = [out.geometry.union_all().centroid.y, out.geometry.union_all().centroid.x]
        m = folium.Map(location=center, zoom_start=11, control_scale=True)
        mx = max(out["length_m"])
        for _, r in out.iterrows():
            w = 3 + 7 * (r["length_m"]/mx)
            folium.GeoJson(
                r.geometry.__geo_interface__,
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