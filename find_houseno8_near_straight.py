#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, math
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely.ops import unary_union
from pyrosm import OSM
import networkx as nx

# ------------------ Geometrie-Utilities ------------------

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
    if nrm == 0.0: return 0.0
    ap = coords - a
    cross = np.abs(ap[:,0]*ab[1] - ap[:,1]*ab[0])
    return float(np.max(cross / nrm))

def stroke_metrics(ls: LineString):
    c = np.asarray(ls.coords, dtype=float)
    if c.shape[0] < 2: return 0.0, 0.0, 0.0
    seg = c[1:] - c[:-1]
    L = float(np.sum(np.hypot(seg[:,0], seg[:,1])))
    chord = float(np.hypot(*(c[-1] - c[0])))
    s = chord / L if L > 0 else 0.0
    dev = max_perp_dev_xy(c, c[0], c[-1])
    return L, s, dev

def longest_straight_prefix(ls: LineString, ref_bearing: float,
                            max_dev_m: float, angle_tol: float, min_keep: float):
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

# ------------------ Daten laden ------------------

def load_small_roads(osm: OSM, include_tracks=False) -> gpd.GeoDataFrame:
    keep = {"unclassified","residential","service","tertiary"}
    if include_tracks: keep.add("track")
    gdf = osm.get_data_by_custom_criteria(
        custom_filter={"highway": list(keep)},
        filter_type="keep",
        keep_nodes=True,
        extra_attributes=["highway","surface","tracktype"]
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
    gdf = gdf[gdf.geometry.type == "LineString"][["geometry","highway"]]
    return gdf.reset_index(drop=True)

def load_houseno8_buildings(osm: OSM) -> gpd.GeoDataFrame:
    bld = osm.get_data_by_custom_criteria(
        custom_filter={"building": True, "addr:housenumber": "8"},
        filter_type="keep",
        keep_nodes=False
    )
    if bld is None or bld.empty:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=4326)
    bld = bld.set_crs(4326, allow_override=True).to_crs(3857)
    bld = bld.explode(index_parts=False, ignore_index=True)
    bld = bld[bld.geometry.type.isin(["Polygon","MultiPolygon"])].copy()
    bld["centroid"] = bld.geometry.centroid
    return bld[["geometry","centroid"]].reset_index(drop=True)

def build_urban_union(osm: OSM) -> gpd.GeoSeries:
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
        return gpd.GeoSeries([], crs=3857)
    union = unary_union(pd.concat(polys, ignore_index=True))
    return gpd.GeoSeries([union], crs=3857)

# ------------------ Graph ------------------

def build_graph(edges: gpd.GeoDataFrame):
    G = nx.Graph()
    def key(pt: Point): return (round(pt.x, 1), round(pt.y, 1))  # 0.1 m Snap
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

def follow_straight_prefix(G, E, start_edge_idx, angle_tol, max_dev_m, min_keep_m, max_total_len_m):
    # finde Kante
    uv = None; data0 = None
    for u,v,d in G.edges(data=True):
        if d["idx"] == start_edge_idx: uv=(u,v); data0=d; break
    if uv is None: return None
    stroke_parts = [data0["geom"]]
    br0 = float(E.loc[start_edge_idx,"bearing"])

    def step(dir_uv):
        prev_node, cur_node, prev_bearing = dir_uv
        total_len = 0.0
        parts = []
        used_local = set()
        while total_len < max_total_len_m:
            # beste Anschlusskante
            best=None; best_d=None; best_idx=None; best_geom=None
            for nbr in G.neighbors(cur_node):
                d = G.get_edge_data(cur_node, nbr)
                idx = d["idx"]
                if idx == start_edge_idx or idx in used_local: continue
                dtheta = ang_diff(prev_bearing, float(E.loc[idx,"bearing"]))
                if best is None or dtheta < best_d:
                    best=(nbr, idx); best_d=dtheta; best_geom=d["geom"]
            if best is None or best_d is None or best_d > angle_tol: break
            next_node, eidx = best
            geom = best_geom
            # Richtung korrigieren
            if Point(geom.coords[0]).distance(G.nodes[cur_node]["pt"]) > Point(geom.coords[-1]).distance(G.nodes[cur_node]["pt"]):
                geom = LineString(list(geom.coords)[::-1])
            prefix, Lp, new_bearing, reached_end = longest_straight_prefix(
                geom, prev_bearing, max_dev_m=max_dev_m, angle_tol=angle_tol, min_keep=min_keep_m
            )
            used_local.add(eidx)
            if prefix is None: break
            parts.append(prefix); total_len += Lp
            prev_bearing = new_bearing
            if not reached_end: break
            prev_node, cur_node = cur_node, next_node
        return parts

    # beide Richtungen
    right = step((uv[0], uv[1], br0))
    left  = step((uv[1], uv[0], br0))
    if left: stroke_parts = left[::-1] + stroke_parts
    if right: stroke_parts = stroke_parts + right

    # stitch
    coords=[]
    for i, seg in enumerate(stroke_parts):
        cs=list(seg.coords)
        if i>0 and coords and coords[-1]==cs[0]: coords.extend(cs[1:])
        else: coords.extend(cs)
    if len(coords)<2: return None
    return LineString(coords)

# ------------------ Pipeline ------------------

def main():
    ap = argparse.ArgumentParser(description="Finde Hausnummer 8 ländlich, 50–200 m von kleiner Straße, mit geradem Straßenabschnitt.")
    ap.add_argument("--pbf", required=True)
    ap.add_argument("--include-tracks", action="store_true", help="tracks nur wenn befestigt")
    ap.add_argument("--dmin", type=float, default=50.0, help="Mindestabstand Haus->Straße (m)")
    ap.add_argument("--dmax", type=float, default=200.0, help="Maxabstand Haus->Straße (m)")
    ap.add_argument("--theta", type=float, default=1.2, help="Winkel-Toleranz je Schritt (°)")
    ap.add_argument("--maxdev", type=float, default=6.0, help="max. Querabweichung zum Chord (m)")
    ap.add_argument("--stepmin", type=float, default=80.0, help="min. Präfix je Schritt (m)")
    ap.add_argument("--winlen", type=float, default=1500.0, help="max. Gesamtlänge des verfolgten Strokes (m)")
    ap.add_argument("--minL", type=float, default=800.0, help="min. Stroke-Länge (m)")
    ap.add_argument("--minS", type=float, default=0.985, help="min. Straightness (Chord/L)")
    ap.add_argument("--rural-quot", type=float, default=0.9, help="Mindestanteil außerhalb urbaner Flächen")
    ap.add_argument("--topn", type=int, default=100)
    ap.add_argument("--out", default="houseno8_candidates")
    args = ap.parse_args()

    osm = OSM(args.pbf)
    roads = load_small_roads(osm, include_tracks=args.include_tracks)
    houses = load_houseno8_buildings(osm)
    if roads.empty or houses.empty:
        print("[INFO] Keine passenden Daten."); return

    # Urban union
    urb = build_urban_union(osm)
    if not urb.empty and not urb.iloc[0].is_empty:
        U = urb.iloc[0]
        houses["urban_len"] = houses.geometry.intersection(U).area
        houses = houses[houses["urban_len"] == 0].drop(columns=["urban_len"])

    # Nearest small road & Distanz
    roads_sindex = roads.sindex
    dmin, dmax = args.dmin, args.dmax
    sel_rows = []
    for i, r in houses.iterrows():
        c = r["centroid"]
        # Nachbarn in 300 m Box vorfiltern
        cand_idx = list(roads_sindex.query(c.buffer(dmax).bounds))
        if not cand_idx: continue
        dists = roads.iloc[cand_idx].geometry.distance(c)
        j = int(np.argmin(dists))
        dist = float(dists.iloc[j])
        if dist < dmin or dist > dmax:
            continue
        sel_rows.append((i, cand_idx[j], dist))

    if not sel_rows:
        print("[INFO] Keine Hausnummer 8 in Distanzfenster."); return

    # Graph bauen
    G, E = build_graph(roads)

    # Für jeden Kandidaten: Stroke um die nächstgelegene Kante
    cand = []
    for (ih, ie, dist) in sel_rows:
        stroke = follow_straight_prefix(G, E, ie,
                                        angle_tol=args.theta, max_dev_m=args.maxdev,
                                        min_keep_m=args.stepmin, max_total_len_m=args.winlen)
        if stroke is None: continue
        L, S, D = stroke_metrics(stroke)
        if L < args.minL or S < args.minS or D > args.maxdev:
            continue
        # Rural-Quote entlang Stroke
        try:
            urb_len = stroke.intersection(urb.iloc[0]).length if not urb.empty else 0.0
        except Exception:
            urb_len = 0.0
        rq = max(L - urb_len, 0.0) / L if L > 0 else 1.0
        if rq < args.rural_quot: continue
        cand.append((ih, ie, dist, stroke, L, S, D, rq))

    if not cand:
        print("[INFO] Keine Kandidaten nach Kriterien."); return

    # Top-Ranking nach L, dann S
    cand.sort(key=lambda t: (t[4], t[5]), reverse=True)
    cand = cand[:args.topn]

    # Ausgabe GDFs (WGS84)
    g_strokes = []
    g_points  = []
    for rank,(ih,ie,dist,stroke,L,S,D,RQ) in enumerate(cand, start=1):
        g_strokes.append({"rank":rank,"length_m":L,"straight":S,"max_dev_m":D,"rural_quot":RQ,
                          "dist_house_m":dist,"geometry":stroke})
        g_points.append({"rank":rank,"geometry":houses.iloc[ih]["centroid"]})
    gdf_st = gpd.GeoDataFrame(g_strokes, crs=3857).to_crs(4326)
    gdf_pt = gpd.GeoDataFrame(g_points,  crs=3857).to_crs(4326)

    stem = args.out
    gdf_st.to_file(f"{stem}_strokes.geojson", driver="GeoJSON")
    gdf_pt.to_file(f"{stem}_houses.geojson", driver="GeoJSON")
    gdf_st.drop(columns="geometry").to_csv(f"{stem}_strokes.csv", index=False)

    print(f"[OK] {len(gdf_st)} Kandidaten → {stem}_strokes.geojson / {stem}_houses.geojson / {stem}_strokes.csv")

    # Folium-Karte
    try:
        import folium
        center = [gdf_pt.geometry.unary_union.centroid.y, gdf_pt.geometry.unary_union.centroid.x]
        m = folium.Map(location=center, zoom_start=11, control_scale=True)
        folium.GeoJson(gdf_st.__geo_interface__,
                       name="gerade Strokes",
                       style_function=lambda f: {"color":"#155e75","weight":6,"opacity":0.95}).add_to(m)
        folium.GeoJson(gdf_pt.__geo_interface__,
                       name="Haus-Nr 8",
                       marker=folium.CircleMarker(radius=5, color="#dc2626", fill=True, fill_opacity=0.9)).add_to(m)
        folium.LayerControl().add_to(m)
        m.save(f"{stem}_map.html")
        print(f"[OK] Karte: {stem}_map.html")
    except Exception as e:
        print(f"[WARN] Folium-Map nicht erstellt: {e}")

if __name__ == "__main__":
    main()
