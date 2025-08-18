#!/usr/bin/env python3
"""Locate buildings by house number and detect straight road segments.

This script combines the functionality of :mod:`find_housenumber_direction`
and a straight-line detector that merges collinear way segments.  It always
runs **both** analyses on the same input PBF file and writes a single
interactive HTML map showing

* buildings with a target ``addr:housenumber`` north/south/east/west of a
  road of a given highway classification and their distance to that road,
* long and straight road segments whose course can continue across changes
  in ``highway`` classification but remain nearly collinear.

The HTML map contains separate layers for roads (optional), houses and straight
segments and can be viewed in any web browser.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import pyproj

from find_housenumber_direction import (
    Building,
    find_buildings,
    prefilter_pbf,
)
from find_straight_ways_v06 import WayCollector, folium

import math
import numpy as np
import networkx as nx
from shapely.geometry import LineString, Point


def bearing180(p0: Point, p1: Point) -> float:
    ang = math.degrees(math.atan2(p1.y - p0.y, p1.x - p0.x))
    if ang < 0:
        ang += 360.0
    return ang if ang < 180.0 else ang - 180.0


def ang_diff(a: float, b: float) -> float:
    d = abs(a - b)
    return d if d <= 90.0 else 180.0 - d


def max_perp_dev(coords: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    n = math.hypot(ab[0], ab[1])
    if n == 0:
        return 0.0
    ap = coords - a
    return float(np.max(np.abs(ap[:, 0] * ab[1] - ap[:, 1] * ab[0]) / n))


def stitch_lines(lines: list[LineString]) -> LineString | None:
    coords = []
    for i, ls in enumerate(lines):
        cs = list(ls.coords)
        if i and coords and coords[-1] == cs[0]:
            coords.extend(cs[1:])
        else:
            coords.extend(cs)
    return LineString(coords) if len(coords) >= 2 else None


def grow_from_edge(
    G: nx.Graph,
    edge_lookup: dict[int, tuple],
    start_idx: int,
    theta: float = 1.0,
    dmax: float = 6.0,
    step_min: float = 60.0,
    max_total: float = 5000.0,
) -> LineString | None:
    """Baue maximale kollineare Kette um eine Startkante (beidseitig)."""

    try:
        uv = edge_lookup[start_idx][:2]
        e0 = edge_lookup[start_idx][2]
    except KeyError:
        return None

    def forward(u, v, br_ref):
        parts = []
        cur = v
        br = br_ref
        total = 0.0
        used = {start_idx}
        acc_coords: list[tuple[float, float]] = []
        while total < max_total:
            best = None
            best_d = None
            best_e = None
            for nbr in G.neighbors(cur):
                d = G.get_edge_data(cur, nbr)
                idx = d["idx"]
                if idx in used:
                    continue
                cand_br = d["bearing"]
                dth = ang_diff(br, cand_br)
                if best_d is None or dth < best_d:
                    best = (cur, nbr)
                    best_d = dth
                    best_e = d
            if best_e is None or best_d is None or best_d > theta:
                break
            used.add(best_e["idx"])

            geom = best_e["geom"]
            if Point(geom.coords[0]).distance(G.nodes[cur]["pt"]) > Point(
                geom.coords[-1]
            ).distance(G.nodes[cur]["pt"]):
                coords = list(geom.coords)[::-1]
            else:
                coords = list(geom.coords)
            if not acc_coords:
                acc_coords = coords
            else:
                acc_coords.extend(coords[1:])
            c = np.asarray(acc_coords, float)
            dev = max_perp_dev(c, c[0], c[-1])
            if dev > dmax:
                break
            seg_len = LineString(coords).length
            if seg_len < step_min:
                break
            parts.append(LineString(coords))
            total += seg_len
            br = best_e["bearing"]
            cur = best[1]
        return parts

    base = e0["geom"]
    p0, p1 = Point(base.coords[0]), Point(base.coords[-1])
    br0 = e0["bearing"]

    left = forward(uv[1], uv[0], br0)
    right = forward(uv[0], uv[1], br0)

    chain = []
    if left:
        chain.extend([LineString(list(ls.coords)[::-1]) for ls in left][::-1])
    chain.append(base)
    if right:
        chain.extend(right)

    return stitch_lines(chain)


def extract_straight_chains(
    segments,
    min_length: float,
    min_straightness: float,
    road_type: str,
    theta: float,
    dmax: float,
) -> list[dict]:
    """Merge collinear segments and evaluate their straightness."""

    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326", "EPSG:3857", always_xy=True
    )
    transformer_inv = pyproj.Transformer.from_crs(
        "EPSG:3857", "EPSG:4326", always_xy=True
    )
    G = nx.Graph()
    edges: dict[int, dict] = {}
    for idx, seg in enumerate(segments):
        lats = [lat for lat, _ in seg.geometry]
        lons = [lon for _, lon in seg.geometry]
        xs, ys = transformer.transform(lons, lats)
        coords = list(zip(xs, ys))
        geom = LineString(coords)
        p0 = Point(coords[0])
        p1 = Point(coords[-1])
        G.add_node((p0.x, p0.y), pt=p0)
        G.add_node((p1.x, p1.y), pt=p1)
        G.add_edge(
            (p0.x, p0.y),
            (p1.x, p1.y),
            idx=idx,
            geom=geom,
            bearing=bearing180(p0, p1),
            highway=seg.highway,
            name=seg.name,
        )
        edges[idx] = {
            "geom": geom,
            "bearing": bearing180(p0, p1),
            "highway": seg.highway,
            "name": seg.name,
        }

    edge_lookup = {
        d["idx"]: (u, v, d) for u, v, d in G.edges(data=True)
    }

    seen: set[tuple[float, float, float, float]] = set()
    candidates: list[dict] = []
    for idx, data in edges.items():
        if data["highway"] != road_type:
            continue
        line = grow_from_edge(G, edge_lookup, idx, theta=theta, dmax=dmax)
        if line is None:
            continue
        coords = np.asarray(line.coords, float)
        h = (
            round(coords[0][0], 1),
            round(coords[0][1], 1),
            round(coords[-1][0], 1),
            round(coords[-1][1], 1),
        )
        if h in seen:
            continue
        seen.add(h)
        xs, ys = coords[:, 0], coords[:, 1]
        lons, lats = transformer_inv.transform(xs, ys)
        path = list(zip(lats, lons))
        length = line.length
        chord = math.hypot(coords[-1][0] - coords[0][0], coords[-1][1] - coords[0][1])
        if length >= min_length and chord > 0:
            straight = chord / length
            if straight >= min_straightness:
                candidates.append(
                    {
                        "id": idx,
                        "length_m": length,
                        "straightness": straight,
                        "geometry": path,
                        "highway": data["highway"],
                        **({"name": data["name"]} if data["name"] else {}),
                    }
                )
    candidates.sort(key=lambda x: x["length_m"], reverse=True)
    return candidates


def create_combined_map(
    buildings: Iterable[Building],
    roads: Iterable[Sequence[Tuple[float, float]]],
    straight_segments: Iterable[dict],
    out: Path,
    road_type: str,
    show_roads: bool,
) -> None:
    """Write an HTML map containing buildings, roads and straight segments."""

    buildings = list(buildings)
    segments = list(straight_segments)
    if buildings:
        center_lat = sum(b.lat for b in buildings) / len(buildings)
        center_lon = sum(b.lon for b in buildings) / len(buildings)
    elif segments:
        center_lat, center_lon = segments[0]["geometry"][0]
    else:
        raise ValueError("no buildings or straight segments to plot")

    m = folium.Map(location=[center_lat, center_lon])

    if show_roads:
        road_group = folium.FeatureGroup(name=f"{road_type} roads")
        road_color = "green" if road_type == "unclassified" else "gray"
        for road in roads:
            folium.PolyLine(
                [(lat, lon) for lon, lat in road], color=road_color, weight=2
            ).add_to(road_group)
        road_group.add_to(m)

    b_group = folium.FeatureGroup(name="houses")
    for b in buildings:
        folium.CircleMarker(
            location=[b.lat, b.lon],
            radius=4,
            weight=1,
            color="red",
            fill=True,
            fill_opacity=0.8,
            popup=f"{b.distance:.1f} m",
        ).add_to(b_group)
    b_group.add_to(m)

    s_group = folium.FeatureGroup(name="straight segments")
    for c in segments:
        folium.PolyLine(
            c["geometry"], color="blue", tooltip=f"Segment {c['id']}"
        ).add_to(s_group)
    s_group.add_to(m)

    folium.LayerControl().add_to(m)
    m.save(str(out))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pbf", help="Path to OSM PBF file")
    parser.add_argument("--out", default="analysis.html", help="Output HTML file")

    # parameters for housenumber search
    parser.add_argument("--min-dist", type=float, default=50.0, help="Minimum distance in metres")
    parser.add_argument("--max-dist", type=float, default=150.0, help="Maximum distance in metres")
    parser.add_argument("--other-housenumber", default="6", help="Reference addr:housenumber")
    parser.add_argument(
        "--min-hn-dist", type=float, default=100.0, help="Minimum distance to reference house in metres"
    )
    parser.add_argument(
        "--max-hn-dist", type=float, default=300.0, help="Maximum distance to reference house in metres"
    )
    parser.add_argument("--prefilter", action="store_true", help="Run osmium tags-filter before processing")
    parser.add_argument("--processes", type=int, default=None, help="Number of worker processes")
    parser.add_argument("--housenumber", default="8", help="Target addr:housenumber")
    parser.add_argument(
        "--road-type",
        default="unclassified",
        help="Highway classification to analyse (for houses and straight segments)",
    )
    parser.add_argument(
        "--no-roads",
        action="store_true",
        help="Do not include road layer in the output HTML map",
    )
    parser.add_argument(
        "--direction",
        choices=["north", "south", "east", "west"],
        default="north",
        help="Direction of buildings relative to the road",
    )

    # parameters for straight segment detection
    parser.add_argument(
        "--min-length", type=float, default=250.0, help="Minimum segment length in meters (default: 250)"
    )
    parser.add_argument(
        "--min-straightness",
        type=float,
        default=0.99,
        help="Minimum straightness ratio (default: 0.99)",
    )
    parser.add_argument("--top", type=int, default=5, help="Number of top segments to print (default: 5)")
    parser.add_argument("--json", type=str, default=None, help="Optional path to write full JSON results")
    parser.add_argument("--oneway", type=str, default=None, help="Filter ways by oneway tag value")
    parser.add_argument("--access", type=str, default=None, help="Filter ways by access tag value")
    parser.add_argument("--no-primary", action="store_true", help="Exclude primary roads")
    parser.add_argument("--no-secondary", action="store_true", help="Exclude secondary roads")
    parser.add_argument(
        "--theta",
        type=float,
        default=1.0,
        help="Maximum angle difference in degrees when extending segments",
    )
    parser.add_argument(
        "--dmax",
        type=float,
        default=6.0,
        help="Maximum perpendicular deviation in metres when extending segments",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pbf = Path(args.pbf)
    if args.prefilter:
        pbf = prefilter_pbf(pbf, args.road_type, args.housenumber, args.other_housenumber)

    buildings, roads = find_buildings(
        pbf,
        args.min_dist,
        args.max_dist,
        args.housenumber,
        args.other_housenumber,
        args.min_hn_dist,
        args.max_hn_dist,
        args.road_type,
        args.direction,
        args.processes,
    )
    print(f"Found {len(buildings)} buildings")

    collector = WayCollector(
        args.oneway,
        args.access,
        include_primary=not args.no_primary,
        include_secondary=not args.no_secondary,
    )
    collector.apply_file(str(pbf), locations=True)
    segments = collector.segments
    candidates = extract_straight_chains(
        segments,
        args.min_length,
        args.min_straightness,
        args.road_type,
        args.theta,
        args.dmax,
    )
    top_candidates = candidates[: args.top]
    for c in top_candidates:
        name_part = f" {c['name']}" if "name" in c else ""
        print(
            f"Segment {c['id']}{name_part} ({c['highway']}): length "
            f"{c['length_m']:.1f} m, straightness {c['straightness']:.4f}"
        )

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(candidates, f, indent=2)

    if folium is None:
        raise RuntimeError("folium is required for HTML output but is not installed")
    create_combined_map(
        buildings,
        roads,
        top_candidates,
        Path(args.out),
        args.road_type,
        show_roads=not args.no_roads,
    )

    if args.prefilter:
        os.remove(pbf)


if __name__ == "__main__":
    main()
