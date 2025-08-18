#!/usr/bin/env python3
"""Combined utilities for housenumber search and straight road detection.

This script exposes the features of :mod:`find_housenumber_direction` and
:mod:`find_straight_ways_v06` under a single command line interface.  Use the
``housenumber`` subcommand to locate buildings with a given house number that
lie north/south/east/west of a classified road.  Use the ``straight``
subcommand to analyse the input PBF for long and straight road segments.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pyproj

from find_housenumber_direction import (
    create_map as create_house_map,
    find_buildings,
    prefilter_pbf,
)
from find_straight_ways_v06 import WayCollector, extract_straight_sections, folium


def _run_housenumber(args: argparse.Namespace) -> None:
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
    create_house_map(buildings, roads, Path(args.out), args.road_type)
    if args.prefilter:
        os.remove(pbf)


def _run_straight(args: argparse.Namespace) -> None:
    geod = pyproj.Geod(ellps="WGS84")
    collector = WayCollector(
        args.oneway,
        args.access,
        include_primary=not args.no_primary,
        include_secondary=not args.no_secondary,
    )
    collector.apply_file(args.pbf, locations=True)
    candidates = extract_straight_sections(
        collector.segments, geod, args.min_length, args.min_straightness
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
    if args.map and folium is None:
        raise RuntimeError("folium is required for --map but is not installed")
    if args.map and top_candidates:
        m = folium.Map(location=top_candidates[0]["geometry"][0], zoom_start=12)
        for c in top_candidates:
            folium.PolyLine(c["geometry"], tooltip=f"Segment {c['id']}").add_to(m)
        m.save(args.map)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    hn = sub.add_parser("housenumber", help="Find buildings by housenumber")
    hn.add_argument("pbf", help="Path to OSM PBF file")
    hn.add_argument("--out", default="houses.html", help="Output HTML file")
    hn.add_argument("--min-dist", type=float, default=50.0, help="Minimum distance in metres")
    hn.add_argument("--max-dist", type=float, default=150.0, help="Maximum distance in metres")
    hn.add_argument("--other-housenumber", default="6", help="Reference addr:housenumber")
    hn.add_argument(
        "--min-hn-dist", type=float, default=100.0, help="Minimum distance to reference house in metres"
    )
    hn.add_argument(
        "--max-hn-dist", type=float, default=300.0, help="Maximum distance to reference house in metres"
    )
    hn.add_argument("--prefilter", action="store_true", help="Run osmium tags-filter before processing")
    hn.add_argument("--processes", type=int, default=None, help="Number of worker processes")
    hn.add_argument("--housenumber", default="8", help="Target addr:housenumber")
    hn.add_argument("--road-type", default="unclassified", help="Highway classification")
    hn.add_argument(
        "--direction",
        choices=["north", "south", "east", "west"],
        default="north",
        help="Direction of buildings relative to the road",
    )
    hn.set_defaults(func=_run_housenumber)

    st = sub.add_parser("straight", help="Find straight road segments")
    st.add_argument("pbf", help="Path to OSM PBF file")
    st.add_argument(
        "--min-length", type=float, default=250.0, help="Minimum segment length in meters (default: 250)"
    )
    st.add_argument(
        "--min-straightness",
        type=float,
        default=0.99,
        help="Minimum straightness ratio (default: 0.99)",
    )
    st.add_argument("--top", type=int, default=5, help="Number of top segments to print (default: 5)")
    st.add_argument("--json", type=str, default=None, help="Optional path to write full JSON results")
    st.add_argument("--map", type=str, default=None, help="Optional path to write an interactive HTML map")
    st.add_argument("--oneway", type=str, default=None, help="Filter ways by oneway tag value")
    st.add_argument("--access", type=str, default=None, help="Filter ways by access tag value")
    st.add_argument("--no-primary", action="store_true", help="Exclude primary roads")
    st.add_argument("--no-secondary", action="store_true", help="Exclude secondary roads")
    st.set_defaults(func=_run_straight)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
