#!/usr/bin/env python3
"""Locate buildings by house number and detect straight road segments.

This script combines the functionality of :mod:`find_housenumber_direction`
and :mod:`find_straight_ways_v06`.  It always runs **both** analyses on the
same input PBF file and writes a single interactive HTML map showing

* buildings with a target ``addr:housenumber`` north/south/east/west of a
  road of a given highway classification and their distance to that road,
* long and straight road segments of the same highway class that meet a
  minimum length and straightness threshold.

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
from find_straight_ways_v06 import WayCollector, extract_straight_sections, folium


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

    geod = pyproj.Geod(ellps="WGS84")
    collector = WayCollector(
        args.oneway,
        args.access,
        include_primary=not args.no_primary,
        include_secondary=not args.no_secondary,
    )
    collector.apply_file(str(pbf), locations=True)
    segments = [s for s in collector.segments if s.highway == args.road_type]
    candidates = extract_straight_sections(
        segments, geod, args.min_length, args.min_straightness
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
