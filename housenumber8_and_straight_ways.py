#!/usr/bin/env python3
"""Find house number 8 buildings and nearby straight road segments.

This utility combines :mod:`find_housenumber8_far` and
:mod:`find_straight_ways_v06`. It searches an OSM PBF file for buildings with
``addr:housenumber=8`` that are a certain distance from an ``unclassified``
road and for straight road segments meeting configurable thresholds. The
results are visualised on an interactive HTML map where buildings are shown as
red points and road segments as blue lines.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import folium
import pyproj

from find_housenumber8_far import Building, find_buildings
from find_straight_ways_v06 import WayCollector, extract_straight_sections


def create_map(
    buildings: Iterable[Building],
    segments: List[dict],
    out: Path,
) -> None:
    """Create a folium map with *buildings* and *segments* and save to *out*."""
    buildings = list(buildings)
    locs: List[List[float]] = []
    for b in buildings:
        locs.append([b.lat, b.lon])
    for seg in segments:
        locs.extend(seg["geometry"])
    if not locs:
        return
    center_lat = sum(lat for lat, _ in locs) / len(locs)
    center_lon = sum(lon for _, lon in locs) / len(locs)
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    for seg in segments:
        folium.PolyLine(seg["geometry"], color="blue", weight=3).add_to(m)
    for b in buildings:
        folium.CircleMarker(
            location=[b.lat, b.lon],
            radius=4,
            weight=1,
            color="red",
            fill=True,
            fill_opacity=0.8,
            popup=f"{b.distance:.1f} m",
        ).add_to(m)
    m.save(str(out))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pbf", help="Path to OSM PBF file")
    parser.add_argument(
        "--out",
        default="map.html",
        help="Output HTML file (default: map.html)",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=50.0,
        help="Minimum distance of houses to unclassified roads in metres (default: 50)",
    )
    parser.add_argument(
        "--max-dist",
        type=float,
        default=150.0,
        help="Maximum distance of houses to unclassified roads in metres (default: 150)",
    )
    parser.add_argument(
        "--min-length",
        type=float,
        default=250.0,
        help="Minimum road segment length in metres (default: 250)",
    )
    parser.add_argument(
        "--min-straightness",
        type=float,
        default=0.99,
        help="Minimum straightness ratio (default: 0.99)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of top straight segments to include (default: 5)",
    )
    parser.add_argument(
        "--oneway",
        type=str,
        default=None,
        help="Filter ways by oneway tag value",
    )
    parser.add_argument(
        "--access",
        type=str,
        default=None,
        help="Filter ways by access tag value",
    )
    parser.add_argument(
        "--no-primary",
        action="store_true",
        help="Exclude primary roads",
    )
    parser.add_argument(
        "--no-secondary",
        action="store_true",
        help="Exclude secondary roads",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pbf = Path(args.pbf)

    buildings = find_buildings(pbf, args.min_dist, args.max_dist)

    geod = pyproj.Geod(ellps="WGS84")
    collector = WayCollector(
        args.oneway,
        args.access,
        include_primary=not args.no_primary,
        include_secondary=not args.no_secondary,
    )
    collector.apply_file(str(pbf), locations=True)
    segments = extract_straight_sections(
        collector.segments, geod, args.min_length, args.min_straightness
    )
    top_segments = segments[: args.top]

    print(
        f"Found {len(buildings)} buildings and {len(segments)} straight segments"
    )
    create_map(buildings, top_segments, Path(args.out))


if __name__ == "__main__":
    main()
