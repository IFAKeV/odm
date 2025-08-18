#!/usr/bin/env python3
"""Locate buildings with house number 8 that are 50–150 m from an unclassified road.

The script scans an OSM PBF file and searches for building polygons that carry
``addr:housenumber=8``. For each such building the distance to the nearest
``highway=unclassified`` way is calculated. Buildings whose distance falls
within the given range are written to a small interactive HTML map so that the
results can be inspected visually.

Example
-------
    python find_housenumber8_far.py pbf/bremen-latest.osm.pbf \\
        --out houses.html

The script prints the number of matches to stdout and writes the HTML file.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import cos, hypot, radians
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import folium
import osmium


@dataclass
class Building:
    lon: float
    lat: float
    distance: float


class Collector(osmium.SimpleHandler):
    """Collect buildings with housenumber=8 and unclassified roads."""

    def __init__(self) -> None:
        super().__init__()
        self.buildings: List[List[Tuple[float, float]]] = []
        self.roads: List[List[Tuple[float, float]]] = []

    def way(self, w: osmium.osm.Way) -> None:  # type: ignore[override]
        coords: List[Tuple[float, float]] = []
        for n in w.nodes:
            if not n.location.valid():
                return
            coords.append((n.lon, n.lat))
        if not coords:
            return

        if w.tags.get("highway") == "unclassified":
            self.roads.append(coords)

        if (
            w.is_closed()
            and w.tags.get("building")
            and w.tags.get("addr:housenumber") == "8"
        ):
            self.buildings.append(coords)


def _point_to_segment_distance(
    pt: Tuple[float, float], p1: Tuple[float, float], p2: Tuple[float, float]
) -> float:
    """Approximate distance in metres from *pt* to the line segment ``p1``–``p2``."""

    lon0, lat0 = pt
    lat_ref = radians(lat0)

    def to_xy(lon: float, lat: float) -> Tuple[float, float]:
        x = radians(lon) * cos(lat_ref)
        y = radians(lat)
        return x, y

    px, py = to_xy(*pt)
    x1, y1 = to_xy(*p1)
    x2, y2 = to_xy(*p2)
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return hypot(px - x1, py - y1) * 6_371_000
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    if t < 0:
        x, y = x1, y1
    elif t > 1:
        x, y = x2, y2
    else:
        x = x1 + t * dx
        y = y1 + t * dy
    return hypot(px - x, py - y) * 6_371_000


def _centroid(coords: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    lon = sum(lon for lon, _ in coords) / len(coords)
    lat = sum(lat for _, lat in coords) / len(coords)
    return lon, lat


def find_buildings(
    pbf: Path, min_dist: float, max_dist: float
) -> List[Building]:
    collector = Collector()
    collector.apply_file(str(pbf), locations=True)

    buildings: List[Building] = []
    for coords in collector.buildings:
        centroid = _centroid(coords)
        min_d = float("inf")
        for road in collector.roads:
            for p1, p2 in zip(road, road[1:]):
                d = _point_to_segment_distance(centroid, p1, p2)
                if d < min_d:
                    min_d = d
        if min_dist <= min_d <= max_dist:
            buildings.append(Building(centroid[0], centroid[1], min_d))
    return buildings


def create_map(buildings: Iterable[Building], out: Path) -> None:
    buildings = list(buildings)
    if not buildings:
        return
    center_lat = sum(b.lat for b in buildings) / len(buildings)
    center_lon = sum(b.lon for b in buildings) / len(buildings)
    m = folium.Map(location=[center_lat, center_lon])
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
        default="houses.html",
        help="Output HTML file (default: houses.html)",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=50.0,
        help="Minimum distance in metres (default: 50)",
    )
    parser.add_argument(
        "--max-dist",
        type=float,
        default=150.0,
        help="Maximum distance in metres (default: 150)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pbf = Path(args.pbf)
    buildings = find_buildings(pbf, args.min_dist, args.max_dist)
    print(f"Found {len(buildings)} buildings")
    create_map(buildings, Path(args.out))


if __name__ == "__main__":
    main()
