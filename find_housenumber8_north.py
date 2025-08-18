#!/usr/bin/env python3
"""Locate house number 8 buildings north of unclassified roads.

The script scans an (optionally pre-filtered) OSM PBF file for buildings that
carry ``addr:housenumber=8``. It computes the distance of each building to the
nearest ``highway=unclassified`` segment and keeps those that are north of the
road and between the given distance bounds. An interactive HTML map is written
with separate layers for roads and buildings.

Example
-------
    # pre-filtering using osmium (recommended)
    osmium tags-filter pbf input.osm.pbf \
        w/highway=unclassified w/building addr:housenumber=8 -o filtered.pbf
    python find_housenumber8_north.py filtered.pbf --out houses.html

    # alternatively let the script do the filtering
    python find_housenumber8_north.py input.osm.pbf --prefilter --out houses.html
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import subprocess
import tempfile
from dataclasses import dataclass
from math import cos, hypot, radians
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import folium
import osmium
from rtree import index


@dataclass
class Building:
    lon: float
    lat: float
    distance: float


class Collector(osmium.SimpleHandler):
    """Collect building centroids and road segments."""

    def __init__(self) -> None:
        super().__init__()
        self.buildings: List[Tuple[float, float]] = []
        self.roads: List[List[Tuple[float, float]]] = []
        self.segments: List[Tuple[float, float, float, float, float]] = []

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
            for (lon1, lat1), (lon2, lat2) in zip(coords, coords[1:]):
                self.segments.append((lon1, lat1, lon2, lat2, max(lat1, lat2)))
        if (
            w.is_closed()
            and w.tags.get("building")
            and w.tags.get("addr:housenumber") == "8"
        ):
            centroid = _centroid(coords)
            self.buildings.append(centroid)


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


def _deg_buffer(lat: float, metres: float) -> Tuple[float, float]:
    lat_deg = metres / 111_320
    lon_deg = metres / (111_320 * cos(radians(lat)))
    return lon_deg, lat_deg


# global storage for multiprocessing workers
SEGMENTS: List[Tuple[float, float, float, float, float]]
RINDEX: index.Index


def _init_worker(segments: List[Tuple[float, float, float, float, float]]) -> None:
    global SEGMENTS, RINDEX
    SEGMENTS = segments
    RINDEX = index.Index()
    for i, (lon1, lat1, lon2, lat2, _) in enumerate(SEGMENTS):
        min_lon = min(lon1, lon2)
        max_lon = max(lon1, lon2)
        min_lat = min(lat1, lat2)
        max_lat = max(lat1, lat2)
        RINDEX.insert(i, (min_lon, min_lat, max_lon, max_lat))


def _process_chunk(
    chunk: Sequence[Tuple[float, float]], min_dist: float, max_dist: float
) -> List[Building]:
    results: List[Building] = []
    for lon, lat in chunk:
        lon_buf, lat_buf = _deg_buffer(lat, max_dist)
        min_d = float("inf")
        for i in RINDEX.intersection(
            (lon - lon_buf, lat - lat_buf, lon + lon_buf, lat + lat_buf)
        ):
            lon1, lat1, lon2, lat2, max_lat = SEGMENTS[i]
            if max_lat >= lat:
                continue
            d = _point_to_segment_distance((lon, lat), (lon1, lat1), (lon2, lat2))
            if d < min_d:
                min_d = d
        if min_dist <= min_d <= max_dist:
            results.append(Building(lon, lat, min_d))
    return results


def _chunkify(
    seq: Sequence[Tuple[float, float]], size: int
) -> Iterable[Sequence[Tuple[float, float]]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def prefilter_pbf(src: Path) -> Path:
    """Run osmium tags-filter to reduce the PBF to relevant objects."""
    fd, name = tempfile.mkstemp(suffix=".osm.pbf")
    os.close(fd)
    tmp = Path(name)
    cmd = [
        "osmium",
        "tags-filter",
        str(src),
        "w/highway=unclassified",
        "w/building",
        "addr:housenumber=8",
        "-o",
        str(tmp),
        "--overwrite",
    ]
    subprocess.run(cmd, check=True)
    return tmp


def find_buildings(
    pbf: Path, min_dist: float, max_dist: float, processes: int | None = None
) -> Tuple[List[Building], List[List[Tuple[float, float]]]]:
    collector = Collector()
    collector.apply_file(str(pbf), locations=True)
    if not collector.buildings:
        return [], collector.roads
    with mp.Pool(
        processes=processes, initializer=_init_worker, initargs=(collector.segments,)
    ) as pool:
        chunks = list(_chunkify(collector.buildings, 100))
        results = pool.starmap(
            _process_chunk, [(chunk, min_dist, max_dist) for chunk in chunks]
        )
    buildings = [b for chunk in results for b in chunk]
    return buildings, collector.roads


def create_map(
    buildings: Iterable[Building], roads: Iterable[Sequence[Tuple[float, float]]], out: Path
) -> None:
    buildings = list(buildings)
    if not buildings:
        return
    center_lat = sum(b.lat for b in buildings) / len(buildings)
    center_lon = sum(b.lon for b in buildings) / len(buildings)
    m = folium.Map(location=[center_lat, center_lon])
    road_group = folium.FeatureGroup(name="unclassified roads")
    for road in roads:
        folium.PolyLine([(lat, lon) for lon, lat in road], color="blue", weight=2).add_to(
            road_group
        )
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
    folium.LayerControl().add_to(m)
    m.save(str(out))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pbf", help="Path to OSM PBF file")
    parser.add_argument("--out", default="houses.html", help="Output HTML file")
    parser.add_argument(
        "--min-dist", type=float, default=50.0, help="Minimum distance in metres"
    )
    parser.add_argument(
        "--max-dist", type=float, default=150.0, help="Maximum distance in metres"
    )
    parser.add_argument(
        "--prefilter",
        action="store_true",
        help="Run osmium tags-filter before processing",
    )
    parser.add_argument(
        "--processes", type=int, default=None, help="Number of worker processes"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pbf = Path(args.pbf)
    if args.prefilter:
        pbf = prefilter_pbf(pbf)
    buildings, roads = find_buildings(pbf, args.min_dist, args.max_dist, args.processes)
    print(f"Found {len(buildings)} buildings")
    create_map(buildings, roads, Path(args.out))
    if args.prefilter:
        os.remove(pbf)


if __name__ == "__main__":
    main()
