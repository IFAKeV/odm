#!/usr/bin/env python3
"""Render selected highways from an OSM PBF file to PNG overlays.

The script extracts a subset of ways with the ``highway`` tag from a given
OSM PBF file and renders each supported highway type into a separate PNG image
on top of OpenStreetMap tiles. Supported types are ``motorway``, ``primary``,
``path``, ``secondary``, ``tertiary``, ``trunk`` and ``unclassified``.
Additionally, a combined image of ``primary``, ``secondary``, ``tertiary`` and
``unclassified`` roads is produced. All generated images share the same extent
and can therefore be stacked on top of each other in an image editor.

Example
-------
    python find_straight_ways_visual_v02.py pbf/zielgebiet.pbf \
        --outdir out --scale 10000

``out/highway_residential.png`` then contains all residential roads, while
``out/highway_tertiary.png`` contains all tertiary roads, etc.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import osmium
from staticmap import StaticMap, Line
from staticmap.staticmap import _lat_to_y, _lon_to_x

ALLOWED_HIGHWAYS = {
    "motorway",
    "primary",
    "path",
    "secondary",
    "tertiary",
    "trunk",
    "unclassified",
}


class HighwayCollector(osmium.SimpleHandler):
    """Collect all highway ways from an OSM PBF file."""

    def __init__(self) -> None:
        super().__init__()
        self.ways: Dict[str, List[List[Tuple[float, float]]]] = defaultdict(list)
        self.min_lon = 180.0
        self.max_lon = -180.0
        self.min_lat = 90.0
        self.max_lat = -90.0

    def way(self, w: osmium.osm.Way) -> None:  # type: ignore[override]
        hw = w.tags.get("highway")
        if hw is None or len(w.nodes) < 2 or hw not in ALLOWED_HIGHWAYS:
            return

        coords: List[Tuple[float, float]] = []
        for n in w.nodes:
            if not n.location.valid():
                continue
            lon = n.lon
            lat = n.lat
            coords.append((lon, lat))
            if lon < self.min_lon:
                self.min_lon = lon
            if lon > self.max_lon:
                self.max_lon = lon
            if lat < self.min_lat:
                self.min_lat = lat
            if lat > self.max_lat:
                self.max_lat = lat
        if len(coords) >= 2:
            self.ways[hw].append(coords)


def calculate_zoom(
    bbox: Tuple[float, float, float, float],
    width: int,
    height: int,
    tile_size: int = 256,
) -> int:
    """Determine a zoom level that fits ``bbox`` into ``width``/``height``."""

    min_lon, min_lat, max_lon, max_lat = bbox
    for z in range(17, -1, -1):
        w = (_lon_to_x(max_lon, z) - _lon_to_x(min_lon, z)) * tile_size
        h = (_lat_to_y(min_lat, z) - _lat_to_y(max_lat, z)) * tile_size
        if w <= width and h <= height:
            return z
    return 0


def render_highways(
    highways: Dict[str, List[List[Tuple[float, float]]]],
    bbox: Tuple[float, float, float, float],
    scale: float,
    line_width: int,
    outdir: str,
) -> None:
    min_lon, min_lat, max_lon, max_lat = bbox
    width = max(int((max_lon - min_lon) * scale) + 1, 1)
    height = max(int((max_lat - min_lat) * scale) + 1, 1)
    center = [(min_lon + max_lon) / 2.0, (min_lat + max_lat) / 2.0]
    zoom = calculate_zoom(bbox, width, height)

    os.makedirs(outdir, exist_ok=True)

    for hw, ways in highways.items():
        m = StaticMap(
            width,
            height,
            url_template="https://a.tile.openstreetmap.org/{z}/{x}/{y}.png",
        )
        for coords in ways:
            m.add_line(Line(coords, "white", line_width))
        image = m.render(zoom=zoom, center=center)
        filename = os.path.join(outdir, f"highway_{hw}.png")
        image.save(filename)
        print(f"Wrote {filename}")

    combined_keys = ["primary", "secondary", "tertiary", "unclassified"]
    combined: List[List[Tuple[float, float]]] = []
    for key in combined_keys:
        combined.extend(highways.get(key, []))
    if combined:
        m = StaticMap(
            width,
            height,
            url_template="https://a.tile.openstreetmap.org/{z}/{x}/{y}.png",
        )
        for coords in combined:
            m.add_line(Line(coords, "white", line_width))
        image = m.render(zoom=zoom, center=center)
        filename = os.path.join(
            outdir, "highway_primary_secondary_tertiary_unclassified.png"
        )
        image.save(filename)
        print(f"Wrote {filename}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pbf", help="Path to OSM PBF file")
    parser.add_argument(
        "--outdir",
        default=".",
        help="Output directory for PNG files (default: current directory)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=10000.0,
        help="Pixels per degree (default: 10000)",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=1,
        help="Line width in pixels (default: 1)",
    )
    args = parser.parse_args()

    handler = HighwayCollector()
    handler.apply_file(args.pbf, locations=True)

    bbox = (handler.min_lon, handler.min_lat, handler.max_lon, handler.max_lat)
    render_highways(handler.ways, bbox, args.scale, args.line_width, args.outdir)


if __name__ == "__main__":
    main()
