#!/usr/bin/env python3
"""Render a simple base map from an OSM PBF file.

The script creates a PNG image showing a rudimentary map consisting of
landuse areas, water bodies, buildings and highways.  The resulting image
shares the same extent and scale as the highway overlays produced by
``find_straight_ways_visual_v01.py`` so that both can be stacked for
visual comparison.

Example
-------
    python render_full_map.py pbf/zielgebiet.pbf --out map.png --scale 10000
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import osmium
from PIL import Image, ImageDraw

# Colors used for rendering
LAND_COLOR = (222, 239, 207, 255)  # pale green
WATER_COLOR = (170, 211, 223, 255)  # light blue
BUILDING_COLOR = (234, 223, 191, 255)  # light brown
ROAD_COLOR = (200, 200, 200, 255)  # grey


class MapCollector(osmium.SimpleHandler):
    """Collect simple map features from an OSM PBF file.

    The bounding box is determined solely by highway geometries to match the
    output of ``find_straight_ways_visual_v01.py``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.highways: List[List[Tuple[float, float]]] = []
        self.buildings: List[List[Tuple[float, float]]] = []
        self.water: List[List[Tuple[float, float]]] = []
        self.land: List[List[Tuple[float, float]]] = []
        self.min_lon = 180.0
        self.max_lon = -180.0
        self.min_lat = 90.0
        self.max_lat = -90.0

    def _update_bbox(self, coords: List[Tuple[float, float]]) -> None:
        for lon, lat in coords:
            if lon < self.min_lon:
                self.min_lon = lon
            if lon > self.max_lon:
                self.max_lon = lon
            if lat < self.min_lat:
                self.min_lat = lat
            if lat > self.max_lat:
                self.max_lat = lat

    def way(self, w: osmium.osm.Way) -> None:  # type: ignore[override]
        if len(w.nodes) < 2:
            return

        coords: List[Tuple[float, float]] = []
        for n in w.nodes:
            if not n.location.valid():
                continue
            coords.append((n.lon, n.lat))
        if not coords:
            return

        hw = w.tags.get("highway")
        if hw is not None:
            self.highways.append(coords)
            self._update_bbox(coords)

        if w.is_closed():
            if w.tags.get("building"):
                self.buildings.append(coords)
            landuse = w.tags.get("landuse")
            natural = w.tags.get("natural")
            if landuse in {
                "forest",
                "grass",
                "meadow",
                "farmland",
                "residential",
                "commercial",
                "industrial",
                "park",
                "recreation_ground",
            } or natural in {"wood", "scrub", "grassland"}:
                self.land.append(coords)
            if (
                natural == "water"
                or w.tags.get("waterway")
                or landuse in {"reservoir", "basin"}
            ):
                self.water.append(coords)
        else:
            if w.tags.get("waterway"):
                self.water.append(coords)


def project(
    lon: float,
    lat: float,
    min_lon: float,
    min_lat: float,
    scale_x: float,
    scale_y: float,
    height: int,
) -> Tuple[int, int]:
    """Project geographical coordinates to image pixel coordinates."""

    x = int((lon - min_lon) * scale_x)
    y = int(height - (lat - min_lat) * scale_y)
    return x, y


def render_map(
    data: MapCollector,
    bbox: Tuple[float, float, float, float],
    scale: float,
    line_width: int,
    outfile: str,
) -> None:
    min_lon, min_lat, max_lon, max_lat = bbox
    width = max(int((max_lon - min_lon) * scale) + 1, 1)
    height = max(int((max_lat - min_lat) * scale) + 1, 1)
    scale_x = width / (max_lon - min_lon) if max_lon > min_lon else 1.0
    scale_y = height / (max_lat - min_lat) if max_lat > min_lat else 1.0

    img = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)

    for coords in data.land:
        pixels = [
            project(lon, lat, min_lon, min_lat, scale_x, scale_y, height)
            for lon, lat in coords
        ]
        if len(pixels) >= 3:
            draw.polygon(pixels, fill=LAND_COLOR)

    for coords in data.water:
        pixels = [
            project(lon, lat, min_lon, min_lat, scale_x, scale_y, height)
            for lon, lat in coords
        ]
        if len(pixels) >= 2:
            if len(pixels) >= 3:
                draw.polygon(pixels, fill=WATER_COLOR)
            else:
                draw.line(pixels, fill=WATER_COLOR, width=1)

    for coords in data.buildings:
        pixels = [
            project(lon, lat, min_lon, min_lat, scale_x, scale_y, height)
            for lon, lat in coords
        ]
        if len(pixels) >= 3:
            draw.polygon(pixels, fill=BUILDING_COLOR)

    for coords in data.highways:
        pixels = [
            project(lon, lat, min_lon, min_lat, scale_x, scale_y, height)
            for lon, lat in coords
        ]
        if len(pixels) >= 2:
            draw.line(pixels, fill=ROAD_COLOR, width=line_width)

    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
    img.save(outfile)
    print(f"Wrote {outfile}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pbf", help="Path to OSM PBF file")
    parser.add_argument(
        "--out",
        default="map.png",
        help="Output PNG file (default: map.png)",
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
        help="Highway line width in pixels (default: 1)",
    )
    args = parser.parse_args()

    collector = MapCollector()
    collector.apply_file(args.pbf, locations=True)

    bbox = (collector.min_lon, collector.min_lat, collector.max_lon, collector.max_lat)
    render_map(collector, bbox, args.scale, args.line_width, args.out)


if __name__ == "__main__":
    main()
