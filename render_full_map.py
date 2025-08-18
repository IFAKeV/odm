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
import json
import os
from typing import List, Tuple

import osmium
from PIL import Image, ImageDraw, ImageFont

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
        self.places: List[Tuple[float, float, str]] = []
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

    def node(self, n: osmium.osm.Node) -> None:  # type: ignore[override]
        if not n.location.valid():
            return
        place = n.tags.get("place")
        name = n.tags.get("name")
        if place and name:
            if place in {"city", "town", "village", "hamlet", "suburb", "locality"}:
                self.places.append((n.lon, n.lat, name))


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
    width: int,
    height: int,
    line_width: int,
    outfile: str,
    font_size: int,
) -> None:
    """Render *data* inside *bbox* using fixed *width*/*height*.

    ``width`` and ``height`` are taken from metadata when available so that the
    resulting image matches previously generated highway overlays.
    """

    min_lon, min_lat, max_lon, max_lat = bbox
    scale_x = width / (max_lon - min_lon) if max_lon > min_lon else 1.0
    scale_y = height / (max_lat - min_lat) if max_lat > min_lat else 1.0

    img = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    def inside_bbox(coords: List[Tuple[float, float]]) -> bool:
        return all(
            min_lon <= lon <= max_lon and min_lat <= lat <= max_lat for lon, lat in coords
        )

    for coords in data.land:
        if not inside_bbox(coords):
            continue
        pixels = [
            project(lon, lat, min_lon, min_lat, scale_x, scale_y, height)
            for lon, lat in coords
        ]
        if len(pixels) >= 3:
            draw.polygon(pixels, fill=LAND_COLOR)

    for coords in data.water:
        if not inside_bbox(coords):
            continue
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
        if not inside_bbox(coords):
            continue
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

    for lon, lat, name in data.places:
        if not (min_lon <= lon <= max_lon and min_lat <= lat <= max_lat):
            continue
        x, y = project(lon, lat, min_lon, min_lat, scale_x, scale_y, height)
        draw.text((x + 2, y - 2), name, fill=(0, 0, 0, 255), font=font)

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
        "--bounds",
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        type=float,
        nargs=4,
        help="Explicit geographical bounds to render",
    )
    parser.add_argument(
        "--bounds-json",
        default=None,
        help="Path to bounds metadata JSON file (default: none)",
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
    parser.add_argument(
        "--font-size",
        type=int,
        default=12,
        help="Font size for place labels (default: 12)",
    )
    args = parser.parse_args()

    collector = MapCollector()
    collector.apply_file(args.pbf, locations=True)

    bbox: Tuple[float, float, float, float]
    width: int
    height: int

    if args.bounds_json:
        with open(args.bounds_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
        bbox = (
            meta["min_lon"],
            meta["min_lat"],
            meta["max_lon"],
            meta["max_lat"],
        )
        width = int(meta["width"])
        height = int(meta["height"])
    elif args.bounds:
        bbox = tuple(args.bounds)  # type: ignore[assignment]
        width = max(int((bbox[2] - bbox[0]) * args.scale) + 1, 1)
        height = max(int((bbox[3] - bbox[1]) * args.scale) + 1, 1)
    else:
        bbox = (collector.min_lon, collector.min_lat, collector.max_lon, collector.max_lat)
        width = max(int((bbox[2] - bbox[0]) * args.scale) + 1, 1)
        height = max(int((bbox[3] - bbox[1]) * args.scale) + 1, 1)

    render_map(collector, bbox, width, height, args.line_width, args.out, args.font_size)


if __name__ == "__main__":
    main()
