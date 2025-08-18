#!/usr/bin/env python3
"""Render selected highways from an OSM PBF file to PNG overlays.

The script extracts a subset of ways with the ``highway`` tag from a given
OSM PBF file and renders each supported highway type into a separate PNG image
with a transparent background. Supported types are ``motorway``, ``primary``,
``path``, ``secondary``, ``tertiary``, ``trunk`` and ``unclassified``.
Additionally, a combined image of ``primary``, ``secondary``, ``tertiary`` and
``unclassified`` roads is produced. All generated images share the same extent
and can therefore be stacked on top of each other in an image editor.

Example
-------
    python find_straight_ways_visual_v03.py pbf/zielgebiet.pbf \
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
from PIL import Image, ImageDraw

ALLOWED_HIGHWAYS = {
    "motorway",
    "primary",
    "path",
    "secondary",
    "tertiary",
    "trunk",
    "unclassified",
}

HIGHWAY_COLORS = {
    "motorway": "red",
    "primary": "blue",
    "path": "green",
    "secondary": "orange",
    "tertiary": "purple",
    "trunk": "black",
    "unclassified": "gray",
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
    scale_x = width / (max_lon - min_lon) if max_lon > min_lon else 1.0
    scale_y = height / (max_lat - min_lat) if max_lat > min_lat else 1.0

    os.makedirs(outdir, exist_ok=True)

    for hw, ways in highways.items():
        img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        for coords in ways:
            pixels = [
                project(lon, lat, min_lon, min_lat, scale_x, scale_y, height)
                for lon, lat in coords
            ]
            draw.line(pixels, fill=(255, 255, 255, 255), width=line_width)
        filename = os.path.join(outdir, f"highway_{hw}.png")
        img.save(filename)
        print(f"Wrote {filename}")

    combined_keys = ["primary", "secondary", "tertiary", "unclassified"]
    combined: List[List[Tuple[float, float]]] = []
    for key in combined_keys:
        combined.extend(highways.get(key, []))
    if combined:
        img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        for coords in combined:
            pixels = [
                project(lon, lat, min_lon, min_lat, scale_x, scale_y, height)
                for lon, lat in coords
            ]
            draw.line(pixels, fill=(255, 255, 255, 255), width=line_width)
        filename = os.path.join(
            outdir, "highway_primary_secondary_tertiary_unclassified.png"
        )
        img.save(filename)
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
    parser.add_argument(
        "--map",
        help="Optional path to save an interactive HTML map",
    )
    args = parser.parse_args()

    handler = HighwayCollector()
    handler.apply_file(args.pbf, locations=True)

    bbox = (handler.min_lon, handler.min_lat, handler.max_lon, handler.max_lat)

    if args.map:
        try:
            import folium
        except ImportError as exc:  # pragma: no cover - runtime check
            raise RuntimeError("folium is required when using --map") from exc

        center_lat = (handler.min_lat + handler.max_lat) / 2.0
        center_lon = (handler.min_lon + handler.max_lon) / 2.0
        m = folium.Map(location=[center_lat, center_lon])

        for hw, ways in handler.ways.items():
            fg = folium.FeatureGroup(name=hw)
            color = HIGHWAY_COLORS.get(hw, "black")
            for coords in ways:
                fg.add_child(
                    folium.PolyLine(
                        [(lat, lon) for lon, lat in coords],
                        color=color,
                        weight=args.line_width,
                    )
                )
            m.add_child(fg)

        folium.LayerControl().add_to(m)
        m.save(args.map)
        print(f"Wrote {args.map}")

    render_highways(handler.ways, bbox, args.scale, args.line_width, args.outdir)


if __name__ == "__main__":
    main()
