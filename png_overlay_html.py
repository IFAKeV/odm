#!/usr/bin/env python3
"""Generate simple HTML files to display PNG overlays on a map.

The script writes a Leaflet map for each PNG file in a directory, using
``folium`` to overlay the image.  The geographical bounds of the image must be
provided explicitly.  Each generated HTML file is placed alongside the input
PNG file by default.

Example
-------
    python png_overlay_html.py out/ \\
        --bounds 7.0 49.0 7.5 49.5 --opacity 0.5

This command will create an HTML file for every ``.png`` in ``out/`` that shows
it semi-transparently on top of a base map so that the underlying streets can
be identified.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import folium


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "png_dir",
        help="Directory containing PNG files created by find_straight_ways_visual_v01.py",
    )
    parser.add_argument(
        "--bounds",
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        type=float,
        nargs=4,
        required=True,
        help="Geographical bounding box covering all generated PNGs",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory for HTML files (default: same as png_dir)",
    )
    parser.add_argument(
        "--opacity",
        type=float,
        default=0.5,
        help="Opacity of the overlay image (default: 0.5)",
    )
    return parser.parse_args()


def create_html(
    image: Path, bounds: Tuple[float, float, float, float], outdir: Path, opacity: float
) -> None:
    """Write a HTML file displaying *image* as overlay within *bounds*.

    Parameters
    ----------
    image:
        Path to the PNG file.
    bounds:
        Tuple ``(min_lon, min_lat, max_lon, max_lat)`` describing the extent of
        the image.
    outdir:
        Directory where the HTML file should be written.
    opacity:
        Opacity of the overlay image.  ``1.0`` is fully opaque and ``0.0``
        hides the overlay completely.
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    center_lat = (min_lat + max_lat) / 2.0
    center_lon = (min_lon + max_lon) / 2.0

    m = folium.Map(location=[center_lat, center_lon])
    folium.raster_layers.ImageOverlay(
        image=os.path.basename(image),
        bounds=[[min_lat, min_lon], [max_lat, max_lon]],
        opacity=opacity,
    ).add_to(m)
    m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])

    out_path = outdir / f"{image.stem}.html"
    m.save(str(out_path))
    print(f"Wrote {out_path}")


def main() -> None:
    args = parse_args()
    png_dir = Path(args.png_dir)
    outdir = Path(args.outdir) if args.outdir else png_dir
    outdir.mkdir(parents=True, exist_ok=True)

    bounds = tuple(args.bounds)  # type: ignore[assignment]

    for image in sorted(png_dir.glob("*.png")):
        create_html(image, bounds, outdir, args.opacity)


if __name__ == "__main__":
    main()
