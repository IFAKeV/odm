#!/usr/bin/env python3
"""Find the longest straight lines in PNG images of a directory.

For each PNG file the script loads the image in grayscale, performs edge
extraction and uses a probabilistic Hough transform to detect line segments.
The longest segments are reported and can optionally be visualised by drawing
them onto the original image. By default, the five longest segments are
returned but this can be customised.

Example
-------
Process images in ``images/`` and save visualisations to ``out/``::

    python longest_lines.py images --visualize --out out

This script requires :mod:`opencv-python` (or ``scikit-image`` as an
alternative implementation).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

Line = Tuple[float, Tuple[int, int, int, int]]


def detect_longest_lines(
    image: np.ndarray, min_line_length: int, max_line_gap: int, count: int
) -> List[Line]:
    """Return the longest line segments detected in *image*.

    The image should be a single-channel (grayscale or binary) array. The
    function applies Canny edge detection followed by a probabilistic Hough
    transform. Results are sorted by length in descending order and the *count*
    longest segments are returned.
    """

    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=50,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    results: List[Line] = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            length = float(np.hypot(x2 - x1, y2 - y1))
            results.append((length, (x1, y1, x2, y2)))
        results.sort(key=lambda x: x[0], reverse=True)
    return results[:count]


def process_image(
    path: Path,
    min_line_length: int,
    max_line_gap: int,
    count: int,
    visualize: bool,
    out_dir: Path,
) -> List[Line]:
    """Process a single image and optionally write a visualisation."""

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"failed to read {path}")

    # Binarise using Otsu's method which works for a variety of inputs
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    longest = detect_longest_lines(binary, min_line_length, max_line_gap, count)

    if visualize and longest:
        overlay = np.zeros((*image.shape, 4), dtype=np.uint8)
        for _, (x1, y1, x2, y2) in longest:
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 255, 255), 3)
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / path.name), overlay)

    return longest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", help="Directory containing PNG images")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Write images with the longest lines drawn as overlays",
    )
    parser.add_argument(
        "--out",
        default="output",
        help="Directory to write visualisations (default: output)",
    )
    parser.add_argument(
        "--min-line-length",
        type=int,
        default=50,
        help="Minimum line length in pixels (default: 50)",
    )
    parser.add_argument(
        "--max-line-gap",
        type=int,
        default=10,
        help="Maximum allowed gap between line segments in pixels (default: 10)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of longest line segments to report (default: 5)",
    )
    args = parser.parse_args()

    directory = Path(args.directory)
    if not directory.is_dir():
        raise RuntimeError(f"{directory} is not a directory")

    pngs = sorted(directory.glob("*.png"))
    if not pngs:
        print("No PNG files found")
        return

    out_dir = Path(args.out)
    for png in pngs:
        longest = process_image(
            png,
            args.min_line_length,
            args.max_line_gap,
            args.count,
            args.visualize,
            out_dir,
        )
        print(png.name)
        for length, (x1, y1, x2, y2) in longest:
            print(f"  ({x1}, {y1}) -> ({x2}, {y2}) length {length:.1f}")


if __name__ == "__main__":
    main()
