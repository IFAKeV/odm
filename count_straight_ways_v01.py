#!/usr/bin/env python3
"""Count straight road sections of a given length in an OSM PBF file.

The script scans an OSM PBF file for ways tagged with ``highway=*`` and
measures their length and straightness. Ways that are at least the requested
length and straight enough are counted and the total number is printed.

Example
-------
    python count_straight_ways_v01.py pbf/saarland-latest.osm.pbf --length 1000 \\
        --min-straightness 0.99
"""

from __future__ import annotations

import argparse

import osmium
import pyproj


class StraightWayCounter(osmium.SimpleHandler):
    """OSM handler that counts long, straight ways."""

    def __init__(self, min_length: float, min_ratio: float) -> None:
        super().__init__()
        self.geod = pyproj.Geod(ellps="WGS84")
        self.min_length = min_length
        self.min_ratio = min_ratio
        self.count = 0

    def way(self, w: osmium.osm.Way) -> None:  # type: ignore[override]
        highway = w.tags.get("highway")
        if highway is None or len(w.nodes) < 2:
            return

        lats = [n.lat for n in w.nodes]
        lons = [n.lon for n in w.nodes]
        length = 0.0
        for i in range(len(lons) - 1):
            length += self.geod.inv(lons[i], lats[i], lons[i + 1], lats[i + 1])[2]
        straight = self.geod.inv(lons[0], lats[0], lons[-1], lats[-1])[2]
        if length >= self.min_length and straight > 0:
            ratio = straight / length
            if ratio >= self.min_ratio:
                self.count += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pbf", help="Path to OSM PBF file")
    parser.add_argument(
        "--length",
        type=float,
        required=True,
        help="Minimum way length in meters to count",
    )
    parser.add_argument(
        "--min-straightness",
        type=float,
        default=0.99,
        help="Minimum straightness ratio (default: 0.99)",
    )
    args = parser.parse_args()

    handler = StraightWayCounter(args.length, args.min_straightness)
    handler.apply_file(args.pbf, locations=True)
    print(handler.count)


if __name__ == "__main__":
    main()
