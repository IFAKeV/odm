#!/usr/bin/env python3
"""Find long, straight ways in an OSM PBF file.

The script scans an OSM PBF file for ways tagged with ``highway=*`` and measures
their length and straightness (ratio of the geodesic distance between start and
end node to the actual path length). Ways that exceed the given thresholds are
reported.

Example:
    python find_straight_ways.py pbf/saarland-latest.osm.pbf --min-length 1000 --min-straightness 0.99
"""

import argparse
import json
from typing import List

import osmium
import pyproj

try:
    import folium  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    folium = None


class StraightWayHandler(osmium.SimpleHandler):
    """OSM handler that collects long, straight ways."""

    def __init__(self, min_length: float, min_ratio: float):
        super().__init__()
        self.geod = pyproj.Geod(ellps="WGS84")
        self.min_length = min_length
        self.min_ratio = min_ratio
        self.candidates = []

    def way(self, w: osmium.osm.Way) -> None:  # type: ignore[override]
        highway = w.tags.get("highway")
        if highway is None or len(w.nodes) < 2:
            return

        lats = [n.lat for n in w.nodes]
        lons = [n.lon for n in w.nodes]
        geometry: List[List[float]] = [[lat, lon] for lat, lon in zip(lats, lons)]
        length = 0.0
        for i in range(len(lons) - 1):
            length += self.geod.inv(lons[i], lats[i], lons[i + 1], lats[i + 1])[2]
        straight = self.geod.inv(lons[0], lats[0], lons[-1], lats[-1])[2]

        if length >= self.min_length and straight > 0:
            ratio = straight / length
            if ratio >= self.min_ratio:
                self.candidates.append(
                    {
                        "id": w.id,
                        "highway": highway,
                        "length_m": length,
                        "straightness": ratio,
                        "start": [lats[0], lons[0]],
                        "end": [lats[-1], lons[-1]],
                        "geometry": geometry,
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pbf", help="Path to OSM PBF file")
    parser.add_argument(
        "--min-length",
        type=float,
        default=1000.0,
        help="Minimum way length in meters (default: 1000)",
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
        help="Number of top ways to print (default: 5)",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Optional path to write full JSON results",
    )
    parser.add_argument(
        "--map",
        type=str,
        default=None,
        help="Optional path to write an interactive HTML map",
    )
    args = parser.parse_args()

    handler = StraightWayHandler(args.min_length, args.min_straightness)
    handler.apply_file(args.pbf, locations=True)

    candidates = sorted(handler.candidates, key=lambda x: x["length_m"], reverse=True)

    for c in candidates[: args.top]:
        print(
            f"Way {c['id']} ({c['highway']}): length {c['length_m']:.1f} m, "
            f"straightness {c['straightness']:.4f}"
        )

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(candidates, f, indent=2)

    if args.map and folium is None:
        raise RuntimeError("folium is required for --map but is not installed")
    if args.map and candidates:
        m = folium.Map(location=candidates[0]["geometry"][0], zoom_start=12)
        for c in candidates:
            folium.PolyLine(
                c["geometry"], tooltip=f"Way {c['id']} ({c['highway']})"
            ).add_to(m)
        m.save(args.map)


if __name__ == "__main__":
    main()
