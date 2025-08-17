#!/usr/bin/env python3
"""Find long, straight road runs in an OSM PBF file using Shapely.

This v04 script experiments with a different merging strategy compared to
``find_straight_ways_v02.py``. All relevant road segments are converted to
Shapely ``LineString`` objects. Connected segments are merged using
``linemerge`` after an optional Ramer--Douglas--Peucker simplification. The
longest straight sub runs are then searched within each merged line by
sliding a window over its coordinate sequence. The straightness of a run is
measured as the ratio between geodesic distance of its end points and its
actual length.

Example:
    python find_straight_ways_v04.py pbf/saarland-latest.osm.pbf \
        --min-length 1000 --min-straightness 0.99 --simplify 0.0001
"""

from __future__ import annotations

import argparse
import json
from typing import Iterable, List, Tuple

import osmium
import pyproj
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge, unary_union

try:  # pragma: no cover - optional dependency
    import folium  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    folium = None


class SegmentCollector(osmium.SimpleHandler):
    """Collect relevant OSM way geometries as Shapely LineStrings."""

    def __init__(
        self, include_primary: bool = True, include_secondary: bool = True
    ) -> None:
        super().__init__()
        self.include_primary = include_primary
        self.include_secondary = include_secondary
        self.segments: List[LineString] = []

    def way(self, w: osmium.osm.Way) -> None:  # type: ignore[override]
        highway = w.tags.get("highway")
        if highway is None or len(w.nodes) < 2:
            return
        if highway in {"motorway", "motorway_link"}:
            return
        if not self.include_primary and highway in {"primary", "primary_link"}:
            return
        if not self.include_secondary and highway in {"secondary", "secondary_link"}:
            return
        coords = [(n.lon, n.lat) for n in w.nodes]
        self.segments.append(LineString(coords))


def geodesic_length(coords: Iterable[Tuple[float, float]], geod: pyproj.Geod) -> float:
    """Return the geodesic length of a coordinate sequence."""

    length = 0.0
    pairs = list(coords)
    for (lon1, lat1), (lon2, lat2) in zip(pairs[:-1], pairs[1:]):
        length += geod.inv(lon1, lat1, lon2, lat2)[2]
    return length


def straight_runs(
    line: LineString, geod: pyproj.Geod, min_len: float, min_ratio: float
) -> List[dict]:
    """Return straight sub runs within ``line`` satisfying the constraints."""

    coords = list(line.coords)
    runs: List[dict] = []
    n = len(coords)
    for i in range(n - 1):
        length = 0.0
        for j in range(i + 1, n):
            lon1, lat1 = coords[j - 1]
            lon2, lat2 = coords[j]
            length += geod.inv(lon1, lat1, lon2, lat2)[2]
            straight = geod.inv(
                coords[i][0], coords[i][1], coords[j][0], coords[j][1]
            )[2]
            if length >= min_len and straight > 0:
                ratio = straight / length
                if ratio >= min_ratio:
                    geometry = [[lat, lon] for lon, lat in coords[i : j + 1]]
                    runs.append(
                        {
                            "length_m": length,
                            "straightness": ratio,
                            "start": geometry[0],
                            "end": geometry[-1],
                            "geometry": geometry,
                        }
                    )
                else:
                    break
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pbf", help="Path to OSM PBF file")
    parser.add_argument(
        "--min-length",
        type=float,
        default=1000.0,
        help="Minimum run length in meters (default: 1000)",
    )
    parser.add_argument(
        "--min-straightness",
        type=float,
        default=0.99,
        help="Minimum straightness ratio (default: 0.99)",
    )
    parser.add_argument(
        "--simplify",
        type=float,
        default=0.0,
        help="Simplify geometry with the given tolerance in degrees before merging",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of top runs to print (default: 5)",
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
    parser.add_argument(
        "--no-primary", action="store_true", help="Exclude primary roads"
    )
    parser.add_argument(
        "--no-secondary", action="store_true", help="Exclude secondary roads"
    )
    args = parser.parse_args()

    geod = pyproj.Geod(ellps="WGS84")

    collector = SegmentCollector(
        include_primary=not args.no_primary, include_secondary=not args.no_secondary
    )
    collector.apply_file(args.pbf, locations=True)

    lines: List[LineString] = collector.segments
    if args.simplify > 0:
        lines = [line.simplify(args.simplify) for line in lines]
    if not lines:
        return

    merged = linemerge(unary_union(MultiLineString(lines)))
    merged_lines: List[LineString]
    if isinstance(merged, LineString):
        merged_lines = [merged]
    else:
        merged_lines = list(merged.geoms)

    candidates: List[dict] = []
    for line in merged_lines:
        candidates.extend(straight_runs(line, geod, args.min_length, args.min_straightness))

    candidates.sort(key=lambda c: c["length_m"], reverse=True)
    top_candidates = candidates[: args.top]
    for idx, c in enumerate(top_candidates, 1):
        print(
            f"Run {idx}: length {c['length_m']:.1f} m, "
            f"straightness {c['straightness']:.4f}"
        )

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(candidates, f, indent=2)

    if args.map and folium is None:
        raise RuntimeError("folium is required for --map but is not installed")
    if args.map and top_candidates:
        m = folium.Map(location=top_candidates[0]["geometry"][0], zoom_start=12)
        for c in top_candidates:
            folium.PolyLine(c["geometry"], tooltip=f"{c['length_m']:.1f} m").add_to(m)
        m.save(args.map)


if __name__ == "__main__":
    main()
