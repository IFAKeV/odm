#!/usr/bin/env python3
"""Find straight road segments in an OSM PBF file.

This v06 script evaluates each OSM way segment individually without merging
adjacent segments. It reports those segments whose length and straightness
exceed configurable thresholds. The straightness of a segment is measured as
the ratio between the geodesic distance of its end points and the actual path
length. Motorways are ignored and primary or secondary roads can be excluded
via command line flags. Optionally the results can be exported as JSON or
visualised on an interactive HTML map.

Example:
    python find_straight_ways_v06.py pbf/saarland-latest.osm.pbf \
        --min-length 250 --min-straightness 0.995
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import List, Optional

import osmium
import pyproj

try:  # pragma: no cover - optional dependency
    import folium  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    folium = None


@dataclass
class SegmentInfo:
    """Information about a single OSM way segment."""

    id: int
    geometry: List[List[float]]
    highway: str
    name: Optional[str]
    oneway: Optional[str]
    access: Optional[str]


class WayCollector(osmium.SimpleHandler):
    """Collect relevant OSM ways as individual segments."""

    def __init__(
        self,
        oneway_filter: str | None = None,
        access_filter: str | None = None,
        include_primary: bool = True,
        include_secondary: bool = True,
        only_unclassified: bool = False,
    ) -> None:
        super().__init__()
        self.oneway_filter = oneway_filter
        self.access_filter = access_filter
        self.include_primary = include_primary
        self.include_secondary = include_secondary
        self.only_unclassified = only_unclassified
        self.segments: List[SegmentInfo] = []

    def way(self, w: osmium.osm.Way) -> None:  # type: ignore[override]
        highway = w.tags.get("highway")
        if highway is None or len(w.nodes) < 2:
            return
        if highway in {"motorway", "motorway_link"}:
            return
        if self.only_unclassified and highway != "unclassified":
            return
        if not self.include_primary and highway in {"primary", "primary_link"}:
            return
        if not self.include_secondary and highway in {"secondary", "secondary_link"}:
            return
        name = w.tags.get("name")
        oneway = w.tags.get("oneway")
        access = w.tags.get("access")
        if self.oneway_filter is not None and oneway != self.oneway_filter:
            return
        if self.access_filter is not None and access != self.access_filter:
            return
        lats = [n.lat for n in w.nodes]
        lons = [n.lon for n in w.nodes]
        geometry: List[List[float]] = [[lat, lon] for lat, lon in zip(lats, lons)]
        self.segments.append(
            SegmentInfo(
                id=w.id,
                geometry=geometry,
                highway=highway,
                name=name,
                oneway=oneway,
                access=access,
            )
        )


def extract_straight_sections(
    segments: List[SegmentInfo],
    geod: pyproj.Geod,
    min_length: float,
    min_straightness: float,
) -> List[dict]:
    """Filter segments for length and straightness thresholds."""

    candidates: List[dict] = []
    for seg in segments:
        geometry = seg.geometry
        length = 0.0
        for j in range(len(geometry) - 1):
            lat1, lon1 = geometry[j]
            lat2, lon2 = geometry[j + 1]
            length += geod.inv(lon1, lat1, lon2, lat2)[2]
        straight = geod.inv(
            geometry[0][1], geometry[0][0], geometry[-1][1], geometry[-1][0]
        )[2]
        if length >= min_length and straight > 0:
            ratio = straight / length
            if ratio >= min_straightness:
                cand = {
                    "id": seg.id,
                    "length_m": length,
                    "straightness": ratio,
                    "start": geometry[0],
                    "end": geometry[-1],
                    "geometry": geometry,
                    "highway": seg.highway,
                }
                if seg.name:
                    cand["name"] = seg.name
                candidates.append(cand)
    candidates.sort(key=lambda x: x["length_m"], reverse=True)
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pbf", help="Path to OSM PBF file")
    parser.add_argument(
        "--min-length",
        type=float,
        default=250.0,
        help="Minimum segment length in meters (default: 250)",
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
        help="Number of top segments to print (default: 5)",
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
        "--oneway",
        type=str,
        default=None,
        help="Filter ways by oneway tag value",
    )
    parser.add_argument(
        "--access",
        type=str,
        default=None,
        help="Filter ways by access tag value",
    )
    parser.add_argument(
        "--no-primary",
        action="store_true",
        help="Exclude primary roads",
    )
    parser.add_argument(
        "--no-secondary",
        action="store_true",
        help="Exclude secondary roads",
    )
    parser.add_argument(
        "--only-unclassified",
        action="store_true",
        help="Process only highway=unclassified",
    )
    args = parser.parse_args()

    geod = pyproj.Geod(ellps="WGS84")
    collector = WayCollector(
        args.oneway,
        args.access,
        include_primary=not args.no_primary,
        include_secondary=not args.no_secondary,
        only_unclassified=args.only_unclassified,
    )
    collector.apply_file(args.pbf, locations=True)

    candidates = extract_straight_sections(
        collector.segments, geod, args.min_length, args.min_straightness
    )
    top_candidates = candidates[: args.top]

    for c in top_candidates:
        name_part = f" {c['name']}" if "name" in c else ""
        print(
            f"Segment {c['id']}{name_part} ({c['highway']}): length "
            f"{c['length_m']:.1f} m, straightness {c['straightness']:.4f}"
        )

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(candidates, f, indent=2)

    if args.map and folium is None:
        raise RuntimeError("folium is required for --map but is not installed")
    if args.map and top_candidates:
        m = folium.Map(location=top_candidates[0]["geometry"][0], zoom_start=12)
        for c in top_candidates:
            folium.PolyLine(
                c["geometry"], tooltip=f"Segment {c['id']}"
            ).add_to(m)
        m.save(args.map)


if __name__ == "__main__":
    main()
