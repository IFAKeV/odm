#!/usr/bin/env python3
"""Find long, straight road runs in an OSM PBF file.

This V02 script joins adjacent road segments before measuring their length
and straightness. It considers ways tagged as ``highway=track``,
``highway=service`` or ``highway=unclassified`` and merges directly connected
segments with the same ``highway`` and ``name`` (if present). The merging uses
a graph search that respects a configurable maximum angular deviation and can
optionally filter by ``oneway`` or ``access`` tags. The straightness of each
merged run is calculated as the ratio between the geodesic distance of
its end points and the actual path length.

Example:
    python find_straight_ways_v02.py pbf/saarland-latest.osm.pbf \
        --min-length 1000 --min-straightness 0.99
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
    nodes: List[int]
    geometry: List[List[float]]
    highway: str
    name: Optional[str]
    oneway: Optional[str]
    access: Optional[str]


class WayCollector(osmium.SimpleHandler):
    """Collect relevant OSM ways grouped by (highway, name) with graph edges."""

    def __init__(
        self, oneway_filter: str | None = None, access_filter: str | None = None
    ) -> None:
        super().__init__()
        self.oneway_filter = oneway_filter
        self.access_filter = access_filter
        self.groups: Dict[
            Tuple[str, str | None], Dict[str, object]
        ] = defaultdict(lambda: {"segments": [], "node_neighbors": defaultdict(list)})

    def way(self, w: osmium.osm.Way) -> None:  # type: ignore[override]
        highway = w.tags.get("highway")
        if highway not in {"track", "service", "unclassified"}:
            return
        if len(w.nodes) < 2:
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
        nodes = [n.ref for n in w.nodes]
        geometry: List[List[float]] = [[lat, lon] for lat, lon in zip(lats, lons)]

        segment = SegmentInfo(
            id=w.id,
            nodes=nodes,
            geometry=geometry,
            highway=highway,
            name=name,
            oneway=oneway,
            access=access,
        )
        group = self.groups[(highway, name)]
        group["segments"].append(segment)
        group["node_neighbors"][nodes[0]].append(segment)
        group["node_neighbors"][nodes[-1]].append(segment)


def merge_segments(
    segments: List[SegmentInfo],
    node_neighbors: Dict[int, List[SegmentInfo]],
    geod: pyproj.Geod,
    min_len: float,
    min_ratio: float,
    max_angle: float,
) -> List[Dict[str, object]]:
    """Merge connected segments into longer runs using a graph search."""

    def direction_allowed(seg: SegmentInfo, from_node: int) -> bool:
        oneway = seg.oneway
        if not oneway or oneway.lower() in {"no", "0", "false"}:
            return True
        if oneway in {"-1", "reverse"}:
            return from_node == seg.nodes[-1]
        return from_node == seg.nodes[0]

    def extend_path(
        node: int,
        prev_az: float,
        ids: List[int],
        geometry: List[List[float]],
        insert_front: bool,
        base_oneway: Optional[str],
        base_access: Optional[str],
    ) -> Tuple[int, List[int], List[List[float]], float]:
        cur_node = node
        azimuth = prev_az
        while True:
            neighbors = [s for s in node_neighbors[cur_node] if s.id not in visited]
            valid: List[Tuple[SegmentInfo, int, float, bool]] = []
            for cand in neighbors:
                if cand.oneway != base_oneway or cand.access != base_access:
                    continue
                if cur_node == cand.nodes[0]:
                    if not direction_allowed(cand, cand.nodes[0]):
                        continue
                    next_node = cand.nodes[-1]
                    cand_az = geod.inv(
                        cand.geometry[0][1],
                        cand.geometry[0][0],
                        cand.geometry[1][1],
                        cand.geometry[1][0],
                    )[0]
                    forward = True
                else:
                    if not direction_allowed(cand, cand.nodes[-1]):
                        continue
                    next_node = cand.nodes[0]
                    cand_az = geod.inv(
                        cand.geometry[-1][1],
                        cand.geometry[-1][0],
                        cand.geometry[-2][1],
                        cand.geometry[-2][0],
                    )[0]
                    forward = False
                diff = abs((cand_az - azimuth + 180) % 360 - 180)
                if diff <= max_angle:
                    valid.append((cand, next_node, cand_az, forward))
            if len(valid) != 1:
                break
            cand, next_node, cand_az, forward = valid[0]
            visited.add(cand.id)
            if insert_front:
                ids.insert(0, cand.id)
                if forward:
                    geometry = cand.geometry[:-1] + geometry
                else:
                    geometry = cand.geometry[1:][::-1] + geometry
            else:
                ids.append(cand.id)
                if forward:
                    geometry.extend(cand.geometry[1:])
                else:
                    geometry.extend(cand.geometry[-2::-1])
            cur_node = next_node
            azimuth = cand_az
        return cur_node, ids, geometry, azimuth

    visited: set[int] = set()
    candidates: List[Dict[str, object]] = []

    for seg in segments:
        if seg.id in visited:
            continue
        visited.add(seg.id)
        ids = [seg.id]
        geometry = seg.geometry[:]
        start = seg.nodes[0]
        end = seg.nodes[-1]

        az = geod.inv(
            geometry[-2][1], geometry[-2][0], geometry[-1][1], geometry[-1][0]
        )[0]
        end, ids, geometry, az = extend_path(
            end, az, ids, geometry, False, seg.oneway, seg.access
        )
        az = geod.inv(
            geometry[1][1], geometry[1][0], geometry[0][1], geometry[0][0]
        )[0]
        start, ids, geometry, _ = extend_path(
            start, az, ids, geometry, True, seg.oneway, seg.access
        )

        length = 0.0
        for j in range(len(geometry) - 1):
            lat1, lon1 = geometry[j]
            lat2, lon2 = geometry[j + 1]
            length += geod.inv(lon1, lat1, lon2, lat2)[2]
        straight = geod.inv(
            geometry[0][1], geometry[0][0], geometry[-1][1], geometry[-1][0]
        )[2]
        if length >= min_len and straight > 0:
            ratio = straight / length
            if ratio >= min_ratio:
                candidates.append(
                    {
                        "ids": ids,
                        "length_m": length,
                        "straightness": ratio,
                        "start": geometry[0],
                        "end": geometry[-1],
                        "geometry": geometry,
                    }
                )
    return candidates


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
        "--max-angle",
        type=float,
        default=10.0,
        help="Maximum allowed deviation angle in degrees between consecutive segments (default: 10)",
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
    args = parser.parse_args()

    geod = pyproj.Geod(ellps="WGS84")
    collector = WayCollector(args.oneway, args.access)
    collector.apply_file(args.pbf, locations=True)

    candidates: List[Dict[str, object]] = []
    for (highway, name), data in collector.groups.items():
        segs = data["segments"]
        neighbors = data["node_neighbors"]
        merged = merge_segments(
            segs,
            neighbors,
            geod,
            args.min_length,
            args.min_straightness,
            args.max_angle,
        )
        for c in merged:
            c["highway"] = highway
            if name:
                c["name"] = name
            candidates.append(c)

    candidates.sort(key=lambda x: x["length_m"], reverse=True)

    top_candidates = candidates[: args.top]

    for c in top_candidates:
        name_part = f" {c['name']}" if "name" in c else ""
        print(
            f"Run {'/'.join(map(str, c['ids']))}{name_part} ({c['highway']}): "
            f"length {c['length_m']:.1f} m, straightness {c['straightness']:.4f}"
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
                c["geometry"], tooltip=f"Run {'/'.join(map(str, c['ids']))}"
            ).add_to(m)
        m.save(args.map)


if __name__ == "__main__":
    main()
