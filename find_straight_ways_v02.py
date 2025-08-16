#!/usr/bin/env python3
"""Find long, straight road runs in an OSM PBF file.

This V02 script joins adjacent road segments before measuring their length
and straightness. It considers ways tagged as ``highway=track``,
``highway=service`` or ``highway=unclassified`` and merges directly connected
segments with the same ``highway`` and ``name`` (if present). The straightness
of each merged run is calculated as the ratio between the geodesic distance of
its end points and the actual path length.

Example:
    python find_straight_ways_v02.py pbf/saarland-latest.osm.pbf \
        --min-length 1000 --min-straightness 0.99
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Dict, List, Tuple

import osmium
import pyproj

try:  # pragma: no cover - optional dependency
    import folium  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    folium = None


RelevantWay = Dict[str, object]


class WayCollector(osmium.SimpleHandler):
    """Collect relevant OSM ways grouped by (highway, name)."""

    def __init__(self) -> None:
        super().__init__()
        self.groups: Dict[Tuple[str, str | None], List[RelevantWay]] = defaultdict(list)

    def way(self, w: osmium.osm.Way) -> None:  # type: ignore[override]
        highway = w.tags.get("highway")
        if highway not in {"track", "service", "unclassified"}:
            return
        if len(w.nodes) < 2:
            return
        name = w.tags.get("name")
        lats = [n.lat for n in w.nodes]
        lons = [n.lon for n in w.nodes]
        nodes = [n.ref for n in w.nodes]
        geometry: List[List[float]] = [[lat, lon] for lat, lon in zip(lats, lons)]
        self.groups[(highway, name)].append(
            {"id": w.id, "nodes": nodes, "geometry": geometry}
        )


def merge_segments(
    segments: List[RelevantWay], geod: pyproj.Geod, min_len: float, min_ratio: float
) -> List[Dict[str, object]]:
    """Merge connected segments into longer runs and calculate metrics."""

    node_to_segments: Dict[int, List[int]] = defaultdict(list)
    for idx, seg in enumerate(segments):
        nodes = seg["nodes"]  # type: ignore[index]
        node_to_segments[nodes[0]].append(idx)
        node_to_segments[nodes[-1]].append(idx)

    visited: set[int] = set()
    candidates: List[Dict[str, object]] = []

    for i, seg in enumerate(segments):
        if i in visited:
            continue
        visited.add(i)
        nodes = seg["nodes"]  # type: ignore[index]
        geometry = seg["geometry"][:]  # type: ignore[index]
        ids = [seg["id"]]  # type: ignore[index]
        start = nodes[0]
        end = nodes[-1]

        # extend forward from end
        cur = end
        while True:
            next_idxs = [idx for idx in node_to_segments[cur] if idx not in visited]
            if len(next_idxs) != 1:
                break
            nxt = segments[next_idxs[0]]
            visited.add(next_idxs[0])
            ids.append(nxt["id"])  # type: ignore[index]
            nxt_nodes = nxt["nodes"]  # type: ignore[index]
            nxt_geom = nxt["geometry"]  # type: ignore[index]
            if nxt_nodes[0] == cur:
                geometry.extend(nxt_geom[1:])
                cur = nxt_nodes[-1]
            else:
                geometry.extend(nxt_geom[-2::-1])
                cur = nxt_nodes[0]
        end = cur

        # extend backward from start
        cur = start
        while True:
            next_idxs = [idx for idx in node_to_segments[cur] if idx not in visited]
            if len(next_idxs) != 1:
                break
            nxt = segments[next_idxs[0]]
            visited.add(next_idxs[0])
            ids.insert(0, nxt["id"])  # type: ignore[index]
            nxt_nodes = nxt["nodes"]  # type: ignore[index]
            nxt_geom = nxt["geometry"]  # type: ignore[index]
            if nxt_nodes[-1] == cur:
                geometry = nxt_geom[:-1] + geometry
                cur = nxt_nodes[0]
            else:
                geometry = nxt_geom[1:][::-1] + geometry
                cur = nxt_nodes[-1]
        start = cur

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
    args = parser.parse_args()

    geod = pyproj.Geod(ellps="WGS84")
    collector = WayCollector()
    collector.apply_file(args.pbf, locations=True)

    candidates: List[Dict[str, object]] = []
    for (highway, name), segs in collector.groups.items():
        merged = merge_segments(segs, geod, args.min_length, args.min_straightness)
        for c in merged:
            c["highway"] = highway
            if name:
                c["name"] = name
            candidates.append(c)

    candidates.sort(key=lambda x: x["length_m"], reverse=True)

    for c in candidates[: args.top]:
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
    if args.map and candidates:
        m = folium.Map(location=candidates[0]["geometry"][0], zoom_start=12)
        for c in candidates:
            folium.PolyLine(
                c["geometry"], tooltip=f"Run {'/'.join(map(str, c['ids']))}"
            ).add_to(m)
        m.save(args.map)


if __name__ == "__main__":
    main()
