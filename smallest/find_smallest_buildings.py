#!/usr/bin/env python3
"""Find the smallest buildings with house numbers in an OSM PBF file.

The script scans all ways tagged with ``building=*`` and ``addr:housenumber``
and reports the smallest footprints by area.  Results are printed as links to
OpenStreetMap.  Optionally a CSV file can be written.
"""

from __future__ import annotations

import argparse
import math
from typing import List, Optional, Sequence, Tuple

try:  # optional dependency
    import osmium  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    osmium = None

Coord = Tuple[float, float]  # (lon, lat)


def _area_m2(coords: Sequence[Coord]) -> float:
    """Approximate polygon area in square metres.

    Converts lon/lat coordinates to a local metric projection using the first
    vertex and applies the shoelace formula.  The approximation is sufficient
    for ranking small buildings.
    """

    if len(coords) < 3:
        return 0.0
    lon0, lat0 = coords[0]
    lat0_rad = math.radians(lat0)
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(lat0_rad)
    xs: List[float] = []
    ys: List[float] = []
    for lon, lat in coords:
        xs.append((lon - lon0) * m_per_deg_lon)
        ys.append((lat - lat0) * m_per_deg_lat)
    area = 0.0
    for i in range(len(coords)):
        j = (i + 1) % len(coords)
        area += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(area) / 2.0


if osmium is not None:
    class BuildingCollector(osmium.SimpleHandler):
        """Collect buildings with house numbers."""

        def __init__(self, housenumber: Optional[str] = None) -> None:
            super().__init__()
            self.housenumber = housenumber
            self.buildings: List[Tuple[int, List[Coord]]] = []

        def way(self, w: osmium.osm.Way) -> None:  # type: ignore[override]
            hn = w.tags.get("addr:housenumber")
            if not w.tags.get("building") or not hn:
                return
            if self.housenumber and hn != self.housenumber:
                return
            if len(w.nodes) < 3:
                return
            coords: List[Coord] = []
            for n in w.nodes:
                if not n.location.valid():
                    return
                coords.append((n.lon, n.lat))
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            self.buildings.append((w.id, coords))
else:  # pragma: no cover - used when osmium is missing
    class BuildingCollector:  # type: ignore[no-redef]
        def __init__(self, housenumber: Optional[str] = None) -> None:
            raise RuntimeError("osmium is required to collect buildings")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pbf", help="Path to OSM PBF file")
    parser.add_argument("--limit", type=int, default=10,
                        help="Number of buildings to report")
    parser.add_argument("--out", help="Optional CSV output path")
    parser.add_argument("--html", help="Optional HTML output path")
    parser.add_argument("--housenumber", help="Filter by specific house number")
    args = parser.parse_args()

    if osmium is None:  # pragma: no cover - runtime check
        raise RuntimeError("osmium is required but not installed")

    collector = BuildingCollector(args.housenumber)
    collector.apply_file(args.pbf, locations=True)

    items = []
    for bid, coords in collector.buildings:
        area = _area_m2(coords)
        items.append((area, bid))
    items.sort(key=lambda x: x[0])
    if args.limit:
        items = items[: args.limit]

    for area, bid in items:
        url = f"https://www.openstreetmap.org/way/{bid}"
        print(f"{url} {area:.2f} m^2")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fp:
            for area, bid in items:
                url = f"https://www.openstreetmap.org/way/{bid}"
                fp.write(f"{url},{area:.2f}\n")

    if args.html:
        with open(args.html, "w", encoding="utf-8") as fp:
            fp.write("<!DOCTYPE html>\n<html><body><ul>\n")
            for area, bid in items:
                url = f"https://www.openstreetmap.org/way/{bid}"
                fp.write(
                    f'<li><a href="{url}">{url}</a> {area:.2f} m^2</li>\n'
                )
            fp.write("</ul></body></html>\n")


if __name__ == "__main__":
    main()
