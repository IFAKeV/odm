#!/usr/bin/env python3
"""Find T-shaped building footprints in an OSM PBF file.

The script scans all ways tagged with ``building=*`` in a given PBF file and
checks whether their polygon outlines resemble the letter ``T``.  Matching
buildings are printed as links to OpenStreetMap and can optionally be exported
to an interactive HTML map.

Example
-------

```
python find_t_shaped_buildings.py pbf/bremen-latest.osm.pbf \
    --out t_buildings.txt --map t_buildings.html
```

The ``--out`` option writes the list of links to a text file.  Using ``--map``
requires ``folium`` to be installed and produces an HTML map with a dedicated
layer for the detected buildings.
"""

from __future__ import annotations

import argparse
import math
from typing import Iterable, List, Sequence, Tuple

try:  # pragma: no cover - runtime import
    import osmium  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    osmium = None

from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

try:  # pragma: no cover - optional dependency
    import folium  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    folium = None


Coord = Tuple[float, float]  # (lon, lat)


def _remove_colinear(coords: Sequence[Coord], tol: float = 1e-9) -> List[Coord]:
    """Remove colinear points from a closed polygon coordinate sequence."""

    if not coords:
        return []
    if coords[0] != coords[-1]:
        coords = list(coords) + [coords[0]]
    result: List[Coord] = [coords[0]]
    for i in range(1, len(coords) - 1):
        a = coords[i - 1]
        b = coords[i]
        c = coords[i + 1]
        cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(cross) > tol:
            result.append(b)
    result.append(result[0])
    return result


def is_t_shape(coords: Sequence[Coord]) -> bool:
    """Return ``True`` if the polygon outline matches a simple T shape."""

    simple = _remove_colinear(coords)
    if len(simple) - 1 != 8:
        return False

    poly = orient(Polygon(simple), sign=1.0)
    if not poly.is_valid or poly.area == 0:
        return False

    ex_coords = list(poly.exterior.coords)
    angles: List[float] = []
    for i in range(len(ex_coords) - 1):
        dx = ex_coords[i + 1][0] - ex_coords[i][0]
        dy = ex_coords[i + 1][1] - ex_coords[i][1]
        angles.append(math.degrees(math.atan2(dy, dx)))

    base = angles[0]
    for a in angles[1:]:
        diff = abs((a - base) % 180)
        if not (diff < 1 or abs(diff - 90) < 1):
            return False

    concave = 0
    for i in range(1, len(ex_coords) - 1):
        ax = ex_coords[i][0] - ex_coords[i - 1][0]
        ay = ex_coords[i][1] - ex_coords[i - 1][1]
        bx = ex_coords[i + 1][0] - ex_coords[i][0]
        by = ex_coords[i + 1][1] - ex_coords[i][1]
        cross = ax * by - ay * bx
        if cross < 0:
            concave += 1
    return concave == 2


if osmium is not None:
    class BuildingCollector(osmium.SimpleHandler):
        """Collect building ways from an OSM PBF file."""

        def __init__(self) -> None:
            super().__init__()
            self.buildings: List[Tuple[int, List[Coord]]] = []

        def way(self, w: osmium.osm.Way) -> None:  # type: ignore[override]
            if w.tags.get("building") and len(w.nodes) >= 4:
                coords: List[Coord] = []
                for n in w.nodes:
                    if not n.location.valid():
                        return
                    coords.append((n.lon, n.lat))
                if coords:
                    self.buildings.append((w.id, coords))
else:  # pragma: no cover - used when osmium is missing
    class BuildingCollector:  # type: ignore[no-redef]
        def __init__(self) -> None:
            raise RuntimeError("osmium is required to collect buildings")


def _write_map(matches: Iterable[Tuple[int, List[Coord]]], outfile: str) -> None:
    if folium is None:  # pragma: no cover - runtime check
        raise RuntimeError("folium is required for --map but is not installed")

    matches = list(matches)
    if not matches:
        return

    first_lat = matches[0][1][0][1]
    first_lon = matches[0][1][0][0]
    m = folium.Map(location=[first_lat, first_lon], zoom_start=18)
    fg = folium.FeatureGroup(name="T-shaped buildings")
    for bid, coords in matches:
        fg.add_child(
            folium.Polygon([(lat, lon) for lon, lat in coords], tooltip=str(bid))
        )
    m.add_child(fg)
    folium.LayerControl().add_to(m)
    m.save(outfile)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pbf", help="Path to OSM PBF file")
    parser.add_argument("--out", help="Write list of OSM links to this file")
    parser.add_argument("--map", help="Optional path to write an HTML map")
    args = parser.parse_args()

    if osmium is None:  # pragma: no cover - runtime check
        raise RuntimeError("osmium is required but not installed")

    collector = BuildingCollector()
    collector.apply_file(args.pbf, locations=True)

    matches: List[Tuple[int, List[Coord]]] = []
    links: List[str] = []
    for bid, coords in collector.buildings:
        if is_t_shape(coords):
            matches.append((bid, coords))
            links.append(f"https://www.openstreetmap.org/way/{bid}")

    for link in links:
        print(link)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(links))

    if args.map:
        _write_map(matches, args.map)


if __name__ == "__main__":
    main()

