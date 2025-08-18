import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

# Ensure repository root is on the import path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from find_straight_ways_visual_v01 import HighwayCollector


def test_base_map_matches_overlay(tmp_path: Path) -> None:
    pbf = Path(__file__).resolve().parent.parent / "pbf" / "bremen-latest.osm.pbf"
    scale = 1000.0

    collector = HighwayCollector()
    collector.apply_file(str(pbf), locations=True)
    bbox = (collector.min_lon, collector.min_lat, collector.max_lon, collector.max_lat)
    width = max(int((bbox[2] - bbox[0]) * scale) + 1, 1)
    height = max(int((bbox[3] - bbox[1]) * scale) + 1, 1)
    meta = {
        "min_lon": bbox[0],
        "min_lat": bbox[1],
        "max_lon": bbox[2],
        "max_lat": bbox[3],
        "width": width,
        "height": height,
        "scale": scale,
    }
    bounds_file = tmp_path / "bounds.json"
    bounds_file.write_text(json.dumps(meta))

    # Generate highway overlays
    subprocess.run(
        [
            sys.executable,
            "find_straight_ways_visual_v01.py",
            str(pbf),
            "--outdir",
            str(tmp_path),
            "--scale",
            str(scale),
        ],
        check=True,
    )

    # Render base map using the same bounds metadata
    map_png = tmp_path / "map.png"
    subprocess.run(
        [
            sys.executable,
            "render_full_map.py",
            str(pbf),
            "--out",
            str(map_png),
            "--bounds-json",
            str(bounds_file),
        ],
        check=True,
    )

    overlay_png = tmp_path / "highway_primary.png"
    if not overlay_png.exists():
        overlay_png = (
            tmp_path / "highway_primary_secondary_tertiary_unclassified.png"
        )

    base_img = Image.open(map_png)
    overlay_img = Image.open(overlay_png)

    assert base_img.size == overlay_img.size
