import subprocess
import sys
from pathlib import Path


def test_count_straight_ways() -> None:
    pbf = Path(__file__).resolve().parent.parent / "pbf" / "bremen-latest.osm.pbf"
    result = subprocess.run(
        [
            sys.executable,
            "count_straight_ways_v01.py",
            str(pbf),
            "--length",
            "1000",
            "--min-straightness",
            "0.999",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "36"
