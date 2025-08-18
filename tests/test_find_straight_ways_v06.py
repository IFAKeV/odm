import subprocess
import sys
from pathlib import Path


def test_only_unclassified() -> None:
    pbf = Path(__file__).resolve().parent.parent / "pbf" / "bremen-latest.osm.pbf"
    result = subprocess.run(
        [
            sys.executable,
            "find_straight_ways_v06.py",
            str(pbf),
            "--min-length",
            "1000",
            "--min-straightness",
            "0.999",
            "--top",
            "5",
            "--only-unclassified",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    lines = [line for line in result.stdout.splitlines() if line.startswith("Segment")]
    assert lines, "No segments returned"
    assert all("(unclassified)" in line for line in lines)

