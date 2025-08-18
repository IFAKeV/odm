import re
import subprocess
import sys
from pathlib import Path


def test_find_housenumber8_far(tmp_path: Path) -> None:
    pbf = Path(__file__).resolve().parent.parent / "pbf" / "bremen-latest.osm.pbf"
    out = tmp_path / "houses.html"
    result = subprocess.run(
        [
            sys.executable,
            "find_housenumber8_far.py",
            str(pbf),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.exists()
    m = re.search(r"Found (\d+) buildings", result.stdout)
    assert m and int(m.group(1)) >= 100
