import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from find_t_shaped_buildings import is_t_shape


def test_is_t_shape_positive() -> None:
    coords = [
        (0.0, 0.0),
        (4.0, 0.0),
        (4.0, 1.0),
        (2.5, 1.0),
        (2.5, 3.0),
        (1.5, 3.0),
        (1.5, 1.0),
        (0.0, 1.0),
        (0.0, 0.0),
    ]
    assert is_t_shape(coords)


def test_is_t_shape_negative() -> None:
    coords = [
        (0.0, 0.0),
        (2.0, 0.0),
        (2.0, 1.0),
        (0.0, 1.0),
        (0.0, 0.0),
    ]
    assert not is_t_shape(coords)

