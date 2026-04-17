"""Tests for ``simulation/scripts/kml_to_waypoints.py``.

These tests exercise the pure stdlib KML/KMZ → local-ENU conversion
pipeline. They avoid ROS, Gazebo and torch on purpose so they can run
inside the training venv (or even plain CPython) without the sim stack.
"""
from __future__ import annotations

import math
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "simulation" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import kml_to_waypoints  # noqa: E402

FIXTURE_KML = SCRIPTS_DIR / "fixtures" / "synthetic_campus.kml"

# Expected ground-truth metres-per-degree at campus latitudes. At lat=37°
# and using the WGS84 equatorial radius (6378137 m), the small-angle
# tangent-plane approximation gives:
#   0.001° lat  -> 111.319 m north
#   0.001° lon  -> 88.922 m east  (|cos(37°)| ≈ 0.7986)
LAT_37_DEG_PER_0_001_DEG_LAT_M = 111.319
LAT_37_DEG_PER_0_001_DEG_LON_M = 88.922


def test_latlon_to_enu_zero_origin() -> None:
    x, y = kml_to_waypoints.latlon_to_enu(
        lat=37.0, lon=-120.0, origin_lat=37.0, origin_lon=-120.0
    )
    assert abs(x) < 1e-6, f"x should be 0 at origin, got {x}"
    assert abs(y) < 1e-6, f"y should be 0 at origin, got {y}"


def test_latlon_to_enu_known_offset() -> None:
    x_east, y_north = kml_to_waypoints.latlon_to_enu(
        lat=37.001, lon=-119.999, origin_lat=37.0, origin_lon=-120.0
    )
    assert abs(y_north - LAT_37_DEG_PER_0_001_DEG_LAT_M) < 0.5, (
        f"y_north={y_north} should be within 0.5 m of "
        f"{LAT_37_DEG_PER_0_001_DEG_LAT_M}"
    )
    assert abs(x_east - LAT_37_DEG_PER_0_001_DEG_LON_M) < 0.5, (
        f"x_east={x_east} should be within 0.5 m of "
        f"{LAT_37_DEG_PER_0_001_DEG_LON_M}"
    )
    x_west, _ = kml_to_waypoints.latlon_to_enu(
        lat=37.0, lon=-120.001, origin_lat=37.0, origin_lon=-120.0
    )
    assert x_west < 0, "going west of origin must give negative x_east"
    assert abs(abs(x_west) - LAT_37_DEG_PER_0_001_DEG_LON_M) < 0.5


def test_parse_linestring() -> None:
    rows = kml_to_waypoints.convert(
        FIXTURE_KML, origin_lat=37.0, origin_lon=-120.0
    )

    assert len(rows) == 5, f"expected 5 waypoints, got {len(rows)}"

    x0, y0, s0 = rows[0]
    assert abs(x0) < 1e-6 and abs(y0) < 1e-6, (
        f"first waypoint should be at (0, 0), got ({x0}, {y0})"
    )
    assert s0 == 0.0

    xs = np.array([r[0] for r in rows])
    ys = np.array([r[1] for r in rows])

    assert np.all(np.diff(ys[:3]) >= 0), (
        f"y should be monotonically non-decreasing across the first 3 "
        f"waypoints (north leg), got ys={ys[:3]}"
    )
    assert np.all(np.diff(xs[2:]) <= 0), (
        f"x should be monotonically non-increasing across the last 3 "
        f"waypoints (west leg), got xs={xs[2:]}"
    )

    expected_y_step = LAT_37_DEG_PER_0_001_DEG_LAT_M / 10.0
    expected_x_step = -LAT_37_DEG_PER_0_001_DEG_LON_M / 10.0
    assert abs(ys[1] - expected_y_step) < 0.5
    assert abs(ys[2] - 2 * expected_y_step) < 0.5
    assert abs(xs[3] - expected_x_step) < 0.5
    assert abs(xs[4] - 2 * expected_x_step) < 0.5


def test_parse_kmz() -> None:
    expected = kml_to_waypoints.convert(
        FIXTURE_KML, origin_lat=37.0, origin_lon=-120.0
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        kmz_path = Path(tmpdir) / "synthetic_campus.kmz"
        with zipfile.ZipFile(kmz_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(FIXTURE_KML, arcname="doc.kml")

        actual = kml_to_waypoints.convert(
            kmz_path, origin_lat=37.0, origin_lon=-120.0
        )

    assert len(actual) == len(expected)
    for (ax, ay, as_), (ex, ey, es) in zip(actual, expected):
        assert math.isclose(ax, ex, abs_tol=1e-9)
        assert math.isclose(ay, ey, abs_tol=1e-9)
        assert math.isclose(as_, es, abs_tol=1e-9)


def test_arc_length_monotonic() -> None:
    rows = kml_to_waypoints.convert(
        FIXTURE_KML, origin_lat=37.0, origin_lon=-120.0
    )
    s_values = [r[2] for r in rows]
    assert s_values[0] == 0.0
    diffs = np.diff(s_values)
    assert np.all(diffs > 0), (
        f"cumulative arc-length must be strictly increasing, got s={s_values}"
    )
    assert s_values[-1] > 0.0


if __name__ == "__main__":
    # Direct-exec fallback for hosts without pytest installed. Useful on
    # the Windows dev box where the training venv may not yet be set up.
    tests = [
        test_latlon_to_enu_zero_origin,
        test_latlon_to_enu_known_offset,
        test_parse_linestring,
        test_parse_kmz,
        test_arc_length_monotonic,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
        except Exception as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
        else:
            print(f"PASS {fn.__name__}")
    if failed:
        raise SystemExit(1)
