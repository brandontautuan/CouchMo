"""Unit tests for the pure-pursuit waypoint expert.

Runs under pytest OR as a plain ``python training/tests/test_pure_pursuit.py``
on hosts that don't have pytest installed (direct-exec fallback at
module bottom). Both modes are expected to work identically on Windows,
macOS, and Linux — the tests depend only on stdlib + NumPy + the
pure-Python :class:`PurePursuitExpert`.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERT_PKG_SRC = REPO_ROOT / "simulation" / "src" / "couchmo_expert"
sys.path.insert(0, str(EXPERT_PKG_SRC))

from couchmo_expert.waypoint_expert_node import (  # noqa: E402
    PurePursuitExpert,
    STEER_POS_DIR,
    load_waypoints_csv,
)

FIXTURE_CSV = (
    REPO_ROOT / "simulation" / "scripts" / "fixtures" / "synthetic_campus.csv"
)


def _make_expert(waypoints: np.ndarray, **overrides: float) -> PurePursuitExpert:
    """Build a :class:`PurePursuitExpert` with sensible test defaults.

    The defaults match ``config/waypoint_follower.yaml`` so these tests
    exercise the same parameter surface the ROS wrapper ships with.
    """
    params: dict[str, object] = dict(
        waypoints=waypoints,
        lookahead_min_m=1.0,
        lookahead_max_m=4.0,
        lookahead_speed_cap_mps=2.0,
        max_throttle=0.6,
        curvature_throttle_gain=1.0,
        wheelbase_m=0.6,
        max_curvature_at_full_steer=1.0,
    )
    params.update(overrides)
    return PurePursuitExpert(**params)  # type: ignore[arg-type]


def _straight_east_waypoints(n: int = 6, spacing_m: float = 1.0) -> np.ndarray:
    """Return ``n`` waypoints on the +x axis, spaced ``spacing_m`` apart."""
    return np.array(
        [[float(i) * spacing_m, 0.0] for i in range(n)], dtype=np.float64
    )


def test_straight_path_zero_steer() -> None:
    """Pose on centerline of a straight east path → near-zero steer."""
    expert = _make_expert(_straight_east_waypoints())
    steer, _, _ = expert.compute(
        pose_xy=(2.0, 0.0), yaw_rad=0.0, speed_mps=1.0
    )
    assert abs(steer) < 1e-3, (
        f"expected |steer| < 1e-3 on centerline, got {steer}"
    )


def test_lateral_offset_corrects_toward_centerline() -> None:
    """Pose 1 m north of east-going path → positive (right) steer.

    Sign convention pinned from ``serial_controller.py:67`` — see the
    ``STEER_POS_DIR`` comment in :mod:`couchmo_expert.waypoint_expert_node`.
    """
    assert STEER_POS_DIR == -1.0, (
        "STEER_POS_DIR should be -1.0 to match serial_controller.py:67 "
        "(positive steer = right)"
    )
    expert = _make_expert(_straight_east_waypoints())
    steer, _, _ = expert.compute(
        pose_xy=(2.0, 1.0), yaw_rad=0.0, speed_mps=1.0
    )
    assert steer > 0.0, (
        f"expected positive (right, toward south centerline) steer, got {steer}"
    )


def test_lateral_offset_opposite_side_inverse_sign() -> None:
    """Offsets on opposite sides of the centerline give opposite steer signs."""
    expert = _make_expert(_straight_east_waypoints())
    steer_north, _, _ = expert.compute(
        pose_xy=(2.0, 1.0), yaw_rad=0.0, speed_mps=1.0
    )
    steer_south, _, _ = expert.compute(
        pose_xy=(2.0, -1.0), yaw_rad=0.0, speed_mps=1.0
    )
    assert steer_north > 0.0, f"north offset should steer right, got {steer_north}"
    assert steer_south < 0.0, f"south offset should steer left, got {steer_south}"
    assert np.sign(steer_north) == -np.sign(steer_south), (
        f"opposite sides must produce opposite signs: "
        f"north={steer_north}, south={steer_south}"
    )


def test_throttle_decreases_on_curvature() -> None:
    """Throttle on the 90° L-bend segment is lower than on the straight leg."""
    wps = load_waypoints_csv(FIXTURE_CSV)
    expert = _make_expert(wps)

    # Straight leg: pose mid-way along the first (northbound) segment.
    _, thr_straight, info_straight = expert.compute(
        pose_xy=(0.0, 5.0), yaw_rad=math.pi / 2.0, speed_mps=1.0
    )
    # Corner: pose exactly at the bend waypoint, still heading north.
    # Target lies ~8.89 m due west, maximal lateral → high curvature.
    _, thr_turn, info_turn = expert.compute(
        pose_xy=(0.0, 22.263898), yaw_rad=math.pi / 2.0, speed_mps=1.0
    )
    assert thr_turn < thr_straight, (
        f"corner throttle should be < straight throttle; "
        f"got straight={thr_straight} (target_idx={info_straight['target_idx']}), "
        f"turn={thr_turn} (target_idx={info_turn['target_idx']})"
    )


def test_throttle_capped_in_zero_one() -> None:
    """Random poses near the path → throttle always in [0, 1]."""
    wps = load_waypoints_csv(FIXTURE_CSV)
    expert = _make_expert(wps)
    rng = np.random.default_rng(seed=0)
    for _ in range(200):
        pose = (float(rng.uniform(-25.0, 10.0)), float(rng.uniform(-5.0, 30.0)))
        yaw = float(rng.uniform(-math.pi, math.pi))
        speed = float(rng.uniform(0.0, 3.0))
        _, throttle, _ = expert.compute(
            pose_xy=pose, yaw_rad=yaw, speed_mps=speed
        )
        assert 0.0 <= throttle <= 1.0, (
            f"throttle {throttle} outside [0, 1] at pose={pose}, "
            f"yaw={yaw}, speed={speed}"
        )


def test_steer_capped_in_negative_one_to_one() -> None:
    """Random poses near the path → steer always in [-1, 1]."""
    wps = load_waypoints_csv(FIXTURE_CSV)
    expert = _make_expert(wps)
    rng = np.random.default_rng(seed=1)
    for _ in range(200):
        pose = (float(rng.uniform(-25.0, 10.0)), float(rng.uniform(-5.0, 30.0)))
        yaw = float(rng.uniform(-math.pi, math.pi))
        speed = float(rng.uniform(0.0, 3.0))
        steer, _, _ = expert.compute(
            pose_xy=pose, yaw_rad=yaw, speed_mps=speed
        )
        assert -1.0 <= steer <= 1.0, (
            f"steer {steer} outside [-1, 1] at pose={pose}, "
            f"yaw={yaw}, speed={speed}"
        )


def test_lookahead_finds_correct_target() -> None:
    """Chosen target is the first waypoint beyond position + lookahead."""
    wps = _straight_east_waypoints(n=6, spacing_m=1.0)
    expert = _make_expert(wps)
    # At speed=1.0, lookahead = 1.0 + (4.0-1.0) * (1.0/2.0) = 2.5 m.
    # Pose at x=0.3 on centerline → closest = wp0, cumulative walk yields
    # target=wp3 (cumulative 3.0 m >= 2.5). The "first beyond x=0.3+2.5=2.8"
    # check independently gives wp3 (x=3.0).
    _, _, info = expert.compute(
        pose_xy=(0.3, 0.0), yaw_rad=0.0, speed_mps=1.0
    )
    pose_x = 0.3
    threshold = pose_x + info["lookahead_m"]
    beyond_mask = wps[:, 0] > threshold
    assert beyond_mask.any(), "no waypoints beyond threshold — bad test fixture"
    expected_idx = int(np.argmax(beyond_mask))
    assert info["target_idx"] == expected_idx, (
        f"target_idx={info['target_idx']} != expected {expected_idx} "
        f"(lookahead={info['lookahead_m']} m, threshold x={threshold})"
    )


def test_finishes_path_emits_zero_throttle() -> None:
    """Pose past the last waypoint → throttle == 0 and info['done'] is True."""
    wps = load_waypoints_csv(FIXTURE_CSV)
    expert = _make_expert(wps)
    last = wps[-1]
    prev = wps[-2]
    direction = last - prev
    direction = direction / float(np.linalg.norm(direction))
    past_pose = last + direction * 2.0
    _, throttle, info = expert.compute(
        pose_xy=(float(past_pose[0]), float(past_pose[1])),
        yaw_rad=float(math.atan2(direction[1], direction[0])),
        speed_mps=1.0,
    )
    assert throttle == 0.0, (
        f"expected throttle=0 past end of path, got {throttle}"
    )
    assert info["done"] is True, (
        f"expected info['done']=True past end of path, got {info['done']}"
    )


if __name__ == "__main__":
    tests = sorted(
        (name, fn)
        for name, fn in globals().items()
        if name.startswith("test_") and callable(fn)
    )
    failed: list[tuple[str, BaseException]] = []
    for name, fn in tests:
        try:
            fn()
        except BaseException as exc:  # noqa: BLE001 — surface every failure
            print(f"FAIL {name}: {exc!r}")
            failed.append((name, exc))
        else:
            print(f"PASS {name}")
    print()
    if failed:
        print(f"{len(failed)}/{len(tests)} tests FAILED")
        sys.exit(1)
    print(f"{len(tests)}/{len(tests)} tests PASSED")
