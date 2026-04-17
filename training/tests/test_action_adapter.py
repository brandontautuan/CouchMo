"""Unit tests for the steer/throttle → cmd_vel action adapter.

Runs under pytest OR as a plain
``python training/tests/test_action_adapter.py`` on hosts that don't
have pytest installed (direct-exec fallback at module bottom). Both
modes are expected to work on Windows, macOS, and Linux — the tests
depend only on stdlib + the pure-Python :func:`steer_throttle_to_twist`
function. ``rclpy`` is NOT required.

Sign convention (pinned from ``serial_controller.py`` lines 123-124):
    ``("turn left", -0.5, 0.3)`` and ``("turn right", 0.5, 0.3)``
    → positive steer = right turn in IRL.

In this adapter angular.z > 0 means "right turn" (the policy path
deliberately mirrors IRL convention rather than REP-103). The simple
linear maps are::

    linear_x  = clamp(throttle, 0, 1) * MAX_LINEAR_MPS
    angular_z = clamp(steer, -1, 1)   * MAX_ANGULAR_RPS
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERT_PKG_SRC = REPO_ROOT / "simulation" / "src" / "couchmo_expert"
sys.path.insert(0, str(EXPERT_PKG_SRC))

from couchmo_expert.steer_throttle_to_cmd_vel import (  # noqa: E402
    steer_throttle_to_twist,
    MAX_LINEAR_MPS,
    MAX_ANGULAR_RPS,
)


# ---------------------------------------------------------------------------
# Core mapping tests (four cases from the plan spec)
# ---------------------------------------------------------------------------


def test_throttle_zero_gives_zero_linear_x() -> None:
    """throttle=0 → linear_x == 0 regardless of steer."""
    linear_x, angular_z = steer_throttle_to_twist(steer=0.0, throttle=0.0)
    assert linear_x == 0.0, (
        f"expected linear_x == 0.0 for throttle=0, got {linear_x}"
    )
    # Steer non-zero with zero throttle still gives zero linear_x.
    linear_x2, _ = steer_throttle_to_twist(steer=0.5, throttle=0.0)
    assert linear_x2 == 0.0, (
        f"expected linear_x == 0.0 for throttle=0 (steer=0.5), got {linear_x2}"
    )


def test_straight_throttle_gives_zero_angular_z() -> None:
    """steer=0, throttle>0 → angular_z == 0 and linear_x > 0."""
    linear_x, angular_z = steer_throttle_to_twist(steer=0.0, throttle=0.5)
    assert angular_z == 0.0, (
        f"expected angular_z == 0.0 for steer=0, got {angular_z}"
    )
    assert linear_x > 0.0, (
        f"expected linear_x > 0 for throttle=0.5, got {linear_x}"
    )


def test_positive_steer_positive_throttle_gives_positive_angular_z() -> None:
    """steer>0, throttle>0 → angular_z > 0 (right turn, matches IRL convention).

    Sign pinned from serial_controller.py:123-124:
        ("turn right", 0.5, 0.3) — positive steer = right turn in IRL.
    The adapter preserves this: positive steer → positive angular_z.
    """
    linear_x, angular_z = steer_throttle_to_twist(steer=0.5, throttle=0.3)
    assert angular_z > 0.0, (
        f"expected angular_z > 0 for steer=0.5 (right turn), got {angular_z}"
    )
    assert linear_x > 0.0, (
        f"expected linear_x > 0 for throttle=0.3, got {linear_x}"
    )


def test_mapping_monotonic_in_throttle() -> None:
    """linear_x grows monotonically as throttle increases (within [0, 1])."""
    previous_lx = -1.0
    throttle_values = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    for thr in throttle_values:
        linear_x, _ = steer_throttle_to_twist(steer=0.0, throttle=thr)
        assert linear_x > previous_lx, (
            f"expected strict increase: linear_x={linear_x} "
            f"not > previous={previous_lx} at throttle={thr}"
        )
        previous_lx = linear_x


def test_mapping_odd_in_steer() -> None:
    """angular_z is an odd function of steer: f(-s) == -f(s).

    This means the mapping is symmetric around zero steer (no built-in
    bias) — verified at several steer magnitudes.
    """
    for steer_pos in [0.1, 0.25, 0.5, 0.75, 1.0]:
        _, angular_z_pos = steer_throttle_to_twist(
            steer=steer_pos, throttle=0.5
        )
        _, angular_z_neg = steer_throttle_to_twist(
            steer=-steer_pos, throttle=0.5
        )
        assert abs(angular_z_pos + angular_z_neg) < 1e-12, (
            f"mapping not odd: f({steer_pos})={angular_z_pos}, "
            f"f({-steer_pos})={angular_z_neg}; sum={angular_z_pos + angular_z_neg}"
        )


# ---------------------------------------------------------------------------
# Clamping tests — ensures out-of-range inputs don't escape the adapter
# ---------------------------------------------------------------------------


def test_throttle_clamped_above_one() -> None:
    """throttle > 1.0 clamps to 1.0 → linear_x == MAX_LINEAR_MPS."""
    linear_x, _ = steer_throttle_to_twist(steer=0.0, throttle=2.0)
    expected = MAX_LINEAR_MPS
    assert abs(linear_x - expected) < 1e-12, (
        f"expected linear_x == {expected} (clamped throttle=2→1), "
        f"got {linear_x}"
    )


def test_negative_throttle_clamped_to_zero() -> None:
    """throttle < 0 clamps to 0 → linear_x == 0 (no reverse drive)."""
    linear_x, _ = steer_throttle_to_twist(steer=0.0, throttle=-0.5)
    assert linear_x == 0.0, (
        f"expected linear_x == 0 for negative throttle, got {linear_x}"
    )


def test_steer_clamped_above_one() -> None:
    """steer > 1.0 clamps to 1.0 → angular_z == MAX_ANGULAR_RPS."""
    _, angular_z = steer_throttle_to_twist(steer=3.0, throttle=0.5)
    expected = MAX_ANGULAR_RPS
    assert abs(angular_z - expected) < 1e-12, (
        f"expected angular_z == {expected} (clamped steer=3→1), "
        f"got {angular_z}"
    )


def test_steer_clamped_below_negative_one() -> None:
    """steer < -1.0 clamps to -1.0 → angular_z == -MAX_ANGULAR_RPS."""
    _, angular_z = steer_throttle_to_twist(steer=-3.0, throttle=0.5)
    expected = -MAX_ANGULAR_RPS
    assert abs(angular_z - expected) < 1e-12, (
        f"expected angular_z == {expected} (clamped steer=-3→-1), "
        f"got {angular_z}"
    )


# ---------------------------------------------------------------------------
# Known-value spot checks
# ---------------------------------------------------------------------------


def test_full_forward_no_steer() -> None:
    """throttle=1, steer=0 → linear_x == MAX_LINEAR_MPS, angular_z == 0."""
    linear_x, angular_z = steer_throttle_to_twist(steer=0.0, throttle=1.0)
    assert abs(linear_x - MAX_LINEAR_MPS) < 1e-12, (
        f"expected linear_x == {MAX_LINEAR_MPS}, got {linear_x}"
    )
    assert angular_z == 0.0, f"expected angular_z == 0, got {angular_z}"


def test_full_right_turn() -> None:
    """steer=1, throttle=1 → linear_x == MAX_LINEAR_MPS, angular_z == MAX_ANGULAR_RPS."""
    linear_x, angular_z = steer_throttle_to_twist(steer=1.0, throttle=1.0)
    assert abs(linear_x - MAX_LINEAR_MPS) < 1e-12, (
        f"expected linear_x == {MAX_LINEAR_MPS}, got {linear_x}"
    )
    assert abs(angular_z - MAX_ANGULAR_RPS) < 1e-12, (
        f"expected angular_z == {MAX_ANGULAR_RPS}, got {angular_z}"
    )


def test_full_left_turn() -> None:
    """steer=-1, throttle=1 → linear_x == MAX_LINEAR_MPS, angular_z == -MAX_ANGULAR_RPS."""
    linear_x, angular_z = steer_throttle_to_twist(steer=-1.0, throttle=1.0)
    assert abs(linear_x - MAX_LINEAR_MPS) < 1e-12, (
        f"expected linear_x == {MAX_LINEAR_MPS}, got {linear_x}"
    )
    assert abs(angular_z - (-MAX_ANGULAR_RPS)) < 1e-12, (
        f"expected angular_z == {-MAX_ANGULAR_RPS}, got {angular_z}"
    )


# ---------------------------------------------------------------------------
# Direct-exec fallback (matches test_pure_pursuit.py pattern)
# ---------------------------------------------------------------------------

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
