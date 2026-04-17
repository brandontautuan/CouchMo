"""Unit tests for eval_policy.check_thresholds (go/no-go gate).

Pure-python — does not need torch / metadrive / onnxruntime, so it always
runs, unlike test_metadrive_eval.py which skips on dep-light hosts. This
is the one contract that guards real-hardware rollout, so it deserves
coverage even in the minimal test environment.
"""
from __future__ import annotations

from training.metadrive.eval_policy import GO_NO_GO, check_thresholds


def _report(collision: float, off_road: float, length: float) -> dict:
    return {
        "episodes": 500,
        "collision_rate": collision,
        "off_road_rate": off_road,
        "mean_episode_length": length,
    }


def test_all_thresholds_met_returns_empty():
    r = _report(
        collision=GO_NO_GO["collision_rate_max"] - 0.01,
        off_road=GO_NO_GO["off_road_rate_max"] - 0.01,
        length=GO_NO_GO["mean_episode_length_min"] + 1.0,
    )
    assert check_thresholds(r) == []


def test_boundary_values_are_passing():
    """The gate uses strict >, so values exactly at the threshold pass."""
    r = _report(
        collision=GO_NO_GO["collision_rate_max"],
        off_road=GO_NO_GO["off_road_rate_max"],
        length=GO_NO_GO["mean_episode_length_min"],
    )
    assert check_thresholds(r) == []


def test_collision_rate_fail_reported():
    r = _report(
        collision=GO_NO_GO["collision_rate_max"] + 0.01,
        off_road=0.0,
        length=GO_NO_GO["mean_episode_length_min"] + 1.0,
    )
    failures = check_thresholds(r)
    assert len(failures) == 1
    assert "collision_rate" in failures[0]


def test_off_road_rate_fail_reported():
    r = _report(
        collision=0.0,
        off_road=GO_NO_GO["off_road_rate_max"] + 0.01,
        length=GO_NO_GO["mean_episode_length_min"] + 1.0,
    )
    failures = check_thresholds(r)
    assert len(failures) == 1
    assert "off_road_rate" in failures[0]


def test_length_fail_reported():
    r = _report(
        collision=0.0,
        off_road=0.0,
        length=GO_NO_GO["mean_episode_length_min"] - 1.0,
    )
    failures = check_thresholds(r)
    assert len(failures) == 1
    assert "mean_episode_length" in failures[0]


def test_all_thresholds_failing_reports_all_three():
    r = _report(
        collision=GO_NO_GO["collision_rate_max"] + 0.01,
        off_road=GO_NO_GO["off_road_rate_max"] + 0.01,
        length=GO_NO_GO["mean_episode_length_min"] - 1.0,
    )
    failures = check_thresholds(r)
    assert len(failures) == 3
