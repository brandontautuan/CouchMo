"""Tests for ``simulation/scripts/generate_campus_world.py``.

The generator is pure-stdlib XML codegen (no ROS, no Gazebo), so these
tests can run inside any CPython env. Every test drives the actual
``synthetic_campus.csv`` fixture produced by
``kml_to_waypoints.py`` — we never hand-roll a parallel CSV in
tests, because that would let the two tools silently diverge.
"""
from __future__ import annotations

import csv
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "simulation" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import generate_campus_world  # noqa: E402

FIXTURE_CSV = SCRIPTS_DIR / "fixtures" / "synthetic_campus.csv"


def _load_waypoints() -> list[tuple[float, float]]:
    with FIXTURE_CSV.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader)
        return [(float(r[0]), float(r[1])) for r in reader]


def _fixture_total_arc_m() -> float:
    with FIXTURE_CSV.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader)
        last_s = 0.0
        for row in reader:
            last_s = float(row[2])
        return last_s


def test_generated_world_is_well_formed_xml() -> None:
    xml = generate_campus_world.generate(_load_waypoints(), seed=0)
    ET.fromstring(xml)  # raises ParseError on malformed XML


def test_generated_world_contains_world_element() -> None:
    xml = generate_campus_world.generate(
        _load_waypoints(), seed=0, world_name="synthetic_campus"
    )
    root = ET.fromstring(xml)
    assert root.tag == "sdf", f"root must be <sdf>, got <{root.tag}>"
    worlds = root.findall("world")
    assert len(worlds) == 1, f"expected exactly 1 <world>, got {len(worlds)}"
    assert worlds[0].get("name") == "synthetic_campus"


def test_generated_world_includes_ground_plane() -> None:
    xml = generate_campus_world.generate(_load_waypoints(), seed=0)
    root = ET.fromstring(xml)
    world = root.find("world")
    assert world is not None
    uris = [inc.findtext("uri") for inc in world.findall("include")]
    assert "model://ground_plane" in uris, (
        f"expected <include><uri>model://ground_plane</uri></include>, "
        f"got includes={uris}"
    )


def test_generated_world_has_sidewalk_ribbon_models_per_segment() -> None:
    waypoints = _load_waypoints()
    n = len(waypoints)
    xml = generate_campus_world.generate(waypoints, seed=0)
    root = ET.fromstring(xml)
    world = root.find("world")
    assert world is not None
    ribbons = [
        m for m in world.findall("model")
        if (m.get("name") or "").startswith("sidewalk_ribbon_")
    ]
    assert len(ribbons) >= n - 1, (
        f"expected >= {n - 1} sidewalk ribbons for {n} waypoints, "
        f"got {len(ribbons)}"
    )
    for ribbon in ribbons:
        link = ribbon.find("link")
        assert link is not None, f"{ribbon.get('name')} is missing <link>"
        assert link.find("collision") is not None, (
            f"{ribbon.get('name')} is missing <collision>"
        )
        assert link.find("visual") is not None, (
            f"{ribbon.get('name')} is missing <visual>"
        )


def test_sidewalk_ribbon_lengths_match_arc_length_within_5pct() -> None:
    waypoints = _load_waypoints()
    xml = generate_campus_world.generate(waypoints, seed=0)
    root = ET.fromstring(xml)
    world = root.find("world")
    assert world is not None

    total_ribbon_length = 0.0
    for model in world.findall("model"):
        if not (model.get("name") or "").startswith("sidewalk_ribbon_"):
            continue
        box_size = model.find("link/collision/geometry/box/size")
        assert box_size is not None and box_size.text is not None
        length = float(box_size.text.split()[0])
        total_ribbon_length += length

    expected = _fixture_total_arc_m()
    assert expected > 0.0
    rel_err = abs(total_ribbon_length - expected) / expected
    assert rel_err < 0.05, (
        f"ribbon length sum {total_ribbon_length:.3f} m differs from "
        f"arc-length {expected:.3f} m by {rel_err * 100:.2f}% (> 5%)"
    )


def test_deterministic_with_fixed_seed() -> None:
    waypoints = _load_waypoints()
    a = generate_campus_world.generate(waypoints, seed=0)
    b = generate_campus_world.generate(waypoints, seed=0)
    assert a == b, "same seed + same waypoints must produce identical XML"
    assert a.encode("utf-8") == b.encode("utf-8"), (
        "byte-for-byte output must match with --seed 0"
    )


def test_corridor_width_default_3m() -> None:
    xml = generate_campus_world.generate(_load_waypoints(), seed=0)
    match = re.search(r"corridor_width_m=([0-9.]+)", xml)
    assert match is not None, (
        "expected 'corridor_width_m=...' metadata in the generated world "
        "header comment but found none"
    )
    assert float(match.group(1)) == 3.0, (
        f"default corridor width must be 3.0 m, world declares "
        f"{match.group(1)}"
    )


if __name__ == "__main__":
    tests = [
        test_generated_world_is_well_formed_xml,
        test_generated_world_contains_world_element,
        test_generated_world_includes_ground_plane,
        test_generated_world_has_sidewalk_ribbon_models_per_segment,
        test_sidewalk_ribbon_lengths_match_arc_length_within_5pct,
        test_deterministic_with_fixed_seed,
        test_corridor_width_default_3m,
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
