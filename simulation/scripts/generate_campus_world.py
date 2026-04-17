#!/usr/bin/env python3
"""Generate a hybrid Gazebo Classic ``.world`` from waypoint CSV.

Part of the CouchMo Campus Sim codegen pipeline::

    KML/KMZ --kml_to_waypoints.py--> CSV (x_m,y_m,s_m)
        --generate_campus_world.py--> .world (SDF) --> Gazebo

The produced world is intentionally *not* photoreal. It emits:

* a Gazebo Classic standard ``sun`` and ``ground_plane`` include,
* one "sidewalk ribbon" ``<model>`` per consecutive waypoint pair — a
  thin (0.02 m) box with matched ``<collision>`` + ``<visual>`` so the
  couch can drive on it and bumping off it is detectable,
* a handful of deterministic static obstacle boxes (benches/signs)
  placed outside the corridor,
* a sane ODE ``<physics>`` block (1 ms step, 1000 Hz).

Depends only on the Python standard library so it runs identically on
Windows, macOS, and Linux without a virtual env.

Typical CLI usage::

    python generate_campus_world.py \\
        --waypoints simulation/scripts/fixtures/synthetic_campus.csv \\
        --out simulation/src/couchmo_description/worlds/synthetic_campus.world \\
        --seed 0 --world-name synthetic_campus
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

SIDEWALK_HEIGHT_M = 0.02
OBSTACLE_SIZE_M = (0.6, 0.4, 1.0)
OBSTACLE_CLEARANCE_M = 0.5
MAX_OBSTACLES = 5


def _read_waypoints_csv(path: Path) -> list[tuple[float, float]]:
    """Read an ``x_m,y_m,s_m`` CSV and return ``(x, y)`` pairs.

    ``s_m`` is ignored — this generator recomputes per-segment length
    from the Euclidean distance between consecutive points, so the CSV
    only needs two usable columns. The header is required and its
    first two columns must be ``x_m`` and ``y_m``.
    """
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None or header[:2] != ["x_m", "y_m"]:
            raise ValueError(
                f"{path}: expected CSV header starting with 'x_m,y_m', got {header!r}"
            )
        out: list[tuple[float, float]] = []
        for row in reader:
            if not row:
                continue
            out.append((float(row[0]), float(row[1])))
        return out


def _total_arc_length(waypoints: list[tuple[float, float]]) -> float:
    total = 0.0
    for (x0, y0), (x1, y1) in zip(waypoints, waypoints[1:]):
        total += math.hypot(x1 - x0, y1 - y0)
    return total


def _append_comment(parent: ET.Element, text: str) -> None:
    """Append an XML comment as a child of ``parent``.

    ``xml.etree.ElementTree`` has no first-class comment node API, so
    we stash an ``ET.Comment`` factory directly into the tree.
    """
    parent.append(ET.Comment(f" {text} "))


def _add_physics(world: ET.Element) -> None:
    physics = ET.SubElement(world, "physics", {"type": "ode"})
    ET.SubElement(physics, "max_step_size").text = "0.001"
    ET.SubElement(physics, "real_time_factor").text = "1.0"
    ET.SubElement(physics, "real_time_update_rate").text = "1000"
    ode = ET.SubElement(physics, "ode")
    solver = ET.SubElement(ode, "solver")
    ET.SubElement(solver, "type").text = "quick"
    ET.SubElement(solver, "iters").text = "50"


def _add_include(world: ET.Element, uri: str) -> None:
    inc = ET.SubElement(world, "include")
    ET.SubElement(inc, "uri").text = uri


def _add_sidewalk_ribbon(
    world: ET.Element,
    *,
    index: int,
    p0: tuple[float, float],
    p1: tuple[float, float],
    strip_w: float,
) -> float:
    """Append one sidewalk-ribbon ``<model>``; return segment length."""
    x0, y0 = p0
    x1, y1 = p1
    dx = x1 - x0
    dy = y1 - y0
    length = math.hypot(dx, dy)
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    yaw = math.atan2(dy, dx)
    z = 0.5 * SIDEWALK_HEIGHT_M
    size = f"{length:.6f} {strip_w:.6f} {SIDEWALK_HEIGHT_M:.6f}"

    model = ET.SubElement(world, "model", {"name": f"sidewalk_ribbon_{index:03d}"})
    ET.SubElement(model, "static").text = "true"
    ET.SubElement(model, "pose").text = (
        f"{cx:.6f} {cy:.6f} {z:.6f} 0 0 {yaw:.6f}"
    )
    link = ET.SubElement(model, "link", {"name": "link"})

    collision = ET.SubElement(link, "collision", {"name": "col"})
    col_geom = ET.SubElement(collision, "geometry")
    col_box = ET.SubElement(col_geom, "box")
    ET.SubElement(col_box, "size").text = size

    visual = ET.SubElement(link, "visual", {"name": "vis"})
    vis_geom = ET.SubElement(visual, "geometry")
    vis_box = ET.SubElement(vis_geom, "box")
    ET.SubElement(vis_box, "size").text = size
    material = ET.SubElement(visual, "material")
    ET.SubElement(material, "ambient").text = "0.55 0.55 0.55 1"
    ET.SubElement(material, "diffuse").text = "0.65 0.65 0.65 1"

    return length


def _add_obstacle(
    world: ET.Element,
    *,
    index: int,
    px: float,
    py: float,
    yaw: float,
) -> None:
    sx, sy, sz = OBSTACLE_SIZE_M
    size = f"{sx:.6f} {sy:.6f} {sz:.6f}"
    model = ET.SubElement(world, "model", {"name": f"obstacle_{index:03d}"})
    ET.SubElement(model, "static").text = "true"
    ET.SubElement(model, "pose").text = (
        f"{px:.6f} {py:.6f} {0.5 * sz:.6f} 0 0 {yaw:.6f}"
    )
    link = ET.SubElement(model, "link", {"name": "link"})

    collision = ET.SubElement(link, "collision", {"name": "col"})
    col_geom = ET.SubElement(collision, "geometry")
    col_box = ET.SubElement(col_geom, "box")
    ET.SubElement(col_box, "size").text = size

    visual = ET.SubElement(link, "visual", {"name": "vis"})
    vis_geom = ET.SubElement(visual, "geometry")
    vis_box = ET.SubElement(vis_geom, "box")
    ET.SubElement(vis_box, "size").text = size
    material = ET.SubElement(visual, "material")
    ET.SubElement(material, "ambient").text = "0.35 0.25 0.15 1"
    ET.SubElement(material, "diffuse").text = "0.45 0.30 0.20 1"


def _scatter_obstacles(
    world: ET.Element,
    *,
    waypoints: list[tuple[float, float]],
    corridor_width_m: float,
    seed: int,
) -> int:
    """Place up to ``MAX_OBSTACLES`` boxes perpendicular to random segments.

    Uses a local ``random.Random(seed)`` so multi-process callers can
    generate worlds in parallel without the global RNG stomping on
    each other. Obstacles are pushed out by
    ``corridor_width_m/2 + OBSTACLE_CLEARANCE_M`` so they can never
    intersect the sidewalk ribbon.
    """
    n_segments = len(waypoints) - 1
    if n_segments <= 0:
        return 0
    count = min(MAX_OBSTACLES, n_segments)
    rng = random.Random(seed)
    perp_dist = corridor_width_m * 0.5 + OBSTACLE_CLEARANCE_M
    placed = 0
    for i in range(count):
        seg_idx = rng.randrange(n_segments)
        t = rng.random()
        side = 1.0 if rng.random() < 0.5 else -1.0
        x0, y0 = waypoints[seg_idx]
        x1, y1 = waypoints[seg_idx + 1]
        dx = x1 - x0
        dy = y1 - y0
        length = math.hypot(dx, dy)
        if length == 0.0:
            continue
        ux = dx / length
        uy = dy / length
        cx = x0 + t * dx
        cy = y0 + t * dy
        nx = -uy
        ny = ux
        px = cx + side * perp_dist * nx
        py = cy + side * perp_dist * ny
        yaw = math.atan2(dy, dx)
        _add_obstacle(world, index=i, px=px, py=py, yaw=yaw)
        placed += 1
    return placed


def generate(
    waypoints: list[tuple[float, float]],
    *,
    sidewalk_strip_w_m: float = 1.5,
    corridor_width_m: float = 3.0,
    seed: int = 0,
    world_name: str = "synthetic_campus",
    source_csv: str | None = None,
) -> str:
    """Build a Gazebo Classic SDF world string for a waypoint route.

    Parameters
    ----------
    waypoints:
        Ordered list of ``(x_m, y_m)`` points in local ENU metres.
        Consecutive pairs become sidewalk-ribbon segments. A minimum
        of two waypoints is required.
    sidewalk_strip_w_m:
        Width of each sidewalk-ribbon box in metres. The nav-grade
        collision surface the couch actually drives on.
    corridor_width_m:
        Total logical corridor width in metres. Recorded in the world
        header and used to offset obstacles perpendicular to the path.
    seed:
        Deterministic seed for obstacle placement. Same seed + same
        waypoints -> byte-identical output.
    world_name:
        Becomes the SDF ``<world name="...">`` attribute.
    source_csv:
        Optional path string recorded in the world header for
        provenance. Does not affect geometry.

    Returns
    -------
    str
        Complete ``.world`` XML text ending in a newline.
    """
    if len(waypoints) < 2:
        raise ValueError(
            f"generate() needs at least 2 waypoints to form 1 segment, "
            f"got {len(waypoints)}"
        )

    sdf = ET.Element("sdf", {"version": "1.6"})
    world = ET.SubElement(sdf, "world", {"name": world_name})

    _append_comment(
        world,
        f"corridor_width_m={corridor_width_m} "
        f"sidewalk_strip_w_m={sidewalk_strip_w_m} "
        f"seed={seed} "
        f"source_csv={source_csv or 'unknown'}",
    )

    _add_physics(world)
    _add_include(world, "model://sun")
    _add_include(world, "model://ground_plane")

    _append_comment(world, "Sidewalk ribbon strips (one per waypoint segment)")
    for i, (p0, p1) in enumerate(zip(waypoints, waypoints[1:])):
        _add_sidewalk_ribbon(
            world,
            index=i,
            p0=p0,
            p1=p1,
            strip_w=sidewalk_strip_w_m,
        )

    _append_comment(world, "Static obstacle templates (deterministic)")
    _scatter_obstacles(
        world,
        waypoints=waypoints,
        corridor_width_m=corridor_width_m,
        seed=seed,
    )

    ET.indent(sdf, space="  ")
    body = ET.tostring(sdf, encoding="unicode")

    header_comment = (
        "<!-- Auto-generated by simulation/scripts/generate_campus_world.py "
        "- do not edit by hand. -->\n"
        f"<!-- source_csv={source_csv or 'unknown'} "
        f"corridor_width_m={corridor_width_m} "
        f"sidewalk_strip_w_m={sidewalk_strip_w_m} "
        f"seed={seed} -->\n"
    )
    return '<?xml version="1.0"?>\n' + header_comment + body + "\n"


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Generate a Gazebo Classic .world (SDF) from a waypoint CSV."
        ),
    )
    p.add_argument(
        "--waypoints",
        type=Path,
        required=True,
        help="Input CSV with header 'x_m,y_m,s_m'.",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output .world path (parent dirs are created).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic seed for obstacle placement (default: 0).",
    )
    p.add_argument(
        "--corridor-width-m",
        type=float,
        default=3.0,
        help="Logical corridor width in metres (default: 3.0).",
    )
    p.add_argument(
        "--sidewalk-strip-w-m",
        type=float,
        default=1.5,
        help="Sidewalk ribbon box width in metres (default: 1.5).",
    )
    p.add_argument(
        "--world-name",
        type=str,
        default="synthetic_campus",
        help="SDF world name attribute (default: synthetic_campus).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    waypoints_path: Path = args.waypoints
    out_path: Path = args.out

    waypoints = _read_waypoints_csv(waypoints_path)
    if len(waypoints) < 2:
        print(
            f"error: {waypoints_path} has {len(waypoints)} waypoint(s); "
            f"need at least 2 to form a sidewalk segment",
            file=sys.stderr,
        )
        return 2

    world_xml = generate(
        waypoints,
        sidewalk_strip_w_m=args.sidewalk_strip_w_m,
        corridor_width_m=args.corridor_width_m,
        seed=args.seed,
        world_name=args.world_name,
        source_csv=waypoints_path.as_posix(),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(world_xml, encoding="utf-8")
    total = _total_arc_length(waypoints)
    print(
        f"generate_campus_world: wrote {len(waypoints) - 1} segments "
        f"(~{total:.2f} m arc) to {out_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
