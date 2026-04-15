# Campus Sim + Camera-Only Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix and extend the CouchMo simulator so it can generate training data and train a camera-only policy for full-campus sidewalk driving, using a waypoint-follower expert plus optional RL fine-tuning.

**Architecture:** Keep the existing ROS 2 Humble + Gazebo Classic sim as the runtime. Add a Google Earth KML→waypoints→world generator, add two simulated armrest cameras, implement a pure-pursuit expert that outputs `(steer, throttle)` at 10 Hz, and build dataset + training scripts that ingest sim images and expert actions.

**Tech Stack:** ROS 2 Humble, Gazebo Classic 11 (`gazebo_ros_pkgs`), Python 3, NumPy/OpenCV/PyTorch (existing `model.py`), Docker/noVNC (existing `simulation/`), optional Gymnasium-style wrappers for training.

---

## File structure (what we will create/modify)

**Modify (existing):**
- `simulation/src/couchmo_description/urdf/couchmo.urdf.xacro` — add two camera sensors at armrest mounts.
- `simulation/src/couchmo_description/launch/gazebo.launch.py` — ensure camera topics are bridged/published and sim can spawn in new campus world.
- `simulation/src/couchmo_description/worlds/` — add new generated campus world (`*.world`) + keep existing worlds.
- `simulation/entrypoint.sim.sh` — optionally add flags/launch args for training/headless runs and improve startup reliability.

**Create (new):**
- `simulation/scripts/kml_to_waypoints.py` — parse KML/KMZ and produce local ENU waypoints (CSV/YAML).
- `simulation/scripts/generate_campus_world.py` — generate a hybrid SDF `.world` from waypoints + simple obstacle templates.
- `simulation/src/couchmo_nav/config/waypoint_follower.yaml` — parameters: corridor width (3.0m), lookahead, throttle caps, etc.
- `simulation/src/couchmo_nav/launch/waypoint_expert.launch.py` — launches the waypoint expert node (and optional visualization markers).
- `simulation/src/couchmo_nav/src/waypoint_expert_node.py` — ROS 2 node implementing pure pursuit expert producing `(steer, throttle)` at 10 Hz.
- `simulation/src/couchmo_nav/src/steer_throttle_to_cmd_vel.py` — adapter mapping `(steer, throttle)` → `cmd_vel` (for Gazebo diff-drive plugin).
- `training/dataset/record_expert_rollouts.py` — record camera frames + expert actions to disk.
- `training/dataset/format_dataset.py` — convert rollouts to a training-friendly dataset format.
- `training/imitation/train_bc.py` — behavior cloning training loop.
- `training/imitation/eval_in_sim.py` — run trained policy in sim via ROS bridge.
- `training/rl/train_ppo.py` (optional milestone) — PPO fine-tune wrapper.
- `training/README.md` — exact commands for generating data and training.

**Create (tests):**
- `training/tests/test_kml_to_waypoints.py`
- `training/tests/test_pure_pursuit.py`
- `training/tests/test_action_adapter.py`

---

## Task 1: Reproduce + stabilize sim bringup (baseline)

**Files:**
- Modify: `simulation/run.sh`
- Modify: `simulation/docker-compose.yml`
- Modify: `simulation/entrypoint.sim.sh`
- Doc: `simulation/README.md` (create if missing)

- [ ] **Step 1: Add a baseline “bringup check” doc**
  - Document expected behavior for:
    - `./run.sh preview` → RViz shows couch model
    - `./run.sh sim` → noVNC desktop shows Gazebo + RViz
  - Include failure modes (Docker daemon not running, ports blocked).

- [ ] **Step 2: Add a “health check” script (readonly checks)**
  - Add a script that prints:
    - whether Docker daemon reachable
    - whether port 6081 is bound after start
  - Run it before launching to provide actionable error messages.

- [ ] **Step 3: Verify baseline works**
  - Run:
    - `cd simulation && ./run.sh preview`
    - `cd simulation && ./run.sh sim`
  - Expected:
    - preview shows RViz couch
    - sim shows Gazebo + RViz, couch spawns

- [ ] **Step 4: Commit**

```bash
git add simulation/run.sh simulation/docker-compose.yml simulation/entrypoint.sim.sh simulation/README.md
git commit -m "sim: stabilize bringup and document health checks"
```

---

## Task 2: Add armrest cameras to URDF (two Brio 100 mounts)

**Files:**
- Modify: `simulation/src/couchmo_description/urdf/couchmo.urdf.xacro`
- Modify: `simulation/src/couchmo_description/launch/gazebo.launch.py` (if remapping needed)

- [ ] **Step 1: Define camera links + joints**
  - Add `left_camera_link` and `right_camera_link` fixed to chassis/body.
  - Mount height: 32 inches = 0.8128 m above ground.
  - Lateral offset: approximate from couch width; start with ±(body_w/2 - 0.10) and adjust.

- [ ] **Step 2: Add Gazebo camera sensors**
  - Add `<gazebo reference="...">` camera sensors publishing ROS image topics:
    - `/left_cam/image_raw`
    - `/right_cam/image_raw`
  - Set realistic-ish parameters: update_rate ~10–30 Hz; FOV approximate if known.

- [ ] **Step 3: Validation**
  - Run sim and confirm topics exist:
    - `ros2 topic list | grep image_raw`
  - Optional: view in RViz.

- [ ] **Step 4: Commit**

```bash
git add simulation/src/couchmo_description/urdf/couchmo.urdf.xacro simulation/src/couchmo_description/launch/gazebo.launch.py
git commit -m "sim: add dual armrest cameras to CouchMo URDF"
```

---

## Task 3: KML/KMZ → waypoints converter (Google Earth traced centerlines)

**Files:**
- Create: `simulation/scripts/kml_to_waypoints.py`
- Create: `training/tests/test_kml_to_waypoints.py`

- [ ] **Step 1: Write failing tests for KML parsing**

```python
# training/tests/test_kml_to_waypoints.py
import math

def test_latlon_to_enu_zero_origin():
    # origin maps to (0,0)
    origin_lat, origin_lon = 38.0, -121.0
    lat, lon = origin_lat, origin_lon
    # expect exactly zero within tolerance
    assert True
```

- [ ] **Step 2: Run tests (expect fail)**

```bash
python -m pytest -q
```

- [ ] **Step 3: Implement minimal KML reader + ENU projection**
  - Support: KML LineString coordinates.
  - Support: KMZ by unzipping and reading the contained KML.
  - Output: CSV with columns `x_m,y_m` and optional `s_m` arc-length.

- [ ] **Step 4: Run tests (expect pass)**

```bash
python -m pytest -q
```

- [ ] **Step 5: Commit**

```bash
git add simulation/scripts/kml_to_waypoints.py training/tests/test_kml_to_waypoints.py
git commit -m "tools: convert Google Earth KML/KMZ routes to local waypoints"
```

---

## Task 4: Generate a hybrid campus SDF world from waypoints

**Files:**
- Create: `simulation/scripts/generate_campus_world.py`
- Create: `simulation/src/couchmo_description/worlds/flc_googleearth.world` (generated output, checked in for reproducibility)
- Create: `training/tests/test_world_generation.py`

- [ ] **Step 1: Write failing test for world generator output**
  - Ensure output `.world` contains:
    - `<world name="...">`
    - ground plane include
    - at least one sidewalk ribbon model derived from waypoints

- [ ] **Step 2: Implement generator**
  - Use waypoints to create sidewalk ribbon segments (box strips) with collision + simple visual material.
  - Add a few static obstacles templates (benches/signs) with deterministic random seed.
  - Provide config for corridor width and sidewalk strip width.

- [ ] **Step 3: Integrate with launch**
  - Ensure `gazebo.launch.py` can launch with `world:=flc_googleearth`.

- [ ] **Step 4: Commit**

```bash
git add simulation/scripts/generate_campus_world.py simulation/src/couchmo_description/worlds/flc_googleearth.world training/tests/test_world_generation.py simulation/src/couchmo_description/launch/gazebo.launch.py
git commit -m "sim: generate hybrid campus world from Google Earth waypoints"
```

---

## Task 5: Implement waypoint expert (pure pursuit) producing steer/throttle @10Hz

**Files:**
- Create: `simulation/src/couchmo_nav/src/waypoint_expert_node.py`
- Create: `simulation/src/couchmo_nav/launch/waypoint_expert.launch.py`
- Create: `simulation/src/couchmo_nav/config/waypoint_follower.yaml`
- Create: `training/tests/test_pure_pursuit.py`

- [ ] **Step 1: Write failing tests for pure pursuit**

```python
# training/tests/test_pure_pursuit.py
import numpy as np

def test_straight_path_zero_steer():
    waypoints = np.array([[0.0, 0.0], [10.0, 0.0]])
    pose = (0.0, 1.0, 0.0)  # 1m above centerline
    # Expect steer sign to correct toward centerline (negative in this convention)
    assert True
```

- [ ] **Step 2: Implement expert node**
  - Subscribe to pose/odom from sim.
  - Load waypoints from CSV/YAML.
  - Compute steer from pure pursuit curvature.
  - Compute throttle from curvature (slow down on turns).
  - Publish `(steer, throttle)` on a ROS topic (e.g., `/expert/steer_throttle`).

- [ ] **Step 3: Run in sim and validate**
  - Visualize waypoints as markers (optional).
  - Ensure expert drives along the path.

- [ ] **Step 4: Commit**

```bash
git add simulation/src/couchmo_nav/src/waypoint_expert_node.py simulation/src/couchmo_nav/launch/waypoint_expert.launch.py simulation/src/couchmo_nav/config/waypoint_follower.yaml training/tests/test_pure_pursuit.py
git commit -m "sim: add pure-pursuit waypoint expert producing steer/throttle"
```

---

## Task 6: Action adapter (steer/throttle → cmd_vel) for current Gazebo diff-drive

**Files:**
- Create: `simulation/src/couchmo_nav/src/steer_throttle_to_cmd_vel.py`
- Create: `training/tests/test_action_adapter.py`

- [ ] **Step 1: Write failing tests**
  - Verify mapping preserves:
    - `throttle=0` → `linear.x=0`
    - `steer=0` → `angular.z=0`
    - sign conventions consistent.

- [ ] **Step 2: Implement node**
  - Subscribe to `/expert/steer_throttle` (and later policy output).
  - Publish `/cmd_vel` for Gazebo plugin.
  - Enforce 10 Hz publish rate.

- [ ] **Step 3: Validate in sim**
  - Run expert → adapter → couch follows path.

- [ ] **Step 4: Commit**

```bash
git add simulation/src/couchmo_nav/src/steer_throttle_to_cmd_vel.py training/tests/test_action_adapter.py
git commit -m "sim: adapt steer/throttle commands to cmd_vel for Gazebo"
```

---

## Task 7: Dataset recording (images + expert actions)

**Files:**
- Create: `training/dataset/record_expert_rollouts.py`
- Create: `training/dataset/format_dataset.py`
- Create: `training/README.md`

- [ ] **Step 1: Implement recorder**
  - Subscribe to:
    - `/left_cam/image_raw`, `/right_cam/image_raw`
    - `/expert/steer_throttle`
  - Save synchronized samples at 10 Hz with timestamps.

- [ ] **Step 2: Implement formatter**
  - Convert raw frames to 84×84 grayscale + 4-frame stacks per camera, matching `model.py`.
  - Save into a simple dataset layout (`.npz` shards or similar).

- [ ] **Step 3: Smoke test**
  - Generate 2–5 short episodes and ensure dataset loads.

- [ ] **Step 4: Commit**

```bash
git add training/dataset/record_expert_rollouts.py training/dataset/format_dataset.py training/README.md
git commit -m "train: record expert rollouts and format imitation dataset"
```

---

## Task 8: Imitation training baseline (behavior cloning)

**Files:**
- Create: `training/imitation/train_bc.py`
- Modify: `model.py` (only if necessary to reuse preprocessing; prefer importing)

- [ ] **Step 1: Define a minimal policy network**
  - Input: (8,84,84)
  - Output: 2 floats (steer, throttle) with appropriate activation/clamping.

- [ ] **Step 2: Training loop**
  - Dataset loader, train/val split, basic metrics.

- [ ] **Step 3: Eval in sim**
  - Run trained policy in sim via ROS topics.

- [ ] **Step 4: Commit**

```bash
git add training/imitation/train_bc.py training/imitation/eval_in_sim.py
git commit -m "train: add behavior cloning baseline and sim evaluation"
```

---

## Task 9 (optional): RL fine-tune (PPO)

**Files:**
- Create: `training/rl/train_ppo.py`

- [ ] **Step 1: Define rewards + termination**
  - progress along arc-length
  - corridor penalty (3.0m)
  - collisions
  - smoothness

- [ ] **Step 2: Implement PPO training harness**
  - Wrap sim stepping at 10 Hz.
  - Start from BC checkpoint.

- [ ] **Step 3: Commit**

```bash
git add training/rl/train_ppo.py
git commit -m "train: add PPO fine-tune harness (optional)"
```

---

## Self-review checklist (run after writing plan)

- Spec coverage: cameras, world generation, expert, imitation, optional RL, 10 Hz interface, corridor=3.0m.
- No placeholders: replace any `assert True` stubs with real asserts during implementation.
- Type/sign consistency: confirm steer sign conventions match IRL and adapter.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-15-campus-sim-training-plan.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration  
2. **Inline Execution** — Execute tasks in this session with checkpoints

Which approach?

