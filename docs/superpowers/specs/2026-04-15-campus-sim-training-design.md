# CouchMo — Campus Simulation & Camera-Only Training (Design)

**Date:** 2026-04-15  
**Status:** Approved (brainstorming)  
**Owner:** CouchMo team  

## Summary

Build a **hybrid** Gazebo Classic + ROS 2 simulation that is **fast, repeatable, and “accurate enough”** to train a **camera-only** driving policy for the real CouchMo platform.

Key decisions:
- **Sensors (v1):** 2× Logitech **Brio 100** cameras, mounted on **both armrests**, **32 inches** above ground.
- **Primary environment (v1):** **Outdoor campus sidewalks/paths** (full campus).
- **Action interface (IRL parity):** policy outputs **`steer ∈ [-1, 1]`**, **`throttle ∈ [0, 1]`** at **10 Hz**, matching the serial protocol used by `serial_controller.py`.
- **Ground truth / supervision:** a **sim expert** controller (**waypoint follower**) generates unlimited labels (no in-person data collection required).
- **Training strategy:** **imitation first**, then **RL fine-tune** with domain randomization to reduce sim-to-real gap.

## Current repo state (starting point)

The repo already contains a ROS 2 + Gazebo Classic stack under `simulation/`:
- Launch files start Gazebo + RViz + SLAM + Nav2 (bringup).
- The robot description (`couchmo.urdf.xacro`) already models a differential-drive couch and includes plugins (diff drive, IMU, and a 2D lidar).
- The IRL control protocol is defined in `serial_controller.py` and documented in `autonomous_car_research.md`.

This design extends the sim to support **camera-only training** and a **new campus world generator**.

## Goals

- **G1 — Working sim**: `simulation/run.sh sim` reliably launches the stack via Docker/noVNC on macOS.
- **G2 — Training-grade camera sim**: two simulated cameras with correct-ish pose/FOV and controllable noise/randomization.
- **G3 — Expert driving**: an “expert” can complete campus routes without collisions and produces `(steer, throttle)` at 10 Hz.
- **G4 — Dataset generation**: batch generation of many episodes with randomized conditions.
- **G5 — Trainable loop**: imitation training baseline that drives in sim; optional RL fine-tune path.
- **G6 — Sim-to-real alignment**: training assumptions match IRL control rate and action semantics.

## Non-goals (v1)

- Photoreal campus reconstruction.
- Perfect vehicle dynamics modeling (suspension, rider mass shift, tire compliance).
- Full autonomous navigation to arbitrary destinations with route planning.
- Using LiDAR/IMU as primary sensors for the learned policy (the policy is camera-only in v1).

## Requirements (high signal)

### R1. Observation + action contract

- **Observation:** two camera streams (left/right), sampled at **10 Hz** for training.
- **Action:** `(steer, throttle)` where:
  - `steer ∈ [-1, 1]` (left→right)
  - `throttle ∈ [0, 1]` (stop→full)
- **Control rate:** 10 Hz (100 ms cycle), consistent with `autonomous_car_research.md`.

### R2. Sidewalk corridor definition

- The “drivable region” is defined as a corridor around the centerline of width **3.0 m** (configurable).
- “Off-path” is measurable and penalizable for RL, and usable for evaluation.

### R3. Domain randomization for camera-only transfer

At minimum, provide hooks to randomize:
- Lighting intensity / sun angle / shadow strength.
- Ground and building texture palette (swap among a small set).
- Camera effects: noise, blur, exposure jitter (bounded), optional lens distortion approximation.
- Dynamic obstacles: pedestrian spawn locations along/near paths.

## Architecture

### Components

1. **World generator**
   - Input: Google Earth **KML/KMZ** polylines (sidewalk/route centerlines), optional polygons (no-go zones).
   - Output:
     - Gazebo Classic **`.world`** file (SDF) containing hybrid geometry.
     - A normalized **waypoint file** (CSV/YAML) in local meters for the expert and evaluation.

2. **Hybrid campus world (Gazebo SDF)**
   - **Nav-grade geometry**: sidewalk ribbons (collision), curbs, building footprints (simple extruded solids).
   - **Visual-grade cues**: lightweight props (signs/benches/trees) placed from templates with constrained randomization.
   - Physics kept stable; collisions prioritized over rendering fidelity.

3. **Sim sensors**
   - Two Gazebo cameras added to the couch model at the armrest mounts.
   - Camera topics published to ROS for recording and training.

4. **Expert driver (waypoint follower)**
   - Controller: Pure Pursuit (or Stanley) to track the centerline.
   - Outputs `(steer, throttle)` at 10 Hz, with conservative speed scheduling.
   - Optional perturbations for robustness (small lateral offsets, actuation noise).

5. **Training bridge**
   - Records: images + expert actions for imitation dataset.
   - Provides a control interface for learned policy rollouts in sim.
   - Converts `(steer, throttle)` to the sim’s underlying motion control (e.g., wheel speeds / `cmd_vel`), while preserving IRL semantics.

### Data flow (training time)

1. Spawn couch in generated campus world at randomized start positions on/near the path.
2. Run expert at 10 Hz → command sim → sim produces camera frames.
3. Log \((left\_img, right\_img, steer, throttle, metadata)\) at 10 Hz.
4. Train imitation model.
5. (Optional) RL fine-tune in sim starting from imitation weights.

## World generation: Google Earth → Gazebo

### Inputs

- **KML/KMZ** polyline(s) tracing sidewalk/path **centerlines** across full campus.
- Optional polygons:
  - no-go zones (grass, stairs, planters)
  - plazas/wide areas (to allow multiple centerline branches later)

### Outputs

- `*.world` SDF file representing campus geometry (hybrid).
- `waypoints.(csv|yaml)` in local ENU meters (x,y).
- `corridor_width_m` config (default 3.0).

### Coordinate approach

- Choose a campus **local origin** (lat/lon reference).
- Convert KML lat/lon → local **meters** using a standard ENU approximation (valid for campus-scale).
- Keep z flat for v1 (optional elevation later).

### Licensing constraints

Google Earth is used to **trace** geometry and paths; the project generates its own SDF/meshes/textures. Do not import Google-provided 3D assets or imagery directly into the distributed sim.

## Expert controller (B)

### Controller behavior

- Follow waypoint centerline with lookahead distance scaled by speed.
- Throttle scheduling:
  - slower for higher curvature
  - cap max speed based on “comfort” and stability
- Emit actions at exactly **10 Hz**.

### Why this expert

- Generates unlimited supervision cheaply.
- Provides deterministic “good enough” driving to bootstrap imitation.

## Training strategy (B/C)

### Stage 1 — Imitation (behavior cloning)

Train a CNN policy to predict `(steer, throttle)` from the two camera feeds.

Dataset recipe:
- Roll out expert with randomized conditions.
- Include perturbations so the learner sees recovery scenarios (start offsets, noise).
- Target input format can match `autonomous_car_research.md` plan:
  - preprocess to **84×84 grayscale**
  - stack **4 frames per camera** → tensor shape **(8, 84, 84)**.

### Stage 2 — RL fine-tuning (optional but recommended)

Start from imitation checkpoint and fine-tune with rewards:
- **Progress reward** along waypoint arc-length.
- **Corridor reward** for staying within 3.0 m corridor.
- **Smoothness** penalties (jerk/oscillation proxies).
- **Collision penalty** and termination.
- **Off-corridor penalty** and termination.

Domain randomization remains enabled throughout.

## Performance & optimization priorities

- Prefer **training throughput** and determinism over photoreal rendering.
- Keep camera resolution low for training; use higher resolution only for debugging.
- Use headless runs for dataset generation (noVNC only when needed).
- Avoid overly high physics update rates unless required for stability.

## Evaluation plan

In-sim metrics:
- Episode completion rate.
- Mean lateral error from centerline.
- Collision rate (static + dynamic obstacles).
- Time-to-completion (normalized).
- Control smoothness (steer/throttle variation).

Transfer sanity checks (IRL):
- Validate camera pose/FOV alignment to Brio 100 armrest mounts.
- Conservative throttle caps and hard e-stop procedures.
- Short supervised tests on limited routes before campus-wide trials.

## Risks and mitigations

- **Sim-to-real gap (camera-only)**: mitigate with strong domain randomization, camera noise/blur, and conservative policies.
- **World fidelity**: keep geometry correct (paths/curbs) first; visuals are secondary and randomized.
- **Control mismatch**: enforce 10 Hz control and exact `(steer, throttle)` semantics end-to-end.

## Milestones (suggested)

- **M0**: sim launches reliably; couch spawns in a test world.
- **M1**: two cameras publish frames from correct poses.
- **M2**: Google Earth KML → waypoints + generated world; expert drives one long route.
- **M3**: dataset generator produces N episodes; imitation baseline trains and drives in sim.
- **M4**: RL fine-tune improves robustness under randomized conditions.

