# Campus Sim + Camera-Only Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix and extend the CouchMo simulator so it can generate training data, train a camera-only policy for full-campus sidewalk driving, and ship that policy to a **Windows laptop** that drives the real couch — all without forcing any single host OS into the loop.

**Architecture — three surfaces:**

| Surface | Where it runs | Stack | Talks to |
|---|---|---|---|
| `simulation/` | Docker on any desktop (Win/Mac/Linux) | ROS 2 Humble + Gazebo Classic 11 | Writes dataset shards to a host-mounted folder |
| `training/` | Native on the desktop (Windows w/ or w/o GPU) | Python 3, NumPy/OpenCV/PyTorch | Reads dataset, writes `.pt` + `.onnx` |
| `runtime/` | Native on the laptop (Windows, **no NVIDIA GPU**) and debuggable on the desktop | Python 3, ONNX Runtime CPU, OpenCV, pyserial | Loads `.onnx`, drives serial protocol from `serial_controller.py` |

**Hard contracts between surfaces:**

- **Dataset format** — directory of `.npz` shards + `manifest.json` (left, right, steer, throttle, t). Sim writes it; training reads it. No ROS imports in `training/`.
- **Model format** — single `model.onnx` (and `model.pt` kept for re-training). Training writes it; runtime loads it. No torch import on the laptop.
- **Action protocol** — `(steer ∈ [-1,1], throttle ∈ [0,1])` at 10 Hz, identical in sim adapter and on-couch serial layer.
- **Preprocessing** — defined exactly once in `shared/preprocess.py`, imported by both `training/` and `runtime/`.

**Cross-platform principles (apply to every task):**

- All Python uses `pathlib.Path`. No raw `/` separators.
- All host-runnable shell scripts ship in pairs: `*.sh` (bash, LF) for macOS/Linux, `*.ps1` (PowerShell) for Windows. Line endings pinned via `.gitattributes`.
- Docker accessed via `docker compose` (v2). noVNC at `http://localhost:6081/vnc.html` is the only host-side viewer — no X11/XQuartz on either OS.
- Machine-specific values (serial port `COM3` vs `/dev/ttyUSB0`, camera index, dataset root) are CLI args, never hardcoded.
- `runtime/` MUST install with one `pip install -r runtime/requirements.txt` on bare Windows or Mac Python 3.10+. No ROS, no Docker, no torch.

**Branching:** Per user direction, work happens on a feature branch in the existing `CouchMo/` repo (no worktree). Suggested name: `feat/campus-sim`.

---

## File structure (what we will create/modify)

**Modify (existing):**
- `simulation/src/couchmo_description/urdf/couchmo.urdf.xacro` — add two camera sensors at armrest mounts.
- `simulation/src/couchmo_description/launch/gazebo.launch.py` — accept new world arg, ensure camera topics are bridged.
- `simulation/src/couchmo_description/worlds/` — add new generated campus world (`*.world`) + keep existing.
- `simulation/docker-compose.yml` — add bind mount `./training/data:/workspace/data`, optional headless service.
- `simulation/run.sh` — drop XQuartz; use noVNC only; minimal wrapper around `docker compose`.

**Create (new — repo-wide):**
- `.gitattributes` — pin `*.sh eol=lf`, `*.ps1 eol=crlf`, default `text=auto`.
- `.gitignore` additions — `training/data/`, `training/checkpoints/`, `runtime/__pycache__/`, `*.onnx`, `*.pt`.

**Create (new — `simulation/`):**
- `simulation/run.ps1` — Windows PowerShell equivalent of `run.sh`.
- `simulation/scripts/bringup_check.py` — cross-platform health check (Docker reachable, port 6081 free).
- `simulation/scripts/kml_to_waypoints.py` — parse KML/KMZ → local ENU waypoints.
- `simulation/scripts/generate_campus_world.py` — generate hybrid SDF world from waypoints.
- `simulation/scripts/fixtures/synthetic_campus.kml` — 3-segment fake KML so Tasks 3 & 4 can run before real campus KML exists.
- `simulation/src/couchmo_expert/package.xml` — ROS 2 Python package manifest.
- `simulation/src/couchmo_expert/setup.py` — ament_python setup.
- `simulation/src/couchmo_expert/setup.cfg` — entry-point bindings.
- `simulation/src/couchmo_expert/resource/couchmo_expert` — ament resource marker.
- `simulation/src/couchmo_expert/couchmo_expert/__init__.py`
- `simulation/src/couchmo_expert/couchmo_expert/waypoint_expert_node.py` — pure-pursuit expert @10 Hz.
- `simulation/src/couchmo_expert/couchmo_expert/steer_throttle_to_cmd_vel.py` — adapter node.
- `simulation/src/couchmo_expert/couchmo_expert/dataset_recorder_node.py` — writes `.npz` shards.
- `simulation/src/couchmo_expert/launch/waypoint_expert.launch.py`
- `simulation/src/couchmo_expert/launch/record_dataset.launch.py`
- `simulation/src/couchmo_expert/config/waypoint_follower.yaml`

**Create (new — `shared/`, used by both training and runtime):**
- `shared/__init__.py`
- `shared/preprocess.py` — single source of truth for 84×84 grayscale + 4-frame stack → `(8, 84, 84) float32`.
- `shared/dataset_format.py` — read/write helpers for `.npz` shard format and `manifest.json`.

**Create (new — `training/`):**
- `training/pyproject.toml` — pytest config, ruff config, package layout.
- `training/requirements.txt` — torch (CPU/CUDA), numpy, opencv-python, onnx, tqdm.
- `training/requirements-dev.txt` — pytest, ruff.
- `training/conftest.py` — adds `shared/` to `sys.path` for tests.
- `training/README.md` — exact commands per OS.
- `training/dataset/format_dataset.py` — convert raw recorder output to training-ready shards (if needed).
- `training/imitation/model.py` — minimal CNN policy: `(8,84,84) → (steer, throttle)`.
- `training/imitation/train_bc.py` — behavior cloning loop. **No ROS imports.**
- `training/imitation/export_onnx.py` — `torch.onnx.export(...)` + numerical-equivalence check vs `.pt`.
- `training/imitation/eval_in_sim.py` — ROS bridge for in-sim eval. `try: import rclpy except ImportError: ...` — runs only inside the sim container or a Linux ROS host.
- `training/rl/train_ppo.py` — optional, Task 9.
- `training/tests/test_kml_to_waypoints.py`
- `training/tests/test_pure_pursuit.py`
- `training/tests/test_action_adapter.py`
- `training/tests/test_world_generation.py`
- `training/tests/test_preprocess.py`
- `training/tests/test_onnx_export.py`
- `training/tests/test_dataset_format.py`

**Create (new — `runtime/`, the on-laptop loop):**
- `runtime/requirements.txt` — `onnxruntime`, `opencv-python`, `pyserial`, `numpy`. **Nothing else.**
- `runtime/README.md` — install on Windows / Mac, COM port discovery, camera index discovery.
- `runtime/__init__.py`
- `runtime/drive.py` — main entry. Modes: `--source live` and `--source replay`.
- `runtime/camera.py` — OpenCV `VideoCapture` wrapper, dual-cam sync.
- `runtime/control.py` — wraps `serial_controller.py`; cross-platform port (`COM3` or `/dev/...`).
- `runtime/inference.py` — ONNX Runtime session wrapper, uses `shared.preprocess`.
- `runtime/tests/test_inference_smoke.py` — load `.onnx`, run on random input, assert output range.
- `runtime/tests/test_replay.py` — `--source replay` against a tiny fixture episode.

---

## Task 0: Cross-platform scaffolding (prereqs)

**Why this exists:** Tasks 2–9 in earlier drafts assumed the `couchmo_expert` ROS package, the `training/` Python project, and the `shared/` preprocessing module already existed. They don't. This task creates them so later tasks can focus on logic, not boilerplate.

**Note on existing packages:** The repo already contains an `ament_cmake` package `couchmo_nav` holding Nav2 configuration (`config/nav2_params.yaml`) and maps. **Do not modify it.** The new learned-policy nodes go in a separate `ament_python` package called `couchmo_expert`. `couchmo_bringup` continues to consume `couchmo_nav` for Nav2; `couchmo_expert` is independent.

**Files:**
- Create: `.gitattributes`, `.gitignore` updates
- Create: `simulation/src/couchmo_expert/{package.xml, setup.py, setup.cfg, resource/couchmo_expert, couchmo_expert/__init__.py, launch/, config/}`
- Create: `shared/{__init__.py, preprocess.py, dataset_format.py}`
- Create: `training/{pyproject.toml, requirements.txt, requirements-dev.txt, conftest.py, README.md, tests/__init__.py}`
- Create: `runtime/{requirements.txt, README.md, __init__.py, tests/__init__.py}`
- Create: `simulation/scripts/fixtures/synthetic_campus.kml`

- [ ] **Step 1: Branch**
  ```bash
  git checkout -b feat/campus-sim
  ```

- [ ] **Step 2: `.gitattributes` + `.gitignore`**
  - `.gitattributes`:
    ```
    * text=auto
    *.sh text eol=lf
    *.ps1 text eol=crlf
    *.bat text eol=crlf
    *.png binary
    *.npz binary
    *.onnx binary
    *.pt binary
    ```
  - Add to `.gitignore`: `training/data/`, `training/checkpoints/`, `**/__pycache__/`, `*.onnx`, `*.pt`, `.pytest_cache/`, `.venv/`.

- [ ] **Step 3: `couchmo_expert` ROS 2 package skeleton**
  - `package.xml` with `<buildtool_depend>ament_python</buildtool_depend>`, deps on `rclpy`, `sensor_msgs`, `geometry_msgs`, `std_msgs`, `cv_bridge`.
  - `setup.py` declares the package, glob-installs `launch/*.py` and `config/*.yaml`, registers entry points (filled in later tasks).
  - Create empty `couchmo_expert/__init__.py`, `launch/`, `config/` dirs.
  - Verify build inside the sim container: `cd /ros2_ws && colcon build --packages-select couchmo_expert`.

- [ ] **Step 4: `shared/preprocess.py`**
  - Function: `preprocess_pair(left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray` returning `(2, 84, 84) uint8`.
  - Class: `FrameStacker(num_frames=4)` with `.push(pair)` → `(8, 84, 84) float32` in `[0, 1]`.
  - Pure NumPy + OpenCV. Importable from training and runtime with no extra deps.

- [ ] **Step 5: `shared/dataset_format.py`**
  - `write_shard(path, left, right, steer, throttle, t)` — saves `.npz`.
  - `read_shard(path) -> dict` — round-trip.
  - `Manifest` dataclass with `episodes: list[EpisodeMeta]`, JSON serializable.

- [ ] **Step 6: `training/` Python project**
  - `pyproject.toml` with `[tool.pytest.ini_options] testpaths = ["tests"]`, ruff config.
  - `requirements.txt`:
    ```
    torch>=2.2
    numpy>=1.24
    opencv-python>=4.8
    onnx>=1.15
    tqdm>=4.66
    ```
    (Document in README that the desktop chooses `torch` CPU or CUDA build.)
  - `requirements-dev.txt`: `pytest>=8`, `ruff>=0.4`.
  - `conftest.py` adds repo root to `sys.path` so `import shared.preprocess` works in tests.
  - `README.md`: per-OS install commands (Win PowerShell venv, Mac/Linux bash venv).

- [ ] **Step 7: `runtime/` Python project**
  - `requirements.txt`:
    ```
    onnxruntime>=1.17
    opencv-python>=4.8
    pyserial>=3.5
    numpy>=1.24
    ```
  - `README.md`: install on Windows laptop (PowerShell venv), COM port discovery (`Get-WmiObject Win32_SerialPort`), camera index discovery snippet.

- [ ] **Step 8: `shared/preprocess.py` tests**
  - `training/tests/test_preprocess.py`: assert output shape, dtype, value range, determinism.

- [ ] **Step 9: Synthetic KML fixture**
  - `simulation/scripts/fixtures/synthetic_campus.kml` — one `<LineString>` with ~5 lat/lon coords forming an L-shape near the campus origin. Used by Tasks 3 & 4 until real campus KML is provided.

- [ ] **Step 10: Verify**
  ```bash
  cd training && python -m pytest -q
  ```
  Expect: `test_preprocess.py` passes; the other test files exist as empty placeholders or are skipped via `pytest.skip("filled in Task N")`.

- [ ] **Step 11: Commit**
  ```bash
  git add .gitattributes .gitignore shared training runtime simulation/src/couchmo_expert simulation/scripts/fixtures
  git commit -m "scaffold: three-surface project layout (sim/training/runtime) + couchmo_expert package"
  ```

---

## Task 1: Cross-platform sim bringup

**Files:**
- Modify: `simulation/run.sh` (strip XQuartz, use noVNC)
- Create: `simulation/run.ps1`
- Create: `simulation/scripts/bringup_check.py`
- Modify: `simulation/docker-compose.yml` (bind mount + optional headless service)
- Create: `simulation/README.md`

- [ ] **Step 1: `bringup_check.py` (cross-platform Python)**
  - Stdlib only. Checks:
    - `docker version` returns 0
    - port 6081 free or already bound by us
    - host has at least 8 GB RAM (warn, don't fail)
  - Prints actionable error messages.

- [ ] **Step 2: Rewrite `run.sh` minimally**
  - Drop all XQuartz logic.
  - Modes: `sim | preview | shell | headless`.
  - Just calls `python3 scripts/bringup_check.py && docker compose up <service>`.
  - Print noVNC URL `http://localhost:6081/vnc.html` after start.

- [ ] **Step 3: Create `run.ps1`**
  - Same modes as `run.sh`. Uses `python` (Windows PEP 397 launcher) and `docker compose`.
  - Print noVNC URL.
  - Verified by user manually on Windows; subagent only writes the script.

- [ ] **Step 4: Update `docker-compose.yml`**
  - Add bind mount on `sim` service: `- ../training/data:/workspace/data`.
  - Optional `sim_headless` service that runs without noVNC for batch dataset gen.

- [ ] **Step 5: `simulation/README.md`**
  - Per-OS bringup table:
    - Windows: `.\run.ps1 sim`
    - macOS/Linux: `./run.sh sim`
  - Both → open `http://localhost:6081/vnc.html`.
  - Failure modes: Docker daemon not running, port 6081 in use, WSL2 not enabled (Windows).

- [ ] **Step 6: Verify (host-side, where possible)**
  - `python simulation/scripts/bringup_check.py` — should pass on host even without Docker running (prints the actionable error).
  - Live `docker compose up` verification is deferred to user; subagent must NOT block on it.

- [ ] **Step 7: Commit**
  ```bash
  git add simulation/run.sh simulation/run.ps1 simulation/scripts/bringup_check.py simulation/docker-compose.yml simulation/README.md
  git commit -m "sim: cross-platform bringup (.sh + .ps1) using noVNC, drop XQuartz"
  ```

---

## Task 2: Add armrest cameras to URDF (two Brio 100 mounts)

**Files:**
- Modify: `simulation/src/couchmo_description/urdf/couchmo.urdf.xacro`
- Modify: `simulation/src/couchmo_description/launch/gazebo.launch.py` (only if remapping needed)

- [ ] **Step 1: Define camera links + joints**
  - Add `left_camera_link` and `right_camera_link` fixed to chassis.
  - Mount height: 32 inches = 0.8128 m above ground.
  - Lateral offset: ±(body_w/2 - 0.10).

- [ ] **Step 2: Add Gazebo camera sensors**
  - `<gazebo reference="...">` blocks publishing:
    - `/left_cam/image_raw`
    - `/right_cam/image_raw`
  - update_rate 10 Hz, FOV approximating Brio 100 (~78° HFOV).

- [ ] **Step 3: Validation (deferred to user)**
  - Subagent confirms xacro parses (`xacro couchmo.urdf.xacro > /tmp/out.urdf` if available).
  - User runs sim and confirms `ros2 topic list | grep image_raw` shows both topics.

- [ ] **Step 4: Commit**
  ```bash
  git add simulation/src/couchmo_description/urdf/couchmo.urdf.xacro simulation/src/couchmo_description/launch/gazebo.launch.py
  git commit -m "sim: add dual armrest cameras (Brio 100, 32in) to CouchMo URDF"
  ```

---

## Task 3: KML/KMZ → waypoints converter

**Files:**
- Create: `simulation/scripts/kml_to_waypoints.py`
- Create: `training/tests/test_kml_to_waypoints.py`

- [ ] **Step 1: Real failing tests for KML parsing**
  - `test_latlon_to_enu_zero_origin`: origin maps exactly to `(0,0)` within 1e-6 m.
  - `test_latlon_to_enu_known_offset`: a known displacement (e.g. 0.001° lat ≈ 111.32 m) is correct within 0.5 m at campus latitudes.
  - `test_parse_linestring`: synthetic KML fixture from Task 0 yields the expected number of waypoints in the right order.
  - `test_parse_kmz`: same fixture wrapped as `.kmz` (zip) yields identical waypoints.
  - **No `assert True` stubs.**

- [ ] **Step 2: Run tests (expect fail)**
  ```bash
  cd training && python -m pytest tests/test_kml_to_waypoints.py -q
  ```

- [ ] **Step 3: Implement**
  - Use stdlib `xml.etree.ElementTree` for KML, `zipfile` for KMZ. No new deps.
  - ENU projection: tangent-plane approximation, fine for campus-scale.
  - Output: CSV with columns `x_m, y_m, s_m` where `s_m` is cumulative arc-length.
  - CLI: `python kml_to_waypoints.py --in path.kml --out path.csv --origin-lat X --origin-lon Y`.

- [ ] **Step 4: Run tests (expect pass)**

- [ ] **Step 5: Commit**
  ```bash
  git add simulation/scripts/kml_to_waypoints.py training/tests/test_kml_to_waypoints.py
  git commit -m "tools: convert Google Earth KML/KMZ routes to local ENU waypoints"
  ```

---

## Task 4: Generate hybrid campus SDF world from waypoints

**Files:**
- Create: `simulation/scripts/generate_campus_world.py`
- Create: `simulation/src/couchmo_description/worlds/synthetic_campus.world` (generated from fixture; checked in)
- Create: `training/tests/test_world_generation.py`

- [ ] **Step 1: Real failing tests**
  - Output `.world` is well-formed XML.
  - Contains `<world name="...">`, ground plane include, ≥1 sidewalk ribbon model derived from input waypoints.
  - Sidewalk ribbon length within 5% of waypoint arc-length.

- [ ] **Step 2: Implement generator**
  - Read waypoints CSV from Task 3.
  - Emit `.world` with sidewalk ribbon strips (boxes) along consecutive waypoint pairs.
  - Configurable corridor width (default 3.0 m) and visual sidewalk strip width (default 1.5 m).
  - Sprinkle a few static obstacle templates (benches/signs) with deterministic seed.
  - CLI: `python generate_campus_world.py --waypoints path.csv --out path.world [--seed 0]`.

- [ ] **Step 3: Generate the synthetic-campus world from Task 0's fixture**
  ```bash
  python simulation/scripts/kml_to_waypoints.py \
    --in simulation/scripts/fixtures/synthetic_campus.kml \
    --out simulation/scripts/fixtures/synthetic_campus.csv \
    --origin-lat <fixture_origin> --origin-lon <fixture_origin>
  python simulation/scripts/generate_campus_world.py \
    --waypoints simulation/scripts/fixtures/synthetic_campus.csv \
    --out simulation/src/couchmo_description/worlds/synthetic_campus.world
  ```

- [ ] **Step 4: Wire into launch**
  - `gazebo.launch.py` accepts `world:=synthetic_campus` (and `world:=flc_googleearth` later).

- [ ] **Step 5: Commit**
  ```bash
  git add simulation/scripts/generate_campus_world.py simulation/src/couchmo_description/worlds/synthetic_campus.world training/tests/test_world_generation.py simulation/src/couchmo_description/launch/gazebo.launch.py
  git commit -m "sim: generate hybrid campus world from waypoints (synthetic fixture)"
  ```

---

## Task 5: Pure-pursuit waypoint expert @10 Hz

**Files:**
- Create: `simulation/src/couchmo_expert/couchmo_expert/waypoint_expert_node.py`
- Create: `simulation/src/couchmo_expert/launch/waypoint_expert.launch.py`
- Create: `simulation/src/couchmo_expert/config/waypoint_follower.yaml`
- Create: `training/tests/test_pure_pursuit.py`
- Modify: `simulation/src/couchmo_expert/setup.py` (entry point)

- [ ] **Step 1: Real failing tests** (pure logic, no ROS)
  - `test_straight_path_zero_steer`: pose on the centerline → `|steer| < 1e-3`.
  - `test_lateral_offset_corrects_toward_centerline`: pose 1 m above centerline → steer sign points down.
  - `test_throttle_decreases_on_curvature`: high-curvature segment → throttle < cap.
  - `test_throttle_capped_in_zero_one`: random poses → `0 ≤ throttle ≤ 1`, `-1 ≤ steer ≤ 1`.

- [ ] **Step 2: Implement expert as plain Python class** in `waypoint_expert_node.py` plus a thin `rclpy.Node` wrapper at the bottom.
  - Pure-pursuit: lookahead distance scaled by speed.
  - Throttle scheduling: linear taper with curvature, cap from YAML config.
  - Publishes `geometry_msgs/Vector3` (x=steer, y=throttle, z=0.0) on `/expert/steer_throttle` at 10 Hz.

- [ ] **Step 3: YAML config** with `corridor_width_m: 3.0`, `lookahead_min_m`, `lookahead_max_m`, `max_throttle`, `curvature_throttle_gain`.

- [ ] **Step 4: Launch file** — loads waypoints CSV, starts the node.

- [ ] **Step 5: Register entry point in `setup.py`**.

- [ ] **Step 6: Run unit tests (expect pass)**
  ```bash
  cd training && python -m pytest tests/test_pure_pursuit.py -q
  ```

- [ ] **Step 7: Commit**
  ```bash
  git add simulation/src/couchmo_expert/couchmo_expert/waypoint_expert_node.py simulation/src/couchmo_expert/launch/waypoint_expert.launch.py simulation/src/couchmo_expert/config/waypoint_follower.yaml simulation/src/couchmo_expert/setup.py training/tests/test_pure_pursuit.py
  git commit -m "sim: pure-pursuit waypoint expert producing (steer, throttle) at 10 Hz"
  ```

---

## Task 6: Action adapter (steer/throttle → cmd_vel)

**Files:**
- Create: `simulation/src/couchmo_expert/couchmo_expert/steer_throttle_to_cmd_vel.py`
- Create: `training/tests/test_action_adapter.py`
- Modify: `simulation/src/couchmo_expert/setup.py` (entry point)

- [ ] **Step 1: Real failing tests** (pure mapping, no ROS)
  - `throttle=0` → `linear.x == 0`.
  - `steer=0, throttle>0` → `angular.z == 0`, `linear.x > 0`.
  - `steer>0, throttle>0` → `angular.z > 0` (matches IRL sign convention from `serial_controller.py` — verify against that file).
  - Mapping is monotonic in throttle and odd in steer.

- [ ] **Step 2: Implement adapter** as plain function + thin `rclpy.Node` wrapper.
  - Subscribes `/expert/steer_throttle` (later: `/policy/steer_throttle`).
  - Publishes `/cmd_vel` at 10 Hz (latches latest input).

- [ ] **Step 3: Run unit tests (expect pass).**

- [ ] **Step 4: Commit**
  ```bash
  git add simulation/src/couchmo_expert/couchmo_expert/steer_throttle_to_cmd_vel.py training/tests/test_action_adapter.py simulation/src/couchmo_expert/setup.py
  git commit -m "sim: adapter mapping (steer, throttle) to cmd_vel at 10 Hz"
  ```

---

## Task 7: Dataset recorder (writes shared `.npz` shards)

**Files:**
- Create: `simulation/src/couchmo_expert/couchmo_expert/dataset_recorder_node.py`
- Create: `simulation/src/couchmo_expert/launch/record_dataset.launch.py`
- Create: `training/tests/test_dataset_format.py`
- Modify: `simulation/src/couchmo_expert/setup.py` (entry point)

**Container path:** writes to `/workspace/data/<episode_id>/shard_<n>.npz` — host-visible at `training/data/...` via the bind mount from Task 1.

- [ ] **Step 1: Tests for `shared/dataset_format.py` round-trip**
  - Write random shard → read → assert byte-equal arrays.
  - Manifest JSON round-trip preserves episode metadata.

- [ ] **Step 2: Implement recorder node**
  - `message_filters.ApproximateTimeSynchronizer` over:
    - `/left_cam/image_raw`
    - `/right_cam/image_raw`
    - `/expert/steer_throttle`
  - Sample at 10 Hz; flush shards every N samples (e.g. 200).
  - Writes via `shared.dataset_format.write_shard`.
  - Updates `manifest.json` per episode.

- [ ] **Step 3: Smoke test (deferred verification)**
  - User runs sim + recorder for one short episode; confirms `training/data/<id>/` populated.
  - Subagent verifies code and round-trip tests only.

- [ ] **Step 4: Commit**
  ```bash
  git add simulation/src/couchmo_expert/couchmo_expert/dataset_recorder_node.py simulation/src/couchmo_expert/launch/record_dataset.launch.py training/tests/test_dataset_format.py simulation/src/couchmo_expert/setup.py
  git commit -m "sim: dataset recorder writes shared .npz shards via bind-mounted volume"
  ```

---

## Task 8a: Behavior cloning training (native, no ROS)

**Files:**
- Create: `training/imitation/model.py`
- Create: `training/imitation/train_bc.py`
- Create: `training/imitation/__init__.py`

- [ ] **Step 1: Define the policy network** in `model.py`
  - Input: `(8, 84, 84) float32` (matches `shared.preprocess`).
  - Small CNN (think Atari-DQN-ish): 3 conv layers + 2 FC.
  - Output: `(steer, throttle)`. Steer via `tanh`, throttle via `sigmoid`.
  - **Constraint:** must export to ONNX cleanly and run on CPU at < 50 ms per inference.

- [ ] **Step 2: Training loop** in `train_bc.py`
  - CLI: `--data-root ./data --epochs N --batch-size N --device cpu|cuda --out ./checkpoints/bc.pt`.
  - Default `--data-root` is `./data` (matches Task 1 bind mount).
  - Dataset reader uses `shared.dataset_format`.
  - Train/val split, MSE loss on `(steer, throttle)`, basic metrics (steer MAE, throttle MAE).
  - Saves `bc.pt` checkpoint + `bc_meta.json` (config, metrics).

- [ ] **Step 3: Smoke test on tiny synthetic dataset**
  - Generate 50 random samples in test fixture form.
  - Run 1 epoch, assert loss decreases.

- [ ] **Step 4: Commit**
  ```bash
  git add training/imitation/model.py training/imitation/train_bc.py training/imitation/__init__.py
  git commit -m "train: behavior cloning loop (native, no ROS) with shared preprocess"
  ```

---

## Task 8b: ONNX export + numerical equivalence check

**Files:**
- Create: `training/imitation/export_onnx.py`
- Create: `training/tests/test_onnx_export.py`

- [ ] **Step 1: Export script**
  - Loads a `.pt` checkpoint, runs `torch.onnx.export(model, dummy_input, path, opset_version=17, dynamic_axes={'input': {0: 'batch'}})`.
  - Validates with `onnx.checker.check_model`.
  - Writes `model.onnx` next to `model.pt`.

- [ ] **Step 2: Equivalence test**
  - `test_onnx_export.py`: train a 1-step model, export, run same input through both Torch and ONNX Runtime, assert max abs diff < 1e-4.

- [ ] **Step 3: Commit**
  ```bash
  git add training/imitation/export_onnx.py training/tests/test_onnx_export.py
  git commit -m "train: ONNX export with numerical-equivalence test against Torch"
  ```

---

## Task 8c: Eval in sim (ROS-gated)

**Files:**
- Create: `training/imitation/eval_in_sim.py`

- [ ] **Step 1: Implement**
  - At top of file:
    ```python
    try:
        import rclpy
        from rclpy.node import Node
        ROS_AVAILABLE = True
    except ImportError:
        ROS_AVAILABLE = False
    ```
  - If invoked without ROS, print clear message and exit 0 (not 1).
  - Loads `.onnx` (preferred) or `.pt`, subscribes camera topics, publishes `/policy/steer_throttle`.

- [ ] **Step 2: Verify import gate**
  - On bare Windows desktop without ROS installed, `python -m training.imitation.eval_in_sim --help` prints help and exits cleanly.

- [ ] **Step 3: Commit**
  ```bash
  git add training/imitation/eval_in_sim.py
  git commit -m "train: in-sim eval node with optional ROS import gate"
  ```

---

## Task 9 (optional): RL fine-tune (PPO)

**Files:**
- Create: `training/rl/train_ppo.py`
- Create: `training/rl/__init__.py`

- [ ] **Step 1: Rewards + termination**
  - Progress along arc-length, corridor penalty (3.0 m), collision penalty, smoothness penalty.

- [ ] **Step 2: PPO harness**
  - Wrap sim stepping at 10 Hz via the same ROS bridge as `eval_in_sim.py` (gated import).
  - Start from BC checkpoint.
  - Saves both `.pt` and `.onnx`.

- [ ] **Step 3: Commit**
  ```bash
  git add training/rl/train_ppo.py training/rl/__init__.py
  git commit -m "train: PPO fine-tune harness starting from BC checkpoint (optional)"
  ```

---

## Task 10: `runtime/drive.py` — on-laptop inference loop

**This is the deploy target.** Must install with `pip install -r runtime/requirements.txt` on a fresh Windows or Mac Python 3.10+ venv. **Zero ROS, zero Docker, zero Torch.**

**Files:**
- Create: `runtime/drive.py`
- Create: `runtime/camera.py`
- Create: `runtime/control.py`
- Create: `runtime/inference.py`
- Create: `runtime/tests/test_inference_smoke.py`
- Create: `runtime/tests/test_replay.py`

- [ ] **Step 1: `runtime/inference.py`**
  - `class Policy: __init__(onnx_path); predict(stacked_frames) -> (steer, throttle)`.
  - Uses `onnxruntime.InferenceSession` with CPU provider only.
  - Imports and uses `shared.preprocess.FrameStacker` so preprocessing is identical to training.

- [ ] **Step 2: `runtime/camera.py`**
  - `DualCamera(left_index, right_index)` with `.read() -> (left_bgr, right_bgr)`.
  - Cross-platform: OpenCV `VideoCapture` works the same on Win/Mac/Linux.
  - Discovers indices via small `list_cameras()` helper.

- [ ] **Step 3: `runtime/control.py`**
  - Wraps `serial_controller.py`. Accepts port string (`COM3` on Windows, `/dev/tty.usbserial-XXXX` on Mac, `/dev/ttyUSB0` on Linux).
  - Single method: `send(steer, throttle)`.

- [ ] **Step 4: `runtime/drive.py` main**
  - CLI:
    ```
    python -m runtime.drive --source live  --left-cam 0 --right-cam 1 --port COM3 --model model.onnx --rate 10
    python -m runtime.drive --source replay --episode path/to/episode.npz --model model.onnx
    ```
  - 10 Hz loop with monotonic timing (`time.perf_counter`). Logs deadline misses.
  - `--source live`: real cameras + serial.
  - `--source replay`: reads `.npz` episode (via `shared.dataset_format`), feeds frames through Policy, prints predicted vs recorded actions side-by-side. **No serial sent in replay mode.**
  - Graceful shutdown on Ctrl-C (zero throttle final command in live mode).

- [ ] **Step 5: Tests**
  - `test_inference_smoke.py`: load a tiny ONNX model (or generate one in fixture), feed random input, assert outputs in correct ranges.
  - `test_replay.py`: run `--source replay` against a 10-frame fixture episode; assert it completes and prints comparable actions.

- [ ] **Step 6: `runtime/README.md`**
  - Windows install (PowerShell venv, `pip install -r runtime/requirements.txt`).
  - Mac install (bash venv).
  - COM port discovery on each OS.
  - Camera index discovery.
  - One-liner sanity command: `python -m runtime.drive --source replay --episode samples/tiny_episode.npz --model checkpoints/bc.onnx`.

- [ ] **Step 7: Commit**
  ```bash
  git add runtime/
  git commit -m "runtime: on-laptop inference loop (ONNX, OpenCV, pyserial) with replay mode"
  ```

---

## Self-review checklist (run after each task and at end)

**Cross-platform hygiene:**
- All Python uses `pathlib.Path`; no raw `/` separators.
- No POSIX-only assumptions in `training/` or `runtime/` (no `/tmp`, no `os.uname`, no `os.path.expanduser('~/...')` without `Path.home()`).
- All shell scripts have a `.ps1` counterpart where intended for host execution.
- `.gitattributes` line endings respected.

**Surface isolation:**
- `runtime/` imports nothing from `simulation/` or `training/imitation/` (only `shared/`).
- `training/imitation/train_bc.py` and `model.py` import nothing from `rclpy`.
- `simulation/` ROS code does not import from `training/imitation/`.

**Spec coverage:**
- Cameras at 0.8128 m, 10 Hz, both cameras, FOV approximated.
- World generation produces a checked-in synthetic world for repeatability.
- Expert produces `(steer ∈ [-1,1], throttle ∈ [0,1])` at exactly 10 Hz.
- Action adapter sign conventions match `serial_controller.py`.
- Dataset format identical between sim writer and training reader.
- Preprocessing identical between training and runtime (single source: `shared/preprocess.py`).
- ONNX numerically equivalent to `.pt` (max abs diff < 1e-4).
- Runtime installs without torch, ROS, or Docker.

**Discipline:**
- No `assert True` placeholders left in any test.
- No Tasks 8/9 features bled into Task 10 or vice versa.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-15-campus-sim-training-plan.md`.

**Branch:** `feat/campus-sim` (created in Task 0). No worktree per user direction.

**Verification policy on this Windows host:**
- Subagents perform code-level + unit-test verification.
- Live ROS/Gazebo/Docker steps marked "deferred to user" — subagent does NOT block on them.
- User runs sim manually between tasks where needed.

**Two execution options:**

1. **Subagent-Driven (recommended)** — Controller dispatches a fresh subagent per task, two-stage review (spec then quality) between tasks. Fast iteration.
2. **Inline Execution** — Execute tasks in this session with checkpoints.

Which approach?
