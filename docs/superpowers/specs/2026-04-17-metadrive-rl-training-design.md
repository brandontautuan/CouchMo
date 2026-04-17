# CouchMo — MetaDrive RL Training & Sim-to-Real Deployment (Design)

**Date:** 2026-04-17
**Status:** Approved (brainstorming)
**Owner:** CouchMo team
**Supersedes (partially):** `2026-04-15-campus-sim-training-design.md` — the Gazebo campus sim is frozen; RL training moves to MetaDrive. Imitation code in `training/imitation/` is reused as-is.

## Summary

Train a camera-only collision-avoidance policy for CouchMo using **MetaDrive** (no Gazebo), via a **two-stage pipeline**:

1. **Behavior cloning (BC) pretrain** — roll out MetaDrive's built-in IDM expert, train `BCPolicy` on supervised `(obs → action)` pairs.
2. **PPO fine-tune** — initialize PPO's actor from the BC checkpoint, fine-tune in `SafeMetaDriveEnv` with traffic/accident scenarios and domain randomization.

Export to ONNX, deploy via existing `runtime/inference.py` + `serial_controller.py`. Training runs on Google Colab; checkpoints persist to mounted Google Drive so sessions can resume.

Key commitment: the observation/action contract is `(8, 84, 84)` stereo grayscale input and `(steer ∈ [-1,1], throttle ∈ [0,1])` output at 10 Hz, identical across MetaDrive, ONNX, and real hardware. The preprocessing pipeline (`shared.preprocess.preprocess_pair` + `FrameStacker`) is the sim/real boundary and is unchanged.

## Current repo state (starting point)

Already built and reused unchanged:

- `training/imitation/model.py` — `BCPolicy` CNN (8 → 2 channels, tanh/sigmoid heads)
- `training/imitation/train_bc.py` — BC training loop (loads `shared.dataset_format` shards)
- `training/imitation/export_onnx.py` — ONNX export
- `shared/preprocess.py` — `preprocess_pair`, `FrameStacker`
- `shared/dataset_format.py` — shard + `Manifest` format
- `runtime/inference.py` — on-laptop ONNX inference loop with replay mode
- `serial_controller.py` — UART protocol to ESP32 (`"steer,throttle\n"` @ 10 Hz, ACK/ERR)
- `Couchmo.ino` — ESP32 firmware (dual-mode controller/UART, watchdog, kill switch)

Frozen (not used by this pipeline):

- `simulation/` — Gazebo + ROS 2 stack
- `training/imitation/eval_in_sim.py` — Gazebo-specific evaluation node

## Goals

- **G1** — Train a policy that avoids collisions in randomized urban scenarios (MetaDrive `SafeMetaDriveEnv`).
- **G2** — Reuse the existing `BCPolicy` architecture and `shared.preprocess` pipeline with no changes.
- **G3** — Export an ONNX artifact that `runtime/inference.py` loads without modification.
- **G4** — Survive Colab session timeouts via resumable checkpointing on mounted Google Drive.
- **G5** — Provide a phased real-hardware rollout plan with explicit go/no-go gates.

## Non-goals (v1)

- Replicating the specific college campus in sim.
- Photoreal rendering.
- Exact skid-steer dynamics matching in MetaDrive (bicycle model accepted; domain randomization compensates).
- Full navigation (waypoint routing, global planning).
- LiDAR / IMU inputs to the learned policy.
- Gazebo mid-fidelity validation (skipped to avoid campus-modeling time sink).

## Requirements

### R1. Observation + action contract

- **Observation:** `(8, 84, 84)` float32 — 2 cameras × 4 stacked frames × 84×84 grayscale, produced by `shared.preprocess.preprocess_pair` + `FrameStacker(num_frames=4)`.
- **Action:** `(steer ∈ [-1, 1], throttle ∈ [0, 1])` float32.
- **Control rate:** 10 Hz (100 ms per decision).
- This contract is identical across MetaDrive training, ONNX inference, and real-hardware deployment.

### R2. Checkpoint resumability

PPO training must survive Colab session death. Every ≤100k training steps, state must persist to Google Drive, and re-running the training cell must resume rather than restart.

### R3. Go/no-go gate before real hardware

No real-hardware rollout begins until `training/metadrive/eval_policy.py` reports, over 500 held-out episodes:

- Collision rate < 5%
- Off-road rate < 10%
- Mean episode length > 300 steps (of 500 max)

These thresholds are starting points and may be revised as data comes in, but a quantitative gate must be defined before advancing to Stage R2 of the real-car rollout.

## Architecture

### Overall pipeline

```
MetaDrive env (custom wrapper)
  ├── observation: stereo virtual cams → (8, 84, 84)   [matches BCPolicy input]
  └── action:       (steer ∈ [-1,1], throttle ∈ [0,1]) [matches serial_controller]
        │
        ▼
─────── Phase 1: BC pretrain ───────
  collect_bc.py  →  .npz dataset on Drive
  train_bc.py (existing, reused as-is)
  output:  checkpoints/bc_metadrive.pt
        │
        ▼
─────── Phase 2: PPO fine-tune ───────
  train_ppo.py  (stable-baselines3 PPO)
  init actor from bc_metadrive.pt
  randomized collision scenarios on every reset
  output:  checkpoints/ppo_metadrive_best.zip
        │
        ▼
─────── Export ───────
  export_onnx.py --from-ppo  →  model.onnx on Drive
        │
        ▼
─────── Deploy ───────
  runtime/inference.py loads model.onnx
  drives real car via serial_controller.py
```

### Code structure

New directory `training/metadrive/`:

```
training/metadrive/
  __init__.py
  env.py              # CouchMoMetaDriveEnv — gym wrapper; obs/action/reward/randomization
  expert_policy.py    # thin adapter around MetaDrive's built-in IDM expert
  collect_bc.py       # roll out expert → write shards in shared.dataset_format
  train_ppo.py        # SB3 PPO; init actor from BC checkpoint; save to Drive
  eval_policy.py      # post-training go/no-go evaluation on held-out seeds
  README.md
  colab_runner.ipynb  # ~12-cell notebook: git pull, mount Drive, invoke scripts
```

Existing code reused unchanged: `training/imitation/model.py`, `training/imitation/train_bc.py`, `shared/preprocess.py`, `shared/dataset_format.py`, `runtime/inference.py`, `serial_controller.py`.

Existing code with minor edits:

- `training/requirements.txt` — add `metadrive-simulator`, `stable-baselines3[extra]`, `gymnasium`.
- `training/imitation/export_onnx.py` — add `--from-ppo PATH` flag that extracts the policy from an SB3 PPO `.zip`, rewraps it in the `BCPolicy` output-activation layout (tanh on steer, sigmoid on throttle), and exports ONNX.

New tests under `training/tests/`:

- `test_metadrive_env.py` — smoke: `reset()`/`step()` return `(8, 84, 84)` float32 obs and accept a `(2,)` action. Skipped if `metadrive` not installed.
- `test_metadrive_collect.py` — smoke: expert rollout writes N valid shards in `shared.dataset_format` layout.
- `test_ppo_bc_init.py` — BC `.pt` weights load cleanly into a fresh SB3 PPO actor; forward pass on dummy obs produces finite output.
- `test_onnx_export_from_ppo.py` — `--from-ppo` produces an ONNX whose output matches the SB3 model output within 1e-5 on random obs.

### MetaDrive env wrapper

`training/metadrive/env.py` — `CouchMoMetaDriveEnv(gym.Env)`, wrapping `SafeMetaDriveEnv`.

**Observation adapter:**

- Attach two `RGBCamera` sensors to the ego at armrest-equivalent offsets: lateral ±0.35 m, height 0.81 m, mild toe-in.
- Each camera renders 256×256×3 RGB per step.
- Pipe both frames through `shared.preprocess.preprocess_pair()` → `(2, 84, 84) uint8` → `shared.preprocess.FrameStacker(num_frames=4).push()` → `(8, 84, 84) float32`.
- Visual randomization (brightness scale, Gaussian noise) is injected **into the raw RGB** before `preprocess_pair()`, ensuring the preprocessing pipeline itself is identical to real-hardware inference.

**Action adapter:**

- Gym action space: `Box(low=[-1, 0], high=[1, 1], dtype=float32)` — matches the serial protocol exactly.
- Convert to MetaDrive's native `[steering, throttle_brake]`: steer passthrough; throttle ≥ 0 means no brake.
- Steering gain multiplier (uniform 0.85–1.15 per episode) applied inside the adapter to bridge the bicycle ↔ skid-steer dynamics gap.
- Because the policy's action head already outputs bounded values (`tanh` on steer, `sigmoid` on throttle), no rescaling is needed on the training path — SB3's action-space clipping only fires on Gaussian-sampled exploration noise.

**Control rate:**

- `decision_repeat = 10`. MetaDrive physics runs at 100 Hz; policy polls every 10 physics steps = 10 Hz.

**Reward (per step):**

| Term | Weight (starting) | Source |
|---|---|---|
| progress along road | +1.0 / m | `vehicle.last_position` delta |
| collision (terminal) | −50.0 | `vehicle.crash_vehicle` / `crash_object` / `crash_sidewalk` |
| off-drivable-area (terminal) | −20.0 | `vehicle.out_of_road` |
| action smoothness | −0.1 × ‖aₜ − aₜ₋₁‖² | policy history |
| idle penalty | −0.05 if `|v| < 0.1` | vehicle velocity |

Weights are starting points, tunable in env config.

**Termination:**

- Collision → `terminated=True`
- Off-drivable-area → `terminated=True`
- 500 steps elapsed → `truncated=True` (= 50 s of decisions)

### Phase 1 — BC pretrain

`training/metadrive/collect_bc.py`:

- Instantiate `CouchMoMetaDriveEnv` with the IDM expert as the agent policy.
- Roll out ~500 episodes × ~500 steps ≈ 250k `(left_bgr, right_bgr, steer, throttle)` tuples.
- Log **raw BGR** frames (not preprocessed) so the shard format is identical to Gazebo shards — `train_bc.py` applies preprocessing at load.
- Write shards via `shared.dataset_format.Manifest` to `MyDrive/CouchMo/metadrive_bc/`.

CLI: `python -m training.metadrive.collect_bc --data-root /content/drive/MyDrive/CouchMo/metadrive_bc --episodes 500`

**BC training:** reuse `training/imitation/train_bc.py` as-is.

CLI: `python -m training.imitation.train_bc --data-root /content/drive/MyDrive/CouchMo/metadrive_bc --out /content/drive/MyDrive/CouchMo/checkpoints/bc_metadrive.pt --epochs 10`

Output: `bc_metadrive.pt` in the existing `{state_dict, arch}` format.

Expected wall-clock on Colab T4: BC data collection ~30–60 min; BC training ~60–90 min.

### Phase 2 — PPO fine-tune

`training/metadrive/train_ppo.py`:

- Build SB3 `PPO` with a custom `ActorCriticPolicy` subclass (`CouchMoActorCriticPolicy`):
  - **Features extractor** (`CouchMoFeaturesExtractor(BaseFeaturesExtractor)`): wraps `BCPolicy.conv1/conv2/conv3/fc1` → 256-dim feature vector.
  - **Action head**: linear layer → 2 outputs, with `tanh` on steer and `sigmoid` on throttle — **exactly matching `BCPolicy.forward`**. This output is the mean of the action distribution.
  - **Action distribution**: `DiagGaussianDistribution` with learnable `log_std` per dim. `squash_output=False` (squashing is baked into the action head's tanh/sigmoid). Actions sampled during training, clipped to the `Box(low=[-1,0], high=[1,1])` action space.
  - **Value head**: fresh-init linear; no BC equivalent.
- **Weight transfer BC → PPO:** copy `BCPolicy.conv1/conv2/conv3/fc1/fc2` verbatim into the matching layers of `CouchMoActorCriticPolicy`. Because the action-head activations are identical (`tanh` on steer, `sigmoid` on throttle), the copy is a direct 1:1 with no rescaling. Log a parameter diff showing which tensors transferred vs. were fresh-init (expected fresh: value head + `log_std`).
- Parallel rollouts: `SubprocVecEnv(n_envs=8)`.
- Hyperparameters (starting): `learning_rate=1e-4`, `n_steps=256`, `batch_size=64`, `n_epochs=10`, `gamma=0.99`, `gae_lambda=0.95`, `clip_range=0.2`, `total_timesteps=5_000_000`.
- Callbacks:
  - `CheckpointCallback(save_freq=100_000, save_path=output_dir)` → `checkpoint_{step}.zip`
  - `EvalCallback(eval_env, best_model_save_path=output_dir, eval_freq=50_000, n_eval_episodes=20, deterministic=True)` → `best.zip`
  - `CurriculumCallback` (custom): ramps traffic density + accident probability with training step count (see Domain randomization below).
- **Resume logic:** on startup, scan `--output-dir` for `checkpoint_*.zip`; if latest exists, `PPO.load(latest, env=vec_env)`, compute `remaining = total_timesteps - model.num_timesteps`, call `model.learn(total_timesteps=remaining, reset_num_timesteps=False)`.

CLI: `python -m training.metadrive.train_ppo --bc-ckpt /content/drive/MyDrive/CouchMo/checkpoints/bc_metadrive.pt --output-dir /content/drive/MyDrive/CouchMo/checkpoints/ppo_runs/run_001 --total-timesteps 5000000`

Expected wall-clock: 12–24 hrs on Colab T4, likely across 2 sessions.

### Export

Extend `training/imitation/export_onnx.py` with `--from-ppo PATH`:

- Load SB3 PPO `.zip`, extract `features_extractor.conv1/conv2/conv3/fc1` and `action_net` weights.
- Rebuild a `BCPolicy` instance and copy those weights into `conv1/conv2/conv3/fc1/fc2`. Architecture and activations match by construction, so this is a pure 1:1 copy.
- Export via the existing BC ONNX path. The exported graph ends with `tanh` on steer and `sigmoid` on throttle, producing `(steer ∈ [-1,1], throttle ∈ [0,1])` — matching the `serial_controller.py` contract.

Unit test: forward pass on random `(1, 8, 84, 84)` input — ONNX output must match SB3's `model.predict(..., deterministic=True)` output within 1e-5.

`runtime/inference.py` requires zero changes.

## Domain randomization

Load-bearing for sim-to-real when Gazebo is skipped. All randomization lives in `CouchMoMetaDriveEnv`.

**Applied per episode reset:**

| Knob | Range | Where |
|---|---|---|
| Map seed | `num_scenarios = 2000` | MetaDrive config |
| Traffic density | 0.1 – 0.4 | MetaDrive config |
| `accident_prob` | 0.0 – 0.3 | MetaDrive config |
| Random lane count | 2 – 4 | MetaDrive config |
| Random lane width | 3.0 – 4.5 m | MetaDrive config |
| Ego lateral spawn offset | ±0.3 m | env reset override |
| Ego heading offset | ±5° | env reset override |
| Camera pitch jitter | ±3° | camera mount pose on reset |
| Camera lateral jitter | ±2 cm | camera mount pose on reset |
| Global brightness scale | 0.7 – 1.3× | post-render gain |
| Steering gain multiplier | 0.85 – 1.15 | action adapter |

**Applied per step:**

- Camera noise: additive Gaussian σ = 3 on 0–255 RGB values, injected **before** `preprocess_pair()`.
- Action delay: with p = 0.1, re-apply the previous step's action (simulates serial/controller latency).

**Curriculum (`CurriculumCallback`):**

- Steps 0 – 1M: traffic density 0.1, `accident_prob` 0.0, lane count 3, lane width 4.0 m → basic lane following.
- Steps 1M – 3M: traffic density ramps to 0.3, `accident_prob` to 0.2, full lane randomization → introduce collision avoidance.
- Steps 3M – 5M: full randomization → robustness + polish.

**Explicitly not randomized:** observation shape, action space, control rate, preprocessing pipeline. Those are the real-car contract.

## Colab integration

**Drive layout:**

```
MyDrive/CouchMo/
  metadrive_bc/              # BC dataset
    manifest.json
    episodes/
  checkpoints/
    bc_metadrive.pt
    ppo_runs/
      run_001/
        checkpoint_{step}.zip
        best.zip
        eval_metrics.csv
        tb/
  exports/
    couchmo_v1.onnx
```

**`colab_runner.ipynb` cells (~12 total):**

1. `from google.colab import drive; drive.mount('/content/drive')`
2. `!git clone / pull` the repo, `cd` into it
3. `!pip install -r training/requirements.txt`
4. Env smoke test (`python -c "from training.metadrive.env import CouchMoMetaDriveEnv; ..."`)
5. (Optional) `python -m training.metadrive.collect_bc ...` — skipped if dataset on Drive
6. `python -m training.imitation.train_bc ...`
7. `python -m training.metadrive.train_ppo ...` (auto-resumes)
8. `python -m training.imitation.export_onnx --from-ppo ... --out /content/drive/.../exports/couchmo_v1.onnx`
9. `%load_ext tensorboard` + `%tensorboard --logdir /content/drive/.../run_001/tb`
10. `python -m training.metadrive.eval_policy --onnx ... --episodes 500` — go/no-go gate

## Evaluation

**During training (`EvalCallback`):**

- Separate eval env with `num_scenarios = 50` held-out seeds (disjoint from training seed range).
- Every 50k training steps, roll out 20 deterministic eval episodes.
- Metrics logged to TensorBoard + `eval_metrics.csv`:
  - Mean episode reward
  - Collision rate (fraction of episodes ending in crash)
  - Off-road rate
  - Mean forward distance
  - Mean episode length
- `best.zip` saved on eval-reward improvement.

**Post-training gate (`training/metadrive/eval_policy.py`):**

- Load the exported ONNX, roll out 500 deterministic held-out episodes.
- Report the full metric suite.
- Go/no-go thresholds (see R3):
  - Collision rate < 5%
  - Off-road rate < 10%
  - Mean episode length > 300 steps

Failing this gate means do not advance to real hardware; debug first.

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Sim-to-real visual gap (urban roads ≠ sidewalks) | High | Aggressive domain randomization; CLAHE in `preprocess_pair`; grayscale erases most color-palette divergence |
| Dynamics gap (bicycle ↔ skid-steer) | Medium | Steering-gain randomization (±15%); action smoothness penalty; low throttle cap on real car |
| PPO reward hacking (learns to idle) | Medium | Idle penalty + progress reward; `EvalCallback` tracks mean forward distance — alert if flat across several evals |
| Colab session death mid-training | High | Resumable checkpointing every 100k steps to Drive |
| ONNX export tanh/sigmoid mismatch | Medium | Unit test: ONNX output matches SB3 model output within 1e-5 on random obs |
| Catastrophic first real-car run | Catastrophic if unchecked | Throttle cap hardcoded in `runtime/inference.py` for Stages R2+ |

## Real-hardware rollout (phased)

Do not advance to the next stage until the prior stage passes.

**Stage R0 — Replay mode validation (no motion).**
Feed recorded Brio footage through the new ONNX via `runtime/inference.py`'s replay mode. Inspect commanded `(steer, throttle)` by eye — finite, plausible, no NaN, no saturation. No hardware connected.

**Stage R1 — Bench test (motors disconnected).**
Laptop ↔ ESP32 serial, motor outputs unplugged. Live inference from real webcams. Confirm: commands flow at 10 Hz; UART watchdog fires on inference crash; Circle button kills throttle instantly.

**Stage R2 — Parking lot, empty, throttle capped.**
Empty parking lot. Hardcode `throttle = min(throttle, 0.15)` in `runtime/inference.py` (one-line limiter). Operator holds PS4 controller; Circle = kill. Drive straight lines and gentle curves. Purpose: confirm nothing catastrophic at walking speed.

**Stage R3 — Parking lot, cones added.**
Same setup; throttle cap raised to 0.25. Place traffic cones. First real test of collision avoidance transfer.

**Stage R4 — Quiet sidewalk pilot.**
Low-traffic campus sidewalk, daylight only, safety operator walking alongside with kill switch. Throttle cap 0.30. Pass/fail criteria finalized after R3 results.

Advancing past R4 requires explicit team sign-off.

**Emergency stops available at all times:**

1. PS4 Circle button → ESP32 drops to `THROTTLE_REST` (implemented in `Couchmo.ino`).
2. UART watchdog → 500 ms silence → ESP32 drops to rest (implemented).
3. Physical e-stop — still an open item per `autonomous_car_research.md` §6; recommended before Stage R3.

## Milestones

- **M0** — `CouchMoMetaDriveEnv` smoke test passes (obs shape, action shape, one full rollout).
- **M1** — BC dataset collected on Drive; `train_bc.py` produces `bc_metadrive.pt`.
- **M2** — PPO fine-tune runs and resumes across sessions; eval metrics logged.
- **M3** — Post-training eval passes go/no-go thresholds; ONNX exported.
- **M4** — Stage R0 (replay mode) sanity checks pass; R1 bench test passes.
- **M5** — Stage R2 parking-lot drive completes without intervention.
- **M6** — Stage R3 cone avoidance passes.
- **M7** — Stage R4 sidewalk pilot passes.
