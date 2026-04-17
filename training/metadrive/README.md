# CouchMo — MetaDrive RL training

Implements the pipeline specified in
`docs/superpowers/specs/2026-04-17-metadrive-rl-training-design.md`:

```
IDM expert rollouts  ->  BC pretrain  ->  PPO fine-tune  ->  ONNX  ->  serial_controller.py
```

## Install (local dev)

```bash
cd training
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt -r requirements-rl.txt
```

## Local smoke test

```bash
python -m pytest tests/test_metadrive_integration.py -v
```

## Colab flow

Open `colab_runner.ipynb` in Colab and run cells top to bottom.

## Module-by-module

| Module | Purpose |
|---|---|
| `env.py` | Gym env wrapping `SafeMetaDriveEnv`; stereo cam obs + `(steer, throttle)` action. |
| `expert_policy.py` | IDM expert adapter; produces actions in serial-protocol format. |
| `collect_bc.py` | CLI: roll out expert, write shards via `shared.dataset_format`. |
| `policy.py` | `CouchMoFeaturesExtractor` + `CouchMoActorCriticPolicy` (tanh+sigmoid action head for 1:1 BC weight transfer). |
| `train_ppo.py` | SB3 PPO CLI with BC warm start, curriculum, resume from Drive. |
| `eval_policy.py` | ONNX evaluation with R3 go/no-go thresholds; non-zero exit if thresholds fail. |
| `colab_runner.ipynb` | Thin Colab orchestration. |

## Go/no-go gate

Before any real-hardware work, `eval_policy.py` must exit 0 on ≥500 held-out
episodes. Thresholds live in `eval_policy.GO_NO_GO`.

## Real-hardware rollout

See spec §Real-hardware rollout. Stages R0 (replay mode), R1 (bench),
R2 (parking lot, capped), R3 (cones), R4 (sidewalk pilot).
