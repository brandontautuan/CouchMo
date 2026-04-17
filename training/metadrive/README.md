# CouchMo — MetaDrive RL training

This directory implements the MetaDrive-based RL pipeline specified in
`docs/superpowers/specs/2026-04-17-metadrive-rl-training-design.md`.

## Install (local dev)

```bash
cd training
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt -r requirements-rl.txt
```

## Pipeline

1. `python -m training.metadrive.collect_bc --data-root <path> --episodes 500`
2. `python -m training.imitation.train_bc --data-root <path> --out <bc.pt>`
3. `python -m training.metadrive.train_ppo --bc-ckpt <bc.pt> --output-dir <run_dir>`
4. `python -m training.imitation.export_onnx --from-ppo <run_dir/best.zip> --out <model.onnx>`
5. `python -m training.metadrive.eval_policy --onnx <model.onnx> --episodes 500`

See `colab_runner.ipynb` for the Colab-driven flow.
