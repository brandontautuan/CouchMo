# CouchMo Training

Native PyTorch training surface. Reads dataset shards written by `simulation/`,
produces `.pt` checkpoints and `.onnx` exports that the `runtime/` surface loads.

- **Runs on:** Windows, macOS, or Linux with Python 3.10+
- **Talks to `simulation/` via:** files on disk under `--data-root`
- **Talks to `runtime/` via:** exported `.onnx` files
- **Does NOT require:** ROS, Docker, or any simulator dependency

## Install

### Windows (PowerShell)

```powershell
cd training
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt
```

### macOS / Linux (bash or zsh)

```bash
cd training
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt
```

## Run tests

From inside the activated venv, at the `training/` directory:

```bash
python -m pytest -q
```

`conftest.py` prepends the repo root to `sys.path` so the tests can
`import shared.preprocess` without installing the `shared` package.

## Data layout convention

Every script in this project takes `--data-root <path>` and treats that path as
the authoritative dataset root. The default development convention is:

```
training/data/
  manifest.json          # written by shared.dataset_format.Manifest
  episodes/
    2026-04-15T18-22-07_ep0001.npz
    2026-04-15T18-23-55_ep0002.npz
    ...
training/checkpoints/
  <run-id>/best.pt
  <run-id>/best.onnx
```

Both `data/` and `checkpoints/` are gitignored. Pass any other root with
`--data-root <path>` and the scripts will use it unchanged.

## MetaDrive RL pipeline

The `training/metadrive/` subpackage adds a sibling RL pipeline (BC pretrain +
PPO fine-tune in MetaDrive). It reuses this package's `BCPolicy`, `train_bc.py`,
and `export_onnx.py` as-is. Heavy deps live in `requirements-rl.txt`:

```bash
pip install -r requirements-rl.txt
```

See `training/metadrive/README.md` for the full flow.
