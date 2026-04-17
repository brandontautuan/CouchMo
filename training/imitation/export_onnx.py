"""ONNX export for the CouchMo BCPolicy checkpoint.

CLI usage::

    python -m training.imitation.export_onnx --in checkpoints/bc.pt
    python -m training.imitation.export_onnx --in checkpoints/bc.pt --out model.onnx --opset 17 --verify

The ``export()`` function is the importable entry point used by tests.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import onnx
import torch

from training.imitation.model import BCPolicy
from training.imitation.train_bc import BC_META_SCHEMA

# Fixed spatial input shape (channels, height, width) that matches FrameStacker output.
_IN_CHANNELS = 8
_H = 84
_W = 84


def _load_model_from_ppo(zip_path: Path) -> tuple[BCPolicy, int]:
    """Load an SB3 PPO zip and return (BCPolicy with PPO weights copied in, in_channels).

    Layer mapping (see training.metadrive.policy):
        features_extractor.conv1  -> BCPolicy.conv1
        features_extractor.conv2  -> BCPolicy.conv2
        features_extractor.conv3  -> BCPolicy.conv3
        features_extractor.fc1    -> BCPolicy.fc1
        action_net                -> BCPolicy.fc2
    """
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise ImportError(
            "stable-baselines3 is required for --from-ppo; "
            "pip install -r training/requirements-rl.txt"
        ) from exc

    from training.metadrive.policy import CouchMoFeaturesExtractor

    ppo = PPO.load(str(zip_path))
    extractor = ppo.policy.features_extractor
    if not isinstance(extractor, CouchMoFeaturesExtractor):
        raise ValueError(
            f"--from-ppo expects a PPO zip trained with CouchMoActorCriticPolicy; "
            f"got features_extractor={type(extractor).__name__}. "
            f"See training.metadrive.train_ppo."
        )
    in_channels = int(extractor.conv1.in_channels)

    model = BCPolicy(in_channels=in_channels)
    model.conv1.load_state_dict(extractor.conv1.state_dict())
    model.conv2.load_state_dict(extractor.conv2.state_dict())
    model.conv3.load_state_dict(extractor.conv3.state_dict())
    model.fc1.load_state_dict(extractor.fc1.state_dict())
    model.fc2.load_state_dict(ppo.policy.action_net.state_dict())
    model.eval()
    return model, in_channels


def export(
    pt_path: Path,
    onnx_path: Path | None = None,
    opset: int = 17,
    verify: bool = False,
    from_ppo: bool = False,
) -> Path:
    """Export a Task-8a BCPolicy checkpoint to ONNX and validate it.

    Args:
        pt_path:   Path to a ``.pt`` checkpoint produced by ``train_bc.py``.
        onnx_path: Destination ``.onnx`` path.  Defaults to ``pt_path`` with
                   the ``.pt`` extension replaced by ``.onnx``.
        opset:     ONNX opset version (default 17).
        verify:    If *True*, also run a numerical equivalence check (random
                   input through Torch and ONNX Runtime; asserts max abs diff
                   < 1e-4).  Prints PASS/FAIL and raises ``RuntimeError`` on
                   failure.
        from_ppo:  If *True*, interpret ``pt_path`` as a stable-baselines3 PPO
                   ``.zip`` produced by ``training.metadrive.train_ppo``. The
                   custom policy's features_extractor (conv1/conv2/conv3/fc1)
                   and action_net are copied into a fresh BCPolicy, which is
                   what gets exported. Skips the bc_meta sibling check.

    Returns:
        The path to the written ``.onnx`` file.

    Raises:
        FileNotFoundError: if ``pt_path`` does not exist.
        ValueError: if the checkpoint format is not the Task-8a format.
        onnx.checker.ValidationError: if the exported graph fails structural
            validation.
        RuntimeError: if ``verify=True`` and the numerical equivalence check
            fails.
    """
    pt_path = Path(pt_path)
    if not pt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {pt_path}")

    if onnx_path is None:
        onnx_path = pt_path.with_suffix(".onnx")
    onnx_path = Path(onnx_path)

    # ------------------------------------------------------------------
    # Load checkpoint — Task-8a format: {"state_dict": ..., "arch": ...}
    # ------------------------------------------------------------------
    if from_ppo:
        model, in_channels = _load_model_from_ppo(pt_path)
    else:
        # Optionally validate bc_meta sibling as a sanity check.
        meta_path = pt_path.with_name(pt_path.stem + "_meta.json")
        if meta_path.exists():
            import json
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            schema = meta.get("schema", "")
            if schema != BC_META_SCHEMA:
                raise ValueError(
                    f"Sibling meta file {meta_path} has unexpected schema "
                    f"'{schema}'; expected '{BC_META_SCHEMA}'. Is this a Task-8a checkpoint?"
                )

        ckpt = torch.load(str(pt_path), weights_only=True)

        if not isinstance(ckpt, dict) or "state_dict" not in ckpt or "arch" not in ckpt:
            raise ValueError(
                "Checkpoint format not recognized; expected Task-8a format "
                "{'state_dict': ..., 'arch': {'in_channels': int}}. "
                f"Got type={type(ckpt).__name__}, keys={list(ckpt.keys()) if isinstance(ckpt, dict) else 'N/A'}"
            )

        arch = ckpt["arch"]
        model = BCPolicy(**arch)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        in_channels = arch.get("in_channels", _IN_CHANNELS)

    # ------------------------------------------------------------------
    # ONNX export — legacy tracing exporter (NOT dynamo_export)
    # ------------------------------------------------------------------
    dummy_input = torch.zeros(1, in_channels, _H, _W)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch"},
            "output": {0: "batch"},
        },
        # Force the legacy tracing exporter (NOT torch.onnx.dynamo_export).
        # In torch >= 2.6+ dynamo=True is the default; we override it here.
        # The dynamo exporter is still experimental and requires onnxscript.
        dynamo=False,
        # do_constant_folding defaults to True — keep it that way so that
        # check_model operates on a fully-folded graph.
    )

    # ------------------------------------------------------------------
    # Structural validation — always run unconditionally
    # ------------------------------------------------------------------
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    # ------------------------------------------------------------------
    # Optional numerical equivalence check
    # ------------------------------------------------------------------
    if verify:
        import numpy as np
        import onnxruntime as ort

        x = torch.rand(3, in_channels, _H, _W)

        with torch.no_grad():
            y_torch = model(x).numpy()

        sess = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        y_onnx = sess.run(None, {"input": x.numpy()})[0]

        max_diff = float(np.max(np.abs(y_torch - y_onnx)))
        if max_diff < 1e-4:
            print(f"PASS  max_abs_diff={max_diff:.2e}")
        else:
            print(f"FAIL  max_abs_diff={max_diff:.2e}  (threshold=1e-4)")
            raise RuntimeError(
                f"ONNX numerical equivalence check FAILED: "
                f"max abs diff {max_diff:.2e} >= 1e-4"
            )

    return onnx_path


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Export a CouchMo BCPolicy checkpoint to ONNX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--in",
        dest="pt_path",
        required=True,
        type=Path,
        metavar="PATH",
        help="Path to the .pt checkpoint (Task-8a format).",
    )
    parser.add_argument(
        "--out",
        dest="onnx_path",
        type=Path,
        default=None,
        metavar="PATH",
        help="Destination .onnx file.  Defaults to <--in>.onnx (sibling of checkpoint).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        metavar="N",
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run numerical equivalence check (Torch vs ONNX Runtime) after export.",
    )
    parser.add_argument(
        "--from-ppo",
        action="store_true",
        help="Interpret --in as an SB3 PPO .zip checkpoint instead of a Task-8a .pt.",
    )

    args = parser.parse_args(argv)

    try:
        out = export(
            pt_path=args.pt_path,
            onnx_path=args.onnx_path,
            opset=args.opset,
            verify=args.verify,
            from_ppo=args.from_ppo,
        )
        print(f"Exported → {out}")
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as exc:
        # Equivalence check failure.
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
