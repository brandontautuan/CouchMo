"""ONNX policy wrapper for the CouchMo laptop runtime.

Loads a ``.onnx`` file exported by ``training.imitation.export_onnx`` with the
CPU execution provider and exposes a stateful :class:`Policy` that owns the
``FrameStacker`` so inference can run from either raw BGR stereo frames (live
cameras) or from already-preprocessed ``(2, 84, 84)`` stereo pairs (replay
mode).

The preprocessing pipeline is imported verbatim from ``shared.preprocess`` so
the training-time and runtime observation shape/dtype are byte-identical.

Deliberately avoids any dependency on PyTorch, ROS, or the ``onnx`` package —
``onnxruntime`` is the only ML-layer import so ``pip install -r
runtime/requirements.txt`` is sufficient to run live mode.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from shared.preprocess import FrameStacker, preprocess_pair


class Policy:
    """Stateful ONNX policy that produces (steer, throttle) actions.

    One ``Policy`` instance owns one ``FrameStacker`` and one
    ``onnxruntime.InferenceSession``; it is therefore session-scoped — create
    a new one between episodes to reset the frame history.
    """

    def __init__(self, onnx_path: str | Path, num_frames: int = 4) -> None:
        """Load the ONNX model and initialise the frame stacker.

        Args:
            onnx_path: Path to an ``.onnx`` file exported by
                ``training.imitation.export_onnx`` (input name ``"input"``,
                shape ``(batch, 8, 84, 84)`` float32).
            num_frames: Temporal stack depth.  Must match the value used at
                training time (default 4 → 8 stacked grayscale channels).
        """
        onnx_path = Path(onnx_path)
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        self._session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        # Export uses input_names=["input"]; read it back instead of hardcoding
        # so a re-exported model with a different name still works.
        self._input_name = self._session.get_inputs()[0].name
        self._stacker = FrameStacker(num_frames=num_frames)

    # ------------------------------------------------------------------
    # Public inference API
    # ------------------------------------------------------------------

    def predict(
        self, left_bgr: np.ndarray, right_bgr: np.ndarray
    ) -> tuple[float, float]:
        """Run inference on a raw stereo BGR pair from the cameras.

        Args:
            left_bgr: ``(H, W, 3)`` uint8 BGR frame (any H, W).
            right_bgr: ``(H, W, 3)`` uint8 BGR frame (any H, W).

        Returns:
            ``(steer, throttle)`` as Python floats.  Not clamped — the caller
            (or the serial controller) is responsible for clipping.
        """
        pair = preprocess_pair(left_bgr, right_bgr)
        return self.predict_from_pair(pair)

    def predict_from_pair(self, pair: np.ndarray) -> tuple[float, float]:
        """Run inference on an already-preprocessed stereo pair.

        Args:
            pair: ``(2, 84, 84)`` uint8 stereo pair (e.g. from a recorded
                shard that has already been through :func:`preprocess_pair`).

        Returns:
            ``(steer, throttle)`` as Python floats.
        """
        stacked = self._stacker.push(pair)
        return self.predict_from_stacked(stacked)

    def predict_from_stacked(self, stacked: np.ndarray) -> tuple[float, float]:
        """Run inference on an already-stacked observation.

        Args:
            stacked: ``(8, 84, 84)`` float32 in ``[0, 1]`` — the direct output
                of ``FrameStacker.push``.

        Returns:
            ``(steer, throttle)`` as Python floats.
        """
        if stacked.ndim != 3:
            raise ValueError(
                f"stacked must be 3D (C, H, W); got shape {stacked.shape}"
            )

        x = stacked.astype(np.float32, copy=False)[np.newaxis, ...]
        out = self._session.run(None, {self._input_name: x})[0]
        # Model output: (1, 2) float32 = (steer, throttle).
        steer = float(out[0, 0])
        throttle = float(out[0, 1])
        return steer, throttle

    def reset(self) -> None:
        """Drop the buffered frames so the next push starts fresh."""
        self._stacker.reset()
