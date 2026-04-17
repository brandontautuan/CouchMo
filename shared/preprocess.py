"""Deterministic stereo-frame preprocessing shared by simulation, training, and runtime.

Pure NumPy + OpenCV. No torch, no ROS. Importable on any Python 3.10+ host.
"""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np

FRAME_HW: tuple[int, int] = (84, 84)


def preprocess_pair(left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
    """Resize, grayscale, and stack a stereo BGR pair into a (2, 84, 84) uint8 array.

    Args:
        left_bgr: Left eye frame as HxWx3 BGR uint8 (any H, W).
        right_bgr: Right eye frame as HxWx3 BGR uint8 (any H, W).

    Returns:
        (2, 84, 84) uint8 array: channel 0 = left, channel 1 = right.
    """
    if left_bgr.ndim != 3 or left_bgr.shape[2] != 3:
        raise ValueError(f"left_bgr must be HxWx3 BGR; got shape {left_bgr.shape}")
    if right_bgr.ndim != 3 or right_bgr.shape[2] != 3:
        raise ValueError(f"right_bgr must be HxWx3 BGR; got shape {right_bgr.shape}")
    if left_bgr.dtype != np.uint8 or right_bgr.dtype != np.uint8:
        raise ValueError(
            f"inputs must be uint8 BGR; got {left_bgr.dtype}, {right_bgr.dtype}"
        )

    h, w = FRAME_HW
    left_resized = cv2.resize(left_bgr, (w, h), interpolation=cv2.INTER_AREA)
    right_resized = cv2.resize(right_bgr, (w, h), interpolation=cv2.INTER_AREA)

    left_gray = cv2.cvtColor(left_resized, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_resized, cv2.COLOR_BGR2GRAY)

    stacked = np.stack([left_gray, right_gray], axis=0)
    return stacked.astype(np.uint8, copy=False)


class FrameStacker:
    """Rolling buffer that concatenates the last ``num_frames`` stereo pairs.

    On the first push, the frame is repeated ``num_frames`` times so that the
    output always has the same shape. Output is float32 in ``[0, 1]`` ready to
    feed a neural network.
    """

    def __init__(self, num_frames: int = 4) -> None:
        if num_frames < 1:
            raise ValueError(f"num_frames must be >= 1; got {num_frames}")
        self.num_frames = num_frames
        self._buf: deque[np.ndarray] = deque(maxlen=num_frames)

    def reset(self) -> None:
        """Drop all buffered frames; next push behaves as the first push."""
        self._buf.clear()

    def push(self, pair: np.ndarray) -> np.ndarray:
        """Push a ``(2, 84, 84)`` uint8 pair and return the stacked observation.

        Returns:
            (num_frames * 2, 84, 84) float32 array in [0, 1]. Most recent frame
            occupies the first two channels.
        """
        if pair.shape != (2, *FRAME_HW):
            raise ValueError(
                f"pair must have shape (2, {FRAME_HW[0]}, {FRAME_HW[1]}); got {pair.shape}"
            )
        if pair.dtype != np.uint8:
            raise ValueError(f"pair must be uint8; got {pair.dtype}")

        if not self._buf:
            for _ in range(self.num_frames):
                self._buf.append(pair.copy())
        else:
            self._buf.append(pair)

        # Newest first so the most recent frame is at the head of the channel axis.
        ordered = list(self._buf)[::-1]
        stacked = np.concatenate(ordered, axis=0)
        return stacked.astype(np.float32) / np.float32(255.0)
