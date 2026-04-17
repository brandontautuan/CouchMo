"""Stereo USB camera capture for the CouchMo laptop runtime.

Wraps two ``cv2.VideoCapture`` objects behind a tiny context-manager friendly
class so :mod:`runtime.drive` can read synchronised BGR frames without caring
about OpenCV plumbing.  OpenCV's ``VideoCapture`` is the same API on Windows,
macOS, and Linux so nothing in here is platform-specific.

Frame rate and resolution are intentionally configurable at construction time
but default to a reasonable 640x480 @ 30 fps — the downstream
``shared.preprocess.preprocess_pair`` downsizes to 84x84 grayscale anyway.
"""
from __future__ import annotations

import cv2
import numpy as np


class DualCamera:
    """Open two USB cameras and read synchronised BGR frames.

    Usage::

        with DualCamera(left_index=0, right_index=1) as cams:
            left_bgr, right_bgr = cams.read()
    """

    def __init__(
        self,
        left_index: int,
        right_index: int,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ) -> None:
        """Open both capture devices.

        Args:
            left_index: OpenCV camera index for the left eye.
            right_index: OpenCV camera index for the right eye.
            width: Requested capture width in pixels.
            height: Requested capture height in pixels.
            fps: Requested frame rate.  Cameras silently clamp to whatever
                the hardware supports.

        Raises:
            RuntimeError: If either camera fails to open.
        """
        if left_index == right_index:
            raise ValueError(
                f"left_index and right_index must differ; both are {left_index}"
            )

        self._left = cv2.VideoCapture(left_index)
        self._right = cv2.VideoCapture(right_index)

        left_ok = self._left.isOpened()
        right_ok = self._right.isOpened()
        if not (left_ok and right_ok):
            # Release whatever did open before raising.
            self.release()
            failures = []
            if not left_ok:
                failures.append(f"left (index={left_index})")
            if not right_ok:
                failures.append(f"right (index={right_index})")
            raise RuntimeError(
                "Failed to open camera(s): " + ", ".join(failures)
            )

        for cap in (self._left, self._right):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)

        self.left_index = left_index
        self.right_index = right_index

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        """Read one frame from each camera.

        Returns:
            ``(left_bgr, right_bgr)`` as ``(H, W, 3)`` uint8 BGR arrays.

        Raises:
            RuntimeError: If either capture fails.
        """
        ok_l, left = self._left.read()
        ok_r, right = self._right.read()
        if not ok_l or not ok_r:
            raise RuntimeError(
                f"camera read failed: left_ok={ok_l} (index={self.left_index}),"
                f" right_ok={ok_r} (index={self.right_index})"
            )
        return left, right

    def release(self) -> None:
        """Release both capture devices.  Safe to call multiple times."""
        for cap_name in ("_left", "_right"):
            cap = getattr(self, cap_name, None)
            if cap is not None:
                try:
                    cap.release()
                except Exception:  # pragma: no cover — release is best-effort.
                    pass
                setattr(self, cap_name, None)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "DualCamera":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def list_cameras(max_index: int = 6) -> list[int]:
    """Return the list of OpenCV camera indices that opened successfully.

    Tries indices ``0 .. max_index - 1`` and returns those where
    ``cv2.VideoCapture`` opens AND the first ``read()`` succeeds.  Useful as a
    one-liner sanity check before launching :mod:`runtime.drive`.

    Args:
        max_index: Exclusive upper bound on indices to probe.

    Returns:
        Sorted list of successful indices (may be empty).
    """
    if max_index < 0:
        raise ValueError(f"max_index must be >= 0; got {max_index}")

    found: list[int] = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        try:
            if cap.isOpened():
                ok, _frame = cap.read()
                if ok:
                    found.append(i)
        finally:
            cap.release()
    return found
