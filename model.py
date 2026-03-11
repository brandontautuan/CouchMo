"""
model.py
-----------------------
Autonomous Driving — Perception Pipeline
Handles: 2 grayscale webcams → frame stacking → (8, 84, 84) tensor

Libraries: OpenCV, NumPy, scikit-image, PyTorch
"""

import cv2
import numpy as np
import torch
from collections import deque
from skimage import exposure


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

IMG_HEIGHT = 84
IMG_WIDTH  = 84
N_FRAMES   = 4          # frames to stack per camera
N_CAMS     = 2          # left cam + right cam
# Final tensor shape: (N_CAMS * N_FRAMES, H, W) = (8, 84, 84)


# ─────────────────────────────────────────────
# SINGLE FRAME PROCESSOR
# ─────────────────────────────────────────────

class FrameProcessor:
    """
    Processes a single raw webcam frame into a normalized grayscale image.

    Steps:
        1. Convert BGR → Grayscale
        2. Resize to 84x84
        3. Enhance contrast (CLAHE via scikit-image)
        4. Normalize pixels to [0, 1]
    """

    def __init__(self, height: int = IMG_HEIGHT, width: int = IMG_WIDTH):
        self.height = height
        self.width  = width

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Args:
            frame: Raw BGR image from OpenCV (H, W, 3)
        Returns:
            Processed grayscale image (84, 84) float32 in [0, 1]
        """
        # 1. Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. Resize to 84x84
        resized = cv2.resize(gray, (self.width, self.height),
                             interpolation=cv2.INTER_AREA)

        # 3. Contrast-limited adaptive histogram equalization (CLAHE)
        #    Helps normalize across different lighting conditions (important for sim-to-real)
        enhanced = exposure.equalize_adapthist(resized, clip_limit=0.03)

        # 4. Ensure float32 in [0, 1] (equalize_adapthist already returns [0,1])
        normalized = enhanced.astype(np.float32)

        return normalized  # shape: (84, 84)


# ─────────────────────────────────────────────
# FRAME STACK (per camera)
# ─────────────────────────────────────────────

class FrameStack:
    """
    Maintains a rolling stack of N most recent frames for one camera.
    Stacking gives the model temporal info (motion, velocity).

    Shape: (N_FRAMES, 84, 84)
    """

    def __init__(self, n_frames: int = N_FRAMES,
                 height: int = IMG_HEIGHT, width: int = IMG_WIDTH):
        self.n_frames  = n_frames
        self.height    = height
        self.width     = width
        self.frames    = deque(maxlen=n_frames)
        self.processor = FrameProcessor(height, width)

    def reset(self, frame: np.ndarray) -> np.ndarray:
        """
        Initialize the stack by filling all slots with the first frame.
        Call this at the start of every new episode.

        Returns:
            Stacked frames (N_FRAMES, 84, 84)
        """
        processed = self.processor.process(frame)
        for _ in range(self.n_frames):
            self.frames.append(processed)
        return self._get_stack()

    def step(self, frame: np.ndarray) -> np.ndarray:
        """
        Add a new frame and return the updated stack.

        Returns:
            Stacked frames (N_FRAMES, 84, 84)
        """
        processed = self.processor.process(frame)
        self.frames.append(processed)
        return self._get_stack()

    def _get_stack(self) -> np.ndarray:
        # Stack along axis 0: list of (84,84) → (N_FRAMES, 84, 84)
        return np.stack(list(self.frames), axis=0)  # (4, 84, 84)


# ─────────────────────────────────────────────
# DUAL CAMERA PERCEPTION PIPELINE
# ─────────────────────────────────────────────

class PerceptionPipeline:
    """
    Full perception pipeline for two webcams.

    Input:  Two raw BGR frames (left cam, right cam)
    Output: PyTorch tensor of shape (8, 84, 84)
              = 2 cameras × 4 stacked frames × 84×84

    Usage:
        pipeline = PerceptionPipeline()

        # At episode start:
        state = pipeline.reset(left_frame, right_frame)

        # Each step:
        state = pipeline.step(left_frame, right_frame)

        # Feed directly into your CNN:
        tensor = pipeline.to_tensor(state)  # shape: (1, 8, 84, 84)
    """

    def __init__(self, n_frames: int = N_FRAMES,
                 height: int = IMG_HEIGHT, width: int = IMG_WIDTH):
        self.left_stack  = FrameStack(n_frames, height, width)
        self.right_stack = FrameStack(n_frames, height, width)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def reset(self, left_frame: np.ndarray,
              right_frame: np.ndarray) -> np.ndarray:
        """
        Reset both stacks at the start of a new episode.

        Returns:
            Combined state array (8, 84, 84)
        """
        left  = self.left_stack.reset(left_frame)    # (4, 84, 84)
        right = self.right_stack.reset(right_frame)  # (4, 84, 84)
        return np.concatenate([left, right], axis=0) # (8, 84, 84)

    def step(self, left_frame: np.ndarray,
             right_frame: np.ndarray) -> np.ndarray:
        """
        Process one timestep — call this every control cycle (10Hz).

        Returns:
            Combined state array (8, 84, 84)
        """
        left  = self.left_stack.step(left_frame)    # (4, 84, 84)
        right = self.right_stack.step(right_frame)  # (4, 84, 84)
        return np.concatenate([left, right], axis=0) # (8, 84, 84)

    def to_tensor(self, state: np.ndarray) -> torch.Tensor:
        """
        Convert state array to PyTorch tensor ready for CNN input.

        Returns:
            Tensor of shape (1, 8, 84, 84) on the correct device
        """
        tensor = torch.FloatTensor(state)       # (8, 84, 84)
        tensor = tensor.unsqueeze(0)            # (1, 8, 84, 84) — batch dim
        return tensor.to(self.device)


# ─────────────────────────────────────────────
# WEBCAM CAPTURE HELPER
# ─────────────────────────────────────────────

class WebcamCapture:
    """
    Handles opening and reading from two USB webcams.

    Args:
        left_id:  OpenCV device index for left camera  (usually 0)
        right_id: OpenCV device index for right camera (usually 1)
    """

    def __init__(self, left_id: int = 0, right_id: int = 1):
        self.left_cam  = cv2.VideoCapture(left_id)
        self.right_cam = cv2.VideoCapture(right_id)

        # Lock exposure to prevent lighting variation breaking the model
        self.left_cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # manual mode
        self.right_cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

        if not self.left_cam.isOpened():
            raise RuntimeError(f"Could not open left camera (id={left_id})")
        if not self.right_cam.isOpened():
            raise RuntimeError(f"Could not open right camera (id={right_id})")

    def read(self):
        """
        Read one frame from each camera.

        Returns:
            (left_frame, right_frame) as BGR numpy arrays
        Raises:
            RuntimeError if either camera fails to read
        """
        ret_l, left_frame  = self.left_cam.read()
        ret_r, right_frame = self.right_cam.read()

        if not ret_l:
            raise RuntimeError("Failed to read from left camera")
        if not ret_r:
            raise RuntimeError("Failed to read from right camera")

        return left_frame, right_frame

    def release(self):
        """Always call this on shutdown to release camera resources."""
        self.left_cam.release()
        self.right_cam.release()
        cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# QUICK TEST — run this file directly to verify
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing PerceptionPipeline with dummy frames...\n")

    # Simulate raw webcam frames (random noise as placeholder)
    dummy_left  = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dummy_right = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Initialize pipeline
    pipeline = PerceptionPipeline(n_frames=N_FRAMES)

    # Reset (episode start)
    state = pipeline.reset(dummy_left, dummy_right)
    print(f"State shape after reset:  {state.shape}")       # (8, 84, 84)
    print(f"State dtype:              {state.dtype}")       # float32
    print(f"State value range:        [{state.min():.3f}, {state.max():.3f}]")  # [0, 1]

    # Step (each control cycle)
    for i in range(5):
        dummy_left  = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        dummy_right = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        state = pipeline.step(dummy_left, dummy_right)

    tensor = pipeline.to_tensor(state)
    print(f"\nTensor shape (CNN input): {tensor.shape}")    # (1, 8, 84, 84)
    print(f"Tensor device:            {tensor.device}")
    print(f"\n✅ Perception pipeline working correctly.")
    print(f"   Ready to connect to CNN encoder.")