"""CouchMo behavior cloning policy network.

Architecture: Atari-DQN-style CNN → (steer, throttle) with tanh/sigmoid heads.
Input: (batch, 8, 84, 84) float32 in [0, 1] — matches shared.preprocess FrameStacker output.
Output: (batch, 2) float32 — column 0 is steer in [-1, 1], column 1 is throttle in [0, 1].

ONNX-clean: uses only standard nn.Conv2d / nn.Linear / F.relu / torch.tanh / torch.sigmoid /
torch.cat — no LazyLinear, no scripting-only ops.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCPolicy(nn.Module):
    """Behavior cloning policy.

    Args:
        in_channels: Number of stacked input channels.  Default 8 = 4 frames × 2 eyes.
    """

    def __init__(self, in_channels: int = 8) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Compute flatten size from a dry forward pass — no hardcoding, no LazyLinear.
        with torch.no_grad():
            _dummy = torch.zeros(1, in_channels, 84, 84)
            _flat = self._conv_forward(_dummy).shape[1]

        self.fc1 = nn.Linear(_flat, 256)
        self.fc2 = nn.Linear(256, 2)

    def _conv_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.flatten(start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            x: (batch, 8, 84, 84) float32 in [0, 1].

        Returns:
            (batch, 2) float32: steer = col 0 in [-1, 1], throttle = col 1 in [0, 1].
        """
        h = self._conv_forward(x)
        h = F.relu(self.fc1(h))
        out = self.fc2(h)
        steer = torch.tanh(out[:, 0:1])
        throttle = torch.sigmoid(out[:, 1:2])
        return torch.cat([steer, throttle], dim=1)


def param_count(model: nn.Module) -> int:
    """Return total trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import time

    model = BCPolicy()
    model.eval()

    x = torch.zeros(1, 8, 84, 84)

    # Warm-up (avoid cold-start timing noise).
    with torch.no_grad():
        for _ in range(3):
            model(x)

    # Timed forward passes.
    n_runs = 20
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(n_runs):
            model(x)
        elapsed_ms = (time.perf_counter() - t0) / n_runs * 1000

    print(f"BCPolicy param count : {param_count(model):,}")
    print(f"CPU forward-pass mean: {elapsed_ms:.2f} ms  (n={n_runs})")
    print(f"Constraint < 50 ms  : {'PASS' if elapsed_ms < 50 else 'FAIL'}")
