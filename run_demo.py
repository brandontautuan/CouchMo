"""Render MetaDrive's PPO expert driving to an MP4.

Usage:
    ./py39 run_demo.py

Output: demo.mp4 in the current directory (~10 seconds of driving, top-down view).
"""
import cv2
import numpy as np

from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.examples.ppo_expert import expert

OUT = "demo.mp4"
FPS = 10          # lower FPS = car appears slower on playback
NUM_FRAMES = 300  # 30 seconds at 10 FPS
FILM_SIZE = (800, 800)


def main() -> None:
    env = MetaDriveEnv(
        dict(
            use_render=False,
            num_scenarios=20,
            start_seed=5,
            traffic_density=0.1,
            map="SCrRXT",  # mix of straights, curves, roundabout, intersection
        )
    )
    writer = cv2.VideoWriter(
        OUT, cv2.VideoWriter_fourcc(*"mp4v"), FPS, FILM_SIZE
    )
    try:
        env.reset(seed=5)
        for i in range(NUM_FRAMES):
            action = expert(env.agent, deterministic=True)
            _, _, terminated, truncated, _ = env.step(action)
            frame = env.render(
                mode="top_down",
                film_size=FILM_SIZE,
                screen_record=True,
                target_vehicle_heading_up=True,
            )
            # MetaDrive returns RGB; cv2 writes BGR.
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if terminated or truncated:
                env.reset()
            if (i + 1) % 20 == 0:
                print(f"  rendered {i + 1}/{NUM_FRAMES} frames")
    finally:
        writer.release()
        env.close()
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
