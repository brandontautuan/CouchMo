"""Render the expert driving a roundabout loop to demo_circle.mp4."""
import cv2

from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.examples.ppo_expert import expert

OUT = "demo_circle.mp4"
FPS = 10
NUM_FRAMES = 300  # 30s at 10 FPS
FILM_SIZE = (800, 800)


def main() -> None:
    env = MetaDriveEnv(
        dict(
            use_render=False,
            num_scenarios=1,
            start_seed=3,
            traffic_density=0.0,
            map="CCCCCCCC",  # 8 curves chained — a winding loop
        )
    )
    writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), FPS, FILM_SIZE)
    try:
        env.reset(seed=3)
        for i in range(NUM_FRAMES):
            action = expert(env.agent, deterministic=True)
            _, _, terminated, truncated, _ = env.step(action)
            frame = env.render(
                mode="top_down",
                film_size=(2500, 2500),      # big canvas so the whole loop fits
                scaling=3,                    # zoomed out a bit
                screen_size=FILM_SIZE,        # output size
                screen_record=True,
                target_agent_heading_up=False,  # keep map fixed so the curve is visible
            )
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if terminated or truncated:
                env.reset(seed=3)
            if (i + 1) % 30 == 0:
                print(f"  rendered {i + 1}/{NUM_FRAMES} frames")
    finally:
        writer.release()
        env.close()
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
