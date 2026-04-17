"""PPO fine-tune on CouchMoMetaDriveEnv.

CLI usage::

    python -m training.metadrive.train_ppo \
        --bc-ckpt ./checkpoints/bc_metadrive.pt \
        --output-dir ./checkpoints/ppo_runs/run_001 \
        --total-timesteps 5000000

Features:
* Custom ActorCriticPolicy (BCPolicy-compatible action head).
* BC weight transfer via --bc-ckpt.
* SubprocVecEnv for parallel rollouts.
* CheckpointCallback + EvalCallback (held-out seeds).
* CurriculumCallback ramps traffic density + accident probability.
* Resume-from-checkpoint: scans --output-dir for the latest ``checkpoint_*.zip``
  and continues with ``reset_num_timesteps=False`` if found.
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any

import torch

log = logging.getLogger(__name__)

CHECKPOINT_PATTERN = re.compile(r"checkpoint_(\d+)_steps\.zip$")


def _latest_checkpoint(output_dir: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for p in output_dir.glob("checkpoint_*.zip"):
        m = CHECKPOINT_PATTERN.search(p.name)
        if m:
            candidates.append((int(m.group(1)), p))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def _make_env_factory(randomize: bool, scenario_start: int = 0):
    def _factory():
        from training.metadrive.env import CouchMoMetaDriveEnv

        return CouchMoMetaDriveEnv(
            config={
                "num_scenarios": 2000,
                "start_seed": scenario_start,
                "randomize": randomize,
            }
        )
    return _factory


def train(
    bc_ckpt: Path | None,
    output_dir: Path,
    total_timesteps: int,
    n_envs: int = 8,
    checkpoint_freq: int = 100_000,
    eval_freq: int = 50_000,
    eval_episodes: int = 20,
    n_steps: int = 256,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
) -> dict[str, Any]:
    """Run (or resume) a PPO fine-tune. Returns a small status dict for tests."""
    # Lazy imports — heavy optional deps.
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        BaseCallback,
        CallbackList,
        CheckpointCallback,
        EvalCallback,
    )
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    from training.imitation.model import BCPolicy
    from training.metadrive.policy import (
        CouchMoActorCriticPolicy,
        copy_bc_weights_into_policy,
    )

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    vec_cls = DummyVecEnv if n_envs == 1 else SubprocVecEnv
    # Vary start_seed per worker so parallel rollouts see disjoint scenario slices.
    train_env = vec_cls(
        [_make_env_factory(randomize=True, scenario_start=i * 2000) for i in range(n_envs)]
    )
    eval_env = None

    try:
        resumed_from: Path | None = _latest_checkpoint(output_dir)

        if resumed_from is not None:
            log.info("Resuming from %s", resumed_from)
            if bc_ckpt is not None:
                log.warning(
                    "Ignoring --bc-ckpt because resuming from %s (weights already learned)",
                    resumed_from,
                )
            model = PPO.load(str(resumed_from), env=train_env)
            bc_transferred: list[str] = []
        else:
            model = PPO(
                CouchMoActorCriticPolicy,
                train_env,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                tensorboard_log=str(output_dir / "tb"),
                verbose=1,
            )
            bc_transferred = []
            if bc_ckpt is not None:
                ckpt = torch.load(str(bc_ckpt), weights_only=True)
                bc = BCPolicy(**ckpt["arch"])
                bc.load_state_dict(ckpt["state_dict"])
                bc_transferred = copy_bc_weights_into_policy(bc, model.policy)
                log.info("BC weights transferred: %s", bc_transferred)

        callbacks: list[BaseCallback] = []

        if checkpoint_freq > 0:
            callbacks.append(
                CheckpointCallback(
                    save_freq=max(1, checkpoint_freq // max(1, n_envs)),
                    save_path=str(output_dir),
                    name_prefix="checkpoint",
                )
            )

        if eval_freq > 0:
            eval_env = DummyVecEnv([_make_env_factory(randomize=False, scenario_start=10_000)])
            callbacks.append(
                EvalCallback(
                    eval_env,
                    best_model_save_path=str(output_dir),
                    eval_freq=max(1, eval_freq // max(1, n_envs)),
                    n_eval_episodes=eval_episodes,
                    deterministic=True,
                    render=False,
                    log_path=str(output_dir),
                )
            )

        callbacks.append(CurriculumCallback(train_env))

        remaining = total_timesteps - int(model.num_timesteps)
        if remaining > 0:
            model.learn(
                total_timesteps=remaining,
                callback=CallbackList(callbacks),
                reset_num_timesteps=False,
            )
        else:
            log.info(
                "Already at %d / %d timesteps; skipping learn()",
                int(model.num_timesteps),
                total_timesteps,
            )

        final = output_dir / "final.zip"
        model.save(str(final))

        return {
            "num_timesteps": int(model.num_timesteps),
            "resumed_from": str(resumed_from) if resumed_from else None,
            "bc_transferred": bc_transferred,
            "final_path": str(final),
        }
    finally:
        train_env.close()
        if eval_env is not None:
            eval_env.close()


class CurriculumCallback:
    """Ramps env config between phases of training.

    Phases (spec §Domain randomization):
    * [0, 1M):       traffic_density=0.1, accident_prob=0.0
    * [1M, 3M):      traffic_density=0.3, accident_prob=0.2
    * [3M, inf):     traffic_density=0.4, accident_prob=0.3
    """

    def __init__(self, vec_env):
        # SB3 CallbackList auto-detects callbacks via BaseCallback; subclass inline.
        from stable_baselines3.common.callbacks import BaseCallback

        class _Impl(BaseCallback):
            def __init__(self, vec_env) -> None:
                super().__init__()
                self._vec_env = vec_env
                self._phase = -1

            def _on_step(self) -> bool:
                step = int(self.num_timesteps)
                phase = 0 if step < 1_000_000 else (1 if step < 3_000_000 else 2)
                if phase == self._phase:
                    return True
                self._phase = phase
                density, accident = {
                    0: (0.1, 0.0),
                    1: (0.3, 0.2),
                    2: (0.4, 0.3),
                }[phase]
                self._vec_env.env_method(
                    "_update_scenario_config",
                    traffic_density=density,
                    accident_prob=accident,
                )
                return True

        self._impl = _Impl(vec_env)

    def __getattr__(self, name):
        return getattr(self._impl, name)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PPO fine-tune on CouchMoMetaDriveEnv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--bc-ckpt", type=Path, default=None, help="BC .pt checkpoint for warm start.")
    p.add_argument("--output-dir", type=Path, required=True, help="Where checkpoints / TB logs go.")
    p.add_argument("--total-timesteps", type=int, default=5_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--checkpoint-freq", type=int, default=100_000)
    p.add_argument("--eval-freq", type=int, default=50_000)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--n-steps", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    return p


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    args = _build_parser().parse_args(argv)
    try:
        result = train(
            bc_ckpt=args.bc_ckpt,
            output_dir=args.output_dir,
            total_timesteps=args.total_timesteps,
            n_envs=args.n_envs,
            checkpoint_freq=args.checkpoint_freq,
            eval_freq=args.eval_freq,
            eval_episodes=args.eval_episodes,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
        print(result)
    except Exception as exc:  # noqa: BLE001 — top-level CLI catch
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
