"""MetaDrive-based RL training for CouchMo.

See docs/superpowers/specs/2026-04-17-metadrive-rl-training-design.md for the
full design. Public modules:

- env           -- CouchMoMetaDriveEnv, a gym.Env wrapping SafeMetaDriveEnv
- expert_policy -- thin adapter around MetaDrive's IDM expert
- collect_bc    -- CLI that records expert rollouts as shared.dataset_format shards
- policy        -- CouchMoFeaturesExtractor + CouchMoActorCriticPolicy
- train_ppo     -- PPO fine-tune CLI with Drive-backed resume
- eval_policy   -- post-training go/no-go evaluation
"""
