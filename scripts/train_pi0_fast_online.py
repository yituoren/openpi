#!/usr/bin/env python
from __future__ import annotations

import argparse
import dataclasses
import logging
from typing import Any

from openpi.envs import make_env, make_env_config
from openpi.rl.exploration import KeyframeExplorationConfig
from openpi.rl.pi0_fast_online_trainer import Pi0FastOnlineRLConfig, Pi0FastOnlineTrainer
from openpi.rl.pi0_fast_policy import create_trained_pi0_fast_rl_policy
from openpi.rl.pi0_fast_rollout import Pi0FastChunkCollector
from openpi.rl.reward_value import CallableChunkRewardValueProvider
from openpi.training import config as train_config


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--policy.path", dest="policy_path", type=str, required=True)
    p.add_argument("--policy.config", dest="policy_config", type=str, default="pi0_fast_robotwin")
    p.add_argument("--env.type", dest="env_type", type=str, default="robotwin")
    p.add_argument("--env.task", dest="env_task", type=str, default=None)
    p.add_argument("--env.ws_url", dest="env_ws_url", type=str, default="ws://127.0.0.1:8765")
    p.add_argument("--env.observation_height", dest="env_observation_height", type=int, default=480)
    p.add_argument("--env.observation_width", dest="env_observation_width", type=int, default=640)
    p.add_argument("--env.action_dim", dest="env_action_dim", type=int, default=14)
    p.add_argument("--env.max_episode_steps_model", dest="env_max_episode_steps_model", type=int, default=300)
    p.add_argument("--num_envs", type=int, default=1)
    p.add_argument("--rollout_batch_size", type=int, default=128)
    p.add_argument("--mini_batch_size", type=int, default=32)
    p.add_argument("--ppo_epochs", type=int, default=4)
    p.add_argument("--buffer_capacity", type=int, default=1024)
    p.add_argument("--max_policy_lag", type=int, default=4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--entropy_coef", type=float, default=0.0)
    p.add_argument("--fixed_reward", type=float, default=0.0)
    p.add_argument("--enable_keyframe_exploration", action="store_true")
    p.add_argument("--keyframe_threshold", type=float, default=0.7)
    p.add_argument("--explore_dct_dims", type=int, default=0)
    p.add_argument("--total_updates", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _default_reward_value_fn(**kwargs) -> dict[str, float]:
    reward = float(kwargs.pop("fixed_reward"))
    del kwargs
    return {"reward": reward}


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    cfg = train_config.get_config(args.policy_config)
    if hasattr(cfg.model, "use_value_head"):
        cfg = dataclasses.replace(
            cfg,
            model=dataclasses.replace(
                cfg.model,
                use_value_head=True,
                use_keyframe_head=args.enable_keyframe_exploration,
            ),
        )
    policy = create_trained_pi0_fast_rl_policy(
        cfg,
        args.policy_path,
        exploration_config=KeyframeExplorationConfig(
            enabled=args.enable_keyframe_exploration,
            keyframe_threshold=args.keyframe_threshold,
            explore_dct_dims=args.explore_dct_dims,
        ),
    )

    env_cfg = make_env_config(
        args.env_type,
        task=args.env_task,
        ws_url=args.env_ws_url,
        observation_height=args.env_observation_height,
        observation_width=args.env_observation_width,
        action_dim=args.env_action_dim,
        max_episode_steps_model=args.env_max_episode_steps_model,
    )
    envs = make_env(env_cfg, n_envs=args.num_envs, use_async_envs=False)
    vec_env = next(iter(next(iter(envs.values())).values()))

    provider = CallableChunkRewardValueProvider(
        lambda **kwargs: _default_reward_value_fn(**kwargs, fixed_reward=args.fixed_reward)
    )
    collector = Pi0FastChunkCollector(vec_env=vec_env, policy=policy, provider=provider)
    rl_cfg = Pi0FastOnlineRLConfig(
        rollout_batch_size=args.rollout_batch_size,
        mini_batch_size=args.mini_batch_size,
        ppo_epochs=args.ppo_epochs,
        gamma=args.gamma,
        clip_eps=args.clip_eps,
        entropy_coef=args.entropy_coef,
        buffer_capacity=args.buffer_capacity,
        max_policy_lag=args.max_policy_lag,
        total_updates=args.total_updates,
        seed=args.seed,
    )
    trainer = Pi0FastOnlineTrainer(
        cfg=rl_cfg,
        policy=policy,
        collector=collector,
    )
    trainer.train()


if __name__ == "__main__":
    main()
