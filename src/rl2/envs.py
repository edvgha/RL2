"""Gymnasium environment factories.

Gym environments are Python/NumPy-backed and not JAX-pure, so they stay
imperative. JAX touches the state arrays the agents own, not the env itself.
"""
from __future__ import annotations

from typing import Tuple

import gymnasium as gym

EnvPair = Tuple[gym.Env, gym.Env]

_FROZEN_LAKE_MAP = [
    "SFFFFFFF",
    "FFFFFFFF",
    "FFFHFFFF",
    "HFFFFHFF",
    "FFFFFFFF",
    "FHHFFFFF",
    "FHFFFFFH",
    "FFFHFFFG",
]


def frozen_lake_env(is_slippery: bool = True) -> EnvPair:
    """Two FrozenLake-v1 instances: headless for training, human-rendered for rollout."""
    kwargs = dict(is_slippery=is_slippery, desc=_FROZEN_LAKE_MAP, success_rate=0.9)
    return (
        gym.make("FrozenLake-v1", render_mode=None, **kwargs), # type: ignore
        gym.make("FrozenLake-v1", render_mode="human", **kwargs), # type: ignore
    )


def frozen_lake_d_env() -> EnvPair:
    return frozen_lake_env(is_slippery=False)


def frozen_lake_s_env() -> EnvPair:
    return frozen_lake_env(is_slippery=True)


def cliff_walking_env(is_slippery: bool) -> EnvPair:
    return (
        gym.make("CliffWalking-v1", render_mode=None, is_slippery=is_slippery),
        gym.make("CliffWalking-v1", render_mode="human", is_slippery=is_slippery),
    )


def cliff_walking_d_env() -> EnvPair:
    return cliff_walking_env(is_slippery=False)


def cliff_walking_s_env() -> EnvPair:
    return cliff_walking_env(is_slippery=True)


def taxi_env() -> EnvPair:
    env = gym.make("Taxi-v4", render_mode=None)
    env_h = gym.make("Taxi-v4", render_mode="human")
    env.action_space.seed(42)
    env_h.action_space.seed(42)
    return env, env_h
