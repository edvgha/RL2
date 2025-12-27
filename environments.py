"""
Environments
"""
from typing import Tuple
import gymnasium as gym

def frozen_lake_env(is_slippery: bool = True) -> Tuple[gym.Env, gym.Env]:
    """
    Create two instances of FrozenLake-v1: one headless, one with human rendering.
    """
    # custom_map = [
    #     "SHFFFFFF",
    #     "FFHFFFFF",
    #     "FFFHFFFF",
    #     "HFFFFHFF",
    #     "FFFHFFFF",
    #     "FHHFFFHF",
    #     "FHFFHFHF",
    #     "FFFHFFFG",
    #     ]
    custom_map = [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "HFFFFHFF",
        "FFFFFFFF",
        "FHHFFFFF",
        "FHFFFFFF",
        "FFFHFFFG",
        ]
    id_name = "FrozenLake-v1"
    env = gym.make(id=id_name, render_mode=None,
                   is_slippery=is_slippery, desc=custom_map, success_rate=0.9)
    env_h = gym.make(id=id_name, render_mode="human",
                     is_slippery=is_slippery,desc=custom_map, success_rate=0.9)
    return env, env_h

def frozen_lake_d_env() -> Tuple[gym.Env, gym.Env]:
    """
    FrozenLake-v1 discrete
    """
    return frozen_lake_env(is_slippery=False)

def frozen_lake_s_env() -> Tuple[gym.Env, gym.Env]:
    """
    FrozenLake-v1 stochastic
    """
    return frozen_lake_env(is_slippery=True)

def cliff_walking_env(is_slippery: bool) -> Tuple[gym.Env, gym.Env]:
    """
    Create two instances of CliffWalking-v1: one headless, one with human rendering.
    """
    id_name = "CliffWalking-v1"
    env = gym.make(id_name, render_mode=None, is_slippery=is_slippery)
    env_h = gym.make(id_name, render_mode="human", is_slippery=is_slippery)
    return env, env_h


def cliff_walking_d_env() -> Tuple[gym.Env, gym.Env]:
    """
    CliffWalking-v1 discrete
    """
    return cliff_walking_env(is_slippery=False)

def cliff_walking_s_env() -> Tuple[gym.Env, gym.Env]:
    """
    CliffWalking-v1 stochastic
    """
    return cliff_walking_env(is_slippery=True)

def taxi_env() -> Tuple[gym.Env, gym.Env]:
    """
    Create two instances of Taxi-v3: one headless, one with human rendering.
    """
    env = gym.make(id="Taxi-v3", render_mode=None)
    env.action_space.seed(42)
    env_h = gym.make(id="Taxi-v3", render_mode="human")
    env_h.action_space.seed(42)
    return env, env_h
