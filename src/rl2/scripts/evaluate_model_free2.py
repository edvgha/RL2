from __future__ import annotations

import warnings
from enum import Enum
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from rl2.envs import (
    EnvPair,
    cliff_walking_d_env,
    cliff_walking_s_env,
    frozen_lake_d_env,
    frozen_lake_s_env,
    taxi_env,
)
from rl2.model_free.sarsa2 import Sarsa
from rl2.model_free.q_learning2 import QLearning

warnings.filterwarnings("ignore")


class Control(Enum):
    QLEARNING = "Q-Learning"
    SARSA = "SARSA"
    ESARSA = "Expected-SARSA"


def plot_metrics(stats: dict, alg_name: str, env_name: str) -> None:
    """
    Plots total rewards and episode lengths on separate figures (subplots) 
    contained within the same layout window frame.
    """
    # Create 2 distinct figures stacked vertically in 1 window canvas frame
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)
    
    # 1. First Subplot Figure: Total Rewards
    ax1.plot(stats["total_rewards"], color="tab:blue", alpha=0.6)
    ax1.set_ylabel("Total Reward")
    ax1.set_title(f"{alg_name} Training Performance ({env_name})")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # 2. Second Subplot Figure: Episode Lengths
    ax2.plot(stats["episode_lengths"], color="tab:red", alpha=0.6)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Episode Length (Steps)")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(5)
    plt.close()


def evaluate(
    env_func: Callable[[], EnvPair],
    control: Control,
    episodes: int
) -> None:
    env, env_h = env_func()
    env_id = getattr(env.spec, "id", "UnknownEnv")
    print(f"\n[*] Env: {env_id} | Alg: {control.value}") # type: ignore

    optimal_policy, stats = np.array([]), {}
    if control == Control.SARSA:
        sarsa_solver = Sarsa(env, max_episodes=episodes, use_esarsa=False)
        optimal_policy = sarsa_solver()
        stats = sarsa_solver.stats
    elif control == Control.ESARSA:
        sarsa_solver = Sarsa(env, max_episodes=episodes, use_esarsa=True)
        optimal_policy = sarsa_solver()
        stats = sarsa_solver.stats
    elif control == Control.QLEARNING:
        q_learning_solver = QLearning(env, max_episodes=episodes)
        optimal_policy = q_learning_solver()
        stats = q_learning_solver.stats
    env.close()

    state, _ = env_h.reset()
    total_reward, steps = 0.0, 0
    while True:
        action = int(optimal_policy[state])
        state, reward, terminated, truncated, _ = env_h.step(action)
        total_reward += float(reward)
        steps += 1
        print(f"  step {steps}: action={action} reward={reward:.1f} state={state}")
        if terminated or truncated:
            break

    print(f"[+] Episode finished | total_reward={total_reward:.1f} steps={steps}")
    env_h.close()

    plot_metrics(stats, control.value, env_id)


def main() -> None:
    evaluate(frozen_lake_s_env, Control.SARSA, episodes=3000)
    evaluate(frozen_lake_s_env, Control.ESARSA, episodes=3000)
    evaluate(frozen_lake_s_env, Control.QLEARNING, episodes=1500)


if __name__ == "__main__":
    main()

