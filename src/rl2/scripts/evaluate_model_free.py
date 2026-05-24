"""Train tabular model-free agents (Q / Double-Q / SARSA / Expected-SARSA) and roll them out."""
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
from rl2.model_free.q_learning import QLearningType, fit_q_learning
from rl2.model_free.sarsa import SARSAType, fit_sarsa

warnings.filterwarnings("ignore")


class Control(Enum):
    QLEARNING = "Q-Learning"
    DOUBLEQLEARNING = "Double-Q-Learning"
    SARSA = "SARSA"
    ESARSA = "Expected-SARSA"


def plot_learning_curves(history: dict, alg_name: str, env_name: str) -> None:
    _, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward", color="tab:blue")
    ax1.plot(history["episode_rewards"], color="tab:blue", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Epsilon", color="tab:orange")
    ax2.plot(history["epsilons"], color="tab:orange", linestyle="--")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    plt.title(f"{alg_name} Learning Progress ({env_name})")
    plt.show(block=False)
    plt.pause(5)
    plt.close()


def _train(env, control: Control, episodes: int, decay_rate: float, seed: int):
    common = dict(
        episodes=episodes,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.1,
        decay_rate=decay_rate,
        seed=seed,
    )
    if control == Control.QLEARNING:
        return fit_q_learning(env, QLearningType.QLEARNING, **common)
    if control == Control.DOUBLEQLEARNING:
        return fit_q_learning(env, QLearningType.DOUBLEQLEARNING, **common)
    if control == Control.SARSA:
        return fit_sarsa(env, SARSAType.SARSA, **common)
    if control == Control.ESARSA:
        return fit_sarsa(env, SARSAType.ESARSA, **common)
    raise ValueError(f"unknown control: {control}")


def evaluate(
    env_func: Callable[[], EnvPair],
    control: Control,
    episodes: int,
    decay_rate: float,
    seed: int = 0,
) -> None:
    env, env_h = env_func()
    print(f"\n[*] Env: {env.spec.id} | Alg: {control.value}")

    _, policy, history = _train(env, control, episodes, decay_rate, seed)
    env.close()
    policy_np = np.asarray(policy)

    state, _ = env_h.reset()
    total_reward = 0.0
    steps = 0
    while True:
        action = int(policy_np[state])
        state, reward, terminated, truncated, _ = env_h.step(action)
        total_reward += float(reward)
        steps += 1
        print(f"  step {steps}: action={action} reward={reward:.1f} state={state}")
        if terminated or truncated:
            break

    print(f"[+] Episode finished | total_reward={total_reward:.1f} steps={steps}")
    env_h.close()


def main() -> None:
    evaluate(frozen_lake_s_env, Control.QLEARNING, episodes=3000, decay_rate=0.0001)
    evaluate(frozen_lake_s_env, Control.DOUBLEQLEARNING, episodes=3000, decay_rate=0.0001)
    evaluate(frozen_lake_s_env, Control.SARSA, episodes=4000, decay_rate=0.0001)
    evaluate(frozen_lake_s_env, Control.ESARSA, episodes=4000, decay_rate=0.0001)


if __name__ == "__main__":
    main()


__all__ = [
    "Control",
    "evaluate",
    "plot_learning_curves",
    "cliff_walking_d_env",
    "cliff_walking_s_env",
    "frozen_lake_d_env",
    "frozen_lake_s_env",
    "taxi_env",
]
