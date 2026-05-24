"""Train semi-gradient linear SARSA on CartPole-v1 and plot the reward curve."""
from __future__ import annotations

import gymnasium as gym
import matplotlib.pyplot as plt

from rl2.model_free.semi_gradient_sarsa import fit_semi_gradient_sarsa


def main() -> None:
    env = gym.make("CartPole-v1")
    _, _, history = fit_semi_gradient_sarsa(env, episodes=5000)
    env.close()

    rewards = history["episode_rewards"]
    plt.scatter(range(len(rewards)), rewards, s=5, alpha=0.4)
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("Semi-gradient linear SARSA on CartPole-v1")
    plt.show()


if __name__ == "__main__":
    main()
