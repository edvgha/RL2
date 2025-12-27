"""
Main for model free control problem
"""
# pylint: skip-file
import warnings
warnings.filterwarnings("ignore")
from typing import (
    Callable,
    Tuple,
    Optional
)
import numpy as np
from enum import Enum
import gymnasium as gym
import matplotlib.pyplot as plt
from src.model_free.base_algorithm import Algorithm
from src.model_free.q_learning import QLearning, QLEARNINGType
from src.model_free.sarsa import SARSA, SARSAType
from environments import (
    frozen_lake_d_env, 
    frozen_lake_s_env,
    cliff_walking_d_env,
    cliff_walking_s_env,
    taxi_env
)


class Control(Enum):
    """
    Algorithms
    """
    QLEARNING = 'Q-Learning'
    DOUBLEQLEARNING = 'Double-Q-Learning'
    SARSA = 'SARSA'
    ESARSA = 'ESARSA'

def plot_learning_curves(history: dict, alg_name: str, env_name: str):
    """
    Draw plot
    """
    _, ax1 = plt.subplots(figsize=(10, 5))

    # Plot Rewards
    color = 'tab:blue'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward', color=color)
    ax1.plot(history['episode_rewards'], color=color, alpha=0.3)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Epsilon', color=color)
    ax2.plot(history['epsilons'], color=color, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'{alg_name} Learning Progress ({env_name})')
    plt.show(block=False)
    plt.pause(5)
    plt.close()


def evaluate(env_func: Callable[[], Tuple[gym.Env, gym.Env]], control: Control, episodes: int, decay_rate: float):
    """
    :param control: Learning algorithm
    :param env_name: Environment
    """
    env, env_h = env_func()
    print(f"\n🔵 Env: {env.spec.id} | Alg: {control.value}")

    policy = np.zeros([], dtype=np.int16)

    algorithm: Algorithm = None
    if Control.QLEARNING == control:
        algorithm = QLearning(
                        qlearning_type=QLEARNINGType.QLEARNING,
                        env=env,
                        alpha=0.1, gamma=0.99, epsilon=1.0,
                        epsilon_min=0.1, decay_rate=decay_rate)
    elif Control.DOUBLEQLEARNING == control:
        algorithm = QLearning(
                        qlearning_type=QLEARNINGType.DOUBLEQLEARNING,
                        env=env,
                        alpha=0.1, gamma=0.99, epsilon=1.0,
                        epsilon_min=0.1, decay_rate=decay_rate)
    elif Control.SARSA == control:
        algorithm = SARSA(
                        sarsa_type=SARSAType.SARSA,
                        env=env,
                        alpha=0.1, gamma=0.99, epsilon=1.0,
                        epsilon_min=0.1, decay_rate=decay_rate)
    
    elif Control.ESARSA == control:
        algorithm = SARSA(
                        sarsa_type=SARSAType.ESARSA,
                        env=env,
                        alpha=0.1, gamma=0.99, epsilon=1.0,
                        epsilon_min=0.1, decay_rate=decay_rate)
    
    algorithm.fit(episodes=episodes)
    policy = algorithm.policy
    env.close()

    state, _ = env_h.reset()
    step: int = 0
    total_reward: float = 0.0

    while True:
        action = policy[state]

        next_state, reward, terminated, truncated, _ = env_h.step(action)

        state = next_state
        step += 1
        total_reward += float(reward)

        print(f"⚪ {step}: Action={action}, Reward={reward:.1f}, State={state}")

        if terminated or truncated:
            break

    print(f"🟢 Episode finished! Total reward: {total_reward:.1f}, Steps: {step}")
    env_h.close()


if __name__ == '__main__':
    # evaluate(env_func=cliff_walking_d_env, control=Control.QLEARNING, episodes=500, decay_rate=0.005)
    # evaluate(env_func=cliff_walking_d_env, control=Control.DOUBLEQLEARNING, episodes=500, decay_rate=0.005)
    # evaluate(env_func=cliff_walking_d_env, control=Control.SARSA, episodes=500, decay_rate=0.005)
    # evaluate(env_func=cliff_walking_d_env, control=Control.ESARSA, episodes=500, decay_rate=0.005)

    # evaluate(env_func=cliff_walking_s_env, control=Control.QLEARNING, episodes=500, decay_rate=0.005)
    # evaluate(env_func=cliff_walking_s_env, control=Control.DOUBLEQLEARNING, episodes=500, decay_rate=0.005)
    # evaluate(env_func=cliff_walking_s_env, control=Control.SARSA, episodes=500, decay_rate=0.005)
    # evaluate(env_func=cliff_walking_s_env, control=Control.ESARSA, episodes=500, decay_rate=0.005)

    # evaluate(env_func=frozen_lake_d_env, control=Control.QLEARNING, episodes=1000, decay_rate=0.0001)
    # evaluate(env_func=frozen_lake_d_env, control=Control.DOUBLEQLEARNING, episodes=2000, decay_rate=0.0001)
    # evaluate(env_func=frozen_lake_d_env, control=Control.SARSA, episodes=3000, decay_rate=0.0001)
    # evaluate(env_func=frozen_lake_d_env, control=Control.ESARSA, episodes=3000, decay_rate=0.0001)

    evaluate(env_func=frozen_lake_s_env, control=Control.QLEARNING, episodes=3000, decay_rate=0.0001)
    evaluate(env_func=frozen_lake_s_env, control=Control.DOUBLEQLEARNING, episodes=3000, decay_rate=0.0001)
    evaluate(env_func=frozen_lake_s_env, control=Control.SARSA, episodes=4000, decay_rate=0.0001)
    evaluate(env_func=frozen_lake_s_env, control=Control.ESARSA, episodes=4000, decay_rate=0.0001)

    # evaluate(env_func=taxi_env, control=Control.QLEARNING, episodes=3000, decay_rate=0.0001)
    # evaluate(env_func=taxi_env, control=Control.DOUBLEQLEARNING, episodes=3000, decay_rate=0.0001)
    # evaluate(env_func=taxi_env, control=Control.SARSA, episodes=8000, decay_rate=0.0001)
    # evaluate(env_func=taxi_env, control=Control.ESARSA, episodes=8000, decay_rate=0.0001)