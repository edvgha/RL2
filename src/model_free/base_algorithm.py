"""
Base 
"""
from abc import ABC, abstractmethod
import numpy as np
import gymnasium as gym


class Algorithm(ABC):
    """
    Abstract base class
    """
    def __init__(self,
                 env: gym.Env | None = None,
                 alpha: float | None = None,
                 gamma: float | None = None,
                 epsilon: float | None = None,
                 epsilon_min: float | None = None,
                 decay_rate: float | None = None,
                 ) -> None:
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.decay_rate = decay_rate

        self.history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'epsilons': []
        }

        self.Q = np.zeros([env.observation_space.n, env.action_space.n])
        self.policy = np.zeros([env.observation_space.n])

    @abstractmethod
    def fit(self, episodes: int):
        """Derived classes must implement the learning loop (SARSA, Q-Learning, etc.)"""
