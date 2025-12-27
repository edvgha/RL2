"""
    OFF-POLICY
"""
from typing import Tuple
from enum import Enum
import random
import gymnasium as gym
import numpy as np
from .base_algorithm import Algorithm


class QLEARNINGType(Enum):
    """
    Type
    """
    QLEARNING = 'Q-LEARNING'
    DOUBLEQLEARNING = 'DOUBLE-Q-LEARNING'


class QLearning(Algorithm):
    """
    ε > 0, α ∈ (0, 1]
    Init Q(S, A) , Q(terminal, *) = 0

    Loop for each episode:
        Initialize S
        Choose A from S using policy derived from Q (e.g. ε - greedy)
        Loop for each step of episode:
            Take action A, observe R, S'
            Choose A' from S' using policy derived from Q (e.g. ε - greedy)
            Q(S, A) <- Q(S, A) + α [R + γ max_a{Q(S', a)} - Q(S, A)]
            S <- S'
            A <- A'
        Until S is terminal
    """
    def __init__(self,
                 qlearning_type: QLEARNINGType,
                 env: gym.Env,
                 alpha: float,
                 gamma: float,
                 epsilon: float,
                 epsilon_min: float,
                 decay_rate: float
                 ) -> None:
        super().__init__(env=env, alpha=alpha, gamma=gamma, epsilon=epsilon,
                         epsilon_min=epsilon_min, decay_rate=decay_rate)
        self.qlearning_type = qlearning_type

        if self.qlearning_type == QLEARNINGType.DOUBLEQLEARNING:
            self.DoubleQ = np.zeros([env.observation_space.n, env.action_space.n])


    def epsilon_greedy(self, state, episode: int) -> Tuple[int, float]:
        """
        Epsilon-greedy action selection
        """
        current_eps = max(self.epsilon_min, self.epsilon * np.exp(-self.decay_rate * episode))
        if random.uniform(0, 1) < current_eps:
            return self.env.action_space.sample(), current_eps

        if self.qlearning_type == QLEARNINGType.DOUBLEQLEARNING:
            combined_q = self.Q[state] + self.DoubleQ[state]
            max_val = np.max(combined_q)
            actions_with_max_val = np.where(combined_q == max_val)[0]
            action = np.random.choice(actions_with_max_val)
            return action, current_eps

        return int(np.argmax(self.Q[state])), current_eps


    def fit(self, episodes: int):
        """
        Main horse
        """
        for episode in range(episodes):

            state, _ = self.env.reset()

            terminated: bool = False
            truncated: bool = False
            total_reward: float = .0
            current_eps: float = .0
            steps: int = 0

            while not (terminated or truncated):
                action, current_eps = self.epsilon_greedy(state, episode)
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                if self.qlearning_type == QLEARNINGType.DOUBLEQLEARNING:
                    # Q1(S, A) = Q1(S, A) + α[R + γQ2(S', argmax Q1(S',a)) - Q1(S, A)]
                    # Q2(S, A) = Q2(S, A) + α[R + γQ1(S', argmax Q2(S',a)) - Q2(S, A)]
                    if random.uniform(0, 1) < 0.5:
                        best_next_action = np.argmax(self.Q[next_state])
                        td_target = reward + self.gamma * self.DoubleQ[next_state][best_next_action]
                        td_error = td_target - self.Q[state, action]
                        self.Q[state, action] += self.alpha * td_error
                    else:
                        best_next_action = np.argmax(self.DoubleQ[next_state])
                        td_target = reward + self.gamma * self.Q[next_state][best_next_action]
                        td_error = td_target - self.DoubleQ[state, action]
                        self.DoubleQ[state, action] += self.alpha * td_error
                else:
                    # Q(S, A) = Q(S, A) + α[R + γmaxQ(S', a) - Q(S, A)]
                    best_next_action = np.argmax(self.Q[next_state])
                    td_target = reward + self.gamma * self.Q[next_state][best_next_action]
                    td_error = td_target - self.Q[state][action]
                    self.Q[state, action] += self.alpha * td_error

                state = next_state
                total_reward += float(reward)
                steps += 1

            self.history['episode_rewards'].append(total_reward)
            self.history['episode_lengths'].append(steps)
            self.history['epsilons'].append(current_eps)

        if self.qlearning_type == QLEARNINGType.DOUBLEQLEARNING:
            self.policy = np.argmax(self.Q + self.DoubleQ, axis=1)
        else:
            self.policy = np.argmax(self.Q, axis=1)
