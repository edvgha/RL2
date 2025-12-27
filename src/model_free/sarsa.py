"""
    ON-POLICY
"""
from typing import Tuple
from enum import Enum
import random
import gymnasium as gym
import numpy as np
from .base_algorithm import Algorithm

class SARSAType(Enum):
    """
    Type
    """
    SARSA = 'SARSA'
    ESARSA = 'ESARSA'


class SARSA(Algorithm):
    """
        SARSA | ESARSA
    """
    def __init__(self,
                 sarsa_type: SARSAType,
                 env: gym.Env,
                 alpha: float,
                 gamma: float,
                 epsilon: float,
                 epsilon_min: float,
                 decay_rate: float
                 ) -> None:
        super().__init__(env=env, alpha=alpha, gamma=gamma, epsilon=epsilon,
                         epsilon_min=epsilon_min, decay_rate=decay_rate)
        self.sarsa_type = sarsa_type

        self.n_actions = env.action_space.n


    def epsilon_greedy(self, state, episode: int) -> Tuple[int, float]:
        """
        Epsilon-greedy action selection
        """
        current_eps = max(self.epsilon_min, self.epsilon * np.exp(-self.decay_rate * episode))
        if random.uniform(0, 1) < current_eps:
            return self.env.action_space.sample(), current_eps
        return int(np.argmax(self.Q[state])), current_eps


    def fit(self, episodes: int):
        """
        fit
        """
        if self.sarsa_type == SARSAType.SARSA:
            self.sarsa(episodes=episodes)
            return
        self.esarsa(episodes=episodes)


    def expected_q_value(self, is_terminal: bool, next_state: int, eps: float) -> float:
        """
        E[Q(S', A')|S']
        """
        if is_terminal:
            return .0

        best_action = np.argmax(self.Q[next_state])

        non_greedy_prob = eps / self.n_actions
        greedy_prob = (1 - eps) + non_greedy_prob

        expected_value = 0
        for a in range(self.n_actions):
            if a == best_action:
                expected_value += greedy_prob * self.Q[next_state][a]
            else:
                expected_value += non_greedy_prob * self.Q[next_state][a]

        return expected_value


    def esarsa(self, episodes: int):
        """
        ε > 0, α ∈ (0, 1]
        Init Q(S, A) , Q(terminal, *) = 0

        Loop for each episode:
            Initialize S
            Choose A from S using policy derived from Q (e.g. ε - greedy)
            Loop for each step of episode:
                Take action A, observe R, S'
                Choose A' from S' using policy derived from Q (e.g. ε - greedy)
                Q(S, A) <- Q(S, A) + α [R + γ E[Q(S', A')|S'] - Q(S, A)]
                S <- S'
                A <- A'
            Until S is terminal
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

                # Q(S, A) = Q(S, A) + α[R + γE[Q(S', A')|S'] - Q(S, A)]
                td_target = float(reward) + self.gamma * self.expected_q_value((terminated or truncated), next_state, current_eps)
                td_error = td_target - self.Q[state][action]
                self.Q[state, action] += self.alpha * td_error

                state = next_state
                total_reward += float(reward)
                steps += 1

            self.history['episode_rewards'].append(total_reward)
            self.history['episode_lengths'].append(steps)
            self.history['epsilons'].append(current_eps)

        self.policy = np.argmax(self.Q, axis=1)


    def sarsa(self, episodes: int):
        """
        ε > 0, α ∈ (0, 1]
        Init Q(S, A) , Q(terminal, *) = 0

        Loop for each episode:
            Initialize S
            Choose A from S using policy derived from Q (e.g. ε - greedy)
            Loop for each step of episode:
                Take action A, observe R, S'
                Choose A' from S' using policy derived from Q (e.g. ε - greedy)
                Q(S, A) <- Q(S, A) + α [R + γ Q(S', A') - Q(S, A)]
                S <- S'
                A <- A'
            Until S is terminal
        """
        for episode in range(episodes):

            state, _ = self.env.reset()

            terminated: bool = False
            truncated: bool = False
            total_reward: float = .0
            current_eps: float = .0
            steps: int = 0

            action, current_eps = self.epsilon_greedy(state, episode)

            while not (terminated or truncated):
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                next_action, _ = self.epsilon_greedy(next_state, episode)

                # Q(S, A) = Q(S, A) + α[R + γQ(S', A') - Q(S, A)]
                td_target = reward + self.gamma * self.Q[next_state][next_action]
                td_error = td_target - self.Q[state][action]
                self.Q[state, action] += self.alpha * td_error

                state = next_state
                action = next_action
                total_reward += float(reward)
                steps += 1

            self.history['episode_rewards'].append(total_reward)
            self.history['episode_lengths'].append(steps)
            self.history['epsilons'].append(current_eps)

        self.policy = np.argmax(self.Q, axis=1)
            