"""
Semi-gradient linear approximation SARSA
"""
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


class SemiGradientSARSA:
    """
        TODO
    """
    def __init__(self, env, n_bins=(6, 12, 6, 12), alpha=0.1, gamma=0.99, epsilon=0.1):
        super().__init__()
        self.env = env
        self.n_bins = n_bins
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.policy = None

        # Linear Weights: shape (bins_product, n_actions)
        # Since we use state aggregation, W[bin_idx, action] is effectively Q(s, a)
        self.n_features = np.prod(n_bins)
        self.weights = np.zeros((self.n_features, self.env.action_space.n))

        # Define observation bounds for binning
        self.obs_low = np.array([-4.8, -3.0, -0.418, -3.5])
        self.obs_high = np.array([4.8, 3.0, 0.418, 3.5])

        self.rewards = []

    def _get_features(self, state):
        """
        State Aggregation: Maps continuous state to a single discrete bin index.
        """
        ratios = (state - self.obs_low) / (self.obs_high - self.obs_low)
        ratios = np.clip(ratios, 0, 0.9999) # Keep within bounds

        bin_indices = (ratios * np.array(self.n_bins)).astype(int)

        # Convert multi-dim index to a single flat index
        flat_idx, multiplier = 0, 1
        for i, bin_index in enumerate(bin_indices):
            flat_idx += bin_index * multiplier
            multiplier *= self.n_bins[i]
        return flat_idx

    def _get_q(self, state, action):
        """
        Linear approximation: Q(s,a) = w^T * x(s,a)
        """
        feature_idx = self._get_features(state)
        return self.weights[feature_idx, action]

    def _select_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()

        q_values = [self._get_q(state, a) for a in range(self.env.action_space.n)]
        return np.argmax(q_values)

    def fit(self, episodes: int):
        """
        TODO
        """
        for _ in range(episodes):
            state, _ = self.env.reset()
            action = self._select_action(state)

            terminated = False
            truncated = False

            episod_reward = 0

            while not (terminated or truncated):
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                episod_reward += reward

                feat_idx = self._get_features(state)
                q_curr = self.weights[feat_idx, action]

                if terminated or truncated:
                    target = reward
                else:
                    next_action = self._select_action(next_state)
                    # Semi-gradient update: uses current weight for the target
                    q_next = self._get_q(next_state, next_action)
                    target = reward + self.gamma * q_next

                # Update weights (Gradient of Linear Q wrt w is just the feature vector)
                # For state aggregation, this is simply the specific bin weight
                self.weights[feat_idx, action] += self.alpha * (target - q_curr)

                state = next_state
                if not (terminated or truncated):
                    action = next_action

            self.rewards.append(episod_reward)

        self.policy = np.argmax(self.weights, axis=1)

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    agent = SemiGradientSARSA(env)
    agent.fit(episodes=5000)

    plt.scatter(range(len(agent.rewards)), agent.rewards)
    plt.show()
