from typing import Tuple, List, Dict
import numpy as np
import gymnasium as gym

class SarsaFuncApprox:
    def __init__(self, 
                 env: gym.Env, 
                 dims: List[int], 
                 num_features=10, 
                 gamma=0.9, 
                 alpha=0.01, 
                 epsilon=1.0, 
                 decay_rate=0.001, 
                 min_epsilon=0.01, 
                 max_episodes=5000, 
                 max_steps_per_episode=500):
        """
        Args:
            env (gym.Env): A Gymnasium environment (Discrete state and action spaces).
            dims (List[int]): [width, height] describing length of x (columns) and y (rows) axes.
            num_features (int): Total number of features in the approximation feature vector.
            gamma (float): Discount factor for future rewards.
            alpha (float): Learning rate \alpha_t = \alpha > 0[cite: 8].
            epsilon (float): Initial exploration parameter \epsilon \in (0, 1)[cite: 8].
            decay_rate (float): Exponential decay rate applied to epsilon after each episode.
            min_epsilon (float): Floor boundary threshold for epsilon decay.
            max_episodes (int): Total number of episodes to process.
            max_steps_per_episode (int): Maximum length of a single generated episode.
        """
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.min_epsilon = min_epsilon
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        
        # dims[0] is width (x axis length), dims[1] is height (y axis length)
        self.dims = dims 
        self.num_features = num_features

        self.stats = {
            "total_rewards": [],
            "episode_lengths": []
        }

        self._parse_env(env)

    def _parse_env(self, env: gym.Env):
        if not isinstance(env.observation_space, gym.spaces.Discrete):
            raise TypeError("Observation space must be a gymnasium.spaces.Discrete space.")
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise TypeError("Action space must be a gymnasium.spaces.Discrete space.")
            
        self.env = env
        self.num_states = int(env.observation_space.n) # type: ignore
        self.num_actions = int(env.action_space.n) # type: ignore

    def _state_to_xy(self, state: int) -> Tuple[float, float]:
        """Maps a 1D state ID to continuous or discrete (x, y) coordinates."""
        # row index corresponding to y-axis, col index corresponding to x-axis
        y = float(state // self.dims[0])
        x = float(state % self.dims[0])
        return x, y
    
    def _get_features(self, state: int, action: int) -> np.ndarray:
        """
        Constructs the feature vector matching the layout requirement:
        [1, x, y, a, x^2, y^2, a^2, xy, xa, ya]
        
        All elements are normalized to fall strictly within the range [-1, 1].
        """
        raw_x, raw_y = self._state_to_xy(state)
        
        # 1. Normalize x to [-1, 1] using its maximum bounds: self.dims[0] - 1
        # Formula: 2 * (val / max_val) - 1
        max_x = float(self.dims[0] - 1)
        x = 2.0 * (raw_x / max_x) - 1.0 if max_x > 0 else 0.0
        
        # 2. Normalize y to [-1, 1] using its maximum bounds: self.dims[1] - 1
        max_y = float(self.dims[1] - 1)
        y = 2.0 * (raw_y / max_y) - 1.0 if max_y > 0 else 0.0
        
        # 3. Normalize action to [-1, 1] using its maximum bounds: self.num_actions - 1
        max_a = float(self.num_actions - 1)
        a = 2.0 * (float(action) / max_a) - 1.0 if max_a > 0 else 0.0
        
        # 4. Construct the polynomial combinations. 
        # Since x, y, a \in [-1, 1], all down-stream products will also stay in [-1, 1].
        features = np.array([
            1.0,
            x,
            y,
            a,
            x**2,
            y**2,
            a**2,
            x * y,
            x * a,
            y * a
        ], dtype=float)
        
        return features

    def _get_q_value(self, state: int, action: int, w: np.ndarray) -> float:
        """Computes q(state, a) = <features, w>"""
        features = self._get_features(state, action)
        return float(np.dot(features, w))

    def _sample_epsilon_greedy_action(self, state: int, w: np.ndarray) -> int:
        """
        Samples an action following the \pi_t(s) policy rule[cite: 8].
        Handles the edge case where all estimated action-values are identical.
        """
        q_values = [self._get_q_value(state, a, w) for a in range(self.num_actions)]
        
        if np.all(q_values == q_values[0]) or np.random.rand() < self.epsilon:
            return int(np.random.choice(self.num_actions))

        return int(np.argmax(q_values))

    def _extract_optimal_policy(self, w: np.ndarray) -> np.ndarray:
        """Extracts deterministic policy matrix mapping [s] -> best action index."""
        optimal_policy = np.zeros(self.num_states, dtype=int)
        for s in range(self.num_states):
            q_values = [self._get_q_value(s, a, w) for a in range(self.num_actions)]
            optimal_policy[s] = np.argmax(q_values)
        return optimal_policy

    def _extract_state_values(self, w: np.ndarray) -> np.ndarray:
        """Extracts state value array V(s) = max_a Q(s, a, w)."""
        state_values = np.zeros(self.num_states, dtype=float)
        for s in range(self.num_states):
            state_values[s] = max([self._get_q_value(s, a, w) for a in range(self.num_actions)])
        return state_values

    def __call__(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            optimal_policy (np.ndarray): Shape: (num_states,)
            state_values (np.ndarray): Shape: (num_states,)
        """
        # Initial parameter vector w initialized to zeros
        w = np.zeros(self.num_features, dtype=float)

        k = 0
        while k < self.max_episodes:
            total_reward, episode_length = 0.0, 0
            
            s_t, _ = self.env.reset()
            
            # 1. Generate a_0 at s_0 following \pi_0(s_0)[cite: 8]
            a_t = self._sample_epsilon_greedy_action(s_t, w)
            
            for _ in range(self.max_steps_per_episode):
                # 2. Loop until termination or step limit cap[cite: 8]
                
                # Interact with the environment to collect r_{t+1}, s_{t+1}[cite: 8]
                s_tp1, r_tp1, terminated, truncated, _ = self.env.step(a_t)
                print(f's: {s_t}, a: {a_t}, s_next: {s_tp1}, r: {r_tp1}, terminated: {terminated or truncated}, eps: {self.epsilon}')
                
                # Generate a_{t+1} following \pi_t(s_{t+1})[cite: 8]
                a_tp1 = self._sample_epsilon_greedy_action(s_tp1, w)
                
                total_reward += float(r_tp1)
                episode_length += 1
                
                # Compute targets for the objective correction step
                q_current = self._get_q_value(s_t, a_t, w)
                q_next = 0.0 if terminated else self._get_q_value(s_tp1, a_tp1, w)
                
                # Gradient with respect to w is equal to the feature vector extraction array
                grad_q = self._get_features(s_t, a_t)
                
                # Update q-value weights via the function approximation formula:
                # w_{t+1} = w_t + \alpha_t * [r_{t+1} + \gamma * q(s_{t+1}, a_{t+1}, w_t) - q(s_t, a_t, w_t)] * \nabla_w q(s_t, a_t, w_t)[cite: 8]
                td_error = float(r_tp1) + (self.gamma * q_next) - q_current
                w = w + self.alpha * td_error * grad_q
                
                if terminated or truncated:
                    break
                    
                # Transition: s_t <- s_{t+1}, a_t <- a_{t+1}[cite: 8]
                s_t, a_t = s_tp1, a_tp1

            self.stats["total_rewards"].append(total_reward)
            self.stats["episode_lengths"].append(episode_length)
            
            # Decay epsilon after processing the episode timeline
            self.epsilon = max(self.epsilon * np.exp(-self.decay_rate), self.min_epsilon)
            
            k += 1

        print(f"Sarsa Function Approximation optimization finished after {k} episodes.")
        
        optimal_policy = self._extract_optimal_policy(w)
        state_values = self._extract_state_values(w)
            
        return optimal_policy, state_values