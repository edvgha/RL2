from typing import Tuple, List, Dict
import random
import collections
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

# =====================================================================
# 1. Deep Neural Network Module: Q(s, a) function
# =====================================================================
class QNetwork(nn.Module):
    def __init__(self, input_dim: int):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# =====================================================================
# 2. Experience Replay Buffer
# =====================================================================
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)
        
    def push(self, state: int, action: int, reward: float, next_state: int, done: bool):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


# =====================================================================
# 3. Off-Policy Deep Q-Learning Solver (with Uniform Behavior Policy)
# =====================================================================
class DeepQLearningUniformBehavior:
    def __init__(self, 
                 env: gym.Env, 
                 dims: List[int], 
                 num_features=10, 
                 gamma=0.9, 
                 alpha=0.001,
                 max_episodes=5000, 
                 max_steps_per_episode=500,
                 buffer_capacity=50000,
                 batch_size=64,
                 target_update_freq=100,
                 warmup_steps=500):
        """
        Args:
            env (gym.Env): A Gymnasium environment.
            dims (List[int]): [width, height] describing grid axes bounds.
            num_features (int): Total number of features in the feature vector.
            gamma (float): Discount factor for future rewards.
            alpha (float): Learning rate for PyTorch optimizer.
            max_episodes (int): Total number of episodes to process.
            max_steps_per_episode (int): Maximum length of a single generated episode.
            buffer_capacity (int): Max storage capacity of the experience replay buffer.
            batch_size (int): Size of uniformly drawn mini-batches.
            target_update_freq (int): Interval iterations to synchronize target network.
            warmup_steps (int): Initial random steps to seed the replay buffer.
        """
        self.gamma = gamma
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        
        self.dims = dims 
        self.num_features = num_features
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.warmup_steps = warmup_steps

        self.stats = {
            "total_rewards": [],
            "episode_lengths": []
        }

        self._parse_env(env)
        
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        self.main_net = QNetwork(self.num_features)
        self.target_net = QNetwork(self.num_features)
        self.target_net.load_state_dict(self.main_net.state_dict())
        self.target_net.eval()  # Keep target network in evaluation mode
        
        self.optimizer = optim.Adam(self.main_net.parameters(), lr=alpha)
        self.loss_fn = nn.MSELoss()  # Minimize squared loss L(w) = (y^T - q)^2
        
        self.total_iterations = 0

    def _parse_env(self, env: gym.Env):
        if not isinstance(env.observation_space, gym.spaces.Discrete):
            raise TypeError("Observation space must be a gymnasium.spaces.Discrete space.")
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise TypeError("Action space must be a gymnasium.spaces.Discrete space.")
            
        self.env = env
        self.num_states = int(env.observation_space.n) # type: ignore
        self.num_actions = int(env.action_space.n) # type: ignore

    def _state_to_xy(self, state: int) -> Tuple[float, float]:
        """Maps a 1D state ID to discrete (x, y) coordinates."""
        y = float(state // self.dims[0])
        x = float(state % self.dims[0])
        return x, y
    
    def _get_features(self, state: int, action: int) -> np.ndarray:
        """
        Constructs the normalized feature vector matching the layout requirement:
        [1, x, y, a, x^2, y^2, a^2, xy, xa, ya]
        """
        raw_x, raw_y = self._state_to_xy(state)
        
        max_x = float(self.dims[0] - 1)
        x = 2.0 * (raw_x / max_x) - 1.0 if max_x > 0 else 0.0
        
        max_y = float(self.dims[1] - 1)
        y = 2.0 * (raw_y / max_y) - 1.0 if max_y > 0 else 0.0
        
        max_a = float(self.num_actions - 1)
        a = 2.0 * (float(action) / max_a) - 1.0 if max_a > 0 else 0.0
        
        features = np.array([
            1.0, x, y, a, x**2, y**2, a**2, x * y, x * a, y * a
        ], dtype=float)
        
        return features

    def _get_q_value(self, state: int, action: int, evaluate_target=False) -> float:
        """Computes q value using either the main network or the target network."""
        features = self._get_features(state, action)
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        
        net = self.target_net if evaluate_target else self.main_net
        with torch.no_grad():
            q_val = net(features_tensor).item()
        return q_val

    def _sample_behavioral_action(self) -> int:
        """
        Behavior policy \pi_b: Uniformly samples any action from the available
        discrete action set. Completely decoupled from current network weights.
        """
        return int(np.random.choice(self.num_actions))

    def _update_network(self):
        if len(self.replay_buffer) < self.batch_size:
            return
            
        # 1. Uniformly draw a mini-batch of samples from replay buffer B
        mini_batch = self.replay_buffer.sample(self.batch_size)
        
        current_features_list = []
        target_y_list = []
        
        # 2. For each sample (s, a, r, s'), calculate target value
        for state, action, reward, next_state, done in mini_batch:
            # Main network features for current state-action pair
            feat = self._get_features(state, action)
            current_features_list.append(feat)
            
            if done:
                y_target = reward
            else:
                # Calculate target value: y^T = r + gamma * max_a q(s', a, w^T)
                max_q_next = max([self._get_q_value(next_state, a, evaluate_target=True) 
                                  for a in range(self.num_actions)])
                y_target = reward + self.gamma * max_q_next
                
            target_y_list.append([y_target])
            
        # Convert data structures into PyTorch tensors
        features_tensor = torch.FloatTensor(np.array(current_features_list))
        targets_tensor = torch.FloatTensor(np.array(target_y_list))
        
        # 3. Update main network parameter w to minimize Loss(w) using the mini-batch
        self.optimizer.zero_grad()
        predictions = self.main_net(features_tensor)
        loss = self.loss_fn(predictions, targets_tensor)
        loss.backward()
        self.optimizer.step()
        
        self.total_iterations += 1
        
        # 4. Set w^T = w every C iterations
        if self.total_iterations % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.main_net.state_dict())

    def _extract_optimal_policy(self) -> np.ndarray:
        optimal_policy = np.zeros(self.num_states, dtype=int)
        for s in range(self.num_states):
            # Target evaluation uses target network values
            q_values = [self._get_q_value(s, a, evaluate_target=True) for a in range(self.num_actions)]
            optimal_policy[s] = np.argmax(q_values)
        return optimal_policy

    def _extract_state_values(self) -> np.ndarray:
        state_values = np.zeros(self.num_states, dtype=float)
        for s in range(self.num_states):
            state_values[s] = max([self._get_q_value(s, a, evaluate_target=True) for a in range(self.num_actions)])
        return state_values

    def __call__(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            optimal_policy (np.ndarray): Shape: (num_states,)
            state_values (np.ndarray): Shape: (num_states,)
        """
        # --- Warmup Stage: Populate buffer with behavioral uniform samples ---
        s_t, _ = self.env.reset()
        for _ in range(self.warmup_steps):
            a_t = self._sample_behavioral_action()
            s_tp1, r_tp1, terminated, truncated, _ = self.env.step(a_t)
            done = terminated or truncated
            self.replay_buffer.push(s_t, a_t, float(r_tp1), s_tp1, done)
            s_t, _ = self.env.reset() if done else (s_tp1, None)

        # --- Optimization Stage ---
        k = 0
        while k < self.max_episodes:
            total_reward, episode_length = 0.0, 0
            s_t, _ = self.env.reset()
            
            for _ in range(self.max_steps_per_episode):
                # Sample action uniformly following behavior policy \pi_b
                a_t = self._sample_behavioral_action()
                
                # Step environment and store sequence sample in replay buffer B
                s_tp1, r_tp1, terminated, truncated, _ = self.env.step(a_t)
                done = terminated or truncated
                self.replay_buffer.push(s_t, a_t, float(r_tp1), s_tp1, done)
                
                total_reward += float(r_tp1)
                episode_length += 1
                
                # Perform an optimization iteration step on the drawn mini-batches
                self._update_network()
                
                if done:
                    break
                    
                s_t = s_tp1

            self.stats["total_rewards"].append(total_reward)
            self.stats["episode_lengths"].append(episode_length)
            if k % 100 == 0:
                print(f"{k} episodes processed ...")
            k += 1

        print(f"Deep Q-Learning (Off-Policy Uniform Behavior) finished after {k} episodes.")
        
        optimal_policy = self._extract_optimal_policy()
        state_values = self._extract_state_values()
            
        return optimal_policy, state_values