from typing import Tuple, List, Dict
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# =====================================================================
# 1. Deep Neural Network Module (Preference Network)
# =====================================================================
class PreferenceNetwork(nn.Module):
    def __init__(self, input_dim: int):
        super(PreferenceNetwork, self).__init__()

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
# 2. REINFORCE Solver (Monte Carlo Policy Gradient)
# =====================================================================
class REINFORCE:
    def __init__(self, 
                 env: gym.Env, 
                 dims: List[int], 
                 num_features=10, 
                 gamma=0.9, 
                 alpha=0.001,
                 max_episodes=5000, 
                 max_steps_per_episode=500):
        """
        Args:
            env (gym.Env): A Gymnasium environment (Discrete state and action spaces).
            dims (List[int]): [width, height] describing grid axes bounds.
            num_features (int): Total number of features in the feature vector.
            gamma (float): Discount factor for future rewards.
            alpha (float): Learning rate for PyTorch optimizer.
            max_episodes (int): Total number of episodes to process.
            max_steps_per_episode (int): Maximum length of a single generated episode.
        """
        self.gamma = gamma
        self.alpha = alpha
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
        
        # Initialize the Policy Network parameter \theta
        self.policy_net = PreferenceNetwork(self.num_features)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.alpha)

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

    def _get_action_probs(self, state: int) -> torch.Tensor:
        """
        Calculates \pi(a|s, \theta) by extracting features for ALL actions in the given state,
        passing them through the network, and applying a Softmax distribution over the logits.
        """
        features_list = [self._get_features(state, a) for a in range(self.num_actions)]
        features_tensor = torch.FloatTensor(np.array(features_list))
        
        # Get logits and squeeze output to shape (num_actions,)
        logits = self.policy_net(features_tensor).squeeze(-1)
        
        # Convert logits to a probability distribution
        action_probs = torch.softmax(logits, dim=0)
        return action_probs

    def _extract_optimal_policy(self) -> np.ndarray:
        """Extracts deterministic policy matrix mapping [s] -> highest probability action index."""
        optimal_policy = np.zeros(self.num_states, dtype=int)
        for s in range(self.num_states):
            with torch.no_grad():
                probs = self._get_action_probs(s)
                optimal_policy[s] = torch.argmax(probs).item()
        return optimal_policy

    def __call__(self) -> np.ndarray:
        """
        Returns:
            optimal_policy (np.ndarray): Shape: (num_states,)
            state_values (np.ndarray): Shape: (num_states,) array of zeros (REINFORCE does not track state values)
        """
        k = 0
        while k < self.max_episodes:
            saved_log_probs = []
            rewards = []
            
            s_t, _ = self.env.reset()
            
            # 1. Generate an episode following \pi(\theta)
            for _ in range(self.max_steps_per_episode):
                probs = self._get_action_probs(s_t)
                
                m = Categorical(probs)
                action = m.sample()
                
                s_tp1, r_tp1, terminated, truncated, _ = self.env.step(action.item())
                done = terminated or truncated
                
                saved_log_probs.append(m.log_prob(action))
                rewards.append(float(r_tp1))
                
                if done:
                    break
                    
                s_t = s_tp1

            self.stats["total_rewards"].append(sum(rewards))
            self.stats["episode_lengths"].append(len(rewards))
            
            # 2. Value Estimation and Policy Update (Stochastic version)
            returns = []
            G = 0.0
            
            # First, calculate all q_t(s_t, a_t) values backward
            for r in reversed(rewards):
                G = r + self.gamma * G
                returns.insert(0, G)
                
            # Second, perform the stochastic parameter update for each step t
            for log_prob, G_t in zip(saved_log_probs, returns):
                self.optimizer.zero_grad()
                
                # Objective loss for this specific step t: -\ln \pi(a_t|s_t) * q_t
                step_loss = -log_prob * G_t
                
                # Backpropagate and update weights immediately for this step
                step_loss.backward()
                self.optimizer.step()
            
            k += 1

        print(f"REINFORCE (Stochastic) optimization finished after {k} episodes.")
        
        optimal_policy = self._extract_optimal_policy()
            
        return optimal_policy