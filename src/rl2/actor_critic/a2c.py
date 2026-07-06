from typing import Tuple, List, Dict
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# =====================================================================
# 1. Deep Neural Network Modules
# =====================================================================
class ActorNetwork(nn.Module):
    def __init__(self, input_dim: int):
        super(ActorNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class CriticNetwork(nn.Module):
    def __init__(self, input_dim: int):
        super(CriticNetwork, self).__init__()
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
# 2. Advantage Actor-Critic (A2C) Solver
# =====================================================================
class A2C:
    def __init__(self, 
                 env: gym.Env, 
                 dims: List[int],
                 num_actor_features=10,
                 num_critic_features=6,
                 gamma=0.9, 
                 alpha_theta=0.001,
                 alpha_w=0.001,
                 max_episodes=5000, 
                 max_steps_per_episode=500, 
                 use_one_hot_features=False):
        """
        Args:
            env (gym.Env): A Gymnasium environment (Discrete state and action spaces).
            dims (List[int]): [width, height] describing grid axes bounds.
            num_actor_features (int): Number of action network features.
            num_critic_features (int): Number of critic network features.
            gamma (float): Discount factor for future rewards.
            alpha_theta (float): Learning rate for Actor's PyTorch optimizer.
            alpha_w (float): Learning rate for Critic's PyTorch optimizer.
            max_episodes (int): Total number of episodes to process.
            max_steps_per_episode (int): Maximum length of a single generated episode.
            use_one_hot_features (bool): if True use one-hot encoding for actor/critic input
        """
        self.gamma = gamma
        self.alpha_theta = alpha_theta
        self.alpha_w = alpha_w
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.use_one_hot_features = use_one_hot_features
        
        self.dims = dims 

        self.stats = {
            "total_rewards": [],
            "episode_lengths": []
        }

        self._parse_env(env)

        if use_one_hot_features:
            num_actor_features = self.num_states + self.num_actions
            num_critic_features = self.num_states
        
        # Initialize Actor (\theta) and Critic (w) networks
        self.actor_net = ActorNetwork(num_actor_features)
        self.critic_net = CriticNetwork(num_critic_features)
        
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=self.alpha_theta)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=self.alpha_w)
        self.critic_loss_fn = nn.MSELoss()

    def _parse_env(self, env: gym.Env):
        if not isinstance(env.observation_space, gym.spaces.Discrete):
            raise TypeError("Observation space must be a gymnasium.spaces.Discrete space.")
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise TypeError("Action space must be a gymnasium.spaces.Discrete space.")
            
        self.env = env
        self.num_states = int(env.observation_space.n) # type: ignore
        self.num_actions = int(env.action_space.n) # type: ignore

    def _get_actor_features(self, state: int, action: int) -> np.ndarray:
        if self.use_one_hot_features:
            return self._get_actor_features_one_hot(state, action)
        return self._get_actor_features_quadratic(state, action)

    def _get_critic_features(self, state: int) -> np.ndarray:
        if self.use_one_hot_features:
            return self._get_critic_features_one_hot(state)
        return self._get_critic_features_quadratic(state)

    def _get_actor_features_one_hot(self, state: int, action: int) -> np.ndarray:
        """
        Constructs a One-Hot encoded feature vector for the (state, action) pair.
        Outputs a concatenation of [State Vector] + [Action Vector]
        """
        state_vec = np.zeros(self.num_states, dtype=float)
        state_vec[state] = 1.0
        
        action_vec = np.zeros(self.num_actions, dtype=float)
        action_vec[action] = 1.0
        
        return np.concatenate([state_vec, action_vec])

    def _get_critic_features_one_hot(self, state: int) -> np.ndarray:
        """
        Constructs a pure One-Hot encoded state feature vector for v(s, w).
        """
        state_vec = np.zeros(self.num_states, dtype=float)
        state_vec[state] = 1.0
        
        return state_vec

    def _state_to_xy(self, state: int) -> Tuple[float, float]:
        """Maps a 1D state ID to discrete (x, y) coordinates."""
        y = float(state // self.dims[0])
        x = float(state % self.dims[0])
        return x, y
    
    def _get_actor_features_quadratic(self, state: int, action: int) -> np.ndarray:
        """Constructs the feature vector for (s, a) preferences: [1, x, y, a, x^2, y^2, a^2, xy, xa, ya]"""
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

    def _get_critic_features_quadratic(self, state: int) -> np.ndarray:
        """Constructs the pure state feature vector for v(s, w): [1, x, y, x^2, y^2, xy]"""
        raw_x, raw_y = self._state_to_xy(state)
        
        max_x = float(self.dims[0] - 1)
        x = 2.0 * (raw_x / max_x) - 1.0 if max_x > 0 else 0.0
        
        max_y = float(self.dims[1] - 1)
        y = 2.0 * (raw_y / max_y) - 1.0 if max_y > 0 else 0.0
        
        features = np.array([
            1.0, x, y, x**2, y**2, x * y
        ], dtype=float)
        
        return features

    def _get_action_probs(self, state: int) -> torch.Tensor:
        """Calculates \pi(a|s, \theta) distribution for all actions."""
        features_list = [self._get_actor_features(state, a) for a in range(self.num_actions)]
        features_tensor = torch.FloatTensor(np.array(features_list))
        
        logits = self.actor_net(features_tensor).squeeze(-1)
        action_probs = torch.softmax(logits, dim=0)
        return action_probs

    def _get_state_value_tensor(self, state: int) -> torch.Tensor:
        """Computes v(s, w) returning a tensor to preserve the computation graph."""
        features = self._get_critic_features(state)
        features_tensor = torch.FloatTensor(features)
        return self.critic_net(features_tensor)

    def _extract_optimal_policy(self) -> np.ndarray:
        """Extracts deterministic policy matrix mapping [s] -> highest probability action index."""
        optimal_policy = np.zeros(self.num_states, dtype=int)
        for s in range(self.num_states):
            with torch.no_grad():
                probs = self._get_action_probs(s)
                optimal_policy[s] = torch.argmax(probs).item()
        return optimal_policy

    def _extract_state_values(self) -> np.ndarray:
        """Extracts state value array directly using the Critic Network v(s, w)."""
        state_values = np.zeros(self.num_states, dtype=float)
        for s in range(self.num_states):
            with torch.no_grad():
                state_values[s] = self._get_state_value_tensor(s).item()
        return state_values

    def __call__(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            optimal_policy (np.ndarray): Shape: (num_states,)
            state_values (np.ndarray): Shape: (num_states,)
        """
        k = 0
        while k < self.max_episodes:
            total_reward, episode_length = 0.0, 0
            
            s_t, _ = self.env.reset()
            
            for _ in range(self.max_steps_per_episode):
                # 1. Sample action WITHOUT building a computation graph to prevent in-place errors
                with torch.no_grad():
                    probs = self._get_action_probs(s_t)
                    a_t = Categorical(probs).sample().item()
                
                # Observe r_{t+1}, s_{t+1}
                s_tp1, r_tp1, terminated, truncated, _ = self.env.step(a_t)
                done = terminated or truncated
                
                total_reward += float(r_tp1)
                episode_length += 1
                
                # We calculate the log_prob of a_t using the fresh Actor network weights
                probs_current = self._get_action_probs(s_t)
                m_current = Categorical(probs_current)
                log_prob = m_current.log_prob(torch.tensor(a_t))

                # Calculate current and next V-values using the Critic Network
                v_current = self._get_state_value_tensor(s_t)
                
                if done:
                    v_next = torch.tensor([0.0])
                else:
                    # Detach next V-value so gradients don't flow backwards across time steps
                    with torch.no_grad():
                        v_next = self._get_state_value_tensor(s_tp1)

                # 2. Advantage (TD error): \delta_t = r_{t+1} + \gamma v(s_{t+1}, w_t) - v(s_t, w_t)
                target_v = float(r_tp1) + self.gamma * v_next
                
                # Delta must be detached so the Actor doesn't accidentally train the Critic
                delta = target_v - v_current.detach()

                # --- 3. Actor Update (\theta) ---
                # Objective: Maximize \delta_t * \ln \pi(a_t|s_t, \theta_t)
                actor_loss = -log_prob * delta 
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # --- 4. Critic Update (w) ---
                # Objective: Minimize TD Error [r_{t+1} + \gamma * v_{t+1} - v_t]^2
                critic_loss = self.critic_loss_fn(v_current, target_v)
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
                
                if done:
                    break
                    
                # Transition to next step
                s_t = s_tp1

            self.stats["total_rewards"].append(total_reward)
            self.stats["episode_lengths"].append(episode_length)
            k += 1

        print(f"A2C optimization finished after {k} episodes | Total R: {sum(self.stats['total_rewards'])}")
        
        optimal_policy = self._extract_optimal_policy()
        state_values = self._extract_state_values()
            
        return optimal_policy, state_values