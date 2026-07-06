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
# 2. Q-Actor-Critic (QAC) Solver
# =====================================================================
class QAC:
    def __init__(self, 
                 env: gym.Env, 
                 dims: List[int], 
                 num_features=10, 
                 gamma=0.9, 
                 alpha_theta=0.001,  # Actor learning rate
                 alpha_w=0.001,      # Critic learning rate
                 max_episodes=5000, 
                 max_steps_per_episode=500):
        """
        Args:
            env (gym.Env): A Gymnasium environment (Discrete state and action spaces).
            dims (List[int]): [width, height] describing grid axes bounds.
            num_features (int): Total number of features in the feature vector.
            gamma (float): Discount factor for future rewards.
            alpha_theta (float): Learning rate for Actor's PyTorch optimizer.
            alpha_w (float): Learning rate for Critic's PyTorch optimizer.
            max_episodes (int): Total number of episodes to process.
            max_steps_per_episode (int): Maximum length of a single generated episode.
        """
        self.gamma = gamma
        self.alpha_theta = alpha_theta
        self.alpha_w = alpha_w
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        
        self.dims = dims 

        self.stats = {
            "total_rewards": [],
            "episode_lengths": []
        }

        self._parse_env(env)
        
        # Initialize Actor and Critic networks
        self.actor_net = ActorNetwork(num_features)
        self.critic_net = CriticNetwork(num_features)
        
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
        """Calculates \pi(a|s, \theta) distribution for all actions."""
        features_list = [self._get_features(state, a) for a in range(self.num_actions)]
        features_tensor = torch.FloatTensor(np.array(features_list))
        
        logits = self.actor_net(features_tensor).squeeze(-1)
        action_probs = torch.softmax(logits, dim=0)
        return action_probs

    def _get_q_value_tensor(self, state: int, action: int) -> torch.Tensor:
        """Computes q(s, a, w) returning a tensor to preserve the computation graph."""
        features = self._get_features(state, action)
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
        """Extracts state value array V(s) = max_a Q(s, a, w) using the Critic Network."""
        state_values = np.zeros(self.num_states, dtype=float)
        for s in range(self.num_states):
            with torch.no_grad():
                q_vals = [self._get_q_value_tensor(s, a).item() for a in range(self.num_actions)]
                state_values[s] = max(q_vals)
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
            
            # 1. Sample initial action WITHOUT building a computation graph
            with torch.no_grad():
                probs = self._get_action_probs(s_t)
                a_t = Categorical(probs).sample().item()
            
            for _ in range(self.max_steps_per_episode):
                # 2. Observe r_{t+1}, s_{t+1}
                s_tp1, r_tp1, terminated, truncated, _ = self.env.step(a_t)
                done = terminated or truncated
                
                total_reward += float(r_tp1)
                episode_length += 1
                
                # 3. Generate a_{t+1} WITHOUT building a computation graph
                if not done:
                    with torch.no_grad():
                        probs_next = self._get_action_probs(s_tp1)
                        a_tp1 = Categorical(probs_next).sample().item()
                else:
                    a_tp1 = 0

                # We calculate the log_prob of a_t using the CURRENT, fresh network weights
                probs_current = self._get_action_probs(s_t)
                m_current = Categorical(probs_current)
                log_prob = m_current.log_prob(torch.tensor(a_t))

                # Calculate current and next Q-values
                q_current = self._get_q_value_tensor(s_t, a_t) # type: ignore
                
                if done:
                    q_next = torch.tensor([0.0])
                else:
                    # Detach next Q-value so gradients don't flow backwards across time steps
                    with torch.no_grad():
                        q_next = self._get_q_value_tensor(s_tp1, a_tp1) # type: ignore

                # --- 4. Actor Update (\theta) ---
                actor_loss = -log_prob * q_current.detach() 
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # --- 5. Critic Update (w) ---
                target_q = float(r_tp1) + self.gamma * q_next
                critic_loss = self.critic_loss_fn(q_current, target_q)
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
                
                if done:
                    break
                    
                # Transition to next step
                s_t = s_tp1
                a_t = a_tp1

            self.stats["total_rewards"].append(total_reward)
            self.stats["episode_lengths"].append(episode_length)
            k += 1

        print(f"QAC optimization finished after {k} episodes.")
        
        optimal_policy = self._extract_optimal_policy()
        state_values = self._extract_state_values()
            
        return optimal_policy, state_values
