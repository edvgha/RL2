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
# 2. Off-Policy A2C Solver (One-Hot Encoded + Uniform Behavior)
# =====================================================================
class OffPolicyA2C:
    def __init__(self, 
                 env: gym.Env, 
                 dims: List[int],
                 gamma=0.9, 
                 alpha_theta=0.001,  
                 alpha_w=0.001,      
                 max_episodes=5000, 
                 max_steps_per_episode=500):
        """
        Args:
            env (gym.Env): A Gymnasium environment (Discrete state and action spaces).
            dims (List[int]): [width, height] describing grid axes bounds.
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
        
        # Determine feature vector sizes based on one-hot encoding lengths
        self.num_critic_features = self.num_states
        self.num_actor_features = self.num_states + self.num_actions
        
        # Initialize target policy \pi (\theta) and value function v (w) networks
        self.actor_net = ActorNetwork(self.num_actor_features)
        self.critic_net = CriticNetwork(self.num_critic_features)
        
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=self.alpha_theta)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=self.alpha_w)

    def _parse_env(self, env: gym.Env):
        if not isinstance(env.observation_space, gym.spaces.Discrete):
            raise TypeError("Observation space must be a gymnasium.spaces.Discrete space.")
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise TypeError("Action space must be a gymnasium.spaces.Discrete space.")
            
        self.env = env
        self.num_states = int(env.observation_space.n) # type: ignore
        self.num_actions = int(env.action_space.n) # type: ignore
    
    def _get_actor_features(self, state: int, action: int) -> np.ndarray:
        """
        Constructs a One-Hot encoded feature vector for the (state, action) pair.
        Outputs a concatenation of [State Vector] + [Action Vector]
        """
        state_vec = np.zeros(self.num_states, dtype=float)
        state_vec[state] = 1.0
        
        action_vec = np.zeros(self.num_actions, dtype=float)
        action_vec[action] = 1.0
        
        return np.concatenate([state_vec, action_vec])

    def _get_critic_features(self, state: int) -> np.ndarray:
        """
        Constructs a pure One-Hot encoded state feature vector for v(s, w).
        """
        state_vec = np.zeros(self.num_states, dtype=float)
        state_vec[state] = 1.0
        
        return state_vec

    def _get_target_action_probs(self, state: int) -> torch.Tensor:
        """Calculates Target Policy \pi(a|s, \theta) distribution for all actions."""
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
                probs = self._get_target_action_probs(s)
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
        
        # Behavior Policy \beta(a|s) is Uniform for all states
        beta_prob = 1.0 / self.num_actions

        while k < self.max_episodes:
            total_reward, episode_length = 0.0, 0
            
            s_t, _ = self.env.reset()
            
            for _ in range(self.max_steps_per_episode):
                # 1. Generate a_t following Behavior Policy \beta(s_t) uniformly
                a_t = int(np.random.choice(self.num_actions))
                
                # Observe r_{t+1}, s_{t+1}
                s_tp1, r_tp1, terminated, truncated, _ = self.env.step(a_t)
                done = terminated or truncated
                
                total_reward += float(r_tp1)
                episode_length += 1
                
                # --- ATTACH TARGET POLICY COMPUTATION GRAPH ---
                # Calculate \pi(a_t|s_t, \theta_t) using the current Target Actor network
                probs_current = self._get_target_action_probs(s_t)
                m_current = Categorical(probs_current)
                
                pi_prob = probs_current[a_t]
                log_prob = m_current.log_prob(torch.tensor(a_t))

                # Importance Sampling Ratio: \pi(a_t|s_t, \theta_t) / \beta(a_t|s_t)
                rho_t = (pi_prob / beta_prob).detach()

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
                # Objective: Maximize \rho_t * \delta_t * \ln \pi(a_t|s_t, \theta_t)
                actor_loss = -(rho_t * delta * log_prob)
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # --- 4. Critic Update (w) ---
                # Objective: Minimize \rho_t * [r_{t+1} + \gamma * v_{t+1} - v_t]^2
                critic_loss = rho_t * (target_v - v_current) ** 2
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

        print(f"Off-Policy A2C (One-Hot + Uniform Behavior) optimization finished after {k} episodes.")
        
        optimal_policy = self._extract_optimal_policy()
        state_values = self._extract_state_values()
            
        return optimal_policy, state_values