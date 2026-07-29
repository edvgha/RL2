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
    def __init__(self, input_dim: int, output_dim: int):
        super(ActorNetwork, self).__init__()
        # Actor maps S -> A. It outputs 'output_dim' continuous logits.
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class CriticNetwork(nn.Module):
    def __init__(self, input_dim: int):
        super(CriticNetwork, self).__init__()
        # Critic evaluates q(s, a, w), so it outputs a single scalar action-value.
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
# 2. Deterministic Actor-Critic (DPG) Solver
# =====================================================================
class DeterministicActorCritic:
    def __init__(self, 
                 env: gym.Env, 
                 dims: List[int],
                 gamma=0.9, 
                 alpha_theta=0.001,
                 alpha_w=0.001,
                 max_episodes=5000, 
                 max_steps_per_episode=500, 
                 use_one_hot_features=True):
        """
        Args:
            env (gym.Env): A Gymnasium environment (Discrete state and action spaces).
            dims (List[int]): [width, height] describing grid axes bounds.
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

        # Actor takes ONLY the state as input. Critic takes BOTH state and action.
        if use_one_hot_features:
            num_actor_features = self.num_states
            num_critic_features = self.num_states + self.num_actions
        else:
            num_actor_features = 6   # [1, x, y, x^2, y^2, xy]
            num_critic_features = 10 # [1, x, y, a, x^2, y^2, a^2, xy, xa, ya]
        
        # Initialize Actor (\theta) mapping S -> A, and Critic (w) networks.
        self.actor_net = ActorNetwork(num_actor_features, self.num_actions)
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

    def _state_to_xy(self, state: int) -> Tuple[float, float]:
        """Maps a 1D state ID to discrete (x, y) coordinates."""
        y = float(state // self.dims[0])
        x = float(state % self.dims[0])
        return x, y
        
    def _get_state_features(self, state: int) -> np.ndarray:
        """Extracts the state features used as input for the Actor network."""
        if self.use_one_hot_features:
            state_vec = np.zeros(self.num_states, dtype=float)
            state_vec[state] = 1.0
            return state_vec
        else:
            raw_x, raw_y = self._state_to_xy(state)
            max_x = float(self.dims[0] - 1)
            x = 2.0 * (raw_x / max_x) - 1.0 if max_x > 0 else 0.0
            max_y = float(self.dims[1] - 1)
            y = 2.0 * (raw_y / max_y) - 1.0 if max_y > 0 else 0.0
            return np.array([1.0, x, y, x**2, y**2, x * y], dtype=float)

    def _get_action_probs(self, state: int) -> torch.Tensor:
        """Calculates discrete \pi(a|s, \theta) distribution mapping s -> logits."""
        state_features = self._get_state_features(state)
        features_tensor = torch.FloatTensor(state_features)
        
        logits = self.actor_net(features_tensor)
        action_probs = torch.softmax(logits, dim=-1)
        return action_probs

    def _get_mu(self, state: int) -> torch.Tensor:
        """
        Calculates the continuous target policy relaxation \mu(s, \theta)
        Returns the expected continuous action (differentiable).
        """
        probs = self._get_action_probs(state)
        
        if self.use_one_hot_features:
            # The continuous action representation is the probability vector itself
            return probs
        else:
            # The continuous action representation is the expected scalar action value in [-1, 1]
            max_a = float(self.num_actions - 1)
            action_scalars = torch.tensor(
                [2.0 * (a / max_a) - 1.0 if max_a > 0 else 0.0 for a in range(self.num_actions)],
                dtype=torch.float32
            )
            return torch.sum(probs * action_scalars)

    def _action_to_continuous(self, action: int) -> torch.Tensor:
        """Converts a drawn discrete integer action a_t into the continuous representation."""
        if self.use_one_hot_features:
            vec = torch.zeros(self.num_actions, dtype=torch.float32)
            vec[action] = 1.0
            return vec
        else:
            max_a = float(self.num_actions - 1)
            val = 2.0 * (action / max_a) - 1.0 if max_a > 0 else 0.0
            return torch.tensor(val, dtype=torch.float32)

    def _get_critic_tensor(self, state: int, continuous_action: torch.Tensor) -> torch.Tensor:
        """
        Builds the critic feature tensor for q(s, a, w) dynamically in PyTorch.
        This ensures gradients can flow through continuous_action (\mu) back to the Actor.
        """
        if self.use_one_hot_features:
            state_vec = torch.zeros(self.num_states, dtype=torch.float32)
            state_vec[state] = 1.0
            return torch.cat([state_vec, continuous_action], dim=-1)
        else:
            raw_x, raw_y = self._state_to_xy(state)
            max_x = float(self.dims[0] - 1)
            x = 2.0 * (raw_x / max_x) - 1.0 if max_x > 0 else 0.0
            max_y = float(self.dims[1] - 1)
            y = 2.0 * (raw_y / max_y) - 1.0 if max_y > 0 else 0.0

            x_t = torch.tensor(x, dtype=torch.float32)
            y_t = torch.tensor(y, dtype=torch.float32)
            
            features = torch.stack([
                torch.tensor(1.0),
                x_t,
                y_t,
                continuous_action,
                x_t**2,
                y_t**2,
                continuous_action**2,
                x_t * y_t,
                x_t * continuous_action,
                y_t * continuous_action
            ])
            return features

    def _extract_optimal_policy(self) -> np.ndarray:
        optimal_policy = np.zeros(self.num_states, dtype=int)
        for s in range(self.num_states):
            with torch.no_grad():
                probs = self._get_action_probs(s)
                optimal_policy[s] = torch.argmax(probs).item()
        return optimal_policy

    def _extract_state_values(self) -> np.ndarray:
        """Extracts expected state value array V(s) = q(s, \mu(s, \theta))."""
        state_values = np.zeros(self.num_states, dtype=float)
        for s in range(self.num_states):
            with torch.no_grad():
                mu_s = self._get_mu(s)
                q_val = self.critic_net(self._get_critic_tensor(s, mu_s)).item()
                state_values[s] = q_val
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
                # 1. Generate a_t following uniform behavior policy \beta
                a_t = int(np.random.choice(self.num_actions))
                
                # Observe r_{t+1}, s_{t+1}
                s_tp1, r_tp1, terminated, truncated, _ = self.env.step(a_t)
                done = terminated or truncated
                
                total_reward += float(r_tp1)
                episode_length += 1
                
                # Format discrete a_t into the continuous domain for the Critic
                a_t_continuous = self._action_to_continuous(a_t)
                q_current = self.critic_net(self._get_critic_tensor(s_t, a_t_continuous)).squeeze()
                
                if done:
                    target_q = torch.tensor(float(r_tp1), dtype=torch.float32)
                else:
                    with torch.no_grad():
                        # Calculate \mu(s_{t+1}, \theta_t) without gradients
                        mu_next = self._get_mu(s_tp1)
                        q_next = self.critic_net(self._get_critic_tensor(s_tp1, mu_next)).squeeze()
                    # TD error target: r_{t+1} + \gamma q(s_{t+1}, \mu(s_{t+1}, \theta_t), w_t)
                    target_q = float(r_tp1) + self.gamma * q_next

                # --- 2 & 4. TD Error & Critic Update (w) ---
                # Objective: Minimize TD Error loss
                critic_loss = self.critic_loss_fn(q_current, target_q.detach())
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                # --- 3. Actor Update (\theta) ---
                # Objective: Maximize q(s_t, \mu(s_t, \theta_t), w_t) via Deterministic PG
                mu_current_with_grad = self._get_mu(s_t)
                q_mu = self.critic_net(self._get_critic_tensor(s_t, mu_current_with_grad)).squeeze()
                
                # We minimize -q to maximize the critic's value evaluation
                actor_loss = -q_mu 
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                if done:
                    break
                    
                # Transition to next step
                s_t = s_tp1

            self.stats["total_rewards"].append(total_reward)
            self.stats["episode_lengths"].append(episode_length)
            k += 1

        print(f"Deterministic Actor-Critic (DPG) optimization finished after {k} episodes | Total R: {sum(self.stats['total_rewards'])}")
        
        optimal_policy = self._extract_optimal_policy()
        state_values = self._extract_state_values()
            
        return optimal_policy, state_values