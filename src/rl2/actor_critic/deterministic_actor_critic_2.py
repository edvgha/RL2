from typing import Tuple, List, Dict
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

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
            nn.Linear(32, 1),
            nn.Tanh()  # Deterministic action a = \mu(s, \theta) bounded to the normalized axis [-1, 1]
        )
        
        # DDPG-style final layer initialization: start \mu(s) near the center of the
        # action axis so Tanh operates in its high-gradient region rather than a
        # saturated boundary where \nabla_\theta \mu vanishes
        nn.init.uniform_(self.network[4].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.network[4].bias, -3e-3, 3e-3)
        
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
# 2. Deterministic Policy Gradient (DPG) Solver
# =====================================================================
class DPG:
    def __init__(self, 
                 env: gym.Env, 
                 dims: List[int],
                 num_actor_features=6,
                 num_critic_features=10,
                 gamma=0.9, 
                 alpha_theta=0.001,
                 alpha_w=0.001,
                 epsilon=0.5,
                 max_episodes=5000, 
                 max_steps_per_episode=500, 
                 target_update_freq=100,
                 use_one_hot_features=True):
        """
        Algorithm: Deterministic policy gradient (deterministic actor-critic).

        The target policy a = \\mu(s, \\theta) is DETERMINISTIC and outputs a CONTINUOUS
        action on the normalized axis [-1, 1]. Since the environment's action space is
        Discrete, that continuous action is rounded to the nearest valid action index
        when stepping the environment, while \\nabla_a q(s, a, w) is evaluated at the
        continuous point a = \\mu(s, \\theta) via autograd. Note that the Actor consumes
        STATE-ONLY features and the Critic consumes STATE-ACTION features (the roles
        are flipped relative to A2C).

        Experience is generated OFF-POLICY by a given behavior policy \\beta(a|s):
        with probability epsilon a uniform random action, otherwise the (discretized)
        action of \\mu(s, \\theta). Setting epsilon=1.0 recovers the fixed uniform
        behavior policy of off_policy_a2c.py; the default keeps \\beta exploratory but
        biased toward the target policy, which is what a sparse-reward grid needs to
        see enough successful trajectories.

        Because the q-Critic bootstraps off-policy at the greedy action (like Q-learning),
        the TD target is computed with a frozen TARGET Critic network w^T synchronized
        every `target_update_freq` iterations — the same stabilizer used in
        deep_q_learning_off_policy.py (and in DDPG, the deep successor of this algorithm).

        Args:
            env (gym.Env): A Gymnasium environment (Discrete state and action spaces).
            dims (List[int]): [width, height] describing grid axes bounds.
            num_actor_features (int): Number of actor network features.
            num_critic_features (int): Number of critic network features.
            gamma (float): Discount factor for future rewards.
            alpha_theta (float): Learning rate for Actor's PyTorch optimizer.
            alpha_w (float): Learning rate for Critic's PyTorch optimizer.
            epsilon (float): Exploration rate of the behavior policy \\beta(a|s) (1.0 = uniform).
            max_episodes (int): Total number of episodes to process.
            max_steps_per_episode (int): Maximum length of a single generated episode.
            target_update_freq (int): Interval iterations to synchronize the target Critic w^T = w.
            use_one_hot_features (bool): if True use one-hot encoding for actor/critic input
        """
        self.gamma = gamma
        self.alpha_theta = alpha_theta
        self.alpha_w = alpha_w
        self.epsilon = epsilon
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.target_update_freq = target_update_freq
        self.use_one_hot_features = use_one_hot_features
        
        self.dims = dims 

        self.stats = {
            "total_rewards": [],
            "episode_lengths": []
        }

        self._parse_env(env)

        if use_one_hot_features:
            # The Actor consumes the state one-hot; the Critic consumes the state one-hot
            # PLUS the continuous action scalar (an action one-hot would not be
            # differentiable w.r.t. a, which DPG requires for \nabla_a q).
            num_actor_features = self.num_states
            num_critic_features = self.num_states + 1
        
        # Initialize Actor (\theta) and Critic (w) networks
        self.actor_net = ActorNetwork(num_actor_features)
        self.critic_net = CriticNetwork(num_critic_features)
        
        # Frozen target Critic (w^T) used only for the TD bootstrap target
        self.target_critic_net = CriticNetwork(num_critic_features)
        self.target_critic_net.load_state_dict(self.critic_net.state_dict())
        self.target_critic_net.eval()  # Keep target network in evaluation mode
        
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=self.alpha_theta)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=self.alpha_w)
        self.critic_loss_fn = nn.MSELoss()
        
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

    def _discrete_to_continuous(self, action: int) -> float:
        """Maps a discrete action index to its normalized position on the continuous action axis [-1, 1]."""
        max_a = float(self.num_actions - 1)
        return 2.0 * (float(action) / max_a) - 1.0 if max_a > 0 else 0.0

    def _continuous_to_discrete(self, a: float) -> int:
        """Maps a continuous action in [-1, 1] to the nearest discrete action index."""
        max_a = self.num_actions - 1
        if max_a <= 0:
            return 0
        index = int(round((float(a) + 1.0) / 2.0 * max_a))
        return int(np.clip(index, 0, max_a))

    def _get_actor_features(self, state: int) -> np.ndarray:
        if self.use_one_hot_features:
            return self._get_actor_features_one_hot(state)
        return self._get_actor_features_quadratic(state)

    def _get_critic_features(self, state: int, action: torch.Tensor) -> torch.Tensor:
        if self.use_one_hot_features:
            return self._get_critic_features_one_hot(state, action)
        return self._get_critic_features_quadratic(state, action)

    def _get_actor_features_one_hot(self, state: int) -> np.ndarray:
        """
        Constructs a pure One-Hot encoded state feature vector for \\mu(s, \\theta).
        """
        state_vec = np.zeros(self.num_states, dtype=float)
        state_vec[state] = 1.0
        
        return state_vec

    def _get_critic_features_one_hot(self, state: int, action: torch.Tensor) -> torch.Tensor:
        """
        Constructs the feature vector for q(s, a, w) as [State One-Hot] + [continuous action].
        Built with torch ops so gradients can flow through `action` (\\nabla_a q).
        """
        state_vec = torch.zeros(self.num_states)
        state_vec[state] = 1.0
        
        return torch.cat([state_vec, action.reshape(1)])
    
    def _get_actor_features_quadratic(self, state: int) -> np.ndarray:
        """Constructs the pure state feature vector for \\mu(s, \\theta): [1, x, y, x^2, y^2, xy]"""
        raw_x, raw_y = self._state_to_xy(state)
        
        max_x = float(self.dims[0] - 1)
        x = 2.0 * (raw_x / max_x) - 1.0 if max_x > 0 else 0.0
        
        max_y = float(self.dims[1] - 1)
        y = 2.0 * (raw_y / max_y) - 1.0 if max_y > 0 else 0.0
        
        features = np.array([
            1.0, x, y, x**2, y**2, x * y
        ], dtype=float)
        
        return features

    def _get_critic_features_quadratic(self, state: int, action: torch.Tensor) -> torch.Tensor:
        """
        Constructs the feature vector for q(s, a, w): [1, x, y, a, x^2, y^2, a^2, xy, xa, ya]
        Built with torch ops so gradients can flow through `action` (\\nabla_a q).
        """
        raw_x, raw_y = self._state_to_xy(state)
        
        max_x = float(self.dims[0] - 1)
        x = 2.0 * (raw_x / max_x) - 1.0 if max_x > 0 else 0.0
        
        max_y = float(self.dims[1] - 1)
        y = 2.0 * (raw_y / max_y) - 1.0 if max_y > 0 else 0.0
        
        a = action.reshape(1)
        one = torch.ones_like(a)
        
        features = torch.cat([
            one, x * one, y * one, a, (x ** 2) * one, (y ** 2) * one, a ** 2, (x * y) * one, x * a, y * a
        ])
        
        return features

    def _get_deterministic_action_tensor(self, state: int) -> torch.Tensor:
        """Computes a = \\mu(s, \\theta) returning a tensor to preserve the computation graph."""
        features = self._get_actor_features(state)
        features_tensor = torch.FloatTensor(features)
        return self.actor_net(features_tensor)

    def _get_q_value_tensor(self, state: int, action: torch.Tensor, evaluate_target=False) -> torch.Tensor:
        """
        Computes q(s, a, w) returning a tensor to preserve the computation graph.
        `action` stays a tensor so \\nabla_a q is available when a = \\mu(s, \\theta).
        Uses either the main Critic network (w) or the target Critic network (w^T).
        """
        features_tensor = self._get_critic_features(state, action)
        net = self.target_critic_net if evaluate_target else self.critic_net
        return net(features_tensor)

    def _get_greedy_action(self, state: int) -> int:
        """Discretizes \\mu(s, \\theta) to the nearest valid action index (no gradient)."""
        with torch.no_grad():
            a_mu = self._get_deterministic_action_tensor(state)
        return self._continuous_to_discrete(a_mu.item())

    def _sample_behavior_action(self, state: int) -> int:
        """
        Samples a_t from the behavior policy \\beta(a|s): with probability epsilon a
        uniform random action, otherwise the (discretized) greedy action \\mu(s, \\theta).
        """
        if np.random.random() < self.epsilon:
            return int(np.random.choice(self.num_actions))
        return self._get_greedy_action(state)

    def _extract_optimal_policy(self) -> np.ndarray:
        """Extracts deterministic policy matrix mapping [s] -> action index nearest to \\mu(s, \\theta)."""
        optimal_policy = np.zeros(self.num_states, dtype=int)
        for s in range(self.num_states):
            optimal_policy[s] = self._get_greedy_action(s)
        return optimal_policy

    def _extract_state_values(self) -> np.ndarray:
        """Extracts state value array V(s) = q(s, \\mu(s, \\theta), w) under the deterministic target policy."""
        state_values = np.zeros(self.num_states, dtype=float)
        for s in range(self.num_states):
            with torch.no_grad():
                a_mu = torch.tensor([self._discrete_to_continuous(self._get_greedy_action(s))])
                state_values[s] = self._get_q_value_tensor(s, a_mu).item()
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
                # 1. Generate a_t following the behavior policy \beta (OFF-POLICY)
                a_t = self._sample_behavior_action(s_t)
                
                # Observe r_{t+1}, s_{t+1}
                s_tp1, r_tp1, terminated, truncated, _ = self.env.step(a_t)
                done = terminated or truncated
                
                total_reward += float(r_tp1)
                episode_length += 1
                
                # Continuous representation of the executed discrete action a_t
                a_t_cont = torch.tensor([self._discrete_to_continuous(a_t)])
                
                # Calculate q(s_t, a_t, w_t) using the Critic Network
                q_current = self._get_q_value_tensor(s_t, a_t_cont)
                
                if done:
                    q_next = torch.tensor([0.0])
                else:
                    # Bootstrap at the TARGET policy's action \mu(s_{t+1}, \theta_t). Since the env is
                    # Discrete, \mu's action is snapped to the nearest EXECUTABLE index (bootstrapping at
                    # the raw continuous output could chase interpolated q-peaks no action ever attains).
                    # Detach so gradients don't flow backwards across time steps.
                    with torch.no_grad():
                        a_tp1 = torch.tensor([self._discrete_to_continuous(self._get_greedy_action(s_tp1))])
                        q_next = self._get_q_value_tensor(s_tp1, a_tp1, evaluate_target=True)

                # 2. TD error: \delta_t = r_{t+1} + \gamma q(s_{t+1}, \mu(s_{t+1}, \theta_t), w_t) - q(s_t, a_t, w_t)
                target_q = float(r_tp1) + self.gamma * q_next

                # --- 3. Actor Update (\theta) ---
                # Objective: Maximize q(s_t, \mu(s_t, \theta), w_t). Autograd applies the chain rule
                # \theta_{t+1} = \theta_t + \alpha_\theta \nabla_\theta \mu(s_t, \theta_t) \nabla_a q(s_t, a, w_t)|_{a=\mu(s_t)}
                a_mu = self._get_deterministic_action_tensor(s_t)
                actor_loss = -self._get_q_value_tensor(s_t, a_mu)
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # --- 4. Critic Update (w) ---
                # Objective: Minimize TD Error [r_{t+1} + \gamma * q_{t+1} - q_t]^2, which implements
                # w_{t+1} = w_t + \alpha_w \delta_t \nabla_w q(s_t, a_t, w_t) (the factor of 2 is absorbed into \alpha_w).
                # zero_grad() also clears the Critic gradients leaked by the Actor's backward pass above.
                critic_loss = self.critic_loss_fn(q_current, target_q)
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
                
                self.total_iterations += 1
                
                # Set w^T = w every C iterations
                if self.total_iterations % self.target_update_freq == 0:
                    self.target_critic_net.load_state_dict(self.critic_net.state_dict())
                
                if done:
                    break
                    
                # Transition to next step
                s_t = s_tp1

            self.stats["total_rewards"].append(total_reward)
            self.stats["episode_lengths"].append(episode_length)
            k += 1

        print(f"DPG optimization finished after {k} episodes | Total R: {sum(self.stats['total_rewards'])}")
        
        optimal_policy = self._extract_optimal_policy()
        state_values = self._extract_state_values()
            
        return optimal_policy, state_values