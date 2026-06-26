from typing import Tuple, List, Dict
import numpy as np
import gymnasium as gym

class Sarsa:
    def __init__(self, env: gym.Env, 
                 gamma=0.9, 
                 alpha=0.1, 
                 epsilon=1.0, 
                 decay_rate=0.001, 
                 min_epsilon=0.01, 
                 max_episodes=5000, 
                 max_steps_per_episode=500,
                 use_esarsa=False):
        """
            Args:
            env (gym.Env): A Gymnasium environment (must use Discrete state and action spaces).
            gamma (float): Discount factor for future rewards.
            alpha (float): Constant learning rate \alpha_t(s,a) = \alpha > 0.
            epsilon (float): Initial exploration parameter \epsilon \in (0, 1).
            decay_rate (float): Exponential decay rate applied to epsilon after each episode.
            min_epsilon (float): Floor boundary threshold for epsilon decay.
            max_episodes (int): Safety cap to prevent infinite loops (corresponds to total loops).
            max_steps_per_episode (int): Maximum length T of a single generated episode.
            use_esarsa: use as target expected q value
        """
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.min_epsilon = min_epsilon
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.esarsa = use_esarsa

        self.stats = {
            "total_rewards": [],
            "episode_lengths": []
        }

        self._parse_env(env)

        self.state_values = np.zeros(self.num_states, dtype=float)

    def _parse_env(self, env: gym.Env):
        # 1. Verify space compatibility
        if not isinstance(env.observation_space, gym.spaces.Discrete):
            raise TypeError("Observation space must be a gymnasium.spaces.Discrete space.")
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise TypeError("Action space must be a gymnasium.spaces.Discrete space.")
            
        self.env = env
        self.num_states = int(env.observation_space.n) # type: ignore
        self.num_actions = int(env.action_space.n) # type: ignore

    def _update_policy_profile(self, pi: np.ndarray, q: np.ndarray, s: int):
        if np.all(q[s, :] == q[s, 0]):
            return pi
        
        a_star = np.argmax(q[s, :])
        num_A = self.num_actions
        
        for a in range(num_A):
            if a == a_star:
                # \pi_{t+1}(a|s_t) = 1 - (\epsilon * |A(s_t)| - 1) / |A(s_t)|
                pi[s, a] = 1.0 - ((self.epsilon * (num_A - 1)) / num_A)
            else:
                # \pi_{t+1}(a|s_t) = \epsilon / |A(s_t)|
                pi[s, a] = self.epsilon / num_A
        return pi
    
    def _extract_state_values(self, q: np.ndarray) -> np.ndarray:
        """
        q (np.ndarray): State-action value matrix. Shape: (num_states, num_actions)

        Returns:
            state_values (np.ndarray): The maximum value for each state V(s) = max_a Q(s, a). Shape: (num_states,)
        """
        
        for s in range(self.num_states):
            self.state_values[s] = np.max(q[s, :])
        return self.state_values

    def _extract_optimal_policy(self, q: np.ndarray) -> np.ndarray:
        """
        q (np.ndarray): State-action value matrix. Shape: (num_states, num_actions)

        Returns:
            optimal_policy (np.ndarray): Optimal deterministic policy. Shape: (num_states,)
        """
        optimal_policy = np.zeros(self.num_states, dtype=int)
        
        for s in range(self.num_states):
            optimal_policy[s] = np.argmax(q[s, :])
        return optimal_policy

    def __call__(self) -> np.ndarray:
        """
        Returns:
            q (np.ndarray): Final estimated state-action value matrix. Shape: (num_states, num_actions)
            policy (np.ndarray): Optimal deterministic policy map [s] -> best action index. Shape: (num_states,)
            stats (dict): Dictionary tracking 'total_rewards' and 'episode_lengths' per episode.
        """
        # Initial q_0(s,a) for all (s,a) initialized to zeros
        q = np.zeros((self.num_states, self.num_actions))
        
        # Initial \epsilon-greedy policy \pi_0 derived from q_0
        pi = np.zeros((self.num_states, self.num_actions))
        for s in range(self.num_states):
            for a in range(self.num_actions):
                pi[s, a] = 1. / self.num_actions
            # pi = self._update_policy_profile(pi, q, s)

        k = 0
        while k < self.max_episodes:
            total_reward, episode_length = 0.0, 0
            
            s_t, _ = self.env.reset()
            
            # Generate a_0 at s_0 following \pi_0(s_0)
            a_t = np.random.choice(self.num_actions, p=pi[s_t])
            
            for _ in range(self.max_steps_per_episode):
                # **Collect** an experience sample (r_{t+1}, s_{t+1}, a_{t+1}) given (s_t, a_t)
                # Generate r_{t+1}, s_{t+1} by interacting with the environment.
                s_tp1, r_tp1, terminated, truncated, _ = self.env.step(a_t)
                
                # Generate a_{t+1} following \pi_t(s_{t+1})
                a_tp1 = np.random.choice(self.num_actions, p=pi[s_tp1])
                
                # Track metrics data
                total_reward += float(r_tp1)
                episode_length += 1
                
                # Update q-value for (s_t, a_t)
                # q_{t+1}(s_t, a_t) = q_t(s_t, a_t) - \alpha_t(s_t, a_t) * [q_t(s_t, a_t) - (r_{t+1} + \gamma * q_t(s_{t+1}, a_{t+1}))]
                # Note: if s_tp1 is terminal, future expected value becomes 0.0
                future_val = 0.0 if terminated else q[s_tp1, a_tp1]

                if self.esarsa:
                    future_val = 0.0 if terminated else float(np.dot(pi[s_tp1, :], q[s_tp1, :]))

                q[s_t, a_t] = q[s_t, a_t] - self.alpha * (q[s_t, a_t] - (float(r_tp1) + self.gamma * future_val))
                
                # Update policy for s_t
                pi = self._update_policy_profile(pi, q, s_t)

                
                if terminated or truncated:
                    break
                    
                # Transition: s_t <- s_{t+1}, a_t <- a_{t+1}
                s_t, a_t = s_tp1, a_tp1

            # Log metrics per completed episode loop
            self.stats["total_rewards"].append(total_reward)
            self.stats["episode_lengths"].append(episode_length)

            self.epsilon = max(self.epsilon * np.exp(-self.decay_rate), self.min_epsilon)

            k += 1

        print(f"Sarsa optimization algorithm completed execution after {k} episodes.")
        
        optimal_policy = self._extract_optimal_policy(q)
        self._extract_state_values(q)
            
        return optimal_policy