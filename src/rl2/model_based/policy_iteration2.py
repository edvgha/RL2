from typing import Tuple
import numpy as np
import gymnasium as gym

class PolicyIteration:
    def __init__(self, env: gym.Env, gamma=0.9, theta=1e-5, max_iterations=5000):
        """
        Args:
            env (gym.Env): A Gymnasium environment (must use Discrete state and action spaces).
            gamma (float): Discount factor for future rewards.
            theta (float): Convergence threshold (\theta).
            max_iterations (int): Safety cap to prevent infinite loops.
        """
        self.gamma = gamma
        self.theta = theta
        self.max_iterations = max_iterations

        self._parse_env(env)

    def _parse_env(self, env: gym.Env):
        # 1. Verify space compatibility
        if not isinstance(env.observation_space, gym.spaces.Discrete):
            raise TypeError("Observation space must be a gymnasium.spaces.Discrete space.")
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise TypeError("Action space must be a gymnasium.spaces.Discrete space.")
            
        unwrapped_env = env.unwrapped
        if not hasattr(unwrapped_env, 'P'):
            raise AttributeError("The provided environment doesn't expose an explicit transition dictionary 'P'.")
            
        self.num_states = int(unwrapped_env.observation_space.n) # type: ignore
        self.num_actions = int(unwrapped_env.action_space.n) # type: ignore
        
        # 2. Pre-allocate our matrix math primitives
        # P_tensor shape: (num_states, num_actions, num_states)
        # R_matrix shape: (num_states, num_actions)
        self.P_tensor = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.R_matrix = np.zeros((self.num_states, self.num_actions))
        self.terminal_states = set()
        
        # 3. Parse Gymnasium's 'P' dictionary directly into our custom matrices
        for s in range(self.num_states):
            is_terminal_state = True
            for a in range(self.num_actions):
                # unwrapped_env.P[s][a] returns a list of tuples: (prob, next_state, reward, terminated)
                transitions = unwrapped_env.P[s][a] # type: ignore
                
                # Compute expected reward matrix: \sum_{r} p(r | s, a) * r
                # and transition tensor matrix: p(s' | s, a)
                for prob, s_prime, reward, terminated in transitions:
                    self.P_tensor[s, a, s_prime] += prob
                    self.R_matrix[s, a] += prob * reward
                    
                    if not terminated:
                        is_terminal_state = False
                        
            if is_terminal_state:
                self.terminal_states.add(s)

    def _extract_optimal_policy(self, pi: np.ndarray) -> np.ndarray:
        """
        pi (np.ndarray): Policy matrix. Shape: (num_states, num_actions)

        Returns:
            optimal_policy (np.ndarray): Optimal deterministic policy. Shape: (num_states,)
        """
        optimal_policy = np.zeros(self.num_states, dtype=int)
        
        for s in range(self.num_states):
            if s in self.terminal_states:
                optimal_policy[s] = -1
                continue
            optimal_policy[s] = np.argmax(pi[s, :])
        return optimal_policy

    def __call__(self) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Returns:
            v (np.ndarray): Optimal state-value function vector. Shape: (num_states,)
            policy (np.ndarray): Optimal deterministic policy. Shape: (num_states,)
            converged: converged or not
        """
        # Initial Guess: v_0
        v = np.zeros(self.num_states)
        
        # Initial Policy Guess \pi_0: Uniform initialization choosing action 0
        pi = np.zeros((self.num_states, self.num_actions))
        pi[:, 0] = 1.0

        k, converged = 0, False
        
        while k < self.max_iterations:
            # --- 1. Policy Evaluation ---
            v_j = np.zeros(self.num_states)
            j = 0
            
            while j < self.max_iterations:
                delta_j, v_j_prev = 0, np.copy(v_j)
                
                for s in range(self.num_states):
                    if s in self.terminal_states:
                        v_j[s] = 0.0
                        continue
                    
                    state_value_sum = 0.0
                    for a in range(self.num_actions):
                        expected_reward = self.R_matrix[s, a]
                        expected_future_value = np.dot(self.P_tensor[s, a, :], v_j_prev)
                        
                        q_s_a = expected_reward + self.gamma * expected_future_value
                        state_value_sum += pi[s, a] * q_s_a
                        
                    v_j[s] = state_value_sum
                    delta_j = max(delta_j, np.abs(v_j[s] - v_j_prev[s]))
                
                j += 1
                if delta_j < self.theta:
                    break
            
            # Assign evaluated value function state
            v = np.copy(v_j)
            
            # --- 2. Policy Improvement ---
            policy_stable = True
            pi_next = np.zeros_like(pi)
            
            for s in range(self.num_states):
                if s in self.terminal_states:
                    continue
                
                q_values = np.zeros(self.num_actions)
                for a in range(self.num_actions):
                    expected_reward = self.R_matrix[s, a]
                    expected_future_value = np.dot(self.P_tensor[s, a, :], v)
                    
                    q_values[a] = expected_reward + self.gamma * expected_future_value
                
                # Determine greedy action
                best_action = np.argmax(q_values)
                pi_next[s, best_action] = 1.0
                
                if not np.array_equal(pi[s], pi_next[s]):
                    policy_stable = False
            
            pi = pi_next
            k += 1
            
            if policy_stable:
                print(f"Policy Iteration converged perfectly at iteration {k}.")
                converged = True
                break
        else:
            print("Reached maximum processing limit without meeting convergence limits.")

        optimal_policy = self._extract_optimal_policy(pi)
            
        return v, optimal_policy, converged