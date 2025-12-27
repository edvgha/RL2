"""
Value Iteration 

Bellman contraction operator:
    V(s) = max_a[Σ_s' P(s'|s,a) * [R(s,a) + γ * V(s')]]

Algoriths:
    Init: V_0
    Loop:
        V_k+1 = B[V_k]
    Until: |V_k+1 - V_k| < ε

    π(a|s) = argmax_a(Σ_s' P(s'|s,a) * [R(s,a) + γ * V_k+1(s')])

"""
from typing import Tuple
import numpy as np

def value_iteration(P: np.ndarray,
                    R: np.ndarray,
                    gamma: float,
                    theta: float,
                    max_iter=1000
                    ) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Perform Value Iteration to compute the optimal value function and policy.

    Args:
        P (np.ndarray): Transition probabilities, shape [n_states, n_actions, n_states]
        R (np.ndarray): Expected rewards, shape [n_states, n_actions]
        gamma (float): Discount factor (0 <= gamma <= 1)
        theta (float): Convergence threshold
        max_iter (int): Maximum number of iterations

    Returns:
        V (np.ndarray): Optimal value function, shape [n_states]
        policy (np.ndarray): Optimal deterministic policy, shape [n_states]
    """
    done = False
    n_states, n_actions = R.shape

    v_values = np.zeros(n_states, dtype=np.float32)
    policy_actions = np.zeros(n_states, dtype=np.int32)

    for _ in range(max_iter):
        v_values_old = v_values.copy()
        for s in range(n_states):
            q_values = np.zeros(n_actions)
            for a in range(n_actions):
                # Bellman update: R(s,a) + γ * Σ_s' P(s'|s,a) * V(s')
                # R(s, a) - expected reward
                q_values[a] = R[s, a] + gamma * np.dot(P[s, a, :], v_values_old)
            v_values[s] = np.max(q_values)
            policy_actions[s] = np.argmax(q_values)

        if np.max(np.abs(v_values - v_values_old)) < theta:
            done = True
            break

    return v_values, policy_actions, done
