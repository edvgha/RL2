"""
Policy Iteration 

Init:
    V, π 
Repeat:
    • Policy Evaluation:
        Loop:
            v = V(s)
            V(s) = Σ_s' P(s'|s, π(s)) * [R(s, π(s), s') + γ * V(s')]
        Until: |v - V(s)| < ε

    • Policy Improvement:
        π'(s) = argmax_a Σ P(s' | s, a) * [R(s, a, s') + γ * V(s')]

Until π = π'
"""
from typing import Tuple
import numpy as np

def policy_iteration(
        P: np.ndarray,
        R: np.ndarray,
        gamma: float,
        theta: float,
        max_eval_iter: int,
        max_impr_iter: int
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Perform Policy Iteration to find the optimal policy and value function.

    Args:
        P (np.ndarray): Transition probabilities, shape [n_states, n_actions, n_states]
        R (np.ndarray): Expected rewards, shape [n_states, n_actions]
        gamma (float): Discount factor (0 <= gamma < 1)
        theta (float): Convergence threshold for policy evaluation
        max_eval_iter (int): Max iterations for policy evaluation
        max_impr_iter (int): Max policy improvement iterations

    Returns:
        V (np.ndarray): Optimal value function, shape [n_states]
        policy (np.ndarray): Optimal deterministic policy, shape [n_states]
    """
    done = False
    n_states, n_actions = R.shape

    v_values = np.zeros(n_states, dtype=np.float32)
    policy_actions = np.zeros(n_states, dtype=np.int32)

    for _ in range(max_impr_iter):
        # -----------------
        # Policy evaluation
        # -----------------
        for _ in range(max_eval_iter):
            v_values_old = v_values.copy()

            # V(s) = Σ_s' P(s'|s, π(s)) * [R(s, π(s), s') + γ * V(s')]
            for s in range(n_states):
                v_values[s] = R[s, policy_actions[s]] + \
                        gamma * np.dot(P[s, policy_actions[s], :], v_values_old)

            if np.max(np.abs(v_values - v_values_old)) < theta:
                break

        # ------------------
        # Policy improvement
        # ------------------
        stable_policy = True
        # π'(s) = argmax_a Σ P(s' | s, a) * [R(s, a, s') + γ * V(s')]
        for s in range(n_states):
            old_action = policy_actions[s]

            q_values = np.zeros(n_actions)
            for a in range(n_actions):
                q_values[a] = R[s, a] + gamma * np.dot(P[s, a, :], v_values)

            new_action = np.argmax(q_values)

            if old_action != new_action:
                stable_policy = False
                policy_actions[s] = new_action

        if stable_policy:
            done = True
            break
    return v_values, policy_actions, done
