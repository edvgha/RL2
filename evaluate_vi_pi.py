"""
    Evaluate
"""
# pylint: skip-file
import warnings
warnings.filterwarnings("ignore")
from typing import (
    Callable,
    Tuple
)
from enum import Enum
import numpy as np
import gymnasium as gym
from src.model_based.value_iteration import value_iteration
from src.model_based.policy_iteration import policy_iteration
from environments import (
    frozen_lake_d_env, 
    frozen_lake_s_env,
    taxi_env
)

class Algorithm(Enum):
    VALUEITERATION = "ValueIteration"
    POLICYITERATION = "PolicyIteration"

def extract_mdp_from_gym_env(env: gym.Env):
    """
    Returns:
        P (np.ndarray): [n_states, n_actions, n_states]
        R (np.ndarray): [n_states, n_actions] → expected reward
        n_states, n_actions, state_to_idx, action_to_idx
    """
    env = env.unwrapped

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    P = np.zeros((n_states, n_actions, n_states))
    R = np.zeros((n_states, n_actions))

    # env.P is a dict: {state: {action: [(prob, next_state, reward, done), ...]}}
    for s in range(n_states):
        for a in range(n_actions):
            transitions = env.P[s][a]
            for prob, s_next, reward, _ in transitions:
                P[s, a, s_next] += prob
                # Expected reward: R(s,a) = Σ_s' P(s'|s,a) * r(s,a,s')
                R[s, a] += prob * reward

    env.close()
    return P, R, n_states, n_actions

def evaluate(env_func: Callable[[], Tuple[gym.Env, gym.Env]], alg: Algorithm):
    """
        Evaluates specified algorithm on on specified env
    """
    env, env_h = env_func()
    print(f"\n🔵 Env: {env.spec.id} | Alg: {alg.value}")

    P, R, _, _ = \
        extract_mdp_from_gym_env(env)

    if alg == Algorithm.VALUEITERATION:
        _, policy, done = value_iteration(P, R, gamma=0.9, theta=1e-6, max_iter=1000)
        assert done
    else: # alg == Algorithm.POLICYITERATION
        _, policy, done = policy_iteration(P, R, gamma=0.9, theta=1e-6,
                                           max_eval_iter=1000, max_impr_iter=100)
    assert done

    state, _ = env_h.reset(seed=42)
    total_reward: float = 0.0
    steps: int = 0

    while True:
        action = policy[state]

        next_state, reward, terminated, truncated, _ = env_h.step(action)

        total_reward += float(reward)
        steps += 1
        state = next_state

        print(f"⚪ {steps}: Action={action}, Reward={reward:.1f}, State={state}")

        if terminated or truncated:
            break

    print(f"🟢 Episode finished! Total reward: {total_reward:.1f}, Steps: {steps}")
    env.close()

if __name__ == '__main__':
    evaluate(env_func=frozen_lake_d_env, alg=Algorithm.VALUEITERATION)
    evaluate(env_func=frozen_lake_d_env, alg=Algorithm.POLICYITERATION)
    evaluate(env_func=taxi_env, alg=Algorithm.VALUEITERATION)
    evaluate(env_func=taxi_env, alg=Algorithm.POLICYITERATION)
    evaluate(env_func=frozen_lake_s_env, alg=Algorithm.VALUEITERATION)
    evaluate(env_func=frozen_lake_s_env, alg=Algorithm.POLICYITERATION)
    