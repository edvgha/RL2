"""Evaluate Value Iteration and Policy Iteration on Gym toy-text envs."""
from __future__ import annotations

import warnings
from enum import Enum
from typing import Callable, Tuple

import gymnasium as gym
import jax.numpy as jnp
import numpy as np

from rl2.envs import EnvPair, frozen_lake_d_env, frozen_lake_s_env, taxi_env
from rl2.model_based.policy_iteration import policy_iteration
from rl2.model_based.value_iteration import value_iteration

warnings.filterwarnings("ignore")


class Algorithm(Enum):
    VALUEITERATION = "ValueIteration"
    POLICYITERATION = "PolicyIteration"


def extract_mdp_from_gym_env(env: gym.Env) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Pull (P, R) out of env.unwrapped.P"""
    inner = env.unwrapped
    n_states = inner.observation_space.n # type: ignore
    n_actions = inner.action_space.n # type: ignore

    P = np.zeros((n_states, n_actions, n_states), dtype=np.float32)
    R = np.zeros((n_states, n_actions), dtype=np.float32)
    for s in range(n_states):
        for a in range(n_actions):
            for prob, s_next, reward, _ in inner.P[s][a]: # type: ignore
                P[s, a, s_next] += prob
                R[s, a] += prob * reward # expected reward
    return jnp.asarray(P), jnp.asarray(R)


def evaluate(env_func: Callable[[], EnvPair], alg: Algorithm) -> None:
    env, env_h = env_func()
    print(f"\n[*] Env: {env.spec.id} | Alg: {alg.value}") # type: ignore

    P, R = extract_mdp_from_gym_env(env)
    env.close()

    if alg == Algorithm.VALUEITERATION:
        _, policy, converged = value_iteration(P, R, gamma=0.9, theta=1e-6, max_iter=1000)
    else:
        _, policy, converged = policy_iteration(
            P, R, gamma=0.9, theta=1e-6, max_eval_iter=1000, max_impr_iter=100
        )
    assert bool(converged), f"{alg.value} did not converge"

    policy_np = np.asarray(policy)
    state, _ = env_h.reset(seed=42)
    total_reward = 0.0
    steps = 0
    while True:
        action = int(policy_np[state])
        state, reward, terminated, truncated, _ = env_h.step(action)
        total_reward += float(reward)
        steps += 1
        print(f"  step {steps}: action={action} reward={reward:.1f} state={state}")
        if terminated or truncated:
            break

    print(f"[+] Episode finished | total_reward={total_reward:.1f} steps={steps}")
    env_h.close()


def main() -> None:
    evaluate(frozen_lake_d_env, Algorithm.VALUEITERATION)
    evaluate(frozen_lake_d_env, Algorithm.POLICYITERATION)
    evaluate(taxi_env, Algorithm.VALUEITERATION)
    evaluate(taxi_env, Algorithm.POLICYITERATION)
    evaluate(frozen_lake_s_env, Algorithm.VALUEITERATION)
    evaluate(frozen_lake_s_env, Algorithm.POLICYITERATION)


if __name__ == "__main__":
    main()
