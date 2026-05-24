"""Semi-gradient linear SARSA with state aggregation, for CartPole.

State aggregation flattens the 4-D continuous observation into a single bin
index; the linear approximator then has shape [n_bins_total, n_actions], so
with one-hot features it reduces to a tabular Q over the binned state space.
"""
from __future__ import annotations

from functools import partial
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np


_OBS_LOW = jnp.asarray([-4.8, -3.0, -0.418, -3.5], dtype=jnp.float32)
_OBS_HIGH = jnp.asarray([4.8, 3.0, 0.418, 3.5], dtype=jnp.float32)


def _multipliers(n_bins: Tuple[int, ...]) -> jnp.ndarray:
    """Row-major flat-index multipliers for a multi-dim bin grid."""
    mults = np.concatenate([[1], np.cumprod(np.asarray(n_bins[:-1]))])
    return jnp.asarray(mults, dtype=jnp.int32)


@jax.jit
def _state_feature(state, obs_low, obs_high, n_bins_arr, multipliers):
    ratios = jnp.clip((state - obs_low) / (obs_high - obs_low), 0.0, 0.9999)
    bins = (ratios * n_bins_arr).astype(jnp.int32)
    return jnp.sum(bins * multipliers).astype(jnp.int32)


@partial(jax.jit, static_argnames=())
def _linear_sarsa_step(
    W,
    key,
    state,
    action,
    reward,
    next_state,
    done,
    eps,
    alpha,
    gamma,
    obs_low,
    obs_high,
    n_bins_arr,
    multipliers,
):
    feat = _state_feature(state, obs_low, obs_high, n_bins_arr, multipliers)
    next_feat = _state_feature(next_state, obs_low, obs_high, n_bins_arr, multipliers)

    key_e, key_a = jax.random.split(key)
    n_actions = W.shape[1]
    explore = jax.random.uniform(key_e) < eps
    rand_a = jax.random.randint(key_a, (), 0, n_actions)
    greedy_a = jnp.argmax(W[next_feat])
    next_action = jnp.where(explore, rand_a, greedy_a).astype(jnp.int32)

    q_curr = W[feat, action]
    q_next = W[next_feat, next_action]
    target = reward + gamma * q_next * (1.0 - done)
    td = target - q_curr
    W = W.at[feat, action].add(alpha * td)
    return W, next_action


def fit_semi_gradient_sarsa(
    env,
    *,
    episodes: int,
    n_bins: Tuple[int, int, int, int] = (6, 12, 6, 12),
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon: float = 0.1,
    seed: int = 0,
) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
    """Train semi-gradient linear SARSA on a continuous-state env.

    Returns (weight matrix [n_features, n_actions], greedy policy over bins, history).
    """
    n_actions = env.action_space.n
    n_features = int(np.prod(n_bins))
    W = jnp.zeros((n_features, n_actions))

    n_bins_arr = jnp.asarray(n_bins, dtype=jnp.int32)
    multipliers = _multipliers(n_bins)

    key = jax.random.PRNGKey(seed)
    alpha_a = jnp.float32(alpha)
    gamma_a = jnp.float32(gamma)
    eps_a = jnp.float32(epsilon)

    history = {"episode_rewards": [], "episode_lengths": []}

    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0.0
        steps = 0

        key, action_key = jax.random.split(key)
        feat = _state_feature(jnp.asarray(state, dtype=jnp.float32),
                              _OBS_LOW, _OBS_HIGH, n_bins_arr, multipliers)
        explore = jax.random.uniform(action_key) < eps_a
        rand_a = jax.random.randint(action_key, (), 0, n_actions)
        greedy_a = jnp.argmax(W[feat])
        action = int(jnp.where(explore, rand_a, greedy_a))

        while not (done or truncated):
            next_state, reward, done, truncated, _ = env.step(action)
            done_f = float(done or truncated)

            key, step_key = jax.random.split(key)
            W, next_action = _linear_sarsa_step(
                W,
                step_key,
                jnp.asarray(state, dtype=jnp.float32),
                jnp.int32(action),
                jnp.float32(reward),
                jnp.asarray(next_state, dtype=jnp.float32),
                jnp.float32(done_f),
                eps_a,
                alpha_a,
                gamma_a,
                _OBS_LOW,
                _OBS_HIGH,
                n_bins_arr,
                multipliers,
            )

            state = next_state
            action = int(next_action)
            total_reward += float(reward)
            steps += 1

        history["episode_rewards"].append(total_reward)
        history["episode_lengths"].append(steps)

    policy = jnp.argmax(W, axis=1).astype(jnp.int32)
    return W, policy, history


def feature_index(state, n_bins: Tuple[int, ...]) -> int:
    """Public helper: map a raw CartPole state to its discrete bin index."""
    return int(
        _state_feature(
            jnp.asarray(state, dtype=jnp.float32),
            _OBS_LOW,
            _OBS_HIGH,
            jnp.asarray(n_bins, dtype=jnp.int32),
            _multipliers(n_bins),
        )
    )
