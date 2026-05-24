"""Epsilon-greedy action selection and epsilon decay schedule."""
from __future__ import annotations

import jax
import jax.numpy as jnp


@jax.jit
def epsilon_at(episode, eps0, eps_min, decay):
    """eps_t = max(eps_min, eps0 * exp(-decay * episode))."""
    return jnp.maximum(eps_min, eps0 * jnp.exp(-decay * episode))


@jax.jit
def epsilon_greedy(key, q_values, eps):
    """Sample an action under an eps-greedy policy on `q_values`."""
    key_e, key_a = jax.random.split(key)
    n_actions = q_values.shape[0]
    explore = jax.random.uniform(key_e) < eps
    random_action = jax.random.randint(key_a, (), 0, n_actions)
    greedy_action = jnp.argmax(q_values)
    return jnp.where(explore, random_action, greedy_action).astype(jnp.int32)
