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
    """Sample an action under an eps-greedy policy on `q_values`.

    Greedy ties are broken uniformly at random, not by argmax. This matters
    a lot at cold-start: Q is initialized to zeros, so without random tie-
    breaking the greedy action is always 0 (e.g. LEFT in FrozenLake), and
    on sparse-reward envs the agent can get stuck at the start state and
    never reach the goal.
    """
    key_e, key_a, key_t = jax.random.split(key, 3)
    n_actions = q_values.shape[0]
    explore = jax.random.uniform(key_e) < eps
    random_action = jax.random.randint(key_a, (), 0, n_actions)
    max_q = jnp.max(q_values)
    is_max = q_values == max_q
    probs = is_max / jnp.sum(is_max)
    greedy_action = jax.random.choice(key_t, n_actions, p=probs)
    return jnp.where(explore, random_action, greedy_action).astype(jnp.int32)
