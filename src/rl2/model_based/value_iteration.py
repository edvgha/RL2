"""
Value Iteration

Bellman contraction operator:
    V(s) = max_a[Σ_s' P(s'|s,a) * [R(s,a) + γ * V(s')]]

Algorithm:
    Init: V_0
    Loop:
        V_k+1 = B[V_k]
    Until: |V_k+1 - V_k| < ε

    π(a|s) = argmax_a(Σ_s' P(s'|s,a) * [R(s,a) + γ * V_k+1(s')])

Implementation note: the fixed-point iteration is expressed as a single
jax.lax.while_loop, and the entire routine is jit-compiled.
"""
from __future__ import annotations

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=("max_iter",))
def value_iteration(
    P: jnp.ndarray,
    R: jnp.ndarray,
    gamma: float,
    theta: float,
    max_iter: int = 1000,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Returns (V*, greedy policy, converged_flag).

    P: [S, A, S]  transition probabilities
    R: [S, A]     expected one-step reward
    """
    n_states = R.shape[0]

    def cond_fn(carry):
        i, V, V_prev = carry
        return (i < max_iter) & (jnp.max(jnp.abs(V - V_prev)) >= theta)

    def body_fn(carry):
        i, V, _ = carry
        # Q[s,a] = R[s,a] + gamma * sum_{s'} P[s,a,s'] V[s']
        Q = R + gamma * (P @ V)
        V_new = jnp.max(Q, axis=1)
        return (i + 1, V_new, V)

    V0 = jnp.zeros(n_states)
    init = (jnp.int32(0), V0, V0 - jnp.inf)  # V_prev = -inf forces first iteration
    i, V, _ = jax.lax.while_loop(cond_fn, body_fn, init)

    Q = R + gamma * (P @ V)
    policy = jnp.argmax(Q, axis=1).astype(jnp.int32)
    converged = i < max_iter
    return V, policy, converged
