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
        π'(s) = argmax_a Σ_s' P(s'|s,a) * [R(s, a, s') + γ * V(s')]

Until π = π'

Implementation note: the eval fixed-point and the outer improvement loop are
both jax.lax.while_loop calls under a single jit.
"""
from __future__ import annotations

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=("max_eval_iter", "max_impr_iter"))
def policy_iteration(
    P: jnp.ndarray,
    R: jnp.ndarray,
    gamma: float,
    theta: float,
    max_eval_iter: int = 1000,
    max_impr_iter: int = 100,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Returns (V*, optimal policy, stable_flag)."""
    n_states = R.shape[0]
    state_idx = jnp.arange(n_states)

    def policy_eval(V_init, policy):
        P_pi = P[state_idx, policy]  # [S, S]
        R_pi = R[state_idx, policy]  # [S]

        def cond_fn(carry):
            i, V, V_prev = carry
            return (i < max_eval_iter) & (jnp.max(jnp.abs(V - V_prev)) >= theta)

        def body_fn(carry):
            i, V, _ = carry
            V_new = R_pi + gamma * (P_pi @ V)
            return (i + 1, V_new, V)

        init = (jnp.int32(0), V_init, V_init - jnp.inf)
        _, V, _ = jax.lax.while_loop(cond_fn, body_fn, init)
        return V

    def policy_improve(V):
        Q = R + gamma * (P @ V)
        return jnp.argmax(Q, axis=1).astype(jnp.int32)

    def cond_fn(carry):
        i, _, _, stable = carry
        return (i < max_impr_iter) & jnp.logical_not(stable)

    def body_fn(carry):
        i, V, policy, _ = carry
        V_new = policy_eval(V, policy)
        new_policy = policy_improve(V_new)
        stable = jnp.all(new_policy == policy)
        return (i + 1, V_new, new_policy, stable)

    init = (
        jnp.int32(0),
        jnp.zeros(n_states),
        jnp.zeros(n_states, dtype=jnp.int32),
        jnp.bool_(False),
    )
    _, V, policy, stable = jax.lax.while_loop(cond_fn, body_fn, init)
    return V, policy, stable
