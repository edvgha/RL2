"""
SARSA (on-policy TD(0)) and Expected SARSA.

SARSA:
    ε > 0, α ∈ (0, 1]
    Init Q(S, A), Q(terminal, *) = 0

    Loop for each episode:
        Initialize S
        Choose A from S using policy derived from Q (e.g. ε-greedy)
        Loop for each step of episode:
            Take action A, observe R, S'
            Choose A' from S' using policy derived from Q (e.g. ε-greedy)
            Q(S, A) <- Q(S, A) + α [R + γ Q(S', A') - Q(S, A)]
            S <- S'
            A <- A'
        Until S is terminal

Expected SARSA: same loop, but the bootstrap term is the expected next-Q
under the current ε-greedy policy, not a sampled A':

    Q(S, A) <- Q(S, A) + α [R + γ E_π[Q(S', A') | S'] - Q(S, A)]

with  E_π[Q(S', A') | S'] = Σ_a π(a|S') Q(S', a)
and   π(a|S') = (1 - ε) + ε/|A|     if a = argmax_a Q(S', a)
              = ε/|A|               otherwise.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp

from .exploration import epsilon_at, epsilon_greedy


class SARSAType(Enum):
    SARSA = "SARSA"
    ESARSA = "Expected-SARSA"


@jax.jit
def sarsa_update(Q, state, action, reward, next_state, next_action, done, alpha, gamma):
    """Q(s,a) <- Q(s,a) + alpha [r + gamma * Q(s',a') - Q(s,a)]."""
    target = reward + gamma * Q[next_state, next_action] * (1.0 - done)
    td_error = target - Q[state, action]
    return Q.at[state, action].add(alpha * td_error)


@jax.jit
def expected_sarsa_update(Q, state, action, reward, next_state, done, eps, alpha, gamma):
    """Bootstraps off the expected next-Q under the current eps-greedy policy."""
    q_next = Q[next_state]
    n_actions = q_next.shape[0]
    best = jnp.argmax(q_next)
    non_greedy = eps / n_actions
    greedy = (1.0 - eps) + non_greedy
    probs = jnp.where(jnp.arange(n_actions) == best, greedy, non_greedy)
    expected_q = jnp.sum(probs * q_next)
    target = reward + gamma * expected_q * (1.0 - done)
    td_error = target - Q[state, action]
    return Q.at[state, action].add(alpha * td_error)


def _fit_sarsa(env, *, episodes, alpha, gamma, epsilon, epsilon_min, decay_rate, seed):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = jnp.zeros((n_states, n_actions))

    key = jax.random.PRNGKey(seed)
    alpha_a = jnp.float32(alpha)
    gamma_a = jnp.float32(gamma)
    history = {"episode_rewards": [], "episode_lengths": [], "epsilons": []}

    for episode in range(episodes):
        eps = float(epsilon_at(jnp.float32(episode), epsilon, epsilon_min, decay_rate))
        state, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0.0
        steps = 0

        key, action_key = jax.random.split(key)
        action = int(epsilon_greedy(action_key, Q[state], eps))

        while not (done or truncated):
            next_state, reward, done, truncated, _ = env.step(action)
            done_f = float(done or truncated)

            key, action_key = jax.random.split(key)
            next_action = int(epsilon_greedy(action_key, Q[next_state], eps))

            Q = sarsa_update(
                Q,
                jnp.int32(state),
                jnp.int32(action),
                jnp.float32(reward),
                jnp.int32(next_state),
                jnp.int32(next_action),
                jnp.float32(done_f),
                alpha_a,
                gamma_a,
            )

            state = next_state
            action = next_action
            total_reward += float(reward)
            steps += 1

        history["episode_rewards"].append(total_reward)
        history["episode_lengths"].append(steps)
        history["epsilons"].append(eps)

    return Q, jnp.argmax(Q, axis=1).astype(jnp.int32), history


def _fit_expected_sarsa(env, *, episodes, alpha, gamma, epsilon, epsilon_min, decay_rate, seed):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = jnp.zeros((n_states, n_actions))

    key = jax.random.PRNGKey(seed)
    alpha_a = jnp.float32(alpha)
    gamma_a = jnp.float32(gamma)
    history = {"episode_rewards": [], "episode_lengths": [], "epsilons": []}

    for episode in range(episodes):
        eps = float(epsilon_at(jnp.float32(episode), epsilon, epsilon_min, decay_rate))
        state, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0.0
        steps = 0

        while not (done or truncated):
            key, action_key = jax.random.split(key)
            action = int(epsilon_greedy(action_key, Q[state], eps))

            next_state, reward, done, truncated, _ = env.step(action)
            done_f = float(done or truncated)

            Q = expected_sarsa_update(
                Q,
                jnp.int32(state),
                jnp.int32(action),
                jnp.float32(reward),
                jnp.int32(next_state),
                jnp.float32(done_f),
                jnp.float32(eps),
                alpha_a,
                gamma_a,
            )

            state = next_state
            total_reward += float(reward)
            steps += 1

        history["episode_rewards"].append(total_reward)
        history["episode_lengths"].append(steps)
        history["epsilons"].append(eps)

    return Q, jnp.argmax(Q, axis=1).astype(jnp.int32), history


def fit_sarsa(
    env,
    stype: SARSAType,
    *,
    episodes: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    epsilon_min: float,
    decay_rate: float,
    seed: int = 0,
) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
    """Train tabular SARSA or Expected SARSA. Returns (Q, greedy policy, history)."""
    impl = _fit_sarsa if stype == SARSAType.SARSA else _fit_expected_sarsa
    return impl(
        env,
        episodes=episodes,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        decay_rate=decay_rate,
        seed=seed,
    )
