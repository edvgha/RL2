"""
Q-Learning (off-policy TD(0)) and Double-Q-Learning.

Q-Learning:
    ε > 0, α ∈ (0, 1]
    Init Q(S, A), Q(terminal, *) = 0

    Loop for each episode:
        Initialize S
        Loop for each step of episode:
            Choose A from S using policy derived from Q (e.g. ε-greedy)
            Take action A, observe R, S'
            Q(S, A) <- Q(S, A) + α [R + γ max_a Q(S', a) - Q(S, A)]
            S <- S'
        Until S is terminal

Double-Q-Learning:
    Init Q1(S, A), Q2(S, A), both = 0

    Loop for each episode:
        Initialize S
        Loop for each step of episode:
            Choose A from S using policy derived from Q1 + Q2 (e.g. ε-greedy)
            Take action A, observe R, S'
            With prob. 0.5:
                Q1(S, A) <- Q1(S, A) + α [R + γ Q2(S', argmax_a Q1(S', a)) - Q1(S, A)]
            else:
                Q2(S, A) <- Q2(S, A) + α [R + γ Q1(S', argmax_a Q2(S', a)) - Q2(S, A)]
            S <- S'
        Until S is terminal

The per-step TD updates are pure jit-compiled functions; the per-episode
loop stays in Python because Gymnasium env.step is impure.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp

from .exploration import epsilon_at, epsilon_greedy


class QLearningType(Enum):
    QLEARNING = "Q-Learning"
    DOUBLEQLEARNING = "Double-Q-Learning"


@jax.jit
def q_learning_update(Q, state, action, reward, next_state, done, alpha, gamma):
    """Q(s,a) <- Q(s,a) + alpha [r + gamma * max_{a'} Q(s',a') - Q(s,a)]."""
    target = reward + gamma * jnp.max(Q[next_state]) * (1.0 - done)
    td_error = target - Q[state, action]
    return Q.at[state, action].add(alpha * td_error)


@jax.jit
def double_q_learning_update(Q1, Q2, key, state, action, reward, next_state, done, alpha, gamma):
    """Flip a coin: update Q1 using Q2's value at Q1's argmax, or vice versa."""
    flip = jax.random.uniform(key) < 0.5

    def update_first(qs):
        Q1, Q2 = qs
        best = jnp.argmax(Q1[next_state])
        target = reward + gamma * Q2[next_state, best] * (1.0 - done)
        td = target - Q1[state, action]
        return Q1.at[state, action].add(alpha * td), Q2

    def update_second(qs):
        Q1, Q2 = qs
        best = jnp.argmax(Q2[next_state])
        target = reward + gamma * Q1[next_state, best] * (1.0 - done)
        td = target - Q2[state, action]
        return Q1, Q2.at[state, action].add(alpha * td)

    return jax.lax.cond(flip, update_first, update_second, (Q1, Q2))


def fit_q_learning(
    env,
    qtype: QLearningType,
    *,
    episodes: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    epsilon_min: float,
    decay_rate: float,
    seed: int = 0,
) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
    """Train tabular (Double-)Q-Learning. Returns (Q-table, greedy policy, history)."""
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q1 = jnp.zeros((n_states, n_actions))
    Q2 = jnp.zeros((n_states, n_actions)) if qtype == QLearningType.DOUBLEQLEARNING else None

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
            q_for_action = (Q1 + Q2)[state] if qtype == QLearningType.DOUBLEQLEARNING else Q1[state]
            action = int(epsilon_greedy(action_key, q_for_action, eps))

            next_state, reward, done, truncated, _ = env.step(action)
            done_f = float(done or truncated)

            s, a, ns = jnp.int32(state), jnp.int32(action), jnp.int32(next_state)
            r = jnp.float32(reward)
            d = jnp.float32(done_f)

            if qtype == QLearningType.DOUBLEQLEARNING:
                key, upd_key = jax.random.split(key)
                Q1, Q2 = double_q_learning_update(Q1, Q2, upd_key, s, a, r, ns, d, alpha_a, gamma_a)
            else:
                Q1 = q_learning_update(Q1, s, a, r, ns, d, alpha_a, gamma_a)

            state = next_state
            total_reward += float(reward)
            steps += 1

        history["episode_rewards"].append(total_reward)
        history["episode_lengths"].append(steps)
        history["epsilons"].append(eps)

    if qtype == QLearningType.DOUBLEQLEARNING:
        Q = Q1 + Q2
    else:
        Q = Q1
    policy = jnp.argmax(Q, axis=1).astype(jnp.int32)
    return Q, policy, history
