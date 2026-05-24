# RL2: Reinforcement Learning Algorithms in JAX

Classic tabular and linear-features Reinforcement Learning algorithms,
implemented in functional [JAX](https://jax.readthedocs.io/) and exercised
on [Gymnasium](https://gymnasium.farama.org/) toy environments.

Per-step TD updates and the model-based fixed-point iterations are pure
jit-compiled functions; the per-episode loop stays in Python because
Gymnasium's `env.step` is not JAX-pure.

## Algorithms

| Family       | Algorithm                       | Module                                                                    |
| ------------ | ------------------------------- | ------------------------------------------------------------------------- |
| Model-based  | Value Iteration                 | [src/rl2/model_based/value_iteration.py](src/rl2/model_based/value_iteration.py)         |
| Model-based  | Policy Iteration                | [src/rl2/model_based/policy_iteration.py](src/rl2/model_based/policy_iteration.py)       |
| Model-free   | Q-Learning, Double-Q-Learning   | [src/rl2/model_free/q_learning.py](src/rl2/model_free/q_learning.py)                     |
| Model-free   | SARSA, Expected SARSA           | [src/rl2/model_free/sarsa.py](src/rl2/model_free/sarsa.py)                               |
| Approximate  | Semi-gradient linear SARSA      | [src/rl2/model_free/semi_gradient_sarsa.py](src/rl2/model_free/semi_gradient_sarsa.py)   |

Each module's docstring carries the pseudocode and Bellman/TD update
formulas in `Σ / γ / α / ε` notation.

## Project layout

```
src/rl2/
├── envs.py                       # Gym env factories (FrozenLake, CliffWalking, Taxi, CartPole)
├── model_based/
│   ├── value_iteration.py
│   └── policy_iteration.py
├── model_free/
│   ├── exploration.py            # ε-greedy + ε decay schedule
│   ├── q_learning.py             # Q-Learning + Double-Q-Learning
│   ├── sarsa.py                  # SARSA + Expected SARSA
│   └── semi_gradient_sarsa.py    # Linear SARSA with state aggregation (CartPole)
└── scripts/
    ├── evaluate_vi_pi.py         # VI / PI on FrozenLake + Taxi
    ├── evaluate_model_free.py    # Q / Double-Q / SARSA / E-SARSA on FrozenLake
    └── evaluate_cartpole.py      # Semi-gradient linear SARSA on CartPole
```

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/your-username/RL2.git
cd RL2
uv sync
```

`uv sync` creates `.venv/` and installs locked dependencies from `uv.lock`
(JAX, Gymnasium, Matplotlib, Pygame).

## Running

The package exposes three console scripts:

```bash
uv run rl2-vi-pi          # Value Iteration & Policy Iteration on FrozenLake / Taxi
uv run rl2-model-free     # Q / Double-Q / SARSA / Expected-SARSA on slippery FrozenLake
uv run rl2-cartpole       # Semi-gradient linear SARSA on CartPole-v1
```

Each script trains headless, then rolls out the greedy policy in a
human-rendered Gymnasium window (CartPole prints a scatter plot of the
reward curve instead).

## Library usage

```python
import jax.numpy as jnp
from rl2.model_based.value_iteration import value_iteration
from rl2.model_free.q_learning import QLearningType, fit_q_learning

# Model-based: given a known MDP (P, R)
V, policy, converged = value_iteration(P, R, gamma=0.9, theta=1e-6)

# Model-free: given a Gymnasium env
import gymnasium as gym
env = gym.make("FrozenLake-v1", is_slippery=True)
Q, policy, history = fit_q_learning(
    env, QLearningType.QLEARNING,
    episodes=3000, alpha=0.1, gamma=0.99,
    epsilon=1.0, epsilon_min=0.1, decay_rate=1e-4, seed=0,
)
```
