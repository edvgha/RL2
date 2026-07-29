# RL2: Reinforcement Learning Algorithms

Classic tabular and deep Reinforcement Learning algorithms, implemented with NumPy, [JAX](https://jax.readthedocs.io/), and [PyTorch](https://pytorch.org/), and exercised on [Gymnasium](https://gymnasium.farama.org/) toy environments.

The tabular methods use NumPy for TD updates and fixed-point iterations; the semi-gradient linear method is written in functional JAX; and the function-approximation algorithms (Deep Q-Learning, REINFORCE, Actor-Critic, Deterministic Actor-Critic) use PyTorch.

Every solver follows the same shape: construct it with a Gymnasium env plus hyperparameters, call the instance, and read `solver.stats` for the training curves.

## Algorithms

| Family | Algorithm | Module |
| --- | --- | --- |
| **Model-based** | Value Iteration | [model_based/value_iteration.py](src/rl2/model_based/value_iteration.py) |
| **Model-based** | Policy Iteration | [model_based/policy_iteration.py](src/rl2/model_based/policy_iteration.py) |
| **Model-free (tabular)** | Q-Learning | [model_free/q_learning.py](src/rl2/model_free/q_learning.py) |
| **Model-free (tabular)** | SARSA / Expected SARSA (`use_esarsa`) | [model_free/sarsa.py](src/rl2/model_free/sarsa.py) |
| **Linear approximation** | Semi-gradient linear SARSA (JAX, tile-binned CartPole) | [model_free/semi_gradient_sarsa.py](src/rl2/model_free/semi_gradient_sarsa.py) |
| **Value approximation** | SARSA with q(s,a,w) | [value_approximation/sarsa_q_value_approximation.py](src/rl2/value_approximation/sarsa_q_value_approximation.py) |
| **Value approximation** | Q-Learning with q(s,a,w) | [value_approximation/q_learning_q_value_approximation.py](src/rl2/value_approximation/q_learning_q_value_approximation.py) |
| **Value approximation** | Deep Q-Learning, uniform behavior policy (replay buffer + target net) | [value_approximation/deep_q_learning_off_policy.py](src/rl2/value_approximation/deep_q_learning_off_policy.py) |
| **Policy gradient** | REINFORCE (Monte-Carlo policy gradient) | [policy_approximation/reinforce.py](src/rl2/policy_approximation/reinforce.py) |
| **Actor-Critic** | QAC (the simplest actor-critic) | [actor_critic/qac.py](src/rl2/actor_critic/qac.py) |
| **Actor-Critic** | A2C (advantage actor-critic) | [actor_critic/a2c.py](src/rl2/actor_critic/a2c.py) |
| **Actor-Critic** | Off-policy A2C (importance-weighted, uniform β) | [actor_critic/off_policy_a2c.py](src/rl2/actor_critic/off_policy_a2c.py) |
| **Actor-Critic** | Deterministic actor-critic — softmax relaxation of µ(s,θ) | [actor_critic/deterministic_actor_critic.py](src/rl2/actor_critic/deterministic_actor_critic.py) |
| **Actor-Critic** | Deterministic actor-critic — scalar action axis, ε-greedy β, target critic | [actor_critic/deterministic_actor_critic_2.py](src/rl2/actor_critic/deterministic_actor_critic_2.py) |

Each module's docstring or paired Jupyter notebook carries the pseudocode and the Bellman/TD update formulas it implements.

### The two deterministic actor-critic variants

Both implement the same deterministic policy gradient update

$$\theta_{t+1} = \theta_t + \alpha_\theta \nabla_\theta \mu(s_t,\theta_t) \nabla_a q(s_t,a,w_t)\big|_{a=\mu(s_t)}$$

but they differ in how a *deterministic, continuous* action is defined over a Discrete action space:

- **`deterministic_actor_critic.py`** — µ(s,θ) is the softmax probability vector itself (one-hot mode) or its expected scalar (quadratic mode). Actions keep their simplex geometry, so no artificial ordering is imposed; the executed action is drawn from a uniform behavior policy β.
- **`deterministic_actor_critic_2.py`** — µ(s,θ) is a single `Tanh` scalar on the normalized axis `[-1, 1]`, rounded to the nearest action index when stepping the env. β is ε-greedy (`epsilon=1.0` recovers the uniform β), and the TD target is bootstrapped from a frozen target critic synced every `target_update_freq` iterations — off-policy bootstrapping at the greedy µ(s′) otherwise diverges. Trade-off: collapsing the actions onto one axis imposes an ordering, so ∇ₐq can only push µ toward *adjacent* actions.

## Project layout

```text
src/rl2/
├── envs.py                         # Gym env factories: FrozenLake (8x8, custom 5x5), CliffWalking, Taxi
├── model_based/                    # Tabular planners (NumPy)
│   ├── value_iteration.py
│   └── policy_iteration.py
├── model_free/                     # Tabular RL
│   ├── exploration.py              # ε-greedy + ε decay schedule
│   ├── q_learning.py
│   ├── sarsa.py                    # SARSA and Expected SARSA
│   └── semi_gradient_sarsa.py      # JAX, continuous-state (CartPole)
├── value_approximation/            # PyTorch value methods
│   ├── sarsa_q_value_approximation.py
│   ├── q_learning_q_value_approximation.py
│   └── deep_q_learning_off_policy.py
├── policy_approximation/           # PyTorch policy gradients
│   ├── reinforce.py
│   └── reinforce2.py
├── actor_critic/                   # PyTorch actor-critic methods
│   ├── qac.py
│   ├── a2c.py
│   ├── off_policy_a2c.py
│   ├── deterministic_actor_critic.py
│   └── deterministic_actor_critic_2.py
└── scripts/                        # Execution & evaluation entry points
    ├── evaluate_vi_pi.py
    ├── evaluate_model_free.py
    ├── evaluate_cartpole.py
    ├── evaluate_value_approximation.py
    ├── evaluate_reinforce.py
    └── evaluate_actor_critic.py
```

Most modules ship with a sibling `.ipynb` notebook holding the algorithm's pseudocode
(`a2c.ipynb`, `deterministic_actor_critic.ipynb`, `off_policy_a2c.ipynb`, `qac.ipynb`,
`q_learning.ipynb`, `sarsa.ipynb`, `MC_eps_greedy.ipynb`, `linear_value_function.ipynb`,
`value_iteration.ipynb`, `policy_iteration.ipynb`, `reinforce.ipynb`, and the
`value_approximation/` notebooks).

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/edvgha/RL2.git
cd RL2
uv sync
```

`uv sync` creates `.venv/` and installs locked dependencies from `uv.lock`
(Gymnasium, JAX, Matplotlib, Pygame, PyTorch).

## Running

Three console scripts are exposed by `pyproject.toml`:

```bash
uv run rl2-vi-pi          # Value Iteration & Policy Iteration on FrozenLake / Taxi
uv run rl2-model-free     # Q-Learning / SARSA / Expected-SARSA on slippery FrozenLake
uv run rl2-cartpole       # Semi-gradient linear SARSA on CartPole-v1
```

The remaining evaluation scripts run as modules:

```bash
uv run python -m rl2.scripts.evaluate_value_approximation
uv run python -m rl2.scripts.evaluate_reinforce
uv run python -m rl2.scripts.evaluate_actor_critic
```

Each evaluation script trains a solver, replays the learned policy in a human-rendered
env, then plots the reward/length curves and a 3D surface of the state-value function.
Pick which algorithm and environment to run by editing the `evaluate(...)` calls in that
script's `main()` — they are commented in and out rather than parsed from the CLI.

## Library usage

Tabular Q-Learning:

```python
import gymnasium as gym
from rl2.model_free.q_learning import QLearning

env = gym.make("FrozenLake-v1", is_slippery=True)
solver = QLearning(env, gamma=0.9, alpha=0.1, epsilon=1.0,
                   decay_rate=0.001, min_epsilon=0.01, max_episodes=5000)
policy = solver()
rewards = solver.stats["total_rewards"]
```

Actor-critic on the deterministic 8x8 FrozenLake:

```python
from rl2.envs import frozen_lake_d_env
from rl2.actor_critic.a2c import A2C

env, env_human = frozen_lake_d_env()
agent = A2C(env, dims=[8, 8], gamma=0.9, alpha_theta=0.001, alpha_w=0.001)
optimal_policy, state_values = agent()
```

Deterministic policy gradient with one-hot features:

```python
from rl2.envs import frozen_lake_d_env
from rl2.actor_critic.deterministic_actor_critic_2 import DPG

env, env_human = frozen_lake_d_env()
agent = DPG(env, dims=[8, 8], epsilon=0.3, max_episodes=5000,
            use_one_hot_features=True)
optimal_policy, state_values = agent()
```

Model-based planning:

```python
from rl2.envs import frozen_lake_s_env
from rl2.model_based.value_iteration import ValueIteration

env, env_human = frozen_lake_s_env()
v, policy, converged = ValueIteration(env=env, gamma=0.99, theta=1e-6)()
```

## Features

Discrete-grid solvers accept a `dims=[width, height]` argument and support two feature
encodings, selected with `use_one_hot_features`:

- **Quadratic** — grid coordinates normalized to `[-1, 1]`, expanded as
  `[1, x, y, a, x², y², a², xy, xa, ya]` for state-action inputs and
  `[1, x, y, x², y², xy]` for state-only inputs. Generalizes across nearby cells.
- **One-hot** — a plain indicator vector over states (concatenated with the action
  indicator where the network takes a state-action pair). No generalization, but exact.

## License

MIT — see [LICENSE](LICENSE).
