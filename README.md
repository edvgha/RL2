# RL2: Reinforcement Learning Algorithms

Classic tabular and deep Reinforcement Learning algorithms, implemented in pure Python (NumPy) and [PyTorch](https://pytorch.org/), and exercised on [Gymnasium](https://gymnasium.farama.org/) toy environments.

The tabular methods and linear-feature algorithms leverage standard NumPy for efficient TD updates and fixed-point iterations. The deep reinforcement learning algorithms (Deep Q-Learning, REINFORCE, Actor-Critic) utilize PyTorch to implement deep neural networks for robust function approximation.

## Algorithms

| Family                 | Algorithm                       | Module                                                                                                 |
| ---------------------- | ------------------------------- | ------------------------------------------------------------------------------------------------------ |
| **Model-based** | Value Iteration                 | [src/rl2/model_based/value_iteration.py](src/rl2/model_based/value_iteration.py)                       |
| **Model-based** | Policy Iteration                | [src/rl2/model_based/policy_iteration.py](src/rl2/model_based/policy_iteration.py)                     |
| **Model-free** | Q-Learning, Double-Q-Learning   | [src/rl2/model_free/q_learning.py](src/rl2/model_free/q_learning.py)                                   |
| **Model-free** | SARSA, Expected SARSA           | [src/rl2/model_free/sarsa.py](src/rl2/model_free/sarsa.py)                                             |
| **Linear Approx** | Semi-gradient linear SARSA      | [src/rl2/model_free/semi_gradient_sarsa.py](src/rl2/model_free/semi_gradient_sarsa.py)                 |
| **Value Approx (DNN)** | SARSA, Q-Learning, Off-Pol DQN  | [src/rl2/value_approximation/](src/rl2/value_approximation/)                                           |
| **Policy Gradient** | REINFORCE                       | [src/rl2/policy_approximation/reinforce.py](src/rl2/policy_approximation/reinforce.py)                 |
| **Actor-Critic** | QAC, A2C, Off-Policy A2C        | [src/rl2/actor_critic/](src/rl2/actor_critic/)                                                         |

Each module's docstring or paired Jupyter notebook carries the pseudocode and Bellman/TD update formulas.

## Project layout

```text
src/rl2/
├── envs.py                         # Gym env factories (FrozenLake, CartPole, etc.)
├── model_based/                    # Tabular Planners
│   ├── value_iteration.py
│   └── policy_iteration.py
├── model_free/                     # Tabular RL
│   ├── exploration.py              # ε-greedy + ε decay schedule
│   ├── q_learning.py               
│   ├── sarsa.py                    
│   └── semi_gradient_sarsa.py      
├── value_approximation/            # PyTorch Deep Value Methods
│   ├── sarsa_q_value_approximation.py
│   ├── q_learning_q_value_approximation.py
│   └── deep_q_learning_off_policy.py
├── policy_approximation/           # PyTorch Policy Gradients
│   └── reinforce.py
├── actor_critic/                   # PyTorch Actor-Critic Methods
│   ├── qac.py
│   ├── a2c.py
│   └── off_policy_a2c.py
└── scripts/                        # Execution & Evaluation scripts
    ├── evaluate_vi_pi.py           
    ├── evaluate_model_free.py      
    ├── evaluate_cartpole.py        
    ├── evaluate_value_approximation.py
    ├── evaluate_reinforce.py
    └── evaluate_actor_critic.py
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

uv run python -m rl2.scripts.evaluate_value_approximation
uv run python -m rl2.scripts.evaluate_reinforce
uv run python -m rl2.scripts.evaluate_actor_critic
```

## Library usage

```python
import numpy as np
import gymnasium as gym
from rl2.model_free.q_learning import QLearningType, fit_q_learning

env = gym.make("FrozenLake-v1", is_slippery=True)
Q, policy, history = fit_q_learning(
    env, QLearningType.QLEARNING,
    episodes=3000, alpha=0.1, gamma=0.99,
    epsilon=1.0, epsilon_min=0.1, decay_rate=1e-4, seed=0,
)
```

```python
import gymnasium as gym
from rl2.actor_critic.a2c import A2C

env = gym.make("FrozenLake-v1", is_slippery=False)
agent = A2C(env, dims=[8, 8], gamma=0.9, alpha_theta=0.001, alpha_w=0.001)
optimal_policy, state_values = agent()
```
