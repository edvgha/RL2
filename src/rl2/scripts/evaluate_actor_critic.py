from __future__ import annotations

import warnings
from enum import Enum
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from rl2.envs import (
    EnvPair,
    cliff_walking_d_env,
    cliff_walking_s_env,
    frozen_lake_d_env,
    frozen_lake_s_env,
    frozen_lake_s_small_env,
    frozen_lake_d_small_env,
    taxi_env,
)
from rl2.actor_critic.qac import QAC
from rl2.actor_critic.a2c import A2C
from rl2.actor_critic.off_policy_a2c import OffPolicyA2C
from rl2.actor_critic.deterministic_actor_critic import DeterministicActorCritic
from rl2.actor_critic.deterministic_actor_critic_2 import DPG

warnings.filterwarnings("ignore")


class Control(Enum):
    QAC = "QAC"
    A2C = "A2C"
    A2COH = "A2C-OH"
    OFFPOLICYA2C = "OFF-POLICY-A2C"
    DETERMINISTIC_ACTOR_CRITIC = "DETERMINISTIC-ACTOR-CRITIC"
    DETERMINISTIC_ACTOR_CRITIC_2 = "DETERMINISTIC-ACTOR-CRITIC-2"



def plot_metrics(stats: dict, alg_name: str, env_name: str) -> None:
    """
    Plots total rewards and episode lengths on separate figures (subplots) 
    contained within the same layout window frame.
    """
    # Create 2 distinct figures stacked vertically in 1 window canvas frame
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)
    
    # 1. First Subplot Figure: Total Rewards
    ax1.plot(stats["total_rewards"], color="tab:blue", alpha=0.6)
    ax1.set_ylabel("Total Reward")
    ax1.set_title(f"{alg_name} Training Performance ({env_name})")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # 2. Second Subplot Figure: Episode Lengths
    ax2.plot(stats["episode_lengths"], color="tab:red", alpha=0.6)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Episode Length (Steps)")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(5)
    plt.close()


def plot_state_values(values: np.ndarray, alg_name: str, env_name: str, dims: list) -> None:
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Reshape the 1D values array to match the 2D grid layout dimensions[cite: 8]
    grid_values = values.reshape(dims[1], dims[0])
    
    # Create matching X and Y coordinate meshgrid matrices
    x = np.arange(dims[0])
    y = np.arange(dims[1])
    X, Y = np.meshgrid(x, y)
    
    # Generate the 3D surface plot
    surf = ax.plot_surface(X, Y, grid_values, cmap='viridis', edgecolor='none', alpha=0.7)
    
    # --- TEXT VALUE ANNOTATIONS ---
    # Loop through every grid intersection coordinate point
    for r in range(dims[1]):
        for c in range(dims[0]):
            val = grid_values[r, c]
            
            # Slightly offset the text on the Z-axis so it floats cleanly above the surface points
            # Adjust the multiplier (0.02) if your environment rewards scale radically
            z_offset = max(0.01, abs(val) * 0.02) if val != 0 else 0.01
            
            ax.text(
                x=c, 
                y=r, 
                z=val + z_offset, 
                s=f"{val:.2f}", 
                color='black', 
                fontsize=8,
                ha='center', 
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.6)
            )
    # ------------------------------

    # Camera & axis viewing layout parameters
    ax.view_init(elev=30, azim=-135)
    ax.tick_params(axis='x', rotation=15)

    # Labels and Titles
    ax.set_xlabel('X Axis (Columns)')
    ax.set_ylabel('Y Axis (Rows)')
    ax.set_zlabel('State Value V(s)')
    ax.set_title(f"{alg_name} 3D State-Value Function with Point Labels ({env_name})")
    
    # Add a color bar map to easily identify peak values
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Value Magnitude')
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(5)
    plt.close()


def evaluate(
    env_func: Callable[[], EnvPair],
    control: Control,
    episodes: int,
    dims: list
) -> None:
    env, env_h = env_func()
    env_id = getattr(env.spec, "id", "UnknownEnv")
    print(f"\n[*] Env: {env_id} | Alg: {control.value}") # type: ignore

    optimal_policy, stats, state_values = np.array([]), {}, np.array([])
    if control == Control.QAC:
        qac_solver = QAC(env, dims=dims, num_features=10, max_episodes=episodes)
        optimal_policy, state_values = qac_solver()
        stats = qac_solver.stats
    elif control == Control.A2C:
        a2c_solver = A2C(env, num_actor_features=10, num_critic_features=6, dims=dims, max_episodes=episodes)
        optimal_policy, state_values = a2c_solver()
        stats = a2c_solver.stats
    elif control == Control.A2COH:
        a2c_solver = A2C(env, dims=[], max_episodes=episodes, use_one_hot_features=True)
        optimal_policy, state_values = a2c_solver()
        stats = a2c_solver.stats
    elif control == Control.OFFPOLICYA2C:
        off_policy_a2c_solver = OffPolicyA2C(env, max_episodes=episodes)
        optimal_policy, state_values = off_policy_a2c_solver()
        stats = off_policy_a2c_solver.stats
    elif control == Control.DETERMINISTIC_ACTOR_CRITIC:
        discrete_policy_a2c_solver = DeterministicActorCritic(env, dims=dims, max_episodes=episodes)
        optimal_policy, state_values = discrete_policy_a2c_solver()
        stats = discrete_policy_a2c_solver.stats
    elif control == Control.DETERMINISTIC_ACTOR_CRITIC_2:
            dpg_solver = DPG(env, dims=dims, max_episodes=episodes)
            optimal_policy, state_values = dpg_solver()
            stats = dpg_solver.stats

    env.close()

    state, _ = env_h.reset()
    total_reward, steps = 0.0, 0
    while True:
        action = int(optimal_policy[state])
        state, reward, terminated, truncated, _ = env_h.step(action)
        total_reward += float(reward)
        steps += 1
        print(f"  step {steps}: action={action} reward={reward:.1f} state={state}")
        if terminated or truncated:
            break

    print(f"[+] Episode finished | total_reward={total_reward:.1f} steps={steps}")
    env_h.close()

    plot_metrics(stats, control.value, env_id)
    plot_state_values(state_values, control.value, env_id, dims)


def main() -> None:
    # evaluate(frozen_lake_s_env, Control.QAC, episodes=5000, dims=[8, 8])
    # evaluate(frozen_lake_s_env, Control.A2C, episodes=3000, dims=[8, 8])
    # evaluate(frozen_lake_s_env, Control.A2COH, episodes=300, dims=[8, 8])
    # evaluate(frozen_lake_s_env, Control.OFFPOLICYA2C, episodes=3000, dims=[8, 8])
    # evaluate(frozen_lake_d_env, Control.DETERMINISTIC_ACTOR_CRITIC, episodes=5000, dims=[8, 8])
    evaluate(frozen_lake_d_env, Control.DETERMINISTIC_ACTOR_CRITIC_2, episodes=1000, dims=[8, 8])


if __name__ == "__main__":
    main()

