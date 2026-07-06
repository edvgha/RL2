import warnings
from enum import Enum
from typing import Callable
import gymnasium as gym

from rl2.envs import EnvPair, frozen_lake_d_env, frozen_lake_s_env, taxi_env
from rl2.model_based.value_iteration2 import ValueIteration
from rl2.model_based.policy_iteration2 import PolicyIteration

warnings.filterwarnings("ignore")


class Algorithm(Enum):
    VALUEITERATION = "ValueIteration"
    POLICYITERATION = "PolicyIteration"


def evaluate(env_func: Callable[[], EnvPair], alg: Algorithm) -> None:
    env, env_h = env_func()
    print(f"\n[*] Env: {env.spec.id} | Alg: {alg.value}") # type: ignore

    if alg == Algorithm.VALUEITERATION:
        vi_solver = ValueIteration(env=env, gamma=0.99, theta=1e-6)
        optimal_v, optimal_policy, converged = vi_solver()
    else:
        pi_solver = PolicyIteration(env=env, gamma=0.99, theta=1e-6)
        optimal_v, optimal_policy, converged = pi_solver()

    assert converged, f"{alg.value} did not converge"
    env.close()

    state, _ = env_h.reset(seed=42)
    total_reward = 0.0
    steps = 0
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


def main() -> None:
    evaluate(frozen_lake_d_env, Algorithm.VALUEITERATION)
    evaluate(frozen_lake_d_env, Algorithm.POLICYITERATION)
    evaluate(taxi_env, Algorithm.VALUEITERATION)
    evaluate(taxi_env, Algorithm.POLICYITERATION)
    evaluate(frozen_lake_s_env, Algorithm.VALUEITERATION)
    evaluate(frozen_lake_s_env, Algorithm.POLICYITERATION)


if __name__ == "__main__":
    main()
