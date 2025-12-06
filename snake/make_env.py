"""
Environment Factory Utilities for Snake RL Training.
Provides helpers to create vectorized environments for SB3.
"""

from typing import Callable, Optional
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from snake_env import SnakeEnv


def make_snake_env(
    n: int = 20,
    gamma: float = 0.995,
    alpha: float = 0.2,
    max_no_food: Optional[int] = None,
    survival_bonus: float = 0.001,
    seed: int = 0,
    render_mode: Optional[str] = None,
) -> Callable[[], SnakeEnv]:
    """
    Create an environment factory function.

    Args:
        n: Grid size
        gamma: Discount factor for reward shaping
        alpha: Coefficient for distance potential
        max_no_food: Max steps without food before truncation
        survival_bonus: Small reward per step
        seed: Random seed
        render_mode: Render mode for the environment

    Returns:
        Callable that creates a monitored SnakeEnv
    """
    def _init() -> SnakeEnv:
        env = SnakeEnv(
            n=n,
            max_no_food=max_no_food,
            gamma=gamma,
            alpha=alpha,
            survival_bonus=survival_bonus,
            seed=seed,
            render_mode=render_mode,
        )
        env = Monitor(env)
        return env

    return _init


def make_vec_env(
    n: int = 20,
    n_envs: int = 1,
    gamma: float = 0.995,
    alpha: float = 0.2,
    max_no_food: Optional[int] = None,
    survival_bonus: float = 0.001,
    seed: int = 0,
    use_subproc: bool = False,
) -> DummyVecEnv | SubprocVecEnv:
    """
    Create a vectorized environment with multiple parallel instances.

    Args:
        n: Grid size
        n_envs: Number of parallel environments
        gamma: Discount factor for reward shaping
        alpha: Coefficient for distance potential
        max_no_food: Max steps without food
        survival_bonus: Small reward per step
        seed: Base random seed (each env gets seed + i)
        use_subproc: Whether to use SubprocVecEnv (True for parallelism)

    Returns:
        Vectorized environment
    """
    env_fns = [
        make_snake_env(
            n=n,
            gamma=gamma,
            alpha=alpha,
            max_no_food=max_no_food,
            survival_bonus=survival_bonus,
            seed=seed + i,
        )
        for i in range(n_envs)
    ]

    if use_subproc and n_envs > 1:
        return SubprocVecEnv(env_fns)
    else:
        return DummyVecEnv(env_fns)


def make_eval_env(
    n: int = 20,
    seed: int = 42,
    render_mode: Optional[str] = None,
) -> SnakeEnv:
    """
    Create a single environment for evaluation.

    Args:
        n: Grid size
        seed: Random seed
        render_mode: Render mode

    Returns:
        SnakeEnv instance
    """
    return SnakeEnv(
        n=n,
        gamma=0.995,
        alpha=0.2,
        seed=seed,
        render_mode=render_mode,
    )
