"""
Evaluation Utilities for Snake RL Agent.
Functions for evaluating model performance and collecting statistics.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv

from snake_env import SnakeEnv
from make_env import make_eval_env


def evaluate(
    model: PPO,
    env: SnakeEnv,
    n_episodes: int = 50,
    deterministic: bool = True,
    verbose: bool = False,
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Evaluate a trained model over multiple episodes.

    Args:
        model: Trained PPO model
        env: Snake environment
        n_episodes: Number of episodes to run
        deterministic: Whether to use deterministic actions
        verbose: Whether to print progress

    Returns:
        mean_score: Average score (food eaten)
        std_score: Standard deviation of scores
        stats: Dictionary with detailed statistics
    """
    scores = []
    lengths = []
    rewards = []
    death_reasons = defaultdict(int)
    steps_list = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            episode_reward += reward
            episode_steps += 1

        scores.append(info.get("score", 0))
        lengths.append(info.get("length", 0))
        rewards.append(episode_reward)
        steps_list.append(episode_steps)

        if info.get("reason"):
            death_reasons[info["reason"]] += 1

        if verbose and (ep + 1) % 10 == 0:
            print(f"Episode {ep + 1}/{n_episodes}: Score={scores[-1]}, Length={lengths[-1]}")

    mean_score = np.mean(scores)
    std_score = np.std(scores)

    stats = {
        "scores": scores,
        "lengths": lengths,
        "rewards": rewards,
        "steps": steps_list,
        "mean_score": mean_score,
        "std_score": std_score,
        "max_score": max(scores),
        "min_score": min(scores),
        "mean_length": np.mean(lengths),
        "max_length": max(lengths),
        "mean_reward": np.mean(rewards),
        "mean_steps": np.mean(steps_list),
        "death_reasons": dict(death_reasons),
        "perfect_games": sum(1 for s in scores if s >= env.n * env.n - 3),  # Near-perfect
    }

    return mean_score, std_score, stats


def evaluate_vec_env(
    model: PPO,
    vec_env: VecEnv,
    n_episodes: int = 50,
    deterministic: bool = True,
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Evaluate a trained model using a vectorized environment.

    Args:
        model: Trained PPO model
        vec_env: Vectorized environment
        n_episodes: Number of episodes to run
        deterministic: Whether to use deterministic actions

    Returns:
        mean_score: Average score
        std_score: Standard deviation
        stats: Detailed statistics
    """
    scores = []
    lengths = []
    rewards = []
    episode_rewards = [0.0] * vec_env.num_envs
    episode_count = 0

    obs = vec_env.reset()

    while episode_count < n_episodes:
        actions, _ = model.predict(obs, deterministic=deterministic)
        obs, reward_batch, done_batch, infos = vec_env.step(actions)

        for i in range(vec_env.num_envs):
            episode_rewards[i] += reward_batch[i]

            if done_batch[i]:
                episode_count += 1
                if episode_count <= n_episodes:
                    info = infos[i]
                    scores.append(info.get("score", 0))
                    lengths.append(info.get("length", 0))
                    rewards.append(episode_rewards[i])
                    episode_rewards[i] = 0.0

                if episode_count >= n_episodes:
                    break

    mean_score = np.mean(scores)
    std_score = np.std(scores)

    stats = {
        "scores": scores,
        "lengths": lengths,
        "rewards": rewards,
        "mean_score": mean_score,
        "std_score": std_score,
        "max_score": max(scores) if scores else 0,
        "min_score": min(scores) if scores else 0,
        "mean_length": np.mean(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "mean_reward": np.mean(rewards) if rewards else 0,
    }

    return mean_score, std_score, stats


def run_single_episode(
    model: PPO,
    env: SnakeEnv,
    deterministic: bool = True,
    render: bool = False,
    max_steps: int = 10000,
) -> Tuple[List[np.ndarray], List[int], Dict[str, Any]]:
    """
    Run a single episode and collect trajectory.

    Args:
        model: Trained PPO model
        env: Snake environment
        deterministic: Whether to use deterministic actions
        render: Whether to render each step
        max_steps: Maximum steps per episode

    Returns:
        observations: List of observations
        actions: List of actions taken
        info: Final episode info
    """
    observations = []
    actions = []

    obs, info = env.reset()
    observations.append(obs.copy())

    done = False
    step = 0

    while not done and step < max_steps:
        action, _ = model.predict(obs, deterministic=deterministic)
        action = int(action)
        actions.append(action)

        if render:
            env.render()

        obs, reward, terminated, truncated, info = env.step(action)
        observations.append(obs.copy())
        done = terminated or truncated
        step += 1

    return observations, actions, info


def compute_score_percentiles(
    scores: List[float],
    percentiles: List[int] = [10, 25, 50, 75, 90, 95, 99],
) -> Dict[str, float]:
    """
    Compute score percentiles.

    Args:
        scores: List of scores
        percentiles: Percentiles to compute

    Returns:
        Dictionary mapping percentile names to values
    """
    result = {}
    for p in percentiles:
        result[f"p{p}"] = np.percentile(scores, p)
    return result


def compare_models(
    models: Dict[str, PPO],
    board_size: int = 20,
    n_episodes: int = 100,
    seed: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple models on the same environment.

    Args:
        models: Dictionary mapping model names to PPO models
        board_size: Grid size for evaluation
        n_episodes: Number of episodes per model
        seed: Random seed for environment

    Returns:
        Dictionary mapping model names to their statistics
    """
    results = {}
    env = make_eval_env(n=board_size, seed=seed)

    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        mean_score, std_score, stats = evaluate(
            model, env, n_episodes=n_episodes, verbose=True
        )
        stats["name"] = name
        results[name] = stats

    env.close()
    return results


def print_evaluation_summary(stats: Dict[str, Any], name: str = "Model") -> None:
    """
    Print a formatted evaluation summary.

    Args:
        stats: Statistics dictionary from evaluate()
        name: Model name for display
    """
    print(f"\n{'='*50}")
    print(f"Evaluation Summary: {name}")
    print(f"{'='*50}")
    print(f"Episodes: {len(stats['scores'])}")
    print(f"\nScore Statistics:")
    print(f"  Mean:    {stats['mean_score']:.2f} +/- {stats['std_score']:.2f}")
    print(f"  Max:     {stats['max_score']}")
    print(f"  Min:     {stats['min_score']}")

    if stats.get('scores'):
        percentiles = compute_score_percentiles(stats['scores'])
        print(f"\nScore Percentiles:")
        for k, v in percentiles.items():
            print(f"  {k}: {v:.1f}")

    print(f"\nLength Statistics:")
    print(f"  Mean:    {stats['mean_length']:.2f}")
    print(f"  Max:     {stats['max_length']}")

    print(f"\nOther Metrics:")
    print(f"  Mean Reward: {stats['mean_reward']:.2f}")
    print(f"  Mean Steps:  {stats.get('mean_steps', 'N/A')}")

    if stats.get('death_reasons'):
        print(f"\nDeath Reasons:")
        total_deaths = sum(stats['death_reasons'].values())
        for reason, count in sorted(stats['death_reasons'].items()):
            pct = 100 * count / total_deaths if total_deaths > 0 else 0
            print(f"  {reason}: {count} ({pct:.1f}%)")

    if stats.get('perfect_games'):
        print(f"\nPerfect/Near-Perfect Games: {stats['perfect_games']}")

    print(f"{'='*50}\n")
