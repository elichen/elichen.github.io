#!/usr/bin/env python3
"""
PPO training script for Snake using stable-baselines3.
Optimized for achieving perfect gameplay (100% grid fill) through parallel training.
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
from typing import Dict, Any

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, EvalCallback,
    CallbackList, StopTrainingOnNoModelImprovement
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed

# Import our custom environment
from snake_gym_env import SnakeGymEnv


class SnakeMetricsCallback(BaseCallback):
    """Custom callback to track Snake-specific metrics during training."""

    def __init__(self, grid_size: int = 20, log_freq: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.grid_size = grid_size
        self.log_freq = log_freq
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_scores = []
        self.perfect_games = 0
        self.best_score = 0
        self.milestone_counts = {25: 0, 50: 0, 75: 0, 90: 0, 100: 0}

    def _on_step(self) -> bool:
        # Check for completed episodes
        for i, done in enumerate(self.locals['dones']):
            if done:
                info = self.locals['infos'][i]

                if 'episode' in info:
                    self.episode_count += 1
                    self.episode_rewards.append(info['episode']['r'])
                    self.episode_lengths.append(info['episode']['l'])

                if 'score' in info:
                    score = info['score']
                    self.episode_scores.append(score)
                    self.best_score = max(self.best_score, score)

                    # Track milestones
                    max_score = self.grid_size * self.grid_size - 1
                    fill_ratio = score / max_score * 100

                    for milestone in [25, 50, 75, 90, 100]:
                        if fill_ratio >= milestone:
                            self.milestone_counts[milestone] += 1

                    # Check for perfect game
                    if score >= max_score:
                        self.perfect_games += 1
                        print(f"🏆 PERFECT GAME #{self.perfect_games} achieved at episode {self.episode_count}!")

                # Log metrics periodically
                if self.episode_count % self.log_freq == 0 and self.episode_scores:
                    avg_reward = np.mean(self.episode_rewards[-100:])
                    avg_length = np.mean(self.episode_lengths[-100:])
                    avg_score = np.mean(self.episode_scores[-100:])
                    max_score_recent = max(self.episode_scores[-100:])

                    print(f"\n📊 Episode {self.episode_count} Statistics:")
                    print(f"   Avg Reward (100 eps): {avg_reward:.2f}")
                    print(f"   Avg Length (100 eps): {avg_length:.2f}")
                    print(f"   Avg Score (100 eps): {avg_score:.2f}")
                    print(f"   Max Score (100 eps): {max_score_recent}")
                    print(f"   Best Score (all time): {self.best_score}")
                    print(f"   Perfect Games: {self.perfect_games}")
                    print(f"   Milestones reached: ", end="")
                    for milestone, count in self.milestone_counts.items():
                        if count > 0:
                            print(f"{milestone}%:{count} ", end="")
                    print()

                    # Log to tensorboard
                    self.logger.record("custom/avg_score", avg_score)
                    self.logger.record("custom/max_score", self.best_score)
                    self.logger.record("custom/perfect_games", self.perfect_games)

        return True

    def _on_training_end(self) -> None:
        print(f"\n🎯 Training Complete!")
        print(f"   Total Episodes: {self.episode_count}")
        print(f"   Best Score: {self.best_score}")
        print(f"   Perfect Games: {self.perfect_games}")
        print(f"   Final Avg Score (100 eps): {np.mean(self.episode_scores[-100:]):.2f}")


class CurriculumScheduler:
    """Manages curriculum learning by progressively increasing grid size."""

    def __init__(self, stages: Dict[int, Dict[str, Any]]):
        """
        Args:
            stages: Dictionary mapping grid sizes to training configurations
                   {grid_size: {'timesteps': X, 'target_score': Y}}
        """
        self.stages = stages
        self.current_stage = 0
        self.stage_keys = sorted(stages.keys())

    def get_current_config(self) -> Dict[str, Any]:
        """Get configuration for current stage."""
        if self.current_stage >= len(self.stage_keys):
            return None
        grid_size = self.stage_keys[self.current_stage]
        return {'grid_size': grid_size, **self.stages[grid_size]}

    def advance_stage(self) -> bool:
        """Move to next curriculum stage."""
        self.current_stage += 1
        return self.current_stage < len(self.stage_keys)

    def should_advance(self, avg_score: float, grid_size: int) -> bool:
        """Check if criteria met to advance to next stage."""
        if grid_size not in self.stages:
            return False
        target = self.stages[grid_size].get('target_score', float('inf'))
        return avg_score >= target


def create_env(grid_size: int, seed: int = 0) -> gym.Env:
    """Create a single Snake environment."""
    def _init():
        env = SnakeGymEnv(
            grid_size=grid_size,
            enable_connectivity=True,
            enable_milestones=True,
            adaptive_starvation=True
        )
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def train_ppo_snake(
    grid_size: int = 20,
    total_timesteps: int = 10_000_000,
    n_envs: int = 8,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 256,
    n_epochs: int = 10,
    gamma: float = 0.95,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    use_curriculum: bool = True,
    save_freq: int = 100_000,
    eval_freq: int = 50_000,
    log_dir: str = "./logs",
    model_dir: str = "./models",
    verbose: int = 1,
    seed: int = 42
):
    """
    Train PPO agent to play Snake.

    Args:
        grid_size: Initial grid size (or fixed if not using curriculum)
        total_timesteps: Total training timesteps
        n_envs: Number of parallel environments
        learning_rate: Learning rate for PPO
        n_steps: Number of steps per environment per update
        batch_size: Batch size for PPO updates
        n_epochs: Number of epochs for PPO updates
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        clip_range: PPO clipping parameter
        ent_coef: Entropy coefficient for exploration
        vf_coef: Value function coefficient
        max_grad_norm: Maximum gradient norm for clipping
        use_curriculum: Whether to use curriculum learning
        save_freq: Checkpoint save frequency
        eval_freq: Evaluation frequency
        log_dir: Directory for tensorboard logs
        model_dir: Directory for model checkpoints
        verbose: Verbosity level
        seed: Random seed
    """

    # Set random seeds for reproducibility
    set_random_seed(seed)

    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"snake_ppo_{grid_size}x{grid_size}_{timestamp}"
    log_path = os.path.join(log_dir, run_name)
    model_path = os.path.join(model_dir, run_name)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    # Save configuration
    config = {
        'grid_size': grid_size,
        'total_timesteps': total_timesteps,
        'n_envs': n_envs,
        'learning_rate': learning_rate,
        'n_steps': n_steps,
        'batch_size': batch_size,
        'n_epochs': n_epochs,
        'gamma': gamma,
        'gae_lambda': gae_lambda,
        'clip_range': clip_range,
        'ent_coef': ent_coef,
        'vf_coef': vf_coef,
        'max_grad_norm': max_grad_norm,
        'use_curriculum': use_curriculum,
        'seed': seed,
        'timestamp': timestamp
    }
    with open(os.path.join(model_path, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Setup curriculum learning if enabled
    if use_curriculum:
        curriculum = CurriculumScheduler({
            5: {'timesteps': 500_000, 'target_score': 22},    # 90% of 25
            8: {'timesteps': 1_000_000, 'target_score': 54},  # 85% of 64
            12: {'timesteps': 2_000_000, 'target_score': 115}, # 80% of 144
            16: {'timesteps': 5_000_000, 'target_score': 192}, # 75% of 256
            20: {'timesteps': -1, 'target_score': 400}        # Perfect game
        })
        current_config = curriculum.get_current_config()
        grid_size = current_config['grid_size']
        print(f"🎓 Starting curriculum learning with {grid_size}x{grid_size} grid")
    else:
        curriculum = None

    # Create vectorized environment
    print(f"🚀 Creating {n_envs} parallel environments...")

    # Use SubprocVecEnv for true parallelism (macOS specific setup)
    if sys.platform == "darwin":  # macOS
        import multiprocessing
        # macOS requires 'fork' method for SubprocVecEnv
        multiprocessing.set_start_method('fork', force=True)

    env = make_vec_env(
        create_env(grid_size, seed),
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv if n_envs > 1 else DummyVecEnv,
        vec_env_kwargs={'start_method': 'fork'} if n_envs > 1 else {}
    )
    env = VecMonitor(env, filename=os.path.join(log_path, "monitor.csv"))

    # Create evaluation environment
    eval_env = make_vec_env(
        create_env(grid_size, seed + 1000),
        n_envs=1,
        vec_env_cls=DummyVecEnv
    )

    # Create PPO model
    print("🤖 Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        tensorboard_log=log_path,
        verbose=verbose,
        seed=seed
    )

    # Configure logger
    new_logger = configure(log_path, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    # Create callbacks
    callbacks = []

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq // n_envs,  # Adjust for vectorized env
        save_path=model_path,
        name_prefix="snake_ppo",
        save_replay_buffer=False,
        save_vecnormalize=False
    )
    callbacks.append(checkpoint_callback)

    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_path,
        log_path=log_path,
        eval_freq=eval_freq // n_envs,
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )
    callbacks.append(eval_callback)

    # Custom metrics callback
    metrics_callback = SnakeMetricsCallback(grid_size=grid_size, log_freq=100)
    callbacks.append(metrics_callback)

    # Stop on no improvement callback
    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=20,
        min_evals=100,
        verbose=1
    )

    # Combine callbacks
    callback_list = CallbackList(callbacks)

    # Training loop
    print(f"🏋️ Starting training for {total_timesteps:,} timesteps...")
    print(f"   Grid size: {grid_size}x{grid_size}")
    print(f"   Parallel environments: {n_envs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Curriculum learning: {'Enabled' if use_curriculum else 'Disabled'}")
    print("-" * 50)

    start_time = time.time()

    try:
        if use_curriculum:
            # Train with curriculum learning
            total_trained = 0
            while curriculum and total_trained < total_timesteps:
                config = curriculum.get_current_config()
                if config is None:
                    break

                stage_timesteps = min(
                    config['timesteps'] if config['timesteps'] > 0 else total_timesteps - total_trained,
                    total_timesteps - total_trained
                )

                print(f"\n🎯 Training on {config['grid_size']}x{config['grid_size']} grid for {stage_timesteps:,} timesteps")
                print(f"   Target score: {config['target_score']}")

                model.learn(
                    total_timesteps=stage_timesteps,
                    callback=callback_list,
                    reset_num_timesteps=False,
                    progress_bar=True
                )

                total_trained += stage_timesteps

                # Check if should advance to next stage
                mean_reward, std_reward = evaluate_policy(
                    model, eval_env, n_eval_episodes=20, deterministic=True
                )

                # Extract average score from evaluation
                # Note: This is simplified, you may need to track actual scores
                avg_score = mean_reward / 10  # Rough approximation

                print(f"   Stage complete! Avg score: {avg_score:.2f}")

                if curriculum.should_advance(avg_score, config['grid_size']):
                    if curriculum.advance_stage():
                        next_config = curriculum.get_current_config()
                        if next_config:
                            # Create new environment with larger grid
                            print(f"🎓 Advancing to {next_config['grid_size']}x{next_config['grid_size']} grid")
                            env.close()
                            eval_env.close()

                            grid_size = next_config['grid_size']
                            env = make_vec_env(
                                create_env(grid_size, seed),
                                n_envs=n_envs,
                                vec_env_cls=SubprocVecEnv if n_envs > 1 else DummyVecEnv,
                                vec_env_kwargs={'start_method': 'fork'} if n_envs > 1 else {}
                            )
                            env = VecMonitor(env, filename=os.path.join(log_path, f"monitor_{grid_size}.csv"))

                            eval_env = make_vec_env(
                                create_env(grid_size, seed + 1000),
                                n_envs=1,
                                vec_env_cls=DummyVecEnv
                            )

                            # Update model's environment
                            model.set_env(env)
                else:
                    print(f"   Target not met, continuing on current grid size")

        else:
            # Train without curriculum
            model.learn(
                total_timesteps=total_timesteps,
                callback=callback_list,
                reset_num_timesteps=True,
                progress_bar=True
            )

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")

    finally:
        # Save final model
        final_model_path = os.path.join(model_path, "final_model")
        model.save(final_model_path)
        print(f"💾 Final model saved to {final_model_path}")

        # Final evaluation
        print("\n📈 Final evaluation...")
        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=100, deterministic=True
        )
        print(f"   Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

        # Training statistics
        elapsed_time = time.time() - start_time
        print(f"\n⏱️ Training completed in {elapsed_time/3600:.2f} hours")
        print(f"📊 Best score achieved: {metrics_callback.best_score}")
        print(f"🏆 Perfect games: {metrics_callback.perfect_games}")

        # Cleanup
        env.close()
        eval_env.close()


def main():
    """Main entry point with command line arguments."""
    parser = argparse.ArgumentParser(description="Train PPO agent for Snake")

    parser.add_argument('--grid-size', type=int, default=20,
                      help='Grid size for Snake game (default: 20)')
    parser.add_argument('--timesteps', type=int, default=10_000_000,
                      help='Total training timesteps (default: 10M)')
    parser.add_argument('--n-envs', type=int, default=8,
                      help='Number of parallel environments (default: 8)')
    parser.add_argument('--lr', type=float, default=3e-4,
                      help='Learning rate (default: 3e-4)')
    parser.add_argument('--batch-size', type=int, default=256,
                      help='Batch size for PPO (default: 256)')
    parser.add_argument('--no-curriculum', action='store_true',
                      help='Disable curriculum learning')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed (default: 42)')
    parser.add_argument('--verbose', type=int, default=1,
                      help='Verbosity level (default: 1)')

    args = parser.parse_args()

    train_ppo_snake(
        grid_size=args.grid_size,
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        use_curriculum=not args.no_curriculum,
        seed=args.seed,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()