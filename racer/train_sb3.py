"""
PPO Training Script for Racing AI
Trains a racing AI using PPO with continuous control and ray-based sensors
"""

import os
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CheckpointCallback,
    CallbackList
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from racer_env import RacerEnv


class CustomRacingNetwork(BaseFeaturesExtractor):
    """
    Custom network architecture optimized for racing with ray sensors
    """
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)

        # Input size is number of rays + speed + angular velocity
        n_input_features = observation_space.shape[0]

        # Feature extraction layers
        self.net = nn.Sequential(
            nn.Linear(n_input_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        return self.net(observations)


class LapTimeCallback(BaseCallback):
    """
    Custom callback to track lap times and racing metrics
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.best_lap_time = float('inf')
        self.total_laps = 0
        self.episode_count = 0

    def _on_step(self) -> bool:
        # Check if any environment has completed an episode
        for i, done in enumerate(self.locals.get("dones", [])):
            if done:
                info = self.locals["infos"][i]
                self.episode_count += 1

                # Track lap times
                if info.get("laps_completed", 0) > 0:
                    self.total_laps += info["laps_completed"]
                    if info.get("best_lap_time", float('inf')) < self.best_lap_time:
                        self.best_lap_time = info["best_lap_time"]
                        print(f"\n🏁 New best lap time: {self.best_lap_time:.2f} steps")

                # Log episode stats
                if self.verbose > 0 and self.episode_count % 10 == 0:
                    print(f"Episode {self.episode_count}: Laps={info.get('laps_completed', 0)}, "
                          f"Collisions={info.get('collisions', 0)}, "
                          f"Distance={info.get('total_distance', 0):.0f}")

        return True

    def _on_training_end(self) -> None:
        print(f"\n=== Training Complete ===")
        print(f"Total episodes: {self.episode_count}")
        print(f"Total laps completed: {self.total_laps}")
        if self.best_lap_time < float('inf'):
            print(f"Best lap time: {self.best_lap_time:.2f} steps")


def create_env():
    """Create a wrapped environment for training"""
    env = RacerEnv(max_steps=3000)
    env = Monitor(env)
    return env


def train_racing_ai(
    total_timesteps=100000,
    n_envs=4,
    save_dir="models",
    checkpoint_freq=10000
):
    """
    Train the racing AI using PPO

    Args:
        total_timesteps: Total training steps
        n_envs: Number of parallel environments
        save_dir: Directory to save models
        checkpoint_freq: Frequency of model checkpoints
    """

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(save_dir, f"ppo_racer_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    print("=" * 50)
    print("🏎️  Racing AI Training with PPO")
    print("=" * 50)
    print(f"Training for {total_timesteps:,} timesteps")
    print(f"Using {n_envs} parallel environments")
    print(f"Models will be saved to: {run_dir}")
    print("=" * 50)

    # Create parallel training environments
    print("\nCreating training environments...")
    if n_envs > 1:
        env = make_vec_env(create_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    else:
        env = make_vec_env(create_env, n_envs=n_envs, vec_env_cls=DummyVecEnv)

    # Create evaluation environment
    eval_env = create_env()

    # PPO hyperparameters optimized for racing
    ppo_kwargs = dict(
        learning_rate=3e-4,
        n_steps=2048,  # Steps per environment per update
        batch_size=64,
        n_epochs=10,
        gamma=0.99,  # Discount factor
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Entropy coefficient for exploration
        vf_coef=0.5,  # Value function coefficient
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=os.path.join(run_dir, "tensorboard")
    )

    # Policy network configuration
    policy_kwargs = dict(
        features_extractor_class=CustomRacingNetwork,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),  # Separate networks for policy and value
        activation_fn=nn.ReLU
    )

    print("\nInitializing PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        **ppo_kwargs
    )

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=os.path.join(run_dir, "checkpoints"),
        name_prefix="ppo_racer"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(run_dir, "best_model"),
        log_path=os.path.join(run_dir, "eval_logs"),
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    lap_time_callback = LapTimeCallback(verbose=1)

    callbacks = CallbackList([checkpoint_callback, eval_callback, lap_time_callback])

    # Training
    print("\n🚀 Starting training...\n")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    finally:
        # Save final model
        final_model_path = os.path.join(run_dir, "final_model")
        model.save(final_model_path)
        print(f"\n✅ Final model saved to: {final_model_path}")

        # Evaluate final performance
        print("\n📊 Evaluating final model...")
        mean_reward, std_reward = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=10,
            deterministic=True
        )
        print(f"Mean episode reward: {mean_reward:.2f} (+/- {std_reward:.2f})")

    # Clean up
    env.close()
    eval_env.close()

    print("\n" + "=" * 50)
    print("🏁 Training Complete!")
    print(f"Models saved in: {run_dir}")
    print("=" * 50)

    return model, run_dir


def test_trained_model(model_path, n_episodes=5):
    """
    Test a trained model and display performance

    Args:
        model_path: Path to the trained model
        n_episodes: Number of test episodes
    """
    print(f"\n🧪 Testing model: {model_path}")

    # Load model
    model = PPO.load(model_path)

    # Create test environment
    env = create_env()

    # Run test episodes
    lap_times = []
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        steps = 0

        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if info.get("laps_completed", 0) > 0 and info.get("lap_time", 0) > 0:
                lap_times.append(info["lap_time"])

        print(f"Episode {episode + 1}: Reward={total_reward:.2f}, "
              f"Steps={steps}, Laps={info.get('laps_completed', 0)}")

    if lap_times:
        print(f"\n📊 Lap Time Statistics:")
        print(f"  Best: {min(lap_times):.2f} steps")
        print(f"  Average: {np.mean(lap_times):.2f} steps")
        print(f"  Worst: {max(lap_times):.2f} steps")

    env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train or test racing AI with PPO")
    parser.add_argument("--train", action="store_true", help="Train a new model")
    parser.add_argument("--test", type=str, help="Test a trained model (provide path)")
    parser.add_argument("--timesteps", type=int, default=100000,
                       help="Total training timesteps (default: 100000)")
    parser.add_argument("--envs", type=int, default=4,
                       help="Number of parallel environments (default: 4)")

    args = parser.parse_args()

    if args.train:
        # Train new model
        model, save_dir = train_racing_ai(
            total_timesteps=args.timesteps,
            n_envs=args.envs
        )

        # Test the trained model
        best_model_path = os.path.join(save_dir, "best_model", "best_model")
        if os.path.exists(best_model_path + ".zip"):
            test_trained_model(best_model_path)

    elif args.test:
        # Test existing model
        test_trained_model(args.test)

    else:
        # Default: Quick training run
        print("No arguments provided. Running quick training (100k steps)...")
        model, save_dir = train_racing_ai(total_timesteps=100000, n_envs=4)