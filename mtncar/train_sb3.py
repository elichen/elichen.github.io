"""
Train a DQN agent on the custom Mountain Car environment using Stable Baselines3.
"""
import os
import sys

# Add stable-baselines3 to path
sys.path.insert(0, "/Users/elichen/code/stable-baselines3")

import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from mtncar_env import MountainCarEnv


def make_env():
    """Create and wrap the environment."""
    env = MountainCarEnv()
    env = Monitor(env)
    return env


def train_dqn(total_timesteps=100000, eval_freq=5000, save_freq=10000):
    """
    Train a DQN agent on Mountain Car.

    Args:
        total_timesteps: Total number of training timesteps
        eval_freq: Evaluate every n steps
        save_freq: Save checkpoint every n steps
    """
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Create training and evaluation environments
    train_env = DummyVecEnv([make_env])
    eval_env = DummyVecEnv([make_env])

    # Configure DQN to match web app architecture
    policy_kwargs = dict(
        net_arch=[64, 64],  # Matching web app's [64, 64] hidden layers
    )

    # Initialize DQN agent
    # Hyperparameters tuned for Mountain Car
    model = DQN(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=0.001,  # Matching web app
        buffer_size=100000,  # Matching web app
        learning_starts=1000,  # Start learning after collecting initial experience
        batch_size=32,  # Matching web app
        tau=1.0,  # Hard updates (matching web app's target network updates)
        gamma=0.99,  # Discount factor
        train_freq=4,  # Train every 4 steps
        gradient_steps=1,  # Number of gradient steps per update
        target_update_interval=100,  # Update target network every 100 steps
        exploration_fraction=0.5,  # 50% of training for exploration decay
        exploration_initial_eps=1.0,  # Start with full exploration
        exploration_final_eps=0.01,  # End with 1% random actions
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./logs/dqn_mtncar_tensorboard/"
    )

    # Setup callbacks
    # Evaluation callback - saves best model based on eval performance
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best_model",
        log_path="./logs/eval",
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=10
    )

    # Checkpoint callback - saves periodic checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path="./models/checkpoints",
        name_prefix="dqn_mtncar"
    )

    # Combine callbacks
    callback = CallbackList([eval_callback, checkpoint_callback])

    print("=" * 60)
    print("Training DQN on Mountain Car")
    print("=" * 60)
    print(f"Total timesteps: {total_timesteps}")
    print(f"Network architecture: {policy_kwargs['net_arch']}")
    print(f"Learning rate: {model.learning_rate}")
    print(f"Batch size: {model.batch_size}")
    print(f"Buffer size: {model.buffer_size}")
    print("=" * 60)

    # Train the agent
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )

    # Save the final model
    final_model_path = "models/dqn_mtncar_final"
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")

    # Also save the best model's weights separately for easier conversion
    best_model = DQN.load("models/best_model/best_model.zip", env=train_env)
    torch.save(best_model.q_net.state_dict(), "models/best_model_weights.pth")
    print(f"Best model weights saved to: models/best_model_weights.pth")

    return model


def evaluate_model(model_path, n_episodes=10):
    """
    Evaluate a trained model.

    Args:
        model_path: Path to the saved model
        n_episodes: Number of episodes to evaluate
    """
    # Load the model
    env = DummyVecEnv([make_env])
    model = DQN.load(model_path, env=env)

    print(f"\nEvaluating model: {model_path}")
    print("=" * 60)

    # Run evaluation episodes
    successes = 0
    total_rewards = []
    episode_lengths = []

    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += float(reward[0])
            steps += 1

            if done:
                if info[0]["success"]:
                    successes += 1
                total_rewards.append(total_reward)
                episode_lengths.append(steps)
                print(f"Episode {episode + 1}: reward={total_reward:.2f}, steps={steps}, success={info[0]['success']}")

    # Print statistics
    print("=" * 60)
    print(f"Success rate: {successes}/{n_episodes} ({100 * successes / n_episodes:.1f}%)")
    print(f"Average reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Average episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train DQN on Mountain Car")
    parser.add_argument("--train", action="store_true", help="Train a new model")
    parser.add_argument("--eval", action="store_true", help="Evaluate a trained model")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total training timesteps")
    parser.add_argument("--model-path", type=str, default="models/best_model/best_model.zip",
                        help="Path to model for evaluation")

    args = parser.parse_args()

    if args.train:
        # Train a new model
        model = train_dqn(total_timesteps=args.timesteps)

        # Evaluate the trained model
        print("\nEvaluating trained model...")
        evaluate_model("models/best_model/best_model.zip", n_episodes=10)

    elif args.eval:
        # Evaluate existing model
        evaluate_model(args.model_path, n_episodes=10)

    else:
        # Default: train a model
        print("No action specified. Use --train to train or --eval to evaluate.")
        print("Example: python train_sb3.py --train --timesteps 100000")
