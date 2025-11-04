#!/usr/bin/env python3
"""
Test script to verify if the trained PPO model has asymmetric behavior
for blocks on the left vs right side of the robot arm.
"""

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from train_ppo import RobotArmEnv


def test_model_symmetry(model_path: str = "models/robot_arm_ppo_final", n_episodes_per_side: int = 20):
    """Test a trained PPO model for asymmetric behavior"""

    env = RobotArmEnv()
    model = PPO.load(model_path)

    # Test left side
    print("Testing LEFT side performance...")
    left_successes = []
    left_rewards = []
    left_lengths = []

    for episode in range(n_episodes_per_side):
        # Force reset to left side
        obs, _ = env.reset()
        # Override block position to be on left side
        env.block_x = env.arm_base_x - 150 + np.random.uniform(-50, 50)  # Left side
        env.block_y = env.ground_y - env.block_size / 2
        obs = env._get_observation()

        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated

        success = 1 if episode_reward > 50 else 0
        left_successes.append(success)
        left_rewards.append(episode_reward)
        left_lengths.append(episode_length)

        print(f"  Episode {episode + 1}: Reward = {episode_reward:.2f}, Success = {success}")

    # Test right side
    print("\nTesting RIGHT side performance...")
    right_successes = []
    right_rewards = []
    right_lengths = []

    for episode in range(n_episodes_per_side):
        # Force reset to right side
        obs, _ = env.reset()
        # Override block position to be on right side
        env.block_x = env.arm_base_x + 150 + np.random.uniform(-50, 50)  # Right side
        env.block_y = env.ground_y - env.block_size / 2
        obs = env._get_observation()

        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated

        success = 1 if episode_reward > 50 else 0
        right_successes.append(success)
        right_rewards.append(episode_reward)
        right_lengths.append(episode_length)

        print(f"  Episode {episode + 1}: Reward = {episode_reward:.2f}, Success = {success}")

    # Print comparison
    print("\n" + "="*50)
    print("SYMMETRY TEST RESULTS")
    print("="*50)

    print(f"\nLEFT SIDE Performance:")
    print(f"  Success rate: {np.mean(left_successes):.2%}")
    print(f"  Average reward: {np.mean(left_rewards):.2f}")
    print(f"  Average length: {np.mean(left_lengths):.1f}")

    print(f"\nRIGHT SIDE Performance:")
    print(f"  Success rate: {np.mean(right_successes):.2%}")
    print(f"  Average reward: {np.mean(right_rewards):.2f}")
    print(f"  Average length: {np.mean(right_lengths):.1f}")

    print(f"\nDIFFERENCE:")
    success_diff = np.mean(left_successes) - np.mean(right_successes)
    reward_diff = np.mean(left_rewards) - np.mean(right_rewards)

    print(f"  Success rate difference: {abs(success_diff):.2%} ({'LEFT' if success_diff > 0 else 'RIGHT'} better)")
    print(f"  Reward difference: {abs(reward_diff):.2f} ({'LEFT' if reward_diff > 0 else 'RIGHT'} better)")

    if abs(success_diff) > 0.2:
        print("\n⚠️  SIGNIFICANT ASYMMETRY DETECTED!")
        print("The model performs much better on one side than the other.")
        print("This explains why the demo works on one side but not the other.")
    else:
        print("\n✓ Model appears to be relatively symmetric")

    return {
        'left': {'successes': left_successes, 'rewards': left_rewards, 'lengths': left_lengths},
        'right': {'successes': right_successes, 'rewards': right_rewards, 'lengths': right_lengths}
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test PPO model for asymmetric behavior")
    parser.add_argument("--model", type=str, default="models/robot_arm_ppo_final",
                        help="Model path for testing")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of episodes to test per side")

    args = parser.parse_args()

    print(f"Testing model: {args.model}")
    print(f"Running {args.episodes} episodes per side...\n")

    results = test_model_symmetry(args.model, args.episodes)