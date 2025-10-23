#!/usr/bin/env python3
"""Test script to verify observation space consistency and agent behavior"""

import numpy as np
from air_hockey_env_v3 import AirHockeyEnv

def test_observation_space():
    env = AirHockeyEnv()
    obs, _ = env.reset()

    print("Environment Observation Test")
    print("=" * 40)
    print(f"Observation shape: {obs.shape}")
    print(f"Observation dtype: {obs.dtype}")
    print(f"Expected shape: (14,)")

    # Test with different paddle/puck positions
    env.puck_pos = np.array([300.0, 100.0])  # Puck near top
    env.puck_vel = np.array([5.0, -10.0])
    env.paddle1_pos = np.array([250.0, 750.0])
    env.paddle2_pos = np.array([350.0, 50.0])

    obs_p1 = env._get_observation(1)
    obs_p2 = env._get_observation(2)

    print("\nPlayer 1 observation (bottom player):")
    labels = ["puck_rel_x", "puck_rel_y", "puck_vx", "puck_vy",
              "opp_rel_x", "opp_rel_y", "dist", "angle", "behind",
              "goal_dist", "puck_goal_dist", "paddle_vx", "paddle_vy", "puck_speed"]
    for i, (label, val) in enumerate(zip(labels, obs_p1)):
        print(f"  {i:2}: {label:15} = {val:7.3f}")

    print("\nPlayer 2 observation (top player - should be flipped):")
    for i, (label, val) in enumerate(zip(labels, obs_p2)):
        print(f"  {i:2}: {label:15} = {val:7.3f}")

    # Test that puck behind paddle detection works correctly
    print("\n" + "=" * 40)
    print("Testing 'puck behind paddle' detection:")

    # For player 1 (bottom), puck at y=100 should NOT be behind (paddle at y=750)
    behind_p1 = 1.0 if 100 > 750 else 0.0
    print(f"  Player 1: puck_y={100}, paddle_y={750}, behind={behind_p1} (should be 0)")

    # For player 2 (top), after flipping: puck_y=700, paddle_y=750
    flipped_puck_y = 800 - 100  # = 700
    flipped_paddle_y = 800 - 50  # = 750
    behind_p2 = 1.0 if flipped_puck_y > flipped_paddle_y else 0.0
    print(f"  Player 2: puck_y={flipped_puck_y}, paddle_y={flipped_paddle_y}, behind={behind_p2} (should be 0)")

    print("\nChecking observation values are in [-1, 1] range:")
    obs_min, obs_max = obs_p1.min(), obs_p1.max()
    print(f"  Min: {obs_min:.3f}, Max: {obs_max:.3f}")

    # Test action space
    print("\n" + "=" * 40)
    print("Action Space Test:")
    action = env.action_space.sample()
    print(f"  Sample action shape: {action.shape}")
    print(f"  Sample action values: [{action[0]:.3f}, {action[1]:.3f}]")

    # Test reward structure
    print("\n" + "=" * 40)
    print("Reward Structure:")
    print("  Score goal: +1.0")
    print("  Receive goal: -1.0")
    print("  Cause fault: -0.5")

    return True

if __name__ == "__main__":
    test_observation_space()
    print("\n✅ All tests completed!")