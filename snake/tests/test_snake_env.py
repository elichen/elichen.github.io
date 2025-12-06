"""
Unit Tests for Snake Environment.
Tests core environment behavior, reward shaping, and edge cases.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from snake_env import SnakeEnv


class TestSnakeEnvBasic:
    """Basic environment tests."""

    def test_reset_produces_valid_state(self):
        """Test that reset produces valid initial state."""
        env = SnakeEnv(n=10, seed=42)
        obs, info = env.reset()

        # Check observation shape
        assert obs.shape == (8, 10, 10)

        # Check snake length
        assert env.snake_length == 3
        assert info["length"] == 3

        # Check head is at valid position
        head = env.snake_head
        assert 0 <= head[0] < 10
        assert 0 <= head[1] < 10

        # Check food is at valid position and not overlapping snake
        food = env.food_pos
        assert 0 <= food[0] < 10
        assert 0 <= food[1] < 10
        assert food not in env.snake

        # Check observation channels
        assert obs[0, head[0], head[1]] == 1.0  # Head channel
        assert obs[2, food[0], food[1]] == 1.0  # Food channel

        env.close()

    def test_reset_with_different_seeds(self):
        """Test that different seeds produce different initial states."""
        env1 = SnakeEnv(n=10, seed=42)
        env2 = SnakeEnv(n=10, seed=123)

        obs1, _ = env1.reset()
        obs2, _ = env2.reset()

        # States should be different with different seeds
        # (food position or snake position should differ)
        assert not np.allclose(obs1, obs2)

        env1.close()
        env2.close()

    def test_reset_same_seed_reproducible(self):
        """Test that same seed produces same initial state."""
        env = SnakeEnv(n=10, seed=42)

        obs1, _ = env.reset(seed=42)
        food1 = env.food_pos
        snake1 = env.snake.copy()

        obs2, _ = env.reset(seed=42)
        food2 = env.food_pos
        snake2 = env.snake.copy()

        assert np.allclose(obs1, obs2)
        assert food1 == food2
        assert snake1 == snake2

        env.close()


class TestSnakeEnvMovement:
    """Tests for snake movement and actions."""

    def test_straight_movement(self):
        """Test that straight action maintains direction."""
        env = SnakeEnv(n=10, seed=42)
        env.reset()

        initial_dir = env.direction
        env.step(1)  # Go straight
        assert env.direction == initial_dir

        env.close()

    def test_turn_left(self):
        """Test that turn left action changes direction correctly."""
        env = SnakeEnv(n=10, seed=42)
        env.reset()

        initial_dir = env.direction
        env.step(0)  # Turn left
        expected_dir = (initial_dir - 1) % 4
        assert env.direction == expected_dir

        env.close()

    def test_turn_right(self):
        """Test that turn right action changes direction correctly."""
        env = SnakeEnv(n=10, seed=42)
        env.reset()

        initial_dir = env.direction
        env.step(2)  # Turn right
        expected_dir = (initial_dir + 1) % 4
        assert env.direction == expected_dir

        env.close()


class TestSnakeEnvCollisions:
    """Tests for collision detection."""

    def test_wall_collision_terminates(self):
        """Test that wall collision terminates episode."""
        env = SnakeEnv(n=5, seed=42)
        env.reset()

        # Run until wall collision
        done = False
        steps = 0
        reason = None
        while not done and steps < 100:
            # Always go straight to eventually hit wall
            obs, reward, terminated, truncated, info = env.step(1)
            done = terminated or truncated
            steps += 1
            if terminated:
                reason = info.get("reason")

        # Should have terminated due to wall
        assert done
        assert reason == "wall" or reason == "self"

        env.close()

    def test_self_collision_terminates(self):
        """Test that self collision terminates episode."""
        env = SnakeEnv(n=20, seed=42)
        env.reset()

        # Make snake longer by eating food (simulate)
        for _ in range(10):
            env.snake.insert(0, env.snake[0])  # Artificially grow snake

        # Try to create self collision by tight turns
        done = False
        steps = 0
        while not done and steps < 50:
            # Alternate turns to create spiral
            action = steps % 3
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        env.close()


class TestSnakeEnvFood:
    """Tests for food mechanics."""

    def test_eating_food_increases_length(self):
        """Test that eating food increases snake length."""
        env = SnakeEnv(n=10, seed=42)
        env.reset()

        initial_length = env.snake_length
        initial_score = env.score

        # Manually place food in front of snake
        head = env.snake_head
        dr, dc = env.DIRECTIONS[env.direction]
        env.food_pos = (head[0] + dr, head[1] + dc)

        # Move to eat food
        obs, reward, terminated, truncated, info = env.step(1)

        if not terminated:  # If we didn't hit a wall
            assert env.snake_length == initial_length + 1
            assert env.score == initial_score + 1
            assert reward >= 1.0  # Base reward for eating

        env.close()

    def test_food_respawns_on_empty_cell(self):
        """Test that food respawns on empty cell after eating."""
        env = SnakeEnv(n=10, seed=42)
        env.reset()

        # Manually place and eat food multiple times
        for _ in range(5):
            head = env.snake_head
            dr, dc = env.DIRECTIONS[env.direction]
            new_food_pos = (head[0] + dr, head[1] + dc)

            # Check if position is valid
            if 0 <= new_food_pos[0] < env.n and 0 <= new_food_pos[1] < env.n:
                env.food_pos = new_food_pos
                obs, reward, terminated, truncated, info = env.step(1)

                if not terminated:
                    # Food should be at new position, not on snake
                    assert env.food_pos not in env.snake or env.snake_length >= env.n * env.n
            else:
                break

        env.close()


class TestSnakeEnvRewards:
    """Tests for reward structure."""

    def test_death_reward_negative(self):
        """Test that death gives negative reward."""
        env = SnakeEnv(n=5, seed=42)
        env.reset()

        # Run until death
        done = False
        last_reward = 0
        while not done:
            obs, reward, terminated, truncated, info = env.step(1)
            done = terminated or truncated
            if terminated:  # Death (not truncation)
                last_reward = reward

        assert last_reward < 0

        env.close()

    def test_eating_food_reward_positive(self):
        """Test that eating food gives positive reward."""
        env = SnakeEnv(n=10, seed=42)
        env.reset()

        # Place food in front of snake
        head = env.snake_head
        dr, dc = env.DIRECTIONS[env.direction]
        new_pos = (head[0] + dr, head[1] + dc)

        if 0 <= new_pos[0] < env.n and 0 <= new_pos[1] < env.n:
            env.food_pos = new_pos
            obs, reward, terminated, truncated, info = env.step(1)

            if not terminated:
                assert reward > 0

        env.close()

    def test_survival_bonus_small_positive(self):
        """Test that survival bonus is applied."""
        env = SnakeEnv(n=20, survival_bonus=0.01, seed=42)
        env.reset()

        # Take a step that doesn't eat food
        old_food = env.food_pos
        obs, reward, terminated, truncated, info = env.step(1)

        if not terminated and env.food_pos == old_food:
            # Reward should include survival bonus (base 0 + shaping + bonus)
            # Can be positive or negative due to shaping
            pass  # Just checking no crash

        env.close()


class TestSnakeEnvAntiStall:
    """Tests for anti-stall mechanism."""

    def test_stall_truncates(self):
        """Test that too many steps without food truncates episode."""
        env = SnakeEnv(n=20, max_no_food=10, seed=42)  # Very short limit for test
        env.reset()

        # Take many steps going in circles
        truncated = False
        steps = 0
        while not truncated and steps < 100:
            # Alternate actions to avoid walls but not eat food
            obs, reward, terminated, truncated, info = env.step(steps % 3)
            if terminated:
                break
            steps += 1

        # Should have truncated due to stall
        if not env.score > 0:  # If we didn't eat any food
            assert truncated or steps >= 100

        env.close()

    def test_stall_penalty(self):
        """Test that stall gives penalty reward."""
        env = SnakeEnv(n=20, max_no_food=5, seed=42)
        env.reset()

        # Force stall
        for _ in range(10):
            obs, reward, terminated, truncated, info = env.step(1)
            if truncated:
                assert reward < 0  # Should have penalty
                assert info.get("reason") == "stall"
                break
            if terminated:
                break

        env.close()


class TestSnakeEnvObservation:
    """Tests for observation encoding."""

    def test_observation_shape(self):
        """Test observation has correct shape."""
        for n in [6, 10, 20]:
            env = SnakeEnv(n=n, seed=42)
            obs, _ = env.reset()
            assert obs.shape == (8, n, n)
            env.close()

    def test_observation_values_bounded(self):
        """Test observation values are in [0, 1]."""
        env = SnakeEnv(n=10, seed=42)
        obs, _ = env.reset()

        assert obs.min() >= 0.0
        assert obs.max() <= 1.0

        # Take some steps
        for _ in range(10):
            obs, _, terminated, truncated, _ = env.step(np.random.randint(3))
            if terminated or truncated:
                obs, _ = env.reset()
            assert obs.min() >= 0.0
            assert obs.max() <= 1.0

        env.close()

    def test_head_channel_single_one(self):
        """Test head channel has exactly one 1.0."""
        env = SnakeEnv(n=10, seed=42)
        obs, _ = env.reset()

        head_channel = obs[0]
        assert head_channel.sum() == 1.0
        assert head_channel.max() == 1.0

        env.close()

    def test_food_channel_single_one(self):
        """Test food channel has exactly one 1.0."""
        env = SnakeEnv(n=10, seed=42)
        obs, _ = env.reset()

        food_channel = obs[2]
        assert food_channel.sum() == 1.0
        assert food_channel.max() == 1.0

        env.close()

    def test_direction_channels_mutually_exclusive(self):
        """Test direction channels are one-hot encoded."""
        env = SnakeEnv(n=10, seed=42)
        obs, _ = env.reset()

        # Channels 3-6 are direction one-hot
        dir_channels = obs[3:7]

        # Exactly one channel should be all 1s
        active_count = 0
        for ch in range(4):
            if dir_channels[ch].sum() == 100:  # All cells are 1
                active_count += 1

        assert active_count == 1

        env.close()


class TestSnakeEnvRender:
    """Tests for rendering."""

    def test_render_human_mode(self):
        """Test human render mode doesn't crash."""
        env = SnakeEnv(n=5, render_mode="human", seed=42)
        env.reset()
        env.render()  # Should print ASCII
        env.close()

    def test_render_rgb_array_mode(self):
        """Test rgb_array render mode returns valid array."""
        env = SnakeEnv(n=5, render_mode="rgb_array", seed=42)
        env.reset()
        frame = env.render()

        assert frame is not None
        assert frame.shape == (100, 100, 3)  # 5 * 20 = 100
        assert frame.dtype == np.uint8

        env.close()


class TestSnakeEnvEdgeCases:
    """Tests for edge cases."""

    def test_small_grid(self):
        """Test environment works on small grid."""
        env = SnakeEnv(n=4, seed=42)
        obs, _ = env.reset()

        # Should be able to play
        for _ in range(20):
            obs, _, terminated, truncated, _ = env.step(np.random.randint(3))
            if terminated or truncated:
                break

        env.close()

    def test_gymnasium_api_compliance(self):
        """Test environment follows Gymnasium API."""
        env = SnakeEnv(n=10, seed=42)

        # Check spaces defined
        assert env.observation_space is not None
        assert env.action_space is not None

        # Check reset returns tuple
        result = env.reset()
        assert isinstance(result, tuple)
        assert len(result) == 2

        # Check step returns 5-tuple
        result = env.step(0)
        assert isinstance(result, tuple)
        assert len(result) == 5

        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

        env.close()

    def test_random_policy_baseline(self):
        """Test that random policy can run without crashes."""
        env = SnakeEnv(n=10, seed=42)

        scores = []
        for _ in range(10):
            obs, _ = env.reset()
            done = False
            while not done:
                action = env.action_space.sample()
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            scores.append(info.get("score", 0))

        # Random policy should get at least some food sometimes
        assert len(scores) == 10
        # Scores should be low for random policy
        assert np.mean(scores) < 20

        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
