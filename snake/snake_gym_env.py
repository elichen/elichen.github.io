"""
Custom Gym environment for Snake game optimized for PPO training.
Designed to achieve perfect gameplay (100% grid fill) through curriculum learning.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from collections import deque
from typing import Optional, Tuple, Dict, Any


class SnakeGymEnv(gym.Env):
    """
    Snake environment following Gym interface for stable-baselines3 training.

    State representation (24 features):
    - Direction one-hot (4): Current snake direction
    - Food direction (2): Normalized dx, dy to food
    - Danger detection (4): Walls/body in each direction
    - Distance to walls (4): Normalized distance in each direction
    - Snake length (1): Normalized by grid area
    - Grid fill ratio (1): Current coverage percentage
    - Connectivity features (4): Accessible cells in each direction
    - Food distance (1): Manhattan distance normalized
    - Body pattern (3): Straight segments, turns, loops detected

    Actions:
    - 0: Up
    - 1: Right
    - 2: Down
    - 3: Left

    Rewards:
    - Food eaten: +100 + length_bonus + efficiency_bonus
    - Milestone bonuses: 25%, 50%, 75%, 90%, 100% grid fill
    - Wall/self collision: -100
    - Starvation: -50
    - Step cost: -0.01
    - Moving toward food: +1
    - Moving away from food: -0.5
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self,
                 grid_size: int = 20,
                 render_mode: Optional[str] = None,
                 enable_connectivity: bool = True,
                 enable_milestones: bool = True,
                 adaptive_starvation: bool = True):
        """
        Initialize Snake environment.

        Args:
            grid_size: Size of the square grid (default 20x20)
            render_mode: Rendering mode ('human' or 'rgb_array')
            enable_connectivity: Include connectivity features in state
            enable_milestones: Give bonus rewards at grid fill milestones
            adaptive_starvation: Scale starvation timeout with snake length
        """
        super().__init__()

        self.grid_size = grid_size
        self.render_mode = render_mode
        self.enable_connectivity = enable_connectivity
        self.enable_milestones = enable_milestones
        self.adaptive_starvation = adaptive_starvation

        # Define action and observation spaces
        self.action_space = spaces.Discrete(4)

        # State space size depends on features enabled
        state_size = 24 if enable_connectivity else 16
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(state_size,), dtype=np.float32
        )

        # Game state variables
        self.snake = []
        self.direction = (0, -1)  # Start moving up
        self.food = None
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.game_over = False

        # Milestone tracking
        self.milestones_reached = set()
        self.max_score_achieved = 0

        # Statistics for logging
        self.episode_stats = {
            'total_reward': 0,
            'food_eaten': 0,
            'steps': 0,
            'collision_type': None
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        # Reset snake to center of grid
        center = self.grid_size // 2
        self.snake = [(center, center)]
        self.direction = (0, -1)  # Start moving up

        # Generate initial food
        self.food = self._generate_food()

        # Reset counters
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.game_over = False
        self.milestones_reached = set()

        # Reset episode stats
        self.episode_stats = {
            'total_reward': 0,
            'food_eaten': 0,
            'steps': 0,
            'collision_type': None
        }

        return self._get_observation(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        self.steps += 1
        self.steps_since_food += 1

        # Store previous distance to food for reward shaping
        prev_dist = self._manhattan_distance(self.snake[0], self.food)

        # Convert action to direction
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # Up, Right, Down, Left
        new_direction = directions[action]

        # Prevent moving backward into itself (optional)
        if len(self.snake) > 1:
            if (new_direction[0] == -self.direction[0] and
                new_direction[1] == -self.direction[1]):
                new_direction = self.direction  # Keep current direction

        self.direction = new_direction

        # Calculate new head position
        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])

        # Initialize reward
        reward = -0.01  # Small step cost for efficiency

        # Check wall collision
        if (new_head[0] < 0 or new_head[0] >= self.grid_size or
            new_head[1] < 0 or new_head[1] >= self.grid_size):
            self.game_over = True
            reward = -100
            self.episode_stats['collision_type'] = 'wall'

        # Check self collision
        elif new_head in self.snake:
            self.game_over = True
            reward = -100
            self.episode_stats['collision_type'] = 'self'

        # Valid move
        else:
            self.snake.insert(0, new_head)

            # Check if food eaten
            if new_head == self.food:
                self.score += 1
                self.steps_since_food = 0
                self.episode_stats['food_eaten'] += 1

                # Calculate food reward with bonuses
                base_reward = 100
                length_bonus = self.score * 2  # Increasing bonus per food

                # Efficiency bonus for eating food quickly
                efficiency_bonus = 0
                if self.steps_since_food < self.grid_size:
                    efficiency_bonus = 50

                reward = base_reward + length_bonus + efficiency_bonus

                # Milestone bonuses
                if self.enable_milestones:
                    grid_area = self.grid_size * self.grid_size
                    fill_ratio = self.score / grid_area

                    milestones = {
                        0.25: 500,   # 25% filled
                        0.50: 1000,  # 50% filled
                        0.75: 2000,  # 75% filled
                        0.90: 5000,  # 90% filled
                        0.99: 10000, # 99% filled (near perfect)
                        1.00: 50000  # PERFECT GAME!
                    }

                    for threshold, bonus in milestones.items():
                        if fill_ratio >= threshold and threshold not in self.milestones_reached:
                            reward += bonus
                            self.milestones_reached.add(threshold)
                            print(f"🎯 Milestone reached: {int(threshold*100)}% grid filled! Bonus: +{bonus}")

                # Generate new food
                self.food = self._generate_food()

                # Check if perfect game achieved
                if self.score >= self.grid_size * self.grid_size - 1:
                    print("🏆 PERFECT GAME ACHIEVED!")
                    self.game_over = True
            else:
                # Remove tail (snake doesn't grow)
                self.snake.pop()

                # Distance-based reward shaping
                new_dist = self._manhattan_distance(new_head, self.food)
                if new_dist < prev_dist:
                    reward += 1.0  # Moving toward food
                else:
                    reward -= 0.5  # Moving away from food

        # Check starvation
        max_steps = self._get_max_steps_without_food()
        if self.steps_since_food >= max_steps:
            self.game_over = True
            reward = -50
            self.episode_stats['collision_type'] = 'starvation'

        # Update statistics
        self.episode_stats['total_reward'] += reward
        self.episode_stats['steps'] = self.steps

        # Update max score
        self.max_score_achieved = max(self.max_score_achieved, self.score)

        # Prepare info dict
        info = {
            'score': self.score,
            'steps': self.steps,
            'episode_stats': self.episode_stats.copy() if self.game_over else {}
        }

        # Return step results
        observation = self._get_observation()
        terminated = self.game_over
        truncated = False  # Could add time limit here

        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Generate observation vector from current game state."""
        features = []

        # 1. Direction one-hot encoding (4 features)
        direction_map = {(0, -1): 0, (1, 0): 1, (0, 1): 2, (-1, 0): 3}
        direction_onehot = [0] * 4
        direction_onehot[direction_map.get(self.direction, 0)] = 1
        features.extend(direction_onehot)

        head = self.snake[0]

        # 2. Food direction (2 features)
        food_dx = (self.food[0] - head[0]) / self.grid_size
        food_dy = (self.food[1] - head[1]) / self.grid_size
        features.extend([food_dx, food_dy])

        # 3. Danger detection in 4 directions (4 features)
        dangers = []
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            next_pos = (head[0] + dx, head[1] + dy)
            is_danger = (
                next_pos[0] < 0 or next_pos[0] >= self.grid_size or
                next_pos[1] < 0 or next_pos[1] >= self.grid_size or
                next_pos in self.snake[1:]  # Don't include head
            )
            dangers.append(1 if is_danger else 0)
        features.extend(dangers)

        # 4. Distance to walls (4 features)
        wall_distances = [
            head[1] / self.grid_size,  # Distance to top
            (self.grid_size - 1 - head[0]) / self.grid_size,  # Distance to right
            (self.grid_size - 1 - head[1]) / self.grid_size,  # Distance to bottom
            head[0] / self.grid_size   # Distance to left
        ]
        features.extend(wall_distances)

        # 5. Snake length (1 feature)
        max_length = self.grid_size * self.grid_size
        features.append(len(self.snake) / max_length)

        # 6. Grid fill ratio (1 feature)
        features.append(self.score / (self.grid_size * self.grid_size))

        # 7. Food distance (1 feature)
        food_dist = self._manhattan_distance(head, self.food) / (2 * self.grid_size)
        features.append(food_dist)

        # 8. Body pattern features (3 features)
        if len(self.snake) >= 3:
            # Detect straight segments
            straight_count = sum(1 for i in range(1, len(self.snake) - 1)
                               if self._is_straight(i))
            features.append(straight_count / len(self.snake))

            # Detect turns
            turn_count = sum(1 for i in range(1, len(self.snake) - 1)
                           if not self._is_straight(i))
            features.append(turn_count / len(self.snake))

            # Detect potential loops (simplified)
            unique_x = len(set(s[0] for s in self.snake))
            unique_y = len(set(s[1] for s in self.snake))
            loop_indicator = 1.0 - (unique_x * unique_y) / (len(self.snake) ** 2)
            features.append(loop_indicator)
        else:
            features.extend([0, 0, 0])

        # 9. Connectivity features (4 features) - if enabled
        if self.enable_connectivity:
            connectivity = self._calculate_connectivity()
            features.extend(connectivity)

        return np.array(features, dtype=np.float32)

    def _generate_food(self) -> Tuple[int, int]:
        """Generate food at a random empty position."""
        empty_cells = [(x, y) for x in range(self.grid_size)
                      for y in range(self.grid_size)
                      if (x, y) not in self.snake]

        if empty_cells:
            return random.choice(empty_cells)
        else:
            # Grid is full (perfect game)
            return self.snake[0]  # Return head position as placeholder

    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _get_max_steps_without_food(self) -> int:
        """Get maximum steps allowed without eating food."""
        if self.adaptive_starvation:
            # Scale with snake length for longer snakes
            base = self.grid_size * 2
            length_factor = 1 + (len(self.snake) / self.grid_size)
            return int(base * length_factor)
        else:
            return self.grid_size * 2

    def _is_straight(self, index: int) -> bool:
        """Check if snake segment at index is part of a straight line."""
        if index <= 0 or index >= len(self.snake) - 1:
            return False

        prev_seg = self.snake[index - 1]
        curr_seg = self.snake[index]
        next_seg = self.snake[index + 1]

        # Check if all three points are on same horizontal or vertical line
        return ((prev_seg[0] == curr_seg[0] == next_seg[0]) or
                (prev_seg[1] == curr_seg[1] == next_seg[1]))

    def _calculate_connectivity(self) -> list:
        """Calculate number of accessible cells in each direction."""
        head = self.snake[0]
        connectivity = []

        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            accessible = 0
            x, y = head[0] + dx, head[1] + dy

            # BFS to count accessible cells in this direction (limited depth)
            visited = set()
            queue = [(x, y, 0)]  # (x, y, depth)
            max_depth = 5  # Limit search depth for efficiency

            while queue:
                cx, cy, depth = queue.pop(0)

                if depth > max_depth:
                    continue

                if (cx, cy) in visited:
                    continue

                if (cx < 0 or cx >= self.grid_size or
                    cy < 0 or cy >= self.grid_size or
                    (cx, cy) in self.snake):
                    continue

                visited.add((cx, cy))
                accessible += 1

                # Add neighbors
                for ndx, ndy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    queue.append((cx + ndx, cy + ndy, depth + 1))

            # Normalize by max possible accessible cells
            max_accessible = min(max_depth * 4, self.grid_size * self.grid_size)
            connectivity.append(accessible / max_accessible)

        return connectivity

    def render(self):
        """Render the game (placeholder for actual rendering)."""
        if self.render_mode == "human":
            # Simple text representation
            grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]

            # Place snake
            for i, (x, y) in enumerate(self.snake):
                if i == 0:
                    grid[y][x] = 'H'  # Head
                else:
                    grid[y][x] = 'S'  # Body

            # Place food
            grid[self.food[1]][self.food[0]] = 'F'

            # Print grid
            print("\n" + "=" * (self.grid_size + 2))
            for row in grid:
                print("|" + "".join(row) + "|")
            print("=" * (self.grid_size + 2))
            print(f"Score: {self.score} | Steps: {self.steps}")

    def close(self):
        """Clean up resources."""
        pass


# Test the environment
if __name__ == "__main__":
    env = SnakeGymEnv(grid_size=5)
    obs, _ = env.reset()

    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    # Run a few random steps
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        if terminated:
            print(f"Game over! Final score: {info['score']}")
            break