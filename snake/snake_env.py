"""
Snake Environment for Reinforcement Learning
Gymnasium-compatible environment with reward shaping and curriculum support.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any


class SnakeEnv(gym.Env):
    """
    Snake game environment for RL training.

    Observation: 8-channel grid (n x n):
        - Channel 0: Head position (1.0 at head)
        - Channel 1: Body positions (1.0 at all body segments)
        - Channel 2: Food position (1.0 at food)
        - Channel 3-6: Direction one-hot (up, right, down, left)
        - Channel 7: Normalized snake length (broadcast)
        - (Note: removed steps_since_food channel, keeping 8 channels)

    Action space: Discrete(3)
        - 0: Turn left
        - 1: Go straight
        - 2: Turn right

    Rewards:
        - +1.0 for eating food
        - -1.0 for death (wall or self-collision)
        - -0.5 for stall (too many steps without food)
        - Potential-based shaping reward for approaching food
        - Small survival bonus per step
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    # Direction constants: 0=up, 1=right, 2=down, 3=left
    DIRECTIONS = {
        0: (-1, 0),  # up
        1: (0, 1),   # right
        2: (1, 0),   # down
        3: (0, -1),  # left
    }

    def __init__(
        self,
        n: int = 20,
        max_no_food: Optional[int] = None,
        render_mode: Optional[str] = None,
        gamma: float = 0.995,
        alpha: float = 0.2,
        survival_bonus: float = 0.001,
        seed: Optional[int] = None,
    ):
        """
        Initialize the Snake environment.

        Args:
            n: Grid size (n x n)
            max_no_food: Max steps without food before truncation (default: 80 + 4*length)
            render_mode: "human" or "rgb_array"
            gamma: Discount factor for potential-based shaping
            alpha: Coefficient for distance-based potential
            survival_bonus: Small reward per non-terminal step
            seed: Random seed
        """
        super().__init__()

        self.n = n
        self.max_no_food_base = max_no_food
        self.render_mode = render_mode
        self.gamma = gamma
        self.alpha = alpha
        self.survival_bonus = survival_bonus

        # Action space: turn left, straight, turn right
        self.action_space = spaces.Discrete(3)

        # Observation space: 8 channels of n x n grid
        self.n_channels = 8
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_channels, n, n),
            dtype=np.float32,
        )

        # Initialize RNG
        self.rng = np.random.default_rng(seed)

        # Game state (initialized in reset)
        self.snake: list = []
        self.direction: int = 0
        self.food_pos: Tuple[int, int] = (0, 0)
        self.steps_since_food: int = 0
        self.score: int = 0
        self.prev_phi: float = 0.0
        self.total_steps: int = 0

    @property
    def snake_head(self) -> Tuple[int, int]:
        """Get head position (first element of snake list)."""
        return self.snake[0]

    @property
    def snake_length(self) -> int:
        """Get current snake length."""
        return len(self.snake)

    @property
    def max_no_food(self) -> int:
        """Calculate max steps without food based on current length."""
        if self.max_no_food_base is not None:
            return self.max_no_food_base
        return 80 + 4 * self.snake_length

    def _manhattan_distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _compute_phi(self) -> float:
        """
        Compute potential function for reward shaping.
        Phi = -alpha * normalized_manhattan_distance(head, food)
        """
        d = self._manhattan_distance(self.snake_head, self.food_pos)
        max_d = 2 * (self.n - 1)
        d_norm = d / max_d if max_d > 0 else 0.0
        return -self.alpha * d_norm

    def _place_food(self) -> None:
        """Place food at a random empty cell."""
        snake_set = set(self.snake)
        empty_cells = [
            (r, c) for r in range(self.n) for c in range(self.n)
            if (r, c) not in snake_set
        ]
        if empty_cells:
            idx = self.rng.integers(len(empty_cells))
            self.food_pos = empty_cells[idx]
        else:
            # Grid is full (game won)
            self.food_pos = (-1, -1)

    def _get_observation(self) -> np.ndarray:
        """
        Build the observation tensor.
        8 channels: head, body, food, dir_up, dir_right, dir_down, dir_left, length
        """
        obs = np.zeros((self.n_channels, self.n, self.n), dtype=np.float32)

        # Channel 0: Head
        hr, hc = self.snake_head
        obs[0, hr, hc] = 1.0

        # Channel 1: Body (all segments including head)
        for r, c in self.snake:
            obs[1, r, c] = 1.0

        # Channel 2: Food
        if self.food_pos[0] >= 0:  # Valid food position
            obs[2, self.food_pos[0], self.food_pos[1]] = 1.0

        # Channels 3-6: Direction one-hot (broadcast across grid)
        obs[3 + self.direction, :, :] = 1.0

        # Channel 7: Normalized snake length (broadcast)
        normalized_length = self.snake_length / (self.n * self.n)
        obs[7, :, :] = normalized_length

        return obs

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed
            options: Additional options (unused)

        Returns:
            observation: Initial observation
            info: Initial info dict
        """
        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Place snake in center, length 3
        center = self.n // 2

        # Random initial direction
        self.direction = self.rng.integers(4)

        # Build snake aligned with direction
        dr, dc = self.DIRECTIONS[self.direction]
        self.snake = []
        for i in range(3):
            # Head first, then body segments behind
            r = center - i * dr
            c = center - i * dc
            # Clamp to grid
            r = max(0, min(self.n - 1, r))
            c = max(0, min(self.n - 1, c))
            self.snake.append((r, c))

        # Place food
        self._place_food()

        # Reset counters
        self.steps_since_food = 0
        self.score = 0
        self.total_steps = 0
        self.prev_phi = self._compute_phi()

        obs = self._get_observation()
        info = {
            "length": self.snake_length,
            "score": self.score,
            "food_pos": self.food_pos,
        }

        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: 0=turn left, 1=straight, 2=turn right

        Returns:
            observation: New observation
            reward: Reward for this step
            terminated: True if game over (death)
            truncated: True if truncated (stall)
            info: Additional info
        """
        self.total_steps += 1

        # Map relative action to absolute direction
        delta = {0: -1, 1: 0, 2: 1}
        new_dir = (self.direction + delta[action]) % 4
        self.direction = new_dir

        # Compute new head position
        dr, dc = self.DIRECTIONS[new_dir]
        hr, hc = self.snake_head
        new_head = (hr + dr, hc + dc)

        terminated = False
        truncated = False
        base_reward = 0.0
        reason = None

        # Check wall collision
        if not (0 <= new_head[0] < self.n and 0 <= new_head[1] < self.n):
            terminated = True
            base_reward = -1.0
            reason = "wall"
        # Check self collision (excluding tail if it will move)
        elif new_head in self.snake[:-1]:
            terminated = True
            base_reward = -1.0
            reason = "self"
        # Check if tail stays (only if eating food, tail won't move)
        elif new_head == self.snake[-1] and new_head != self.food_pos:
            # Tail will move, so this is safe
            pass
        elif new_head in self.snake:
            terminated = True
            base_reward = -1.0
            reason = "self"

        if not terminated:
            # Check if eating food
            if new_head == self.food_pos:
                # Grow snake (don't remove tail)
                self.snake.insert(0, new_head)
                self.score += 1
                self.steps_since_food = 0
                base_reward = 1.0

                # Check if grid is full (perfect game)
                if self.snake_length >= self.n * self.n:
                    terminated = True
                    reason = "win"
                else:
                    self._place_food()
            else:
                # Move snake (remove tail)
                self.snake.insert(0, new_head)
                self.snake.pop()
                self.steps_since_food += 1

        # Anti-stall check
        if not terminated and self.steps_since_food > self.max_no_food:
            truncated = True
            base_reward += -0.5
            reason = "stall"

        # Reward shaping (potential-based)
        if not terminated:
            phi = self._compute_phi()
            r_shape = self.gamma * phi - self.prev_phi
            self.prev_phi = phi
        else:
            r_shape = 0.0

        # Total reward
        if terminated and reason != "win":
            reward = base_reward  # No shaping on death
        else:
            reward = base_reward + r_shape + self.survival_bonus

        # Build observation
        obs = self._get_observation()

        info = {
            "length": self.snake_length,
            "score": self.score,
            "reason": reason,
            "steps": self.total_steps,
            "food_pos": self.food_pos,
        }

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment.

        Returns:
            RGB array if render_mode is "rgb_array", None otherwise
        """
        if self.render_mode == "human":
            self._render_ascii()
            return None
        elif self.render_mode == "rgb_array":
            return self._render_rgb()
        return None

    def _render_ascii(self) -> None:
        """Print ASCII representation of the game."""
        snake_set = set(self.snake)
        head = self.snake_head

        print(f"\nScore: {self.score}  Length: {self.snake_length}  Steps: {self.total_steps}")
        print("+" + "-" * self.n + "+")

        for r in range(self.n):
            row = "|"
            for c in range(self.n):
                pos = (r, c)
                if pos == head:
                    row += "O"  # Head
                elif pos in snake_set:
                    row += "#"  # Body
                elif pos == self.food_pos:
                    row += "*"  # Food
                else:
                    row += " "
            row += "|"
            print(row)

        print("+" + "-" * self.n + "+")

    def _render_rgb(self) -> np.ndarray:
        """
        Render to RGB array for video logging.

        Returns:
            RGB array of shape (n*cell_size, n*cell_size, 3)
        """
        cell_size = 20
        img_size = self.n * cell_size
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

        # Background
        img[:, :] = [40, 40, 40]  # Dark gray

        # Grid lines
        for i in range(self.n + 1):
            pos = i * cell_size
            img[pos:pos+1, :] = [60, 60, 60]
            img[:, pos:pos+1] = [60, 60, 60]

        # Food
        fr, fc = self.food_pos
        if fr >= 0:
            r1, r2 = fr * cell_size + 2, (fr + 1) * cell_size - 2
            c1, c2 = fc * cell_size + 2, (fc + 1) * cell_size - 2
            img[r1:r2, c1:c2] = [255, 50, 50]  # Red

        # Snake body
        for i, (r, c) in enumerate(self.snake):
            r1, r2 = r * cell_size + 1, (r + 1) * cell_size - 1
            c1, c2 = c * cell_size + 1, (c + 1) * cell_size - 1
            if i == 0:
                img[r1:r2, c1:c2] = [50, 255, 50]  # Bright green (head)
            else:
                img[r1:r2, c1:c2] = [30, 180, 30]  # Darker green (body)

        return img

    def close(self) -> None:
        """Clean up resources."""
        pass


# Register environment with Gymnasium
gym.register(
    id="Snake-v0",
    entry_point="snake_env:SnakeEnv",
    max_episode_steps=10000,
)
