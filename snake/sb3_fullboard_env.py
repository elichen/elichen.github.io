"""Gymnasium-compatible full-board Snake environment for SB3 training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


Action = int
Position = Tuple[int, int]


@dataclass
class RewardConfig:
    """Configurable reward shaping parameters."""

    step_penalty: float = 0.0
    food_reward: float = 1.0
    food_fill_bonus: float = 0.1
    death_penalty: float = -1.0
    distance_scale: float = 0.1
    starvation_penalty: float = -0.5


class FullBoardSnakeEnv(gym.Env):
    """Snake environment that exposes the full grid as observations."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    ACTION_TO_DELTA: List[Tuple[int, int]] = [
        (0, -1),  # Up
        (1, 0),  # Right
        (0, 1),  # Down
        (-1, 0),  # Left
    ]

    def __init__(
        self,
        grid_size: int = 20,
        *,
        vision_channels: int = 4,
        max_steps_without_food: Optional[int] = None,
        reward_config: Optional[RewardConfig] = None,
        enable_self_play_block: bool = True,
        seed: Optional[int] = None,
        observation_grid_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.max_cells = grid_size * grid_size
        self.vision_channels = vision_channels
        self.obs_grid_size = observation_grid_size or grid_size
        if self.obs_grid_size < self.grid_size:
            raise ValueError("observation_grid_size must be >= grid_size")
        self.board_shape = (self.vision_channels, self.obs_grid_size, self.obs_grid_size)
        self.reward_config = reward_config or RewardConfig()
        self.max_steps_without_food = max_steps_without_food or int(2.5 * grid_size * grid_size)
        self.enable_self_play_block = enable_self_play_block
        self._obs_buffer = np.zeros(self.board_shape, dtype=np.float32)

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=self.board_shape,
                    dtype=np.float32,
                ),
                "stats": spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32),
            }
        )

        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.snake: List[Position] = []
        self.direction: Position = (0, -1)
        self.food: Optional[Position] = None
        self.steps: int = 0
        self.steps_since_food: int = 0
        self.prev_distance: float = 0.0
        self.score: int = 0

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #
    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, int]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        center = self.grid_size // 2
        self.snake = [(center, center)]
        self.direction = (0, -1)
        self.food = self._sample_food()
        self.steps = 0
        self.steps_since_food = 0
        self.score = 0
        self.prev_distance = self._manhattan_distance(self.snake[0], self.food)

        obs = self._get_observation()
        info = self._build_info()
        return obs, info

    def step(
        self, action: Action
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        assert self.action_space.contains(action), f"Invalid action {action}"
        reward = self.reward_config.step_penalty

        chosen_delta = self.ACTION_TO_DELTA[action]
        if self.enable_self_play_block and len(self.snake) > 1:
            opposite = (-self.direction[0], -self.direction[1])
            if chosen_delta == opposite:
                chosen_delta = self.direction

        self.direction = chosen_delta
        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)
        self.steps += 1
        self.steps_since_food += 1

        terminated = False
        truncated = False

        # Collision detection
        if self._is_collision(new_head):
            reward += self.reward_config.death_penalty
            terminated = True
        else:
            self.snake.insert(0, new_head)
            ate_food = self.food is not None and new_head == self.food

            if ate_food:
                fill_ratio = len(self.snake) / self.max_cells
                reward += self.reward_config.food_reward + self.reward_config.food_fill_bonus * fill_ratio
                self.score += 1
                self.steps_since_food = 0
                self.food = self._sample_food()
            else:
                self.snake.pop()  # move tail forward

            if self.food is not None:
                new_distance = self._manhattan_distance(new_head, self.food)
                reward += self.reward_config.distance_scale * (self.prev_distance - new_distance)
                self.prev_distance = new_distance

        if not terminated and self.steps_since_food >= self.max_steps_without_food:
            reward += self.reward_config.starvation_penalty
            truncated = True

        obs = self._get_observation()
        info = self._build_info()
        if len(self.snake) >= self.max_cells:
            terminated = True
            info["perfect_game"] = True

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _sample_food(self) -> Optional[Position]:
        empty_cells = {(x, y) for x in range(self.grid_size) for y in range(self.grid_size)}
        for x, y in self.snake:
            empty_cells.discard((x, y))
        if not empty_cells:
            return None
        idx = self.np_random.integers(len(empty_cells))
        return list(empty_cells)[idx]

    def _is_collision(self, position: Position) -> bool:
        x, y = position
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return True
        return position in self.snake

    def _manhattan_distance(self, a: Position, b: Optional[Position]) -> float:
        if b is None:
            return 0.0
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _get_observation(self) -> Dict[str, np.ndarray]:
        self._obs_buffer.fill(0.0)
        head = self.snake[0]

        # Walls channel (index 3) - default everything to wall, then carve out playable area.
        self._obs_buffer[3, :, :] = 1.0
        self._obs_buffer[3, 0 : self.grid_size, 0 : self.grid_size] = 0.0

        # Snake body (channel 1) and head (channel 0)
        for segment in self.snake[1:]:
            if 0 <= segment[0] < self.obs_grid_size and 0 <= segment[1] < self.obs_grid_size:
                self._obs_buffer[1, segment[1], segment[0]] = 1.0
        if 0 <= head[0] < self.obs_grid_size and 0 <= head[1] < self.obs_grid_size:
            self._obs_buffer[0, head[1], head[0]] = 1.0

        # Food channel (index 2)
        if self.food is not None:
            self._obs_buffer[2, self.food[1], self.food[0]] = 1.0

        length_norm = len(self.snake) / self.max_cells
        dist_norm = (
            self._manhattan_distance(head, self.food) / (2 * self.grid_size)
            if self.food is not None
            else 0.0
        )
        steps_norm = min(1.0, self.steps_since_food / self.max_steps_without_food)
        direction_x = float(self.direction[0])
        direction_y = float(self.direction[1])
        score_norm = self.score / self.max_cells

        stats = np.array(
            [length_norm, dist_norm, steps_norm, direction_x, direction_y, score_norm],
            dtype=np.float32,
        )

        return {"board": self._obs_buffer.copy(), "stats": stats}

    def _build_info(self) -> Dict[str, float]:
        return {
            "score": self.score,
            "length": len(self.snake),
            "grid_fill": len(self.snake) / self.max_cells,
            "steps": self.steps,
            "steps_since_food": self.steps_since_food,
        }


def make_env(
    grid_size: int,
    *,
    seed: int,
    reward_config: Optional[RewardConfig] = None,
    observation_grid_size: Optional[int] = None,
) -> gym.Env:
    """Factory used by SB3 VecEnvs."""

    def _init() -> gym.Env:
        env = FullBoardSnakeEnv(
            grid_size=grid_size,
            reward_config=reward_config,
            seed=seed,
            observation_grid_size=observation_grid_size,
        )
        return env

    return _init
