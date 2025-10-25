import gymnasium as gym
import numpy as np
from gymnasium import spaces

class AirHockeyEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()
        self.width, self.height = 600, 800
        self.friction, self.max_speed, self.paddle_speed = 0.98, 25, 10
        self.puck_radius, self.paddle_radius, self.goal_width = 15, 20, 200

        self.puck_pos = np.array([300.0, 400.0])
        self.puck_vel = np.array([0.0, 0.0])
        self.paddle1_pos = np.array([300.0, 750.0])
        self.paddle1_vel = np.array([0.0, 0.0])
        self.paddle2_pos = np.array([300.0, 50.0])
        self.paddle2_vel = np.array([0.0, 0.0])
        self.frame_count = 0
        self.max_frames = 3000
        # V5: 12 features - puck-focused (no opponent observations)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.render_mode = render_mode
        self.use_opponent_obs = False  # V5: No opponent tracking

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.puck_pos = np.array([self.width/2 + np.random.uniform(-50, 50),
                                  self.height/2 + np.random.uniform(-50, 50)])
        self.puck_vel = np.random.uniform(-2, 2, size=2)
        self.paddle1_pos = np.array([self.width/2, self.height - 50.0])
        self.paddle1_vel = np.array([0.0, 0.0])
        self.paddle2_pos = np.array([self.width/2, 50.0])
        self.paddle2_vel = np.array([0.0, 0.0])
        self.frame_count = 0
        return self._get_observation(1), {}

    def step(self, action):
        self.frame_count += 1
        if isinstance(action, dict):
            action_p1 = action.get('player1', np.array([0.0, 0.0]))
            action_p2 = action.get('player2', np.array([0.0, 0.0]))
        else:
            action_p1, action_p2 = action, np.array([0.0, 0.0])

        self._move_paddle(1, action_p1)
        self._move_paddle(2, action_p2)
        self._update_puck()
        goal_scored_by = self._check_goals()

        # Sparse rewards
        reward = 0.0
        if goal_scored_by == 1: reward = 0.67
        elif goal_scored_by == 2: reward = -1.0
        if self.frame_count >= self.max_frames: reward = -0.33

        # DENSE REWARDS to break defensive deadlock
        puck_speed = np.linalg.norm(self.puck_vel)
        reward += 0.001 * puck_speed  # Reward fast puck

        puck_in_opp_half = self.puck_pos[1] < self.height / 2
        reward += 0.01 if puck_in_opp_half else -0.01  # Reward offensive positioning

        puck_moving_to_goal = self.puck_vel[1] < -0.1  # Moving toward opponent goal
        reward += 0.005 if puck_moving_to_goal else 0  # Reward offensive direction

        dist_to_puck = np.linalg.norm(self.puck_pos - self.paddle1_pos)
        reward -= 0.005 * (dist_to_puck / np.sqrt(self.width**2 + self.height**2))  # Reward puck engagement

        terminated = (goal_scored_by > 0) or (self.frame_count >= self.max_frames)
        return self._get_observation(1), reward, terminated, False, {"goal_scored_by": goal_scored_by}

    def _get_observation(self, player):
        if player == 1:
            paddle_pos, paddle_vel = self.paddle1_pos, self.paddle1_vel
            opponent_pos, opponent_vel = self.paddle2_pos, self.paddle2_vel
            puck_pos, puck_vel = self.puck_pos, self.puck_vel
        else:
            paddle_pos = np.array([self.paddle2_pos[0], self.height - self.paddle2_pos[1]])
            paddle_vel = np.array([self.paddle2_vel[0], -self.paddle2_vel[1]])
            opponent_pos = np.array([self.paddle1_pos[0], self.height - self.paddle1_pos[1]])
            opponent_vel = np.array([self.paddle1_vel[0], -self.paddle1_vel[1]])
            puck_pos = np.array([self.puck_pos[0], self.height - self.puck_pos[1]])
            puck_vel = np.array([self.puck_vel[0], -self.puck_vel[1]])

        puck_rel = (puck_pos - paddle_pos) / [self.width, self.height]
        puck_v = puck_vel / self.max_speed
        dist = np.linalg.norm(puck_pos - paddle_pos) / np.sqrt(self.width**2 + self.height**2)
        angle = np.arctan2(puck_pos[1] - paddle_pos[1], puck_pos[0] - paddle_pos[0]) / np.pi
        behind = 1.0 if puck_pos[1] > paddle_pos[1] else 0.0
        goal_dist = (self.height - paddle_pos[1]) / self.height * 2 - 1
        puck_goal_dist = (self.height - puck_pos[1]) / self.height * 2 - 1
        paddle_v = np.clip(paddle_vel / self.max_speed, -1, 1)
        puck_spd = np.linalg.norm(puck_vel) / self.max_speed

        # V5: 12 features - puck-focused only (NO opponent observations)
        return np.array([puck_rel[0], puck_rel[1], puck_v[0], puck_v[1],
                        dist, angle, behind, goal_dist, puck_goal_dist,
                        paddle_v[0], paddle_v[1], puck_spd], dtype=np.float32)

    def get_observation_for_player(self, player):
        """Proper method for getting player observations (fixes evaluation bug)"""
        return self._get_observation(player)

    def _move_paddle(self, player, action):
        paddle_pos = self.paddle1_pos if player == 1 else self.paddle2_pos
        paddle_vel = self.paddle1_vel if player == 1 else self.paddle2_vel
        y_min = self.height/2 + self.paddle_radius if player == 1 else self.paddle_radius
        y_max = self.height - self.paddle_radius if player == 1 else self.height/2 - self.paddle_radius

        dx, dy = action[0] * self.paddle_speed, action[1] * self.paddle_speed * (-1 if player == 2 else 1)
        paddle_vel[0] = paddle_vel[0] * 0.6 + dx * 0.4
        paddle_vel[1] = paddle_vel[1] * 0.6 + dy * 0.4
        paddle_pos[0] = np.clip(paddle_pos[0] + paddle_vel[0], self.paddle_radius, self.width - self.paddle_radius)
        paddle_pos[1] = np.clip(paddle_pos[1] + paddle_vel[1], y_min, y_max)

    def _update_puck(self):
        self.puck_pos += self.puck_vel
        self.puck_vel *= self.friction

        # Wall collisions
        if self.puck_pos[0] <= self.puck_radius or self.puck_pos[0] >= self.width - self.puck_radius:
            self.puck_vel[0] *= -0.8
            self.puck_pos[0] = np.clip(self.puck_pos[0], self.puck_radius, self.width - self.puck_radius)
        if self.puck_pos[1] <= self.puck_radius or self.puck_pos[1] >= self.height - self.puck_radius:
            if not self._in_goal():
                self.puck_vel[1] *= -0.8
                self.puck_pos[1] = np.clip(self.puck_pos[1], self.puck_radius, self.height - self.puck_radius)

        # Paddle collisions
        self._check_paddle_collision(self.paddle1_pos, self.paddle1_vel)
        self._check_paddle_collision(self.paddle2_pos, self.paddle2_vel)

        # Speed limit
        speed = np.linalg.norm(self.puck_vel)
        if speed > self.max_speed:
            self.puck_vel = self.puck_vel / speed * self.max_speed

    def _check_paddle_collision(self, paddle_pos, paddle_vel):
        dist = np.linalg.norm(self.puck_pos - paddle_pos)
        if dist < self.puck_radius + self.paddle_radius and dist > 0:
            normal = (self.puck_pos - paddle_pos) / dist
            overlap = self.puck_radius + self.paddle_radius - dist
            self.puck_pos += normal * overlap
            relative_vel = self.puck_vel - paddle_vel
            vel_along_normal = np.dot(relative_vel, normal)
            if vel_along_normal < 0:
                impulse = -2 * vel_along_normal * normal
                self.puck_vel += impulse
                self.puck_vel += paddle_vel * 0.3

    def _in_goal(self):
        goal_left, goal_right = (self.width - self.goal_width) / 2, (self.width + self.goal_width) / 2
        return goal_left <= self.puck_pos[0] <= goal_right

    def _check_goals(self):
        if self.puck_pos[1] <= 0 and self._in_goal():
            return 1  # Player 1 scores
        elif self.puck_pos[1] >= self.height and self._in_goal():
            return 2  # Player 2 scores
        return 0