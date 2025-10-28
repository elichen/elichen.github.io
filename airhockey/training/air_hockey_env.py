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
        # 8 features: paddle pos, puck pos, paddle vel, puck vel (all in player's half-rink coords 0-1)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.render_mode = render_mode

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
        p1_hit, p2_hit = self._update_puck()
        goal_scored_by = self._check_goals()

        # Dense reward shaping to break defensive Nash equilibrium
        puck_speed = np.linalg.norm(self.puck_vel)
        dist_to_puck = np.linalg.norm(self.puck_pos - self.paddle1_pos)
        offensive_pos = max(0, (self.puck_pos[1] - self.paddle1_pos[1]) / self.height)

        reward = -0.001  # Small penalty per timestep
        reward += 0.001 * puck_speed  # Encourage puck movement
        reward += 0.01 * offensive_pos  # Reward being behind puck (offensive)
        reward -= 0.005 * (dist_to_puck / self.width)  # Stay near puck

        if p1_hit: reward += 0.1  # Reward hitting puck
        if goal_scored_by == 1: reward = 1.0
        elif goal_scored_by == 2: reward = -1.0

        terminated = (goal_scored_by > 0) or (self.frame_count >= self.max_frames)
        return self._get_observation(1), reward, terminated, False, {"goal_scored_by": goal_scored_by}

    def _get_observation(self, player):
        # 8 features: all normalized to full rink 0-1 from player's perspective
        if player == 1:
            # P1: bottom, 0=P1 goal (y=height), 1=P2 goal (y=0)
            paddle_x = self.paddle1_pos[0] / self.width
            paddle_y = (self.height - self.paddle1_pos[1]) / self.height
            puck_x = self.puck_pos[0] / self.width
            puck_y = (self.height - self.puck_pos[1]) / self.height
            paddle_dx = (self.paddle1_vel[0] / self.max_speed + 1) / 2
            paddle_dy = (-self.paddle1_vel[1] / self.max_speed + 1) / 2
            puck_dx = (self.puck_vel[0] / self.max_speed + 1) / 2
            puck_dy = (-self.puck_vel[1] / self.max_speed + 1) / 2
        else:
            # P2: top, 0=P2 goal (y=0), 1=P1 goal (y=height)
            paddle_x = self.paddle2_pos[0] / self.width
            paddle_y = self.paddle2_pos[1] / self.height
            puck_x = self.puck_pos[0] / self.width
            puck_y = self.puck_pos[1] / self.height
            paddle_dx = (self.paddle2_vel[0] / self.max_speed + 1) / 2
            paddle_dy = (self.paddle2_vel[1] / self.max_speed + 1) / 2
            puck_dx = (self.puck_vel[0] / self.max_speed + 1) / 2
            puck_dy = (self.puck_vel[1] / self.max_speed + 1) / 2

        return np.clip([paddle_x, paddle_y, puck_x, puck_y, paddle_dx, paddle_dy, puck_dx, puck_dy], 0, 1).astype(np.float32)

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

        # Paddle collisions - track which paddle hit
        p1_hit = self._check_paddle_collision(self.paddle1_pos, self.paddle1_vel)
        p2_hit = self._check_paddle_collision(self.paddle2_pos, self.paddle2_vel)

        # Speed limit
        speed = np.linalg.norm(self.puck_vel)
        if speed > self.max_speed:
            self.puck_vel = self.puck_vel / speed * self.max_speed

        return p1_hit, p2_hit

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
            return True
        return False

    def _in_goal(self):
        goal_left, goal_right = (self.width - self.goal_width) / 2, (self.width + self.goal_width) / 2
        return goal_left <= self.puck_pos[0] <= goal_right

    def _check_goals(self):
        if self.puck_pos[1] <= 0 and self._in_goal():
            return 1  # Player 1 scores
        elif self.puck_pos[1] >= self.height and self._in_goal():
            return 2  # Player 2 scores
        return 0