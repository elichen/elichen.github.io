import gymnasium as gym
import numpy as np
from gymnasium import spaces

class AirHockeyEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()
        self.width, self.height = 600, 800
        self.friction, self.max_speed, self.paddle_speed = 0.98, 25, 10
        self.puck_radius, self.paddle_radius, self.goal_width, self.goal_posts = 15, 20, 200, 20

        self.puck_pos = np.array([300.0, 400.0])
        self.puck_vel = np.array([0.0, 0.0])
        self.paddle1_pos = np.array([300.0, 750.0])
        self.paddle1_vel = np.array([0.0, 0.0])
        self.paddle2_pos = np.array([300.0, 50.0])
        self.paddle2_vel = np.array([0.0, 0.0])
        self.frame_count = 0
        self.max_frames = 3000
        self.same_position_time = 0
        self.side_wall_time = 0
        self.last_puck_pos = self.puck_pos.copy()
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(12,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        standard_serve = options.get('standard_serve')
        if standard_serve is None:
            standard_serve = self.np_random.random() < .5

        if standard_serve:
            serve_y = options.get('serve_y', self.np_random.choice([self.height / 4, self.height / 2, self.height * 3 / 4]))
            self.puck_pos = np.array([self.width / 2, serve_y], dtype=float)
            self.puck_vel = np.zeros(2)
            self.paddle1_pos = np.array([self.width / 2, self.height - 50], dtype=float)
            self.paddle2_pos = np.array([self.width / 2, 50], dtype=float)
        else:
            self.puck_pos = np.array([
                self.np_random.uniform(self.puck_radius + 50, self.width - self.puck_radius - 50),
                self.np_random.uniform(self.height * 0.2, self.height * 0.8)
            ])
            self.puck_vel = self.np_random.uniform(-5, 5, size=2)
            self.paddle1_pos = np.array([
                self.np_random.uniform(self.paddle_radius + 50, self.width - self.paddle_radius - 50),
                self.np_random.uniform(self.height * 0.65, self.height - 50.0)
            ])
            self.paddle2_pos = np.array([
                self.np_random.uniform(self.paddle_radius + 50, self.width - self.paddle_radius - 50),
                self.np_random.uniform(50.0, self.height * 0.35)
            ])

        self.paddle1_vel = np.array([0.0, 0.0])
        self.paddle2_vel = np.array([0.0, 0.0])

        self.frame_count = 0
        self.same_position_time = 0
        self.side_wall_time = 0
        self.last_puck_pos = self.puck_pos.copy()
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
        unstuck = False
        if not goal_scored_by and self._is_puck_stuck():
            self._unstick_puck()
            unstuck = True

        # Asymmetric rewards and stronger offensive incentives
        puck_speed = np.linalg.norm(self.puck_vel)
        dist_to_puck = np.linalg.norm(self.puck_pos - self.paddle1_pos)
        offensive_pos = max(0, (self.puck_pos[1] - self.paddle1_pos[1]) / self.height)
        puck_toward_goal = max(0, -self.puck_vel[1] / self.max_speed)  # Puck moving toward opponent goal

        reward = -0.002  # Slightly higher timestep penalty
        reward += 0.005 * puck_speed  # 5x: Encourage puck movement
        reward += 0.05 * offensive_pos  # 5x: Reward being behind puck (offensive)
        reward -= 0.01 * (dist_to_puck / self.width)  # 2x: Stay near puck
        reward += 0.1 * puck_toward_goal  # NEW: Reward shots toward goal

        if p1_hit:
            reward += 0.1  # Reward hitting puck
            if self.puck_vel[1] < -10:  # Strong forward hit
                reward += 0.1  # Extra reward for aggressive shots

        # Asymmetric goal rewards
        if goal_scored_by == 1:
            reward = 2.0  # Asymmetric: 2x reward for scoring
        elif goal_scored_by == 2:
            reward = -1.0  # Standard penalty for conceding

        # Timeout penalty to discourage defensive play
        terminated = (goal_scored_by > 0) or (self.frame_count >= self.max_frames)
        if self.frame_count >= self.max_frames:
            reward -= 0.5  # Penalty for timeout
        return self._get_observation(1), reward, terminated, False, {
            "goal_scored_by": goal_scored_by, "p1_hit": p1_hit, "p2_hit": p2_hit, "unstuck": unstuck
        }

    def _get_observation(self, player):
        # 12 features: own paddle, puck, and opponent paddle from the active player's perspective.
        if player == 1:
            # P1: bottom, 0=P1 goal (y=height), 1=P2 goal (y=0)
            paddle_x = self.paddle1_pos[0] / self.width
            paddle_y = (self.height - self.paddle1_pos[1]) / self.height
            puck_x = self.puck_pos[0] / self.width
            puck_y = (self.height - self.puck_pos[1]) / self.height
            # Properly normalize velocities to [0, 1] range with clipping
            paddle_dx = np.clip(self.paddle1_vel[0] / self.max_speed, -1, 1) * 0.5 + 0.5
            paddle_dy = np.clip(-self.paddle1_vel[1] / self.max_speed, -1, 1) * 0.5 + 0.5
            puck_dx = np.clip(self.puck_vel[0] / self.max_speed, -1, 1) * 0.5 + 0.5
            puck_dy = np.clip(-self.puck_vel[1] / self.max_speed, -1, 1) * 0.5 + 0.5
            opponent_x = self.paddle2_pos[0] / self.width
            opponent_y = (self.height - self.paddle2_pos[1]) / self.height
            opponent_dx = np.clip(self.paddle2_vel[0] / self.max_speed, -1, 1) * 0.5 + 0.5
            opponent_dy = np.clip(-self.paddle2_vel[1] / self.max_speed, -1, 1) * 0.5 + 0.5
        else:
            # P2: top, 0=P2 goal (y=0), 1=P1 goal (y=height)
            paddle_x = self.paddle2_pos[0] / self.width
            paddle_y = self.paddle2_pos[1] / self.height
            puck_x = self.puck_pos[0] / self.width
            puck_y = self.puck_pos[1] / self.height
            # Properly normalize velocities to [0, 1] range with clipping
            paddle_dx = np.clip(self.paddle2_vel[0] / self.max_speed, -1, 1) * 0.5 + 0.5
            paddle_dy = np.clip(self.paddle2_vel[1] / self.max_speed, -1, 1) * 0.5 + 0.5
            puck_dx = np.clip(self.puck_vel[0] / self.max_speed, -1, 1) * 0.5 + 0.5
            puck_dy = np.clip(self.puck_vel[1] / self.max_speed, -1, 1) * 0.5 + 0.5
            opponent_x = self.paddle1_pos[0] / self.width
            opponent_y = self.paddle1_pos[1] / self.height
            opponent_dx = np.clip(self.paddle1_vel[0] / self.max_speed, -1, 1) * 0.5 + 0.5
            opponent_dy = np.clip(self.paddle1_vel[1] / self.max_speed, -1, 1) * 0.5 + 0.5

        return np.clip([paddle_x, paddle_y, puck_x, puck_y, paddle_dx, paddle_dy, puck_dx, puck_dy,
                        opponent_x, opponent_y, opponent_dx, opponent_dy], 0, 1).astype(np.float32)

    def get_observation_for_player(self, player):
        """Proper method for getting player observations (fixes evaluation bug)"""
        return self._get_observation(player)

    def _move_paddle(self, player, action):
        paddle_pos = self.paddle1_pos if player == 1 else self.paddle2_pos
        paddle_vel = self.paddle1_vel if player == 1 else self.paddle2_vel
        y_min = self.height/2 + self.paddle_radius if player == 1 else self.paddle_radius
        y_max = self.height - self.paddle_radius if player == 1 else self.height/2 - self.paddle_radius

        requested = np.array([action[0] * self.paddle_speed,
                              action[1] * self.paddle_speed * (-1 if player == 2 else 1)])
        smoothed = paddle_vel * .6 + requested * .4
        previous = paddle_pos.copy()
        paddle_pos[:] = np.clip(previous + smoothed,
                                [self.paddle_radius, y_min],
                                [self.width - self.paddle_radius, y_max])
        paddle_vel[:] = paddle_pos - previous

    def _update_puck(self):
        self.puck_pos += self.puck_vel
        self.puck_vel *= self.friction

        if self.puck_pos[0] - self.puck_radius < 0:
            self.puck_pos[0], self.puck_vel[0] = self.puck_radius, self.puck_vel[0] * -.8
        if self.puck_pos[0] + self.puck_radius > self.width:
            self.puck_pos[0], self.puck_vel[0] = self.width - self.puck_radius, self.puck_vel[0] * -.8
        if not self._goal_side():
            if self.puck_pos[1] - self.puck_radius < 0:
                self.puck_pos[1], self.puck_vel[1] = self.puck_radius, self.puck_vel[1] * -.8
            if self.puck_pos[1] + self.puck_radius > self.height:
                self.puck_pos[1], self.puck_vel[1] = self.height - self.puck_radius, self.puck_vel[1] * -.8

        # Paddle collisions - track which paddle hit
        p1_hit = self._check_paddle_collision(self.paddle1_pos, self.paddle1_vel)
        p2_hit = self._check_paddle_collision(self.paddle2_pos, self.paddle2_vel)

        # Speed limit
        speed = np.linalg.norm(self.puck_vel)
        if speed > self.max_speed:
            self.puck_vel = self.puck_vel / speed * self.max_speed

        return p1_hit, p2_hit

    def _check_paddle_collision(self, paddle_pos, paddle_vel):
        delta = self.puck_pos - paddle_pos
        dist = np.linalg.norm(delta)
        if dist < self.puck_radius + self.paddle_radius:
            angle = np.arctan2(delta[1], delta[0])
            radius = self.puck_radius + self.paddle_radius
            self.puck_pos = paddle_pos + np.array([np.cos(angle), np.sin(angle)]) * radius
            self.puck_vel = paddle_vel * 1.8
            speed = np.linalg.norm(self.puck_vel)
            if 0 < speed < 5:
                self.puck_vel *= 5 / speed
            if speed > self.max_speed:
                self.puck_vel *= self.max_speed / speed
            return True
        return False

    def _in_goal(self):
        goal_left, goal_right = (self.width - self.goal_width) / 2, (self.width + self.goal_width) / 2
        return goal_left < self.puck_pos[0] < goal_right

    def _goal_side(self):
        if not self._in_goal():
            return 0
        if self.puck_pos[1] - self.puck_radius < self.goal_posts:
            return 1
        if self.puck_pos[1] + self.puck_radius > self.height - self.goal_posts:
            return 2
        return 0

    def _check_goals(self):
        return self._goal_side()

    def _is_puck_stuck(self):
        slow = np.all(np.abs(self.puck_vel) < .1)
        in_side_wall_zone = (self.puck_pos[0] < self.puck_radius + 60 or
                             self.puck_pos[0] > self.width - self.puck_radius - 60)
        self.side_wall_time = self.side_wall_time + 1 if in_side_wall_zone else 0
        near_wall = (self.puck_pos[0] - self.puck_radius < 10 or self.puck_pos[0] + self.puck_radius > self.width - 10 or
                     self.puck_pos[1] - self.puck_radius < 10 or self.puck_pos[1] + self.puck_radius > self.height - 10)
        if not near_wall:
            return self.side_wall_time > 180
        if np.linalg.norm(self.puck_pos - self.last_puck_pos) < 1:
            self.same_position_time += 1
        else:
            self.same_position_time = 0
            self.last_puck_pos = self.puck_pos.copy()
        return slow or self.same_position_time > 30 or self.side_wall_time > 180

    def _unstick_puck(self):
        self.puck_pos = self.np_random.uniform([120, 200], [self.width - 120, self.height - 200])
        self.puck_vel = self.np_random.uniform(-2.5, 2.5, 2)
        self.same_position_time = 0
        self.side_wall_time = 0
        self.last_puck_pos = self.puck_pos.copy()
