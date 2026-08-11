import unittest

import numpy as np

try:
    from .air_hockey_env import AirHockeyEnv
except ImportError:
    from air_hockey_env import AirHockeyEnv


class PhysicsParityTest(unittest.TestCase):
    def setUp(self):
        self.env = AirHockeyEnv()
        self.env.reset(seed=11)

    def test_moving_paddle_replaces_puck_velocity_like_browser(self):
        self.env.paddle1_pos = np.array([300.0, 600.0])
        self.env.paddle1_vel = np.array([3.0, -4.0])
        self.env.puck_pos = np.array([300.0, 570.0])
        self.env.puck_vel = np.array([20.0, 7.0])

        self.assertTrue(self.env._check_paddle_collision(self.env.paddle1_pos, self.env.paddle1_vel))
        np.testing.assert_allclose(self.env.puck_pos, [300.0, 565.0], atol=1e-8)
        np.testing.assert_allclose(self.env.puck_vel, [5.4, -7.2], atol=1e-8)

    def test_slow_and_fast_paddle_collision_speed_limits(self):
        self.env.paddle1_pos = np.array([300.0, 600.0])
        self.env.puck_pos = np.array([330.0, 600.0])
        self.env._check_paddle_collision(self.env.paddle1_pos, np.array([1.0, 0.0]))
        np.testing.assert_allclose(self.env.puck_vel, [5.0, 0.0], atol=1e-8)

        self.env.puck_pos = np.array([330.0, 600.0])
        self.env._check_paddle_collision(self.env.paddle1_pos, np.array([20.0, 0.0]))
        np.testing.assert_allclose(self.env.puck_vel, [25.0, 0.0], atol=1e-8)

    def test_goal_thresholds_match_browser(self):
        self.env.puck_pos = np.array([300.0, 34.999])
        self.assertEqual(self.env._goal_side(), 1)
        self.env.puck_pos = np.array([300.0, 35.0])
        self.assertEqual(self.env._goal_side(), 0)
        self.env.puck_pos = np.array([300.0, 765.001])
        self.assertEqual(self.env._goal_side(), 2)
        self.env.puck_pos = np.array([200.0, 0.0])
        self.assertEqual(self.env._goal_side(), 0)

    def test_observation_contains_opponent_in_each_perspective(self):
        self.env.paddle1_pos = np.array([120.0, 700.0])
        self.env.paddle1_vel = np.array([5.0, -10.0])
        self.env.paddle2_pos = np.array([480.0, 100.0])
        self.env.paddle2_vel = np.array([-5.0, 10.0])
        self.env.puck_pos = np.array([360.0, 240.0])
        self.env.puck_vel = np.array([10.0, -20.0])

        p1 = self.env.get_observation_for_player(1)
        p2 = self.env.get_observation_for_player(2)
        self.assertEqual(p1.shape, (12,))
        np.testing.assert_allclose(p1[[0, 1, 8, 9]], [.2, .125, .8, .875])
        np.testing.assert_allclose(p2[[0, 1, 8, 9]], [.8, .125, .2, .875])
        np.testing.assert_allclose(p1[8:12], [.8, .875, .4, .3])
        np.testing.assert_allclose(p2[8:12], [.2, .875, .6, .3])


if __name__ == '__main__':
    unittest.main()
