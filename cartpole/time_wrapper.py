import numpy as np
import gymnasium as gym

class AddTimeInfo(gym.core.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        if self.env.num_envs > 1:
            raise ValueError("AddTimeInfo only supports single environments")
        self.epi_time = -0.5
        if 'dm_control' in env.spec.id:
            self.time_limit = 1000
        else:
            self.time_limit = env.spec.max_episode_steps
        self.obs_space_size = self.observation_space.shape[0] + self.env.num_envs
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_space_size,), dtype=np.float32)
        if not (isinstance(self.action_space, gym.spaces.Box) or isinstance(self.action_space, gym.spaces.Discrete)):
            raise ValueError("Unsupported action space")

    def step(self, action):
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        obs = np.concatenate((obs, np.array([self.epi_time] * self.env.num_envs)))
        self.epi_time += 1.0 / self.time_limit
        return obs, rews, terminateds, truncateds, infos
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.epi_time = -0.5
        obs = np.concatenate((obs, np.array([self.epi_time])))
        return obs, info