import os
import random
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
from air_hockey_env_v3 import AirHockeyEnv

class SelfPlayEnv(gym.Wrapper):
    def __init__(self, env, opponent_pool):
        super().__init__(env)
        self.opponent_pool = opponent_pool
        self.current_opponent = None
        self._select_opponent()

    def reset(self, seed=None, options=None):
        self._select_opponent()
        return self.env.reset(seed=seed, options=options)

    def _select_opponent(self):
        if self.opponent_pool:
            self.current_opponent = random.choice(self.opponent_pool)
        else:
            self.current_opponent = None

    def step(self, action):
        obs_p2 = self.env.get_observation_for_player(2)

        if self.current_opponent is None:
            action_p2 = self.env.action_space.sample()
        else:
            action_p2, _ = self.current_opponent.predict(obs_p2, deterministic=False)
            if isinstance(action_p2, np.ndarray) and action_p2.ndim > 1:
                action_p2 = action_p2[0]

        return self.env.step({'player1': action, 'player2': action_p2})

class OpponentPoolCallback(BaseCallback):
    def __init__(self, save_freq=50000, max_pool_size=20, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.max_pool_size = max_pool_size
        self.opponent_pool = []

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0 and self.n_calls > 0:
            # Clone current model for opponent pool
            self.model.save("temp_model")
            opponent = PPO.load("temp_model")
            self.opponent_pool.append(opponent)

            # Remove oldest opponent if pool is full
            if len(self.opponent_pool) > self.max_pool_size:
                self.opponent_pool.pop(0)

            # Update all environments with new opponent pool
            for env in self.training_env.envs:
                env.opponent_pool = self.opponent_pool.copy()

            if self.verbose > 0:
                print(f"Added checkpoint to opponent pool. Pool size: {len(self.opponent_pool)}")

        return True

def train(args):
    os.makedirs("models", exist_ok=True)
    model_name = "ppo_selfplay_v3"

    # Create environment with empty opponent pool initially
    opponent_pool = []
    env = DummyVecEnv([lambda: SelfPlayEnv(AirHockeyEnv(), opponent_pool)])

    # Create callback for updating opponent pool
    callback = OpponentPoolCallback(save_freq=50000, max_pool_size=20, verbose=1)

    model = PPO("MlpPolicy", env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                policy_kwargs=dict(net_arch=dict(pi=[256, 256, 256, 256], vf=[256, 256, 256, 256])),
                verbose=1,
                device=args.device)

    print(f"Training {model_name} with proper self-play...")
    print(f"- Opponent pool updates every 50k timesteps")
    print(f"- Maximum pool size: 20 past versions")

    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=True)

    model.save(f"models/{model_name}_final")
    print(f"✓ Model saved: models/{model_name}_final.zip")
    return model_name

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=2000000)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    model_name = train(args)
    print(f"\nTo export: python export_to_onnx.py --model models/{model_name}_final.zip")

if __name__ == "__main__":
    main()