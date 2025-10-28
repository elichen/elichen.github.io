import os
import random
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
from air_hockey_env import AirHockeyEnv
from gradient_monitor import GradientMonitorCallback

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
            if hasattr(self.training_env, 'envs'):
                for env in self.training_env.envs:
                    if hasattr(env, 'opponent_pool'):
                        env.opponent_pool = self.opponent_pool[:]  # Use list slicing instead of copy
            else:
                self.training_env.opponent_pool = self.opponent_pool[:]  # Use list slicing instead of copy

            if self.verbose > 0:
                print(f"Added checkpoint to opponent pool. Pool size: {len(self.opponent_pool)}")

        return True

def make_env(opponent_pool):
    def _init():
        env = AirHockeyEnv(render_mode=None)
        return SelfPlayEnv(env, opponent_pool)
    return _init

def train(args):
    os.makedirs("models", exist_ok=True)
    model_name = "ppo_selfplay"

    # Initial empty opponent pool
    opponent_pool = []

    # Create parallel environments for faster training
    if args.n_envs > 1:
        print(f"Creating {args.n_envs} parallel environments with SubprocVecEnv...")
        env = SubprocVecEnv([make_env(opponent_pool) for _ in range(args.n_envs)])
    else:
        env = DummyVecEnv([make_env(opponent_pool)])

    # Create callbacks for updating opponent pool and monitoring gradients
    callbacks = [
        OpponentPoolCallback(save_freq=50000, max_pool_size=20, verbose=1),
        GradientMonitorCallback(check_freq=1000, verbose=1)
    ]

    # Scale batch size with envs to keep gradient steps constant
    batch_size = int(64 * (args.n_envs / 4))

    # Reduced network size for 8-feature observation space
    model = PPO("MlpPolicy", env,
                learning_rate=1e-4,  # Lower for stability
                n_steps=2048,
                batch_size=batch_size,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.1,  # Tighter for stable updates
                ent_coef=0.02,  # Higher for exploration
                vf_coef=0.5,
                max_grad_norm=0.5,
                policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
                verbose=1,
                device=args.device)

    obs_dim = env.envs[0].observation_space.shape[0] if hasattr(env, 'envs') else env.observation_space.shape[0]
    print(f"Training {model_name}...")
    print(f"- {obs_dim}-feature observation space")
    print(f"- {args.n_envs} parallel environments")
    print(f"- Batch size: {batch_size} (scaled for constant gradient steps)")
    print(f"- Opponent pool updates every 50k timesteps")
    print(f"- Maximum pool size: 20 past versions")

    model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=True)

    model.save(f"models/{model_name}_final")
    print(f"✓ Model saved: models/{model_name}_final.zip")

    # Clean up parallel environments
    env.close()

    return model_name

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--n_envs", type=int, default=4, help="Number of parallel environments")
    args = parser.parse_args()

    model_name = train(args)
    print(f"\nTo evaluate: python evaluate_model.py --model models/{model_name}_final.zip")

if __name__ == "__main__":
    main()