import os
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from air_hockey_env import AirHockeyEnv
import gymnasium as gym
from gradient_monitor import GradientMonitorCallback

class CurriculumEnv(gym.Wrapper):
    """Environment wrapper that implements curriculum learning stages"""

    def __init__(self, env, stage='hitting', opponent_model=None):
        super().__init__(env)
        self.stage = stage
        self.opponent_model = opponent_model
        self.hits_this_episode = 0
        self.goals_this_episode = 0

    def reset(self, seed=None, options=None):
        self.hits_this_episode = 0
        self.goals_this_episode = 0
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        # Get opponent action based on stage
        if self.stage == 'hitting':
            # Stage 1: Stationary opponent for easier hitting
            action_p2 = np.array([0.0, 0.0])
        elif self.stage == 'scoring':
            # Stage 2: Weak defensive opponent
            obs_p2 = self.env.get_observation_for_player(2)
            puck_x = obs_p2[2]  # Puck x position
            # Simple opponent that just tracks puck x-position
            action_p2 = np.array([(puck_x - obs_p2[0]) * 2, 0.0])
            action_p2 = np.clip(action_p2, -1, 1)
        else:  # strategy
            # Stage 3: Use trained opponent model
            if self.opponent_model:
                obs_p2 = self.env.get_observation_for_player(2)
                action_p2, _ = self.opponent_model.predict(obs_p2, deterministic=False)
            else:
                action_p2 = self.env.action_space.sample()

        # Combined action for both players
        combined_action = {'player1': action, 'player2': action_p2}
        obs, base_reward, terminated, truncated, info = self.env.step(combined_action)

        # Track hits (velocity change detection)
        prev_puck_vel = getattr(self, 'prev_puck_vel', np.zeros(2))
        curr_puck_vel = self.env.puck_vel
        vel_change = np.linalg.norm(curr_puck_vel - prev_puck_vel)
        hit_detected = vel_change > 5  # Significant velocity change indicates hit
        if hit_detected:
            self.hits_this_episode += 1
        self.prev_puck_vel = curr_puck_vel.copy()

        if info.get('goal_scored_by') == 1:
            self.goals_this_episode += 1

        # Compute stage-specific rewards (pass hit detection to avoid redundant calculation)
        reward = self._compute_stage_reward(base_reward, info, hit_detected)

        return obs, reward, terminated, truncated, info

    def _compute_stage_reward(self, base_reward, info, hit_detected):
        """Compute rewards based on curriculum stage"""

        if self.stage == 'hitting':
            # Stage 1: Focus on hitting the puck
            reward = 0.0

            # Big reward for hitting the puck
            if hit_detected:
                reward += 1.0  # Major reward for hitting

            # Reward for being near the puck
            dist_to_puck = np.linalg.norm(self.env.puck_pos - self.env.paddle1_pos)
            reward -= 0.01 * (dist_to_puck / self.env.width)

            # Small penalty for time
            reward -= 0.001

            # Bonus for goals (but not the main focus)
            if info.get('goal_scored_by') == 1:
                reward += 0.5

        elif self.stage == 'scoring':
            # Stage 2: Focus on scoring goals
            reward = 0.0

            # Moderate reward for hitting
            if hit_detected:
                reward += 0.2

                # Extra reward for hitting toward goal
                if self.env.puck_vel[1] < -10:
                    reward += 0.3

            # Major reward for scoring
            if info.get('goal_scored_by') == 1:
                reward += 5.0  # Big scoring reward
                # First goal bonus
                if self.goals_this_episode == 1:
                    reward += 2.0
            elif info.get('goal_scored_by') == 2:
                reward -= 1.0

            # Reward offensive positioning
            offensive_pos = max(0, (self.env.puck_pos[1] - self.env.paddle1_pos[1]) / self.env.height)
            reward += 0.02 * offensive_pos

            # Small time penalty
            reward -= 0.002

        else:  # strategy
            # Stage 3: Full competitive play with balanced rewards
            reward = base_reward  # Use environment's full reward structure

            # Additional strategic rewards
            if info.get('goal_scored_by') == 1:
                reward += 1.0  # Extra goal bonus
            elif info.get('goal_scored_by') == 2:
                reward -= 0.5

        return reward

def make_env(stage, opponent_model=None):
    def _init():
        env = AirHockeyEnv(render_mode=None)
        return CurriculumEnv(env, stage=stage, opponent_model=opponent_model)
    return _init

def train_stage(stage_name, timesteps, n_envs, prev_model=None, opponent_model=None):
    """Train a single curriculum stage"""
    print(f"\n{'='*60}")
    print(f"STAGE: {stage_name.upper()}")
    print(f"{'='*60}")

    # Create environments
    if n_envs > 1:
        env = SubprocVecEnv([make_env(stage_name, opponent_model) for _ in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(stage_name, opponent_model)])

    # Create or load model
    if prev_model:
        print(f"Loading model from previous stage...")
        model = PPO.load(prev_model, env=env)
        # Optionally adjust learning rate for fine-tuning
        model.learning_rate = 5e-5
    else:
        print(f"Creating new model...")
        batch_size = int(64 * (n_envs / 4))
        model = PPO("MlpPolicy", env,
                    learning_rate=1e-4,
                    n_steps=2048,
                    batch_size=batch_size,
                    n_epochs=10,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.1,  # Standardized with selfplay
                    ent_coef=0.02,  # Standardized with selfplay - higher exploration
                    vf_coef=0.5,
                    max_grad_norm=0.5,
                    policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
                    verbose=1)

    # Train with gradient monitoring
    print(f"Training for {timesteps} timesteps...")
    gradient_callback = GradientMonitorCallback(check_freq=1000, verbose=1)
    model.learn(total_timesteps=timesteps, callback=gradient_callback, progress_bar=True)

    # Save
    model_path = f"models/curriculum_{stage_name}_final"
    model.save(model_path)
    print(f"✓ Model saved: {model_path}.zip")

    env.close()
    return model_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-steps", type=int, default=500000, help="Timesteps for hitting stage")
    parser.add_argument("--stage2-steps", type=int, default=500000, help="Timesteps for scoring stage")
    parser.add_argument("--stage3-steps", type=int, default=1000000, help="Timesteps for strategy stage")
    parser.add_argument("--n_envs", type=int, default=10, help="Number of parallel environments")
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)

    # Stage 1: Learn to hit the puck
    print("\n" + "="*60)
    print("CURRICULUM LEARNING: 3-STAGE TRAINING")
    print("="*60)
    print("\nStage 1: Learning to hit the puck")
    print("- Stationary opponent")
    print("- Reward for puck contact")
    print("- Reward for proximity to puck")

    stage1_model = train_stage("hitting", args.stage1_steps, args.n_envs)

    # Stage 2: Learn to score goals
    print("\nStage 2: Learning to score")
    print("- Weak defensive opponent")
    print("- Major reward for scoring")
    print("- Reward for offensive shots")

    stage2_model = train_stage("scoring", args.stage2_steps, args.n_envs, prev_model=stage1_model)

    # Stage 3: Learn strategy
    print("\nStage 3: Learning strategy")
    print("- Self-play against Stage 2 model")
    print("- Full competitive rewards")
    print("- Balanced offense/defense")

    # Load Stage 2 model as opponent
    opponent = PPO.load(stage2_model)
    stage3_model = train_stage("strategy", args.stage3_steps, args.n_envs,
                               prev_model=stage2_model, opponent_model=opponent)

    print("\n" + "="*60)
    print("CURRICULUM COMPLETE!")
    print("="*60)
    print(f"\nFinal model: {stage3_model}.zip")
    print("\nTo evaluate: python evaluate_model.py --model " + stage3_model + ".zip")

if __name__ == "__main__":
    main()