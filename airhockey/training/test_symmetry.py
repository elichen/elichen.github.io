from air_hockey_env import AirHockeyEnv
from stable_baselines3 import PPO
import numpy as np

env = AirHockeyEnv()
model = PPO.load("models/ppo_selfplay_final.zip")

# Test P1 (bottom) vs Random
wins_p1 = 0
for i in range(20):
    obs = env.reset()[0]
    done = False
    while not done:
        action1, _ = model.predict(env.get_observation_for_player(1), deterministic=True)
        action2 = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step({'player1': action1, 'player2': action2})
        done = terminated or truncated
    if info.get("goal_scored_by") == 1:
        wins_p1 += 1

print(f"P1 (bottom) wins: {wins_p1}/20 = {wins_p1*5}%")

# Test P2 (top) vs Random  
wins_p2 = 0
for i in range(20):
    obs = env.reset()[0]
    done = False
    while not done:
        action1 = env.action_space.sample()
        action2, _ = model.predict(env.get_observation_for_player(2), deterministic=True)
        obs, reward, terminated, truncated, info = env.step({'player1': action1, 'player2': action2})
        done = terminated or truncated
    if info.get("goal_scored_by") == 2:
        wins_p2 += 1

print(f"P2 (top) wins: {wins_p2}/20 = {wins_p2*5}%")
print(f"Difference: {abs(wins_p1 - wins_p2)*5}%")
