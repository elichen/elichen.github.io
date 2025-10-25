import argparse
import numpy as np
from stable_baselines3 import PPO
from air_hockey_env import AirHockeyEnv

class RandomAgent:
    def predict(self, obs, deterministic=False):
        return np.random.uniform(-1, 1, 2), None

class DoNothingAgent:
    def predict(self, obs, deterministic=False):
        return np.array([0.0, 0.0]), None

def evaluate_matchup(env, agent1, agent2, num_episodes=100, player1_name="Player1", player2_name="Player2"):
    wins = {1: 0, 2: 0, "timeout": 0}
    goals = {1: 0, 2: 0}
    episode_lengths = []

    for episode in range(num_episodes):
        obs = env.reset()[0]
        done = False
        steps = 0

        while not done:
            # FIXED: Use proper method for getting observations
            action1, _ = agent1.predict(env.get_observation_for_player(1), deterministic=True)
            action2, _ = agent2.predict(env.get_observation_for_player(2), deterministic=True)

            # Combine actions for environment step
            combined_action = {'player1': action1, 'player2': action2}
            obs, reward, terminated, truncated, info = env.step(combined_action)
            done = terminated or truncated
            steps += 1

            # Track goals
            if "goal_scored_by" in info:
                if info["goal_scored_by"] == 1:
                    goals[1] += 1
                elif info["goal_scored_by"] == 2:
                    goals[2] += 1

        # Determine winner
        if steps >= env.max_frames:
            wins["timeout"] += 1
        elif info.get("goal_scored_by") == 1:
            wins[1] += 1
        elif info.get("goal_scored_by") == 2:
            wins[2] += 1

        episode_lengths.append(steps)

    # Calculate statistics
    win_rate_1 = wins[1] / num_episodes * 100
    win_rate_2 = wins[2] / num_episodes * 100
    timeout_rate = wins["timeout"] / num_episodes * 100
    avg_goals_1 = goals[1] / num_episodes
    avg_goals_2 = goals[2] / num_episodes
    avg_episode_length = np.mean(episode_lengths)

    print(f"\n{'='*60}")
    print(f"{player1_name} vs {player2_name} ({num_episodes} games)")
    print(f"{'='*60}")
    print(f"{player1_name:20} | Wins: {wins[1]:3} ({win_rate_1:5.1f}%) | Avg Goals: {avg_goals_1:.2f}")
    print(f"{player2_name:20} | Wins: {wins[2]:3} ({win_rate_2:5.1f}%) | Avg Goals: {avg_goals_2:.2f}")
    print(f"{'Timeouts':20} | {wins['timeout']:3} ({timeout_rate:5.1f}%)")
    print(f"Avg game length: {avg_episode_length:.0f} steps")

    return {
        "win_rate_1": win_rate_1,
        "win_rate_2": win_rate_2,
        "timeout_rate": timeout_rate,
        "avg_goals_1": avg_goals_1,
        "avg_goals_2": avg_goals_2,
        "avg_length": avg_episode_length
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes")
    args = parser.parse_args()

    env = AirHockeyEnv(render_mode=None)
    print(f"Using environment with {env.observation_space.shape[0]} features")

    # Load trained model
    print(f"Loading model: {args.model}")
    trained_model = PPO.load(args.model)

    # Create baseline agents
    random_agent = RandomAgent()
    do_nothing_agent = DoNothingAgent()

    # Run evaluations
    print(f"\nEvaluating model performance over {args.episodes} episodes each...")

    # Trained vs Random
    results_random = evaluate_matchup(
        env, trained_model, random_agent, args.episodes,
        "Trained Model", "Random Agent"
    )

    # Trained vs Do Nothing
    results_nothing = evaluate_matchup(
        env, trained_model, do_nothing_agent, args.episodes,
        "Trained Model", "Do Nothing"
    )

    # Trained vs itself (for consistency check)
    results_self = evaluate_matchup(
        env, trained_model, trained_model, args.episodes,
        "Trained (P1)", "Trained (P2)"
    )

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY - Model Validation")
    print(f"{'='*60}")

    if results_random["win_rate_1"] > 90:
        print("✓ Model beats random agent >90% (Strong evidence of learning)")
    elif results_random["win_rate_1"] > 70:
        print("⚠ Model beats random agent 70-90% (Moderate learning)")
    else:
        print("✗ Model struggles against random (<70% win rate)")

    if results_nothing["win_rate_1"] > 95:
        print("✓ Model dominates passive opponent (Good offense)")

    if results_random["timeout_rate"] < 30:
        print("✓ Low timeout rate (<30%) - Model can finish games")
    elif results_random["timeout_rate"] < 60:
        print("⚠ Moderate timeout rate (30-60%)")
    else:
        print("✗ High timeout rate (>60%) - Defensive deadlock")

    if abs(results_self["win_rate_1"] - 50) < 15:
        print("✓ Self-play is balanced (Consistent policy)")

    print(f"\nGoal differential vs Random: {results_random['avg_goals_1'] - results_random['avg_goals_2']:.2f}")
    print(f"Goal differential vs DoNothing: {results_nothing['avg_goals_1'] - results_nothing['avg_goals_2']:.2f}")

if __name__ == "__main__":
    main()