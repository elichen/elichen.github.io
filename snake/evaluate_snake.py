#!/usr/bin/env python3
"""
Evaluation script for trained Snake PPO models.
Provides comprehensive testing, visualization, and statistics.
"""

import os
import sys
import time
import json
import argparse
from typing import List, Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import seaborn as sns

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

from snake_gym_env import SnakeGymEnv


class SnakeEvaluator:
    """Comprehensive evaluation suite for Snake agents."""

    def __init__(self, model_path: str, grid_size: int = 20):
        """
        Initialize evaluator with trained model.

        Args:
            model_path: Path to saved PPO model
            grid_size: Grid size for evaluation
        """
        self.model = PPO.load(model_path)
        self.grid_size = grid_size
        self.env = SnakeGymEnv(grid_size=grid_size,
                              enable_connectivity=True,
                              enable_milestones=True,
                              adaptive_starvation=True)

    def evaluate_performance(self, n_episodes: int = 100,
                           deterministic: bool = True,
                           verbose: bool = True) -> Dict[str, Any]:
        """
        Evaluate agent performance over multiple episodes.

        Args:
            n_episodes: Number of episodes to evaluate
            deterministic: Use deterministic actions
            verbose: Print progress

        Returns:
            Dictionary with comprehensive statistics
        """
        scores = []
        lengths = []
        rewards = []
        collision_types = {'wall': 0, 'self': 0, 'starvation': 0, 'perfect': 0}
        milestone_reached = {25: 0, 50: 0, 75: 0, 90: 0, 100: 0}

        max_possible_score = self.grid_size * self.grid_size - 1

        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0

            while not done:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1

            # Collect statistics
            score = info.get('score', 0)
            scores.append(score)
            lengths.append(episode_length)
            rewards.append(episode_reward)

            # Track collision types
            if 'episode_stats' in info and info['episode_stats']:
                collision = info['episode_stats'].get('collision_type', 'unknown')
                if collision in collision_types:
                    collision_types[collision] += 1

            # Track milestone achievements
            fill_ratio = (score / max_possible_score) * 100
            for milestone in [25, 50, 75, 90, 100]:
                if fill_ratio >= milestone:
                    milestone_reached[milestone] += 1

            # Check for perfect game
            if score >= max_possible_score:
                collision_types['perfect'] += 1
                if verbose:
                    print(f"🏆 PERFECT GAME in episode {episode + 1}!")

            if verbose and (episode + 1) % 10 == 0:
                avg_score = np.mean(scores[-10:])
                max_score = max(scores[-10:])
                print(f"Episodes {episode - 8}-{episode + 1}: "
                      f"Avg Score: {avg_score:.1f}, Max: {max_score}")

        # Calculate comprehensive statistics
        stats = {
            'n_episodes': n_episodes,
            'grid_size': self.grid_size,
            'max_possible_score': max_possible_score,

            # Score statistics
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'min_score': min(scores),
            'max_score': max(scores),
            'median_score': np.median(scores),

            # Performance percentiles
            'percentile_25': np.percentile(scores, 25),
            'percentile_75': np.percentile(scores, 75),
            'percentile_90': np.percentile(scores, 90),
            'percentile_95': np.percentile(scores, 95),
            'percentile_99': np.percentile(scores, 99),

            # Length and reward stats
            'mean_length': np.mean(lengths),
            'mean_reward': np.mean(rewards),

            # Success rates
            'fill_rate': np.mean(scores) / max_possible_score * 100,
            'perfect_games': collision_types['perfect'],
            'perfect_rate': collision_types['perfect'] / n_episodes * 100,

            # Collision analysis
            'collisions': collision_types,
            'wall_collision_rate': collision_types['wall'] / n_episodes * 100,
            'self_collision_rate': collision_types['self'] / n_episodes * 100,
            'starvation_rate': collision_types['starvation'] / n_episodes * 100,

            # Milestone achievement rates
            'milestones': {k: v / n_episodes * 100 for k, v in milestone_reached.items()},

            # Raw data for further analysis
            'scores': scores,
            'lengths': lengths,
            'rewards': rewards
        }

        return stats

    def visualize_episode(self, save_path: Optional[str] = None,
                         deterministic: bool = True,
                         fps: int = 10) -> float:
        """
        Visualize a single episode and optionally save as GIF.

        Args:
            save_path: Path to save animation (e.g., 'snake_game.gif')
            deterministic: Use deterministic actions
            fps: Frames per second for animation

        Returns:
            Final score achieved
        """
        obs, _ = self.env.reset()
        done = False
        frames = []
        score = 0

        # Collect frames
        while not done:
            # Render current state
            frame = self._render_frame()
            frames.append(frame)

            # Take action
            action, _ = self.model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            score = info.get('score', score)

        # Add final frame
        frames.append(self._render_frame())

        # Create animation
        if save_path:
            self._create_animation(frames, save_path, fps)
            print(f"💾 Animation saved to {save_path}")

        print(f"🎮 Episode complete! Final score: {score}")
        return score

    def _render_frame(self) -> np.ndarray:
        """Render current game state as numpy array."""
        # Create grid visualization
        grid = np.zeros((self.grid_size, self.grid_size, 3))

        # Draw snake (green gradient)
        for i, (x, y) in enumerate(self.env.snake):
            intensity = 1.0 - (i / len(self.env.snake)) * 0.5
            if i == 0:  # Head
                grid[y, x] = [0, intensity, 0]  # Bright green
            else:  # Body
                grid[y, x] = [0, intensity * 0.7, 0]  # Darker green

        # Draw food (red)
        if self.env.food:
            grid[self.env.food[1], self.env.food[0]] = [1, 0, 0]

        # Add walls (white border)
        grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = [0.3, 0.3, 0.3]

        return grid

    def _create_animation(self, frames: List[np.ndarray], save_path: str, fps: int):
        """Create and save animation from frames."""
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect('equal')
        ax.axis('off')

        im = ax.imshow(frames[0], interpolation='nearest')

        def animate(i):
            im.set_array(frames[i])
            ax.set_title(f"Snake AI - Frame {i+1}/{len(frames)}")
            return [im]

        anim = FuncAnimation(fig, animate, frames=len(frames),
                           interval=1000/fps, blit=True)

        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer)
        plt.close()

    def plot_statistics(self, stats: Dict[str, Any], save_path: Optional[str] = None):
        """
        Create comprehensive visualization of evaluation statistics.

        Args:
            stats: Statistics dictionary from evaluate_performance
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Snake PPO Evaluation - {stats["n_episodes"]} Episodes', fontsize=16)

        # 1. Score distribution
        ax = axes[0, 0]
        ax.hist(stats['scores'], bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(stats['mean_score'], color='red', linestyle='--', label=f'Mean: {stats["mean_score"]:.1f}')
        ax.axvline(stats['max_score'], color='green', linestyle='--', label=f'Max: {stats["max_score"]}')
        ax.set_xlabel('Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Score Distribution')
        ax.legend()

        # 2. Score over episodes
        ax = axes[0, 1]
        episodes = range(1, len(stats['scores']) + 1)
        ax.plot(episodes, stats['scores'], alpha=0.5, label='Individual')
        # Rolling average
        window = min(20, len(stats['scores']) // 5)
        if window > 1:
            rolling_avg = np.convolve(stats['scores'], np.ones(window)/window, mode='valid')
            ax.plot(range(window, len(stats['scores']) + 1), rolling_avg,
                   color='red', linewidth=2, label=f'{window}-episode average')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Score')
        ax.set_title('Score Progression')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Collision type breakdown
        ax = axes[0, 2]
        collision_data = stats['collisions'].copy()
        if collision_data.get('perfect', 0) == 0:
            collision_data.pop('perfect', None)
        if collision_data:
            colors = {'wall': 'red', 'self': 'orange', 'starvation': 'yellow', 'perfect': 'green'}
            wedges, texts, autotexts = ax.pie(collision_data.values(),
                                              labels=collision_data.keys(),
                                              autopct='%1.1f%%',
                                              colors=[colors.get(k, 'gray') for k in collision_data.keys()])
            ax.set_title('Game Ending Types')

        # 4. Milestone achievement rates
        ax = axes[1, 0]
        milestones = list(stats['milestones'].keys())
        achievement_rates = list(stats['milestones'].values())
        bars = ax.bar(milestones, achievement_rates)
        # Color bars based on achievement
        for i, (milestone, rate) in enumerate(zip(milestones, achievement_rates)):
            if rate >= 80:
                bars[i].set_color('green')
            elif rate >= 50:
                bars[i].set_color('yellow')
            else:
                bars[i].set_color('red')
        ax.set_xlabel('Grid Fill %')
        ax.set_ylabel('Achievement Rate %')
        ax.set_title('Milestone Achievement Rates')
        ax.set_ylim(0, 105)
        for i, v in enumerate(achievement_rates):
            ax.text(milestones[i], v + 1, f'{v:.1f}%', ha='center')

        # 5. Score percentiles
        ax = axes[1, 1]
        percentiles = [25, 50, 75, 90, 95, 99]
        percentile_values = [stats[f'percentile_{p}'] for p in percentiles]
        ax.plot(percentiles, percentile_values, marker='o', linewidth=2, markersize=8)
        ax.set_xlabel('Percentile')
        ax.set_ylabel('Score')
        ax.set_title('Score Percentiles')
        ax.grid(True, alpha=0.3)
        for p, v in zip(percentiles, percentile_values):
            ax.annotate(f'{v:.0f}', (p, v), textcoords="offset points",
                       xytext=(0,10), ha='center')

        # 6. Performance summary
        ax = axes[1, 2]
        ax.axis('off')
        summary_text = f"""
Performance Summary
═══════════════════
Mean Score: {stats['mean_score']:.1f} ± {stats['std_score']:.1f}
Max Score: {stats['max_score']} / {stats['max_possible_score']}
Grid Fill Rate: {stats['fill_rate']:.1f}%

Perfect Games: {stats['perfect_games']} ({stats['perfect_rate']:.1f}%)
90% Fill Rate: {stats['milestones'][90]:.1f}%
75% Fill Rate: {stats['milestones'][75]:.1f}%

Collision Types:
• Wall: {stats['wall_collision_rate']:.1f}%
• Self: {stats['self_collision_rate']:.1f}%
• Starvation: {stats['starvation_rate']:.1f}%

Mean Episode Length: {stats['mean_length']:.0f} steps
Mean Episode Reward: {stats['mean_reward']:.1f}
"""
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Statistics plot saved to {save_path}")

        plt.show()

    def test_different_grid_sizes(self, grid_sizes: List[int] = [5, 8, 12, 16, 20],
                                 n_episodes: int = 50) -> Dict[int, Dict[str, Any]]:
        """
        Test model performance on different grid sizes.

        Args:
            grid_sizes: List of grid sizes to test
            n_episodes: Episodes per grid size

        Returns:
            Dictionary mapping grid size to statistics
        """
        results = {}

        for size in grid_sizes:
            print(f"\n🎮 Testing on {size}x{size} grid...")
            self.grid_size = size
            self.env = SnakeGymEnv(grid_size=size,
                                  enable_connectivity=True,
                                  enable_milestones=True,
                                  adaptive_starvation=True)

            stats = self.evaluate_performance(n_episodes, verbose=False)
            results[size] = stats

            print(f"   Mean Score: {stats['mean_score']:.1f}")
            print(f"   Max Score: {stats['max_score']}")
            print(f"   Fill Rate: {stats['fill_rate']:.1f}%")
            print(f"   Perfect Games: {stats['perfect_games']}")

        return results

    def compare_deterministic_vs_stochastic(self, n_episodes: int = 100) -> Dict[str, Dict[str, Any]]:
        """
        Compare deterministic vs stochastic action selection.

        Args:
            n_episodes: Episodes for each mode

        Returns:
            Comparison statistics
        """
        print("\n🎲 Evaluating deterministic policy...")
        det_stats = self.evaluate_performance(n_episodes, deterministic=True, verbose=False)

        print("🎲 Evaluating stochastic policy...")
        stoch_stats = self.evaluate_performance(n_episodes, deterministic=False, verbose=False)

        comparison = {
            'deterministic': det_stats,
            'stochastic': stoch_stats,
            'comparison': {
                'score_diff': det_stats['mean_score'] - stoch_stats['mean_score'],
                'perfect_diff': det_stats['perfect_games'] - stoch_stats['perfect_games'],
                'fill_diff': det_stats['fill_rate'] - stoch_stats['fill_rate']
            }
        }

        print(f"\n📊 Comparison Results:")
        print(f"   Deterministic Mean Score: {det_stats['mean_score']:.1f}")
        print(f"   Stochastic Mean Score: {stoch_stats['mean_score']:.1f}")
        print(f"   Score Advantage: {comparison['comparison']['score_diff']:.1f}")
        print(f"   Deterministic Perfect Games: {det_stats['perfect_games']}")
        print(f"   Stochastic Perfect Games: {stoch_stats['perfect_games']}")

        return comparison


def main():
    """Main entry point for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate trained Snake PPO model")

    parser.add_argument('model_path', type=str,
                      help='Path to trained PPO model')
    parser.add_argument('--grid-size', type=int, default=20,
                      help='Grid size for evaluation (default: 20)')
    parser.add_argument('--episodes', type=int, default=100,
                      help='Number of evaluation episodes (default: 100)')
    parser.add_argument('--visualize', action='store_true',
                      help='Visualize a game episode')
    parser.add_argument('--save-gif', type=str,
                      help='Save visualization as GIF (provide filename)')
    parser.add_argument('--plot-stats', action='store_true',
                      help='Plot evaluation statistics')
    parser.add_argument('--test-sizes', action='store_true',
                      help='Test on multiple grid sizes')
    parser.add_argument('--compare-policies', action='store_true',
                      help='Compare deterministic vs stochastic policies')
    parser.add_argument('--save-results', type=str,
                      help='Save evaluation results to JSON file')

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"❌ Model not found at {args.model_path}")
        sys.exit(1)

    # Initialize evaluator
    print(f"📂 Loading model from {args.model_path}")
    evaluator = SnakeEvaluator(args.model_path, args.grid_size)

    # Run evaluation
    print(f"\n🔬 Evaluating on {args.grid_size}x{args.grid_size} grid for {args.episodes} episodes...")
    stats = evaluator.evaluate_performance(args.episodes)

    # Print summary
    print(f"\n{'='*50}")
    print(f"📊 EVALUATION SUMMARY")
    print(f"{'='*50}")
    print(f"Mean Score: {stats['mean_score']:.1f} ± {stats['std_score']:.1f}")
    print(f"Max Score: {stats['max_score']} / {stats['max_possible_score']}")
    print(f"Grid Fill Rate: {stats['fill_rate']:.1f}%")
    print(f"Perfect Games: {stats['perfect_games']} ({stats['perfect_rate']:.1f}%)")
    print(f"\nPercentiles:")
    for p in [50, 75, 90, 95, 99]:
        print(f"  {p}th: {stats[f'percentile_{p}']:.0f}")
    print(f"\nMilestone Achievement Rates:")
    for milestone, rate in stats['milestones'].items():
        print(f"  {milestone}% grid fill: {rate:.1f}%")
    print(f"{'='*50}")

    # Optional visualizations
    if args.visualize or args.save_gif:
        print(f"\n🎮 Running visualization...")
        score = evaluator.visualize_episode(save_path=args.save_gif)

    if args.plot_stats:
        evaluator.plot_statistics(stats, save_path='evaluation_stats.png')

    if args.test_sizes:
        size_results = evaluator.test_different_grid_sizes()

    if args.compare_policies:
        policy_comparison = evaluator.compare_deterministic_vs_stochastic()

    # Save results if requested
    if args.save_results:
        results = {
            'model_path': args.model_path,
            'evaluation_stats': {k: v for k, v in stats.items()
                               if k not in ['scores', 'lengths', 'rewards']},
            'grid_size': args.grid_size,
            'n_episodes': args.episodes
        }
        with open(args.save_results, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to {args.save_results}")


if __name__ == "__main__":
    main()