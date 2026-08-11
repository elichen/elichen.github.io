"""Adversarial and liveness gates for a deployable air-hockey policy mixture."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from air_hockey_env import AirHockeyEnv
from train_dagger_selfplay import RandomPolicy, ScriptedPolicy, policy_action


class PolicyMixture:
    def __init__(self, paths, weights, seed):
        self.policies = [PPO.load(path, device='cpu') for path in paths]
        self.weights = np.asarray(weights, dtype=np.float64)
        self.weights /= self.weights.sum()
        self.rng = np.random.default_rng(seed)
        self.active = 0
        self.observation_space = self.policies[0].observation_space

    def select(self, exclude=None):
        weights = self.weights.copy()
        if exclude is not None and len(weights) > 1:
            weights[exclude] = 0
            weights /= weights.sum()
        self.active = int(self.rng.choice(len(weights), p=weights))
        return self.active

    def predict(self, obs, deterministic=True):
        return self.policies[self.active].predict(obs, deterministic=deterministic)


def select_episode(policy, exclude=None):
    return policy.select(exclude) if hasattr(policy, 'select') else None


def result_stats(outcomes, lengths):
    values = np.asarray(outcomes, dtype=np.float64)
    points = (values + 1) / 2
    error = points.std(ddof=1) / math.sqrt(len(points)) if len(points) > 1 else 0
    return {
        'wins': int((values > 0).sum()),
        'losses': int((values < 0).sum()),
        'draws': int((values == 0).sum()),
        'score': float(points.mean()),
        'payoff': float(values.mean()),
        'score_ci95': [float(max(0, points.mean() - 1.96 * error)),
                       float(min(1, points.mean() + 1.96 * error))],
        'avg_length': float(np.mean(lengths)),
    }


def bilateral(first, second, games_per_side, seed, max_frames):
    env = AirHockeyEnv()
    env.max_frames = max_frames
    outcomes, lengths = [], []
    for focal_player in (1, 2):
        for episode in range(games_per_side):
            first_index = select_episode(first)
            select_episode(second, first_index if first is second else None)
            env.reset(seed=seed + episode)
            done = False
            steps = 0
            while not done:
                if focal_player == 1:
                    action1, _ = policy_action(first, env.get_observation_for_player(1))
                    action2, _ = policy_action(second, env.get_observation_for_player(2))
                else:
                    action1, _ = policy_action(second, env.get_observation_for_player(1))
                    action2, _ = policy_action(first, env.get_observation_for_player(2))
                _, _, terminated, truncated, info = env.step({'player1': action1, 'player2': action2})
                done = terminated or truncated
                steps += 1
            goal = info['goal_scored_by']
            outcomes.append(0 if not goal else 1 if goal == focal_player else -1)
            lengths.append(steps)
    return result_stats(outcomes, lengths)


def reset_round(env, goal):
    serve_y = env.height / 4 if goal == 1 else env.height * 3 / 4 if goal == 2 else env.height / 2
    env.puck_pos = np.array([env.width / 2, serve_y])
    env.puck_vel = np.zeros(2)
    env.paddle1_pos = np.array([env.width / 2, env.height - 50])
    env.paddle2_pos = np.array([env.width / 2, 50.0])
    env.paddle1_vel = np.zeros(2)
    env.paddle2_vel = np.zeros(2)
    env.same_position_time = 0
    env.side_wall_time = 0
    env.last_puck_pos = env.puck_pos.copy()


def continuous_self_play(paths, weights, seeds, frames, max_scoreless, round_frames):
    reports = []
    for seed in range(seeds):
        bottom = PolicyMixture(paths, weights, seed * 2 + 101)
        top = PolicyMixture(paths, weights, seed * 2 + 102)
        bottom_index = bottom.select()
        top.select(bottom_index)
        env = AirHockeyEnv()
        env.max_frames = frames + 1
        env.reset(seed=seed, options={'standard_serve': True, 'serve_y': env.height / 2})
        goals = timeouts = unsticks = side_run = max_side_run = scoreless = longest_scoreless = 0
        for _ in range(frames):
            action1, _ = policy_action(bottom, env.get_observation_for_player(1))
            action2, _ = policy_action(top, env.get_observation_for_player(2))
            _, _, _, _, info = env.step({'player1': action1, 'player2': action2})
            scoreless += 1
            near_side = env.puck_pos[0] < env.puck_radius + 60 or env.puck_pos[0] > env.width - env.puck_radius - 60
            side_run = side_run + 1 if near_side else 0
            max_side_run = max(max_side_run, side_run)
            unsticks += int(info['unstuck'])
            if info['goal_scored_by']:
                goals += 1
                longest_scoreless = max(longest_scoreless, scoreless)
                scoreless = side_run = 0
                reset_round(env, info['goal_scored_by'])
                bottom_index = bottom.select()
                top.select(bottom_index)
            elif scoreless >= round_frames:
                timeouts += 1
                longest_scoreless = max(longest_scoreless, scoreless)
                scoreless = side_run = 0
                reset_round(env, 0)
                bottom_index = bottom.select()
                top.select(bottom_index)
        longest_scoreless = max(longest_scoreless, scoreless)
        reports.append({'goals': goals, 'timeouts': timeouts, 'unsticks': unsticks, 'max_sidewall_frames': max_side_run,
                        'longest_scoreless_frames': longest_scoreless})
    return {
        'runs': reports,
        'total_goals': sum(run['goals'] for run in reports),
        'total_timeouts': sum(run['timeouts'] for run in reports),
        'total_unsticks': sum(run['unsticks'] for run in reports),
        'max_sidewall_frames': max(run['max_sidewall_frames'] for run in reports),
        'max_scoreless_frames': max(run['longest_scoreless_frames'] for run in reports),
        'passed': (sum(run['goals'] + run['timeouts'] for run in reports) >= seeds and
                   max(run['max_sidewall_frames'] for run in reports) <= 181 and
                   max(run['longest_scoreless_frames'] for run in reports) <= max_scoreless),
    }


def bank_velocity(start_x, start_y, target_x, wall, speed=25):
    if wall == 'left':
        ratio = (target_x - 27 + .8 * start_x) / (.8 * (start_y - 35))
        sign = -1
    else:
        ratio = (1053 - .8 * start_x - target_x) / (.8 * (start_y - 35))
        sign = 1
    vy = -speed / math.sqrt(1 + ratio ** 2)
    return np.array([sign * abs(vy) * ratio, vy])


def bank_shot_defense(policy, seed):
    shots = []
    shot_id = 0
    for defender in (1, 2):
        for wall in ('left', 'right'):
            for start_x in (240, 300, 360):
                for target_x in (250, 350):
                    env = AirHockeyEnv()
                    env.max_frames = 500
                    env.reset(seed=seed + shot_id, options={'standard_serve': True})
                    velocity = bank_velocity(start_x, 600, target_x, wall)
                    if defender == 2:
                        env.puck_pos = np.array([start_x, 600.0])
                        env.puck_vel = velocity
                    else:
                        env.puck_pos = np.array([start_x, 200.0])
                        env.puck_vel = np.array([velocity[0], -velocity[1]])
                    select_episode(policy)
                    saved = False
                    outcome = 'timeout'
                    for frame in range(500):
                        action, _ = policy_action(policy, env.get_observation_for_player(defender))
                        actions = {'player1': np.zeros(2), 'player2': np.zeros(2)}
                        actions[f'player{defender}'] = action
                        _, _, terminated, _, info = env.step(actions)
                        hit = info['p1_hit'] if defender == 1 else info['p2_hit']
                        if hit:
                            saved, outcome = True, 'contact'
                            break
                        if info['goal_scored_by']:
                            saved = info['goal_scored_by'] == defender
                            outcome = 'safe_goal' if saved else 'conceded'
                            break
                        if terminated:
                            saved, outcome = True, 'timeout'
                            break
                    shots.append({'defender': defender, 'wall': wall, 'start_x': start_x,
                                  'target_x': target_x, 'saved': saved, 'outcome': outcome,
                                  'frames': frame + 1})
                    shot_id += 1
    return {'saved': sum(shot['saved'] for shot in shots), 'shots': len(shots),
            'save_rate': sum(shot['saved'] for shot in shots) / len(shots), 'details': shots}


def evaluate(args):
    mixture = PolicyMixture(args.models, args.weights, args.seed)
    incumbent = PPO.load(args.incumbent, device='cpu')
    versus_incumbent = bilateral(mixture, incumbent, args.games_per_side, args.seed + 10000, args.max_frames)
    versus_random = bilateral(mixture, RandomPolicy(args.seed + 1), args.baseline_games_per_side,
                              args.seed + 20000, args.max_frames)
    incumbent_random = bilateral(incumbent, RandomPolicy(args.seed + 2), args.baseline_games_per_side,
                                 args.seed + 20000, args.max_frames)
    versus_scripted = bilateral(mixture, ScriptedPolicy(), args.baseline_games_per_side,
                                args.seed + 30000, args.max_frames)
    incumbent_scripted = bilateral(incumbent, ScriptedPolicy(), args.baseline_games_per_side,
                                   args.seed + 30000, args.max_frames)
    liveness = continuous_self_play(args.models, args.weights, args.liveness_seeds,
                                    args.liveness_frames, args.max_scoreless_frames, args.round_frames)
    mixture_bank = bank_shot_defense(mixture, args.seed + 40000)
    incumbent_bank = bank_shot_defense(incumbent, args.seed + 40000)
    gates = {
        'incumbent_lower_bound': versus_incumbent['score_ci95'][0] > .5,
        'random_regression': versus_random['score'] >= incumbent_random['score'] - .03,
        'scripted_improvement': versus_scripted['score'] >= incumbent_scripted['score'] + .05,
        'draw_rate': versus_incumbent['draws'] / (args.games_per_side * 2) < .8,
        'self_play_liveness': liveness['passed'],
        'bank_shot_regression': mixture_bank['save_rate'] >= incumbent_bank['save_rate'],
    }
    report = {
        'models': args.models, 'weights': args.weights,
        'versus_incumbent': versus_incumbent,
        'candidate_versus_random': versus_random,
        'incumbent_versus_random': incumbent_random,
        'candidate_versus_scripted': versus_scripted,
        'incumbent_versus_scripted': incumbent_scripted,
        'self_play_liveness': liveness,
        'candidate_bank_shots': mixture_bank,
        'incumbent_bank_shots': incumbent_bank,
        'gates': gates,
        'passed': all(gates.values()),
    }
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['models/psro_v3/main_01.zip',
                                                        'models/psro_v3/main_02.zip',
                                                        'models/psro_v2/main_02.zip'])
    parser.add_argument('--weights', nargs='+', type=float, default=[.75, .20, .05])
    parser.add_argument('--incumbent', default='models/ppo_selfplay_final.zip')
    parser.add_argument('--games-per-side', type=int, default=100)
    parser.add_argument('--baseline-games-per-side', type=int, default=40)
    parser.add_argument('--max-frames', type=int, default=1200)
    parser.add_argument('--liveness-seeds', type=int, default=5)
    parser.add_argument('--liveness-frames', type=int, default=5000)
    parser.add_argument('--max-scoreless-frames', type=int, default=3000)
    parser.add_argument('--round-frames', type=int, default=1200)
    parser.add_argument('--output', default='models/psro_v3/adversarial_evaluation.json')
    parser.add_argument('--seed', type=int, default=47)
    args = parser.parse_args()
    if len(args.models) != len(args.weights) or not any(weight > 0 for weight in args.weights):
        parser.error('--models and --weights need equal lengths and at least one positive weight')
    return args


if __name__ == '__main__':
    evaluate(parse_args())
