"""Payoff-matrix PSRO/PFSP training for the browser-parity air-hockey environment."""

import argparse
import json
import math
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from air_hockey_env import AirHockeyEnv
from train_dagger_selfplay import RandomPolicy, ScriptedPolicy, load_teacher_weights, policy_action


def load_policy(path):
    return PPO.load(str(path), device='cpu')


class PSROEnv(gym.Wrapper):
    """Trains one player against a frozen, weighted population."""

    def __init__(self, opponent_paths, weights, seed=0, max_frames=1200):
        super().__init__(AirHockeyEnv())
        self.env.max_frames = max_frames
        self.opponents = [load_policy(path) for path in opponent_paths]
        self.weights = np.asarray(weights, dtype=np.float64)
        self.weights /= self.weights.sum()
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.opponent = self.opponents[0]
        self.learner_player = 1
        self.phi = 0.0

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.opponent = self.opponents[self.rng.choice(len(self.opponents), p=self.weights)]
        self.learner_player = int(self.rng.integers(1, 3))
        self.phi = self._potential()
        info['learner_player'] = self.learner_player
        return self.env.get_observation_for_player(self.learner_player), info

    def _potential(self):
        obs = self.env.get_observation_for_player(self.learner_player)
        distance = np.linalg.norm(obs[2:4] - obs[0:2])
        return .20 * obs[3] + .05 * (1 - min(distance, math.sqrt(2)))

    def step(self, action):
        opponent_player = 3 - self.learner_player
        opponent_obs = self.env.get_observation_for_player(opponent_player)
        opponent_action, _ = policy_action(self.opponent, opponent_obs, deterministic=False)
        actions = {
            f'player{self.learner_player}': np.clip(action, -1, 1),
            f'player{opponent_player}': np.clip(opponent_action, -1, 1),
        }
        _, _, terminated, truncated, info = self.env.step(actions)
        goal = info['goal_scored_by']
        terminal_reward = 1.0 if goal == self.learner_player else -1.0 if goal else 0.0
        next_phi = 0.0 if terminated or truncated else self._potential()
        reward = terminal_reward + .995 * next_phi - self.phi
        self.phi = next_phi
        info['learner_player'] = self.learner_player
        return self.env.get_observation_for_player(self.learner_player), float(reward), terminated, truncated, info


def make_env(paths, weights, seed, max_frames):
    return lambda: PSROEnv(paths, weights, seed, max_frames)


def make_vec_env(paths, weights, n_envs, seed, max_frames):
    factories = [make_env(paths, weights, seed + i, max_frames) for i in range(n_envs)]
    return SubprocVecEnv(factories) if n_envs > 1 else DummyVecEnv(factories)


def new_model(env, source_path, args, ent_coef=None):
    model = PPO(
        'MlpPolicy', env, learning_rate=args.learning_rate, n_steps=args.n_steps,
        batch_size=args.batch_size, n_epochs=args.n_epochs, gamma=.995,
        gae_lambda=.95, clip_range=.15, ent_coef=args.ent_coef if ent_coef is None else ent_coef,
        vf_coef=.5, max_grad_norm=.5,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
        device='cpu', verbose=0,
    )
    load_teacher_weights(model, load_policy(source_path))
    return model


def train_oracle(source_path, opponent_paths, weights, steps, output_path, args, seed, ent_coef=None):
    env = make_vec_env(opponent_paths, weights, args.n_envs, seed, args.max_frames)
    model = new_model(env, source_path, args, ent_coef)
    model.learn(total_timesteps=steps, progress_bar=True)
    model.save(output_path)
    env.close()
    return str(output_path) + '.zip'


def game_outcome(goal, focal_player):
    if not goal:
        return 0.0
    return 1.0 if goal == focal_player else -1.0


def play_bilateral(first, second, games_per_side, seed, max_frames):
    """Returns outcomes for `first`, with identical reset seeds on both sides."""
    env = AirHockeyEnv()
    env.max_frames = max_frames
    outcomes = []
    lengths = []
    for side in (1, 2):
        for episode in range(games_per_side):
            env.reset(seed=seed + episode)
            done = False
            steps = 0
            while not done:
                if side == 1:
                    action1, _ = policy_action(first, env.get_observation_for_player(1))
                    action2, _ = policy_action(second, env.get_observation_for_player(2))
                else:
                    action1, _ = policy_action(second, env.get_observation_for_player(1))
                    action2, _ = policy_action(first, env.get_observation_for_player(2))
                _, _, terminated, truncated, info = env.step({'player1': action1, 'player2': action2})
                done = terminated or truncated
                steps += 1
            outcomes.append(game_outcome(info['goal_scored_by'], side))
            lengths.append(steps)
    values = np.asarray(outcomes)
    points = (values + 1) / 2
    standard_error = float(points.std(ddof=1) / math.sqrt(len(points))) if len(points) > 1 else 0.0
    return {
        'wins': int((values > 0).sum()),
        'losses': int((values < 0).sum()),
        'draws': int((values == 0).sum()),
        'score': float(points.mean()),
        'payoff': float(values.mean()),
        'score_ci95': [float(max(0, points.mean() - 1.96 * standard_error)),
                       float(min(1, points.mean() + 1.96 * standard_error))],
        'avg_length': float(np.mean(lengths)),
    }


def payoff_matrix(paths, games_per_side, seed, max_frames):
    policies = [load_policy(path) for path in paths]
    matrix = np.zeros((len(paths), len(paths)), dtype=np.float64)
    matchups = {}
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            result = play_bilateral(policies[i], policies[j], games_per_side,
                                    seed + 10000 * i + 100 * j, max_frames)
            matrix[i, j] = result['payoff']
            matrix[j, i] = -result['payoff']
            matchups[f'{i}:{j}'] = result
    return matrix, matchups


def regret_matching_nash(matrix, iterations=50000):
    """Approximate a zero-sum Nash distribution without an LP dependency."""
    n = len(matrix)
    row_regret = np.zeros(n)
    column_regret = np.zeros(n)
    row_sum = np.zeros(n)
    column_sum = np.zeros(n)
    for _ in range(iterations):
        positive = np.maximum(row_regret, 0)
        row = positive / positive.sum() if positive.sum() else np.full(n, 1 / n)
        positive = np.maximum(column_regret, 0)
        column = positive / positive.sum() if positive.sum() else np.full(n, 1 / n)
        row_values = matrix @ column
        column_values = -matrix.T @ row
        row_regret += row_values - row @ row_values
        column_regret += column_values - column @ column_values
        row_sum += row
        column_sum += column
    mixture = row_sum / row_sum.sum() + column_sum / column_sum.sum()
    mixture /= mixture.sum()
    exploitability = float(max((matrix @ mixture).max(), (-matrix.T @ mixture).max()))
    return mixture, exploitability


def pfsp_weights(meta, base_payoffs, meta_fraction=.65):
    win_rates = (np.asarray(base_payoffs) + 1) / 2
    hard = np.maximum(.05, (1 - win_rates) ** 2)
    hard /= hard.sum()
    weights = meta_fraction * np.asarray(meta) + (1 - meta_fraction) * hard
    return weights / weights.sum()


def create_seed(teacher_path, output_path, args):
    env = DummyVecEnv([lambda: AirHockeyEnv()])
    model = new_model(env, teacher_path, args)
    model.policy.log_std.data.fill_(-1.5)
    model.save(output_path)
    env.close()
    return str(output_path) + '.zip'


def save_state(output, names, paths, matrix, matchups, meta, exploitability, rounds):
    data = {
        'names': names,
        'paths': paths,
        'payoff_matrix': matrix.tolist(),
        'matchups': matchups,
        'meta_strategy': meta.tolist(),
        'exploitability': exploitability,
        'rounds': rounds,
    }
    with (output / 'league.json').open('w') as handle:
        json.dump(data, handle, indent=2)


def evaluate_candidates(candidate_paths, incumbent_path, games_per_side, seed, max_frames):
    incumbent = load_policy(incumbent_path)
    results = {}
    for i, path in enumerate(candidate_paths):
        results[path] = play_bilateral(load_policy(path), incumbent, games_per_side,
                                       seed + 1000 * i, max_frames)
    return results


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    seed_path = create_seed(args.incumbent, output / 'seed_12', args)
    names = ['incumbent', 'curriculum', 'dagger', 'seed_12']
    paths = [args.incumbent, args.curriculum, args.dagger, seed_path]
    for extra in args.extra_policy:
        if extra not in paths:
            names.append(f'extra_{Path(extra).stem}')
            paths.append(extra)
    main_path = args.main_start or seed_path
    if main_path not in paths:
        names.append(f'main_start_{Path(main_path).stem}')
        paths.append(main_path)
    candidates = []
    rounds = []

    for iteration in range(args.iterations):
        print(f'\nPSRO round {iteration + 1}/{args.iterations}: estimating payoff matrix', flush=True)
        matrix, matchups = payoff_matrix(paths, args.meta_games_per_side,
                                         args.seed + iteration * 100000, args.max_frames)
        meta, exploitability = regret_matching_nash(matrix, args.meta_solver_iterations)
        main_index = paths.index(main_path)
        weights = pfsp_weights(meta, matrix[main_index])
        print('meta:', np.round(meta, 3).tolist(), 'PFSP:', np.round(weights, 3).tolist(),
              f'exploitability={exploitability:.3f}', flush=True)

        main_output = output / f'main_{iteration + 1:02d}'
        main_new = train_oracle(main_path, paths, weights, args.oracle_steps, main_output,
                                args, args.seed + 1000 * iteration)
        exploiter_weights = .2 * meta
        exploiter_weights[main_index] += .8
        exploiter_weights /= exploiter_weights.sum()
        exploiter_output = output / f'exploiter_{iteration + 1:02d}'
        exploiter_new = train_oracle(seed_path, paths, exploiter_weights, args.exploiter_steps,
                                     exploiter_output, args, args.seed + 1000 * iteration + 500,
                                     ent_coef=max(args.ent_coef, .01))
        names.extend([f'main_{iteration + 1:02d}', f'exploiter_{iteration + 1:02d}'])
        paths.extend([main_new, exploiter_new])
        candidates.extend([main_new, exploiter_new])
        main_path = main_new
        rounds.append({
            'round': iteration + 1,
            'training_meta': meta.tolist(),
            'pfsp_weights': weights.tolist(),
            'exploiter_weights': exploiter_weights.tolist(),
            'pre_round_exploitability': exploitability,
        })

    print('\nComputing final league and promotion audit', flush=True)
    matrix, matchups = payoff_matrix(paths, args.meta_games_per_side,
                                     args.seed + 900000, args.max_frames)
    meta, exploitability = regret_matching_nash(matrix, args.meta_solver_iterations)
    save_state(output, names, paths, matrix, matchups, meta, exploitability, rounds)

    quick = evaluate_candidates(candidates, args.incumbent, args.selection_games_per_side,
                                args.seed + 950000, args.max_frames)
    best_path = max(candidates, key=lambda path: quick[path]['score'])
    best = load_policy(best_path)
    incumbent = load_policy(args.incumbent)
    final_incumbent = play_bilateral(best, incumbent, args.gate_games_per_side,
                                     args.seed + 970000, args.max_frames)
    candidate_random = play_bilateral(best, RandomPolicy(args.seed + 1), args.baseline_games_per_side,
                                      args.seed + 980000, args.max_frames)
    incumbent_random = play_bilateral(incumbent, RandomPolicy(args.seed + 2), args.baseline_games_per_side,
                                      args.seed + 980000, args.max_frames)
    candidate_scripted = play_bilateral(best, ScriptedPolicy(), args.baseline_games_per_side,
                                        args.seed + 990000, args.max_frames)
    promoted = (final_incumbent['score'] > .52 and final_incumbent['score_ci95'][0] > .5 and
                candidate_random['score'] >= incumbent_random['score'] - .03)
    audit = {
        'selected': best_path,
        'quick_selection': quick,
        'versus_incumbent': final_incumbent,
        'candidate_versus_random': candidate_random,
        'incumbent_versus_random': incumbent_random,
        'candidate_versus_scripted': candidate_scripted,
        'promotion_rule': 'score > .52, 95% lower bound > .50, random score no more than .03 below incumbent',
        'promoted': promoted,
        'final_meta_strategy': meta.tolist(),
        'final_exploitability': exploitability,
    }
    with (output / 'evaluation.json').open('w') as handle:
        json.dump(audit, handle, indent=2)
    print(json.dumps(audit, indent=2), flush=True)
    return best_path, promoted


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--incumbent', default='models/ppo_selfplay_final.zip')
    parser.add_argument('--curriculum', default='models/curriculum_strategy_final.zip')
    parser.add_argument('--dagger', default='models/dagger_selfplay/dagger_selfplay_final.zip')
    parser.add_argument('--output', default='models/psro')
    parser.add_argument('--main-start', help='12-feature policy to continue as the main lineage')
    parser.add_argument('--extra-policy', action='append', default=[], help='extra frozen league member; repeatable')
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--oracle-steps', type=int, default=500000)
    parser.add_argument('--exploiter-steps', type=int, default=250000)
    parser.add_argument('--n-envs', type=int, default=8)
    parser.add_argument('--max-frames', type=int, default=1200)
    parser.add_argument('--n-steps', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--n-epochs', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--ent-coef', type=float, default=.003)
    parser.add_argument('--meta-games-per-side', type=int, default=24)
    parser.add_argument('--selection-games-per-side', type=int, default=75)
    parser.add_argument('--gate-games-per-side', type=int, default=150)
    parser.add_argument('--baseline-games-per-side', type=int, default=50)
    parser.add_argument('--meta-solver-iterations', type=int, default=50000)
    parser.add_argument('--torch-threads', type=int, default=2)
    parser.add_argument('--seed', type=int, default=23)
    return parser.parse_args()


if __name__ == '__main__':
    train(parse_args())
