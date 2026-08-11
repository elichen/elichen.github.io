import argparse
import json
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from torch.utils.data import DataLoader, TensorDataset

from air_hockey_env import AirHockeyEnv


class ScriptedPolicy:
    def predict(self, obs, deterministic=True):
        x = np.asarray(obs, dtype=np.float32)[..., :8]
        one = x.ndim == 1
        x = x[None] if one else x
        p, q = x[:, :2], x[:, 2:4]
        pv, qv = (x[:, 4:6] - .5) * 2, (x[:, 6:8] - .5) * 2
        toward = qv[:, 1] > -.05
        reachable = q[:, 1] < .54
        goal = np.column_stack((np.full(len(x), .5), np.ones(len(x))))
        shot = goal - q
        shot /= np.maximum(np.linalg.norm(shot, axis=1, keepdims=True), 1e-6)
        behind = q - shot * .055
        steps = np.clip((.065 - q[:, 1]) / (np.minimum(qv[:, 1], -.02) * 25 / 800), 0, 80)
        ix = q[:, 0] + qv[:, 0] * 25 / 600 * steps
        ix = .025 + .95 * (1 - np.abs(np.mod((ix - .025) / .95, 2) - 1))
        defend = np.column_stack((np.clip(ix, .08, .92), np.full(len(x), .065)))
        wait = np.column_stack((np.clip(q[:, 0], .18, .82), np.full(len(x), .20)))
        target = np.where((reachable & toward)[:, None], behind, np.where((qv[:, 1] < -.05)[:, None], defend, wait))
        close = np.linalg.norm(q - p, axis=1) < .09
        target = np.where((close & toward)[:, None], q + shot * .04, target)
        d = target - p
        a = np.column_stack((d[:, 0] * 9 - pv[:, 0] * .35, -d[:, 1] * 12 + pv[:, 1] * .35))
        a = np.clip(a, -1, 1).astype(np.float32)
        return (a[0] if one else a), None


class RandomPolicy:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def predict(self, obs, deterministic=True):
        return self.rng.uniform(-1, 1, 2).astype(np.float32), None


def policy_action(policy, obs):
    n = getattr(getattr(policy, 'observation_space', None), 'shape', (len(obs),))[0]
    return policy.predict(obs[..., :n], deterministic=True)


class LeagueEnv(gym.Wrapper):
    def __init__(self, opponent_paths=(), seed=0, max_frames=1500):
        super().__init__(AirHockeyEnv())
        self.env.max_frames = max_frames
        self.rng = random.Random(seed)
        self.opponents = [ScriptedPolicy(), RandomPolicy(seed)] + [PPO.load(p) for p in opponent_paths]
        self.opponent = self.opponents[0]
        self.phi = 0

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        r = self.rng.random()
        self.opponent = self.opponents[0] if r < .1 else self.opponents[1] if r < .25 else self.opponents[2] if r < .75 or len(self.opponents) == 3 else self.rng.choice(self.opponents[3:])
        self.phi = self._potential()
        return obs, info

    def _potential(self):
        q = self.env.get_observation_for_player(1)[2:4]
        return 2 * q[1]

    def step(self, action):
        obs2 = self.env.get_observation_for_player(2)
        a2, _ = policy_action(self.opponent, obs2)
        obs, _, terminated, truncated, info = self.env.step({'player1': action, 'player2': a2})
        new_phi = self._potential()
        goal = info['goal_scored_by']
        reward = 10.0 if goal == 1 else -10.0 if goal == 2 else new_phi - self.phi - .002
        if info.get('p1_hit'):
            reward += .2 + .3 * max(0, (obs[7] - .5) * 2)
        if terminated and not goal:
            reward -= 10.0
        self.phi = new_phi
        return obs, reward, terminated, truncated, info


def make_league_env(paths, seed, max_frames):
    return lambda: LeagueEnv(paths, seed, max_frames)


def make_vec(paths, n_envs, seed, max_frames):
    fs = [make_league_env(paths, seed + i, max_frames) for i in range(n_envs)]
    return SubprocVecEnv(fs) if n_envs > 1 else DummyVecEnv(fs)


def collect_dagger(model, teacher, opponent, steps, beta, seed):
    env = AirHockeyEnv()
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    xs, ys = [], []
    for _ in range(steps):
        label, _ = policy_action(teacher, obs)
        learned, _ = model.predict(obs, deterministic=True)
        action = label if rng.random() < beta else learned
        obs2 = env.get_observation_for_player(2)
        a2, _ = policy_action(opponent, obs2)
        xs.append(obs.copy())
        ys.append(label)
        obs, _, done, trunc, _ = env.step({'player1': action, 'player2': a2})
        if done or trunc:
            obs, _ = env.reset()
    return np.asarray(xs), np.asarray(ys)


def imitate(model, x, y, epochs, batch_size, device, learning_rate=3e-4):
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=learning_rate)
    model.policy.train()
    losses = []
    for _ in range(epochs):
        for obs, target in loader:
            obs, target = obs.to(device), target.to(device)
            features = model.policy.extract_features(obs, model.policy.features_extractor)
            latent = model.policy.mlp_extractor.forward_actor(features)
            pred = torch.clamp(model.policy.action_net(latent), -1, 1)
            loss = torch.nn.functional.mse_loss(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    return float(np.mean(losses[-len(loader):]))


def play(agent1, agent2, episodes, seed=10000, max_frames=3000):
    env = AirHockeyEnv()
    env.max_frames = max_frames
    out = {'p1': 0, 'p2': 0, 'timeout': 0}
    lengths = []
    for i in range(episodes):
        env.reset(seed=seed + i)
        done = False
        n = 0
        while not done:
            a1, _ = policy_action(agent1, env.get_observation_for_player(1))
            a2, _ = policy_action(agent2, env.get_observation_for_player(2))
            _, _, term, trunc, info = env.step({'player1': a1, 'player2': a2})
            done = term or trunc
            n += 1
        goal = info['goal_scored_by']
        out['p1' if goal == 1 else 'p2' if goal == 2 else 'timeout'] += 1
        lengths.append(n)
    out['avg_length'] = float(np.mean(lengths))
    return out


def score(model, opponent, episodes, max_frames=3000):
    a = play(model, opponent, episodes, max_frames=max_frames)
    b = play(opponent, model, episodes, seed=20000, max_frames=max_frames)
    wins = a['p1'] + b['p2']
    losses = a['p2'] + b['p1']
    return {'wins': wins, 'losses': losses, 'timeouts': a['timeout'] + b['timeout'], 'decisive_win_rate': wins / max(1, wins + losses), 'p1': a, 'p2': b}


def load_teacher_weights(model, teacher):
    target, source = model.policy.state_dict(), teacher.policy.state_dict()
    for key, value in source.items():
        if target[key].shape == value.shape:
            target[key] = value
        elif value.ndim == 2 and target[key].shape[0] == value.shape[0]:
            target[key][:, :value.shape[1]] = value
            target[key][:, value.shape[1]:] = 0
    model.policy.load_state_dict(target)


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    bootstrap_env = DummyVecEnv([lambda: AirHockeyEnv()])
    model = PPO('MlpPolicy', bootstrap_env, learning_rate=args.learning_rate, n_steps=1024, batch_size=256,
                n_epochs=6, gamma=.99, gae_lambda=.95, clip_range=.1, ent_coef=.001,
                policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])), device=args.device, verbose=0)
    teacher = PPO.load(args.teacher)
    if args.resume:
        model.policy.load_state_dict(PPO.load(args.resume).policy.state_dict())
        data = np.load(args.dagger_data)
        x, y = data['observations'], data['actions']
        print(f'Resuming {args.resume} with {len(x)} DAgger anchors')
    else:
        load_teacher_weights(model, teacher)
        x = np.empty((0, model.observation_space.shape[0]), np.float32)
        y = np.empty((0, 2), np.float32)
        for i in range(args.dagger_iterations):
            beta = max(0, 1 - i / max(1, args.dagger_iterations - 1)) ** 2
            xi, yi = collect_dagger(model, teacher, teacher, args.dagger_steps, beta, args.seed + i)
            x, y = np.concatenate((x, xi)), np.concatenate((y, yi))
            loss = imitate(model, x, y, args.bc_epochs, args.batch_size, model.device)
            print(f'DAgger {i + 1}/{args.dagger_iterations}: beta={beta:.2f} samples={len(x)} loss={loss:.6f}')
        model.policy.log_std.data.fill_(-1)
    bootstrap_env.close()
    dagger_path = out / 'dagger_policy'
    model.save(dagger_path)
    np.savez_compressed(out / 'dagger_data.npz', observations=x, actions=y)
    snapshots = [str(dagger_path)]
    per_round = max(1, args.rl_steps // args.rounds)
    curve = []
    for i in range(args.rounds):
        paths = [args.teacher] + (snapshots[-(args.pool_size - 1):] if args.pool_size > 1 else [])
        env = make_vec(paths, args.n_envs, args.seed + i * args.n_envs, args.max_frames)
        model = PPO.load(dagger_path, env=env, device=args.device) if i == 0 else model
        if i:
            model.set_env(env)
        model.learn(per_round, reset_num_timesteps=False, progress_bar=True)
        anchor_loss = None
        if args.anchor_epochs:
            xi, yi = collect_dagger(model, teacher, teacher, args.anchor_steps, 0, args.seed + 100 + i)
            x, y = np.concatenate((x, xi)), np.concatenate((y, yi))
            anchor_loss = imitate(model, x, y, args.anchor_epochs, args.batch_size, model.device, 1e-4)
        snapshot = out / f'snapshot_{i + 1:02d}'
        model.save(snapshot)
        snapshots.append(str(snapshot))
        env.close()
        quick = {
            'round': i + 1,
            'anchor_loss': anchor_loss,
            'teacher': score(model, teacher, 5, args.max_frames),
            'random': score(model, RandomPolicy(args.seed + i), 5, args.max_frames)
        }
        curve.append(quick)
        with open(out / 'training_curve.json', 'w') as f:
            json.dump(curve, f, indent=2)
        print(json.dumps(quick))
    final_path = out / 'dagger_selfplay_final'
    model.save(final_path)
    report = {
        'teacher': score(model, teacher, args.eval_episodes),
        'scripted': score(model, ScriptedPolicy(), args.eval_episodes),
        'random': score(model, RandomPolicy(args.seed), args.eval_episodes)
    }
    with open(out / 'evaluation.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return final_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dagger-iterations', type=int, default=4)
    p.add_argument('--dagger-steps', type=int, default=25000)
    p.add_argument('--bc-epochs', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--rl-steps', type=int, default=2000000)
    p.add_argument('--rounds', type=int, default=10)
    p.add_argument('--pool-size', type=int, default=8)
    p.add_argument('--n-envs', type=int, default=8)
    p.add_argument('--max-frames', type=int, default=1500)
    p.add_argument('--learning-rate', type=float, default=5e-5)
    p.add_argument('--anchor-steps', type=int, default=5000)
    p.add_argument('--anchor-epochs', type=int, default=1)
    p.add_argument('--eval-episodes', type=int, default=100)
    p.add_argument('--teacher', default='models/ppo_selfplay_final.zip')
    p.add_argument('--resume')
    p.add_argument('--dagger-data')
    p.add_argument('--output', default='models/dagger_selfplay')
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=7)
    args = p.parse_args()
    train(args)


if __name__ == '__main__':
    main()
