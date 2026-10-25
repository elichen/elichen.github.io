"""Score a BBF snapshot in Python the way BBF evaluates (Dopamine preprocessing, 0-29 no-ops, epsilon 0.001,
life loss clears the frame stack, 27,000-move cap), with float32 weights. Reference for the page's atari.evaluate().

  python eval_ref.py <params.pkl> <Game> [games]
"""
import sys, os
import numpy as np
import jax, flax, pickle
from bbf_net import network, q_values
from dopamine.discrete_domains import atari_lib

path, game, n = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 20
params = flax.core.unfreeze(pickle.load(open(path, 'rb'))['target'])
num_actions = params['params']['head']['advantage']['net']['kernel'].shape[1] // 51
net = network(num_actions)
rng = np.random.default_rng(0)
envs = [atari_lib.create_atari_environment(game, sticky_actions=False) for _ in range(n)]
for e in envs: e.terminal_on_life_loss = True
states = np.zeros((n, 84, 84, 4), np.uint8)
score, moves, done = np.zeros(n), np.zeros(n, int), np.zeros(n, bool)


def start(i):
    obs = envs[i].reset()
    for _ in range(rng.integers(30)):
        obs, _, term, _ = envs[i].step(0)
        if term: obs = envs[i].reset()
    states[i] = 0
    states[i, ..., -1] = obs[..., 0]


for i in range(n): start(i)
while not done.all():
    live = np.flatnonzero(~done)
    q = q_values(net, params, states)[live]  # fixed batch: one compile
    for k, i in enumerate(live):
        a = int(rng.integers(num_actions)) if rng.random() <= 0.001 else int(q[k].argmax())
        obs, r, term, _ = envs[i].step(a)
        score[i] += r; moves[i] += 1
        if envs[i].game_over or moves[i] >= 27000:
            done[i] = True
            continue
        if term: states[i] = 0
        states[i] = np.roll(states[i], -1, -1); states[i, ..., -1] = obs[..., 0]
print(f'{game} {os.path.basename(path)}: mean {score.mean():.0f} ± {score.std() / np.sqrt(n):.0f} over {n} games; '
      f'scores {sorted(score.astype(int).tolist())}')
