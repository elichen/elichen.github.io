"""Dopamine's AtariPreprocessing (as BBF trains) on a fixed random action sequence, for game_check.mjs."""
import sys, os, json, hashlib
import numpy as np
BBF_WORK = os.path.expanduser('~/code/bbf-work'); sys.path[:0] = [BBF_WORK, os.path.join(BBF_WORK, 'shims')]
from dopamine.discrete_domains import atari_lib

out = {}
for game, name in [('MsPacman', 'ms_pacman'), ('Qbert', 'qbert'), ('RoadRunner', 'road_runner'), ('KungFuMaster', 'kung_fu_master')]:
    env = atari_lib.create_atari_environment(game, sticky_actions=False)
    env.terminal_on_life_loss = True
    env.environment.reset(seed=0)
    obs = env.reset()
    rng = np.random.default_rng(2)
    n = env.action_space.n
    steps = [[0, 0, 0, hashlib.md5(obs.tobytes()).hexdigest()[:12]]]
    acts = []
    for t in range(3000):
        a = int(rng.integers(n)); acts.append(a)
        obs, r, term, _ = env.step(a)
        steps.append([float(r), int(term), int(env.game_over), hashlib.md5(obs.tobytes()).hexdigest()[:12]])
        if env.game_over:
            obs = env.reset(); steps.append([0, 0, 0, hashlib.md5(obs.tobytes()).hexdigest()[:12]]); acts.append(-1)
    out[name] = {'actions': acts, 'steps': steps}
    print(name, 'lives lost', sum(s[1] for s in steps), 'games', sum(s[2] for s in steps))
json.dump(out, open('test/game.json', 'w'))
