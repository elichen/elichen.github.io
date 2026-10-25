"""A few real agent inputs ([84, 84, 4] stacks from Dopamine's preprocessing, random play) per game, for export.py."""
import sys, os
import numpy as np
BBF_WORK = os.path.expanduser(os.environ.get('BBF_WORK', '~/code/bbf-work')); sys.path[:0] = [BBF_WORK, os.path.join(BBF_WORK, 'shims')]
from dopamine.discrete_domains import atari_lib

os.makedirs('test', exist_ok=True)
for game, name in [('MsPacman', 'ms_pacman'), ('Qbert', 'qbert'), ('RoadRunner', 'road_runner'), ('KungFuMaster', 'kung_fu_master'), ('SpaceInvaders', 'space_invaders')]:
    env = atari_lib.create_atari_environment(game, sticky_actions=False)
    rng = np.random.default_rng(3)
    stack = np.zeros((84, 84, 4), np.uint8)
    stack[..., -1] = env.reset()[..., 0]
    states = []
    for t in range(1, 401):
        obs, _, term, _ = env.step(int(rng.integers(env.action_space.n)))
        if term: stack[:] = 0
        stack = np.roll(stack, -1, -1); stack[..., -1] = obs[..., 0]
        if env.game_over: stack[:] = 0; stack[..., -1] = env.reset()[..., 0]
        if t % 50 == 0: states.append(stack.copy())
    np.save(f'test/states_{name}.npy', np.array(states))
    print(name, len(states))
