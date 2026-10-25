"""Native ale-py reference trace for parity.mjs: per-frame reward, lives, game over, and hashes of screen + RAM."""
import hashlib, json, os, sys
import numpy as np
from ale_py import ALEInterface, roms

GAMES = ['ms_pacman', 'qbert', 'road_runner', 'kung_fu_master']
FRAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
STICKY = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
out = {}
for game in GAMES:
    ale = ALEInterface()
    ale.setInt('random_seed', 0)
    ale.setFloat('repeat_action_probability', STICKY)
    ale.loadROM(roms.get_rom_path(game))
    acts = list(ale.getMinimalActionSet())
    rng = np.random.default_rng(1)
    actions, steps = [], []
    for t in range(FRAMES):
        if ale.game_over():
            ale.reset_game()
        a = acts[rng.integers(len(acts))] if t % 4 == 0 else actions[-1]  # hold each action 4 frames, like frame skip
        actions.append(a)
        ia = a.value
        r = ale.act(a)
        h = hashlib.md5(ale.getScreenRGB().tobytes() + ale.getRAM().tobytes()).hexdigest()[:12]
        steps.append([r, ale.lives(), int(ale.game_over()), h])
    out[game] = {"actions": [a.value for a in actions], 'steps': steps, "minimal": [a.value for a in acts]}
    print(game, 'reward', sum(s[0] for s in steps), 'resets', sum(s[2] for s in steps))
os.makedirs('test', exist_ok=True)
out['_sticky'] = STICKY
json.dump(out, open('test/parity.json', 'w'))
