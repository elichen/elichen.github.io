"""Pooled grayscale frames from the four games + cv2 INTER_AREA 84x84 reference, for resize_check.mjs."""
import numpy as np, cv2
from ale_py import ALEInterface, roms
rng = np.random.default_rng(0)
src, ref = [], []
for game in ['ms_pacman', 'qbert', 'road_runner', 'kung_fu_master']:
    ale = ALEInterface(); ale.setInt('random_seed', 0); ale.setFloat('repeat_action_probability', 0.0)
    ale.loadROM(roms.get_rom_path(game)); acts = list(ale.getMinimalActionSet())
    for t in range(3000):
        if ale.game_over(): ale.reset_game()
        a = acts[rng.integers(len(acts))]
        ale.act(a); g1 = ale.getScreenGrayscale(); ale.act(a); g2 = ale.getScreenGrayscale()
        if t % 15 == 0:
            p = np.maximum(g1, g2); src.append(p); ref.append(cv2.resize(p, (84, 84), interpolation=cv2.INTER_AREA))
np.array(src, np.uint8).tofile('test/resize_src.bin'); np.array(ref, np.uint8).tofile('test/resize_ref.bin'); print(len(src), 'frames')
