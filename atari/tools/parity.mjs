// Replays test/parity.json (from parity_ref.py, native ale-py) in the wasm build and compares every frame.
import { readFileSync } from 'node:fs';
import createALEModule from '@farama/ale-wasm';
import { createHash } from 'node:crypto';

const ref = JSON.parse(readFileSync(new URL('./test/parity.json', import.meta.url)));
const pkg = new URL('./node_modules/@farama/ale-wasm/', import.meta.url).pathname;
const sticky = ref._sticky ?? 0; delete ref._sticky;
const ALE = await createALEModule({ locateFile: f => pkg + f });
console.log('wasm ALE', ALE.ALEInterface.getVersion());
let ok = true;
for (const [game, { actions, steps, minimal }] of Object.entries(ref)) {
  const ale = new ALE.ALEInterface();
  ale.setInt('random_seed', 0);
  ale.setFloat('repeat_action_probability', sticky);
  ale.loadROM(`/roms/${game}.bin`);
  const acts = [...ale.getMinimalActionSet()];
  if (acts.join() !== minimal.join()) { console.log(game, 'minimal action set differs', acts, minimal); ok = false; continue; }
  let bad = -1, t0 = performance.now();
  for (let t = 0; t < actions.length; t++) {
    if (ale.gameOver()) ale.resetGame();
    const r = ale.act(actions[t]);
    const h = createHash('md5').update(ale.getScreenRGB()).update(ale.getRAM()).digest('hex').slice(0, 12);
    const s = steps[t];
    if (r !== s[0] || ale.lives() !== s[1] || +ale.gameOver() !== s[2] || h !== s[3]) { bad = t; break; }
  }
  const fps = actions.length / ((performance.now() - t0) / 1000);
  console.log(game.padEnd(16), bad < 0 ? `match ${actions.length} frames` : `MISMATCH at frame ${bad}`, `(${fps.toFixed(0)} frames/s incl. hashing)`);
  if (bad >= 0) ok = false;
}
console.log(ok ? 'PASS' : 'FAIL');
process.exit(ok ? 0 : 1);
