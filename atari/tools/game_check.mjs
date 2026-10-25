// game.js (wasm ALE + preprocess.js) vs Dopamine's AtariPreprocessing (game_ref.py): same actions, compare every
// observation, reward, terminal and game over. Observations may differ in a few resize pixels (see resize_check).
import { readFileSync } from 'node:fs';
import createALEModule from '@farama/ale-wasm';
import { Game } from '../game.js';
import { createHash } from 'node:crypto';
const md5 = a => createHash('md5').update(a).digest('hex').slice(0, 12);

const ref = JSON.parse(readFileSync(new URL('./test/game.json', import.meta.url)));
const pkg = new URL('./node_modules/@farama/ale-wasm/', import.meta.url).pathname;
const ALE = await createALEModule({ locateFile: f => pkg + f, printErr: () => {} });
let ok = true;
for (const [name, { actions, steps }] of Object.entries(ref)) {
  const g = new Game(ALE, `/roms/${name}.bin`, { maxNoops: 0 });
  g.envReset();
  let k = 0, rewardBad = 0, termBad = 0, obsBad = 0;
  const check = (r, term, over) => {
    const s = steps[k++];
    if (r !== s[0]) rewardBad++;
    if (term !== s[1] || over !== s[2]) termBad++;
    if (md5(g.obs) !== s[3]) obsBad++;
  };
  check(0, 0, 0);
  for (const a of actions) {
    if (a === -1) { g.envReset(); check(0, 0, 0); continue; }
    const { reward, terminal } = g.envStep(a);
    check(reward, +terminal, +g.gameOver);
  }
  console.log(name.padEnd(16), `${k} steps, reward mismatches ${rewardBad}, terminal mismatches ${termBad}, observations differing ${obsBad}`);
  // A frame differs when one of its 7,056 pixels is 1 apart (OpenCV rounding, see resize_check): 0-8% of frames.
  if (rewardBad || termBad || obsBad > 0.1 * k) ok = false;
}
console.log(ok ? 'PASS' : 'FAIL');
