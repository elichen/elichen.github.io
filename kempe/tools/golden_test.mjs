// Compare kempe/mech.js against reference outputs from kempe/train/golden.py.
// usage: node kempe/tools/golden_test.mjs golden.json
import { readFileSync } from 'node:fs';
import { Machine, Fitter, Refiner, targetFromPoints, T_SIM } from '../mech.js';

const g = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const target = targetFromPoints(Float64Array.from(g.target));
let worst = 0;
for (const [i, c] of g.cases.entries()) {
  const m = new Machine({ ...c.spec, pos: Float64Array.from(c.spec.pos) });
  const pErr = Math.max(...c.params.map((v, k) => Math.abs(v - m.initial[k])));
  const f = new Fitter(m, target);
  const cv = f.curveOf(m.initial, 0);
  let cErr = 0;
  for (let k = 0; k < 64; k++) cErr = Math.max(cErr, Math.abs(f.cre[k] - c.curve[0][k]), Math.abs(f.cim[k] - c.curve[1][k]));
  const al = target.align(f.cre, f.cim);
  const t0 = performance.now();
  const r = new Refiner(m, target);
  for (let k = 0; k < 30; k++) r.step();
  const ms = performance.now() - t0;
  const err = r.shapeError();
  console.log(`case ${i}: n=${m.n} D=${m.D} params ${pErr.toExponential(1)} curve ${cErr.toExponential(1)} ` +
    `minSin ${cv.minSin.toFixed(4)}/${c.min_sin.toFixed(4)} dist ${al.dist.toFixed(5)}/${c.dist.toFixed(5)} ` +
    `v${al.variant}/${c.variant} s${al.shift}/${c.shift} fit ${err.toFixed(4)}/${c.fit_err.toFixed(4)} (${ms.toFixed(0)} ms)`);
  worst = Math.max(worst, cErr, Math.abs(al.dist - c.dist));
}
console.log(worst < 1e-6 ? 'OK' : `MISMATCH ${worst}`);
