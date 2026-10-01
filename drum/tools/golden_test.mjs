// Compare drum/fem.js with reference meshes and spectra from drum/train/golden.py.
// usage: node drum/tools/golden_test.mjs golden.json
import { readFileSync } from 'node:fs';
import { mesh, modes, crisscrossMesh } from '../fem.js';

let worst = 0;
for (const c of JSON.parse(readFileSync(process.argv[2], 'utf8'))) {
  const t0 = performance.now();
  const m = c.crisscross ? crisscrossMesh(Float64Array.from(c.outline), c.crisscross) : mesh(Float64Array.from(c.outline));
  const t1 = performance.now();
  const { lam } = modes(m, c.lam.length);
  const t2 = performance.now();
  const rel = Math.max(...c.lam.map((v, i) => Math.abs(lam[i] / v - 1)));
  worst = Math.max(worst, rel);
  console.log(`${c.name}: nodes ${m.nodes.length / 2}/${c.nodes} tris ${m.tri.length / 3}/${c.tris} rim ${m.nRim}/${c.nRim} ` +
    `max rel eig diff ${rel.toExponential(2)}  mesh ${(t1 - t0).toFixed(0)} ms, modes ${(t2 - t1).toFixed(0)} ms`);
}
console.log(worst < 1e-8 ? 'OK' : `MISMATCH ${worst}`);
