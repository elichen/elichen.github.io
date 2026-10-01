// Check drum/listener.js against PyTorch reference outputs.
// usage: node drum/tools/listener_test.mjs drum/model/ golden_listener.json
import { readFileSync } from 'node:fs';
import { Listener } from '../listener.js';

const dir = process.argv[2];
const meta = JSON.parse(readFileSync(dir + 'listener.json', 'utf8'));
const bin = readFileSync(dir + meta.file);
const net = new Listener(meta, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength));
let worst = 0;
for (const c of JSON.parse(readFileSync(process.argv[3], 'utf8'))) {
  const t0 = performance.now();
  const r = net.hear(Float64Array.from(c.lam), c.K);
  const ms = performance.now() - t0;
  let d = 0;
  c.outlines.forEach((o, m) => o.forEach((v, i) => { d = Math.max(d, Math.abs(v - r.outlines[m][i])); }));
  const dc = Math.max(...c.confidence.map((v, m) => Math.abs(v - r.confidence[m])));
  worst = Math.max(worst, d, dc);
  console.log(`K=${c.K}: max outline diff ${d.toExponential(2)}, confidence diff ${dc.toExponential(2)} (${ms.toFixed(1)} ms)`);
}
console.log(worst < 1e-3 ? 'OK' : `MISMATCH ${worst}`);
