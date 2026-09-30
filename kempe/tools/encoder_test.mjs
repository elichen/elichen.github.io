// Check the JavaScript encoder against PyTorch reference embeddings.
// usage: node kempe/tools/encoder_test.mjs kempe/data/ golden_encoder.json
import { readFileSync } from 'node:fs';
import { CurveEncoder } from '../encoder.js';

const dir = process.argv[2];
const meta = JSON.parse(readFileSync(dir + 'encoder.json', 'utf8'));
const bin = readFileSync(dir + meta.file);
const enc = new CurveEncoder(meta, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength));
let worst = 0;
for (const c of JSON.parse(readFileSync(process.argv[3], 'utf8'))) {
  const t0 = performance.now();
  const e = enc.embed(Float64Array.from(c.re), Float64Array.from(c.im));
  const ms = performance.now() - t0;
  const diff = Math.max(...c.emb.map((v, i) => Math.abs(v - e[i])));
  worst = Math.max(worst, diff);
  console.log(`${c.name}: max diff ${diff.toExponential(2)} (${ms.toFixed(1)} ms)`);
}
console.log(worst < 1e-4 ? 'OK' : 'MISMATCH');
