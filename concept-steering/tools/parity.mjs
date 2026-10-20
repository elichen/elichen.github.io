// The page's WebGPU model vs PyTorch on the same int8 weights (tools/test/ref.json from export.py):
// layer-12 residual, top concepts by spread, next-token logits, and 30 greedy tokens unsteered + steered (one batch).
import { readFileSync } from 'fs';
import { create, globals } from 'webgpu';
import { Gemma } from '../gemma.js';
import { Tokenizer } from '../tokenizer.js';

Object.assign(globalThis, globals);
const dir = new URL('..', import.meta.url).pathname;
const gpu = create([]);
const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
const device = await adapter.requestDevice();
globalThis.keepGPU = [gpu, adapter, device];   // if Dawn objects are garbage-collected, Dawn segfaults
device.addEventListener?.('uncapturederror', e => console.error('GPU error:', e.error.message));
const man = JSON.parse(readFileSync(dir + 'model/manifest.json'));
const files = {};
for (const f of Object.keys(man.files)) { const b = readFileSync(dir + 'model/' + f); files[f] = b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); }
let t0 = performance.now();
const model = await Gemma.create(device, man, files);
console.log(`loaded in ${(performance.now() - t0).toFixed(0)} ms`);
const stats = new Float32Array(readFileSync(dir + 'data/stats.bin').buffer.slice(0));
const F = man.concepts, best = stats.subarray(0, F), freq = stats.subarray(F, 2 * F);
const ref = JSON.parse(readFileSync(dir + 'tools/test/ref.json'));
let fail = 0;
const check = (ok, msg) => { console.log((ok ? 'ok   ' : 'FAIL ') + msg); if (!ok) fail++; };

for (const r of ref.read) {
  t0 = performance.now();
  const { z, resid, T } = await model.read(r.ids);
  const ms = performance.now() - t0;
  let err = 0, mag = 0;
  [1, T - 1].forEach((p, k) => { for (let i = 0; i < 640; i++) { err = Math.max(err, Math.abs(resid[p * 640 + i] - r.resid[k][i])); mag = Math.max(mag, Math.abs(r.resid[k][i])); } });
  const spread = new Float32Array(F);
  for (let j = 0; j < F; j++) {
    if (!(freq[j] > 1e-6 && freq[j] < 1e-2)) continue;
    let s = 0;
    for (let n = 1; n < T; n++) s += z[n * F + j];
    spread[j] = s / (T - 1) / Math.max(best[j], 1e-6);
  }
  const top = [...spread.keys()].sort((a, b) => spread[b] - spread[a]).slice(0, 10);
  const overlap = top.filter(j => r.spread[0].includes(j)).length;
  check(err / mag < 1e-3, `read ${T} tokens in ${ms.toFixed(0)} ms: residual max err ${err.toExponential(2)} (of ${mag.toFixed(0)}), top-10 concepts ${overlap}/10, top-1 ${top[0]} vs ${r.spread[0][0]}`);
  const [lg] = await model.forward([r.ids], 0);
  const lerr = Math.max(...r.logits[0].map((id, k) => Math.abs(lg[id] - r.logits[1][k])));
  let am = 0; for (let i = 1; i < lg.length; i++) if (lg[i] > lg[am]) am = i;
  check(am === r.logits[0][0] && lerr < 1e-2, `logits: argmax ${am} vs ${r.logits[0][0]}, top-10 max err ${lerr.toExponential(2)}`);
}

// greedy, both rows in one batch: row 0 unsteered, row 1 steered
const [u, s] = ref.generate;
const j = s.concept, strength = stats[2 * F + j];
model.setSteer([null, model.conceptVector(j, s.scale * strength * best[j])]);
const tok = new Tokenizer(JSON.parse(readFileSync(dir + 'data/vocab.json')), readFileSync(dir + 'data/merges.bin').buffer.slice(0));
const ids = tok.encode(u.prompt);
const outs = [[], []];
t0 = performance.now();
let logits = await model.forward([ids, ids], 0);
for (let t = 0; t < 30; t++) {
  const next = logits.map(lg => { let am = 0; for (let i = 1; i < lg.length; i++) if (lg[i] > lg[am]) am = i; return am; });
  next.forEach((id, b) => outs[b].push(id));
  if (t < 29) logits = await model.forward(next.map(id => [id]), ids.length + t);
}
const ms = (performance.now() - t0) / 30;
for (const [b, r] of [[0, u], [1, s]]) {
  const firstDiff = r.ids.findIndex((id, k) => id !== outs[b][k]);
  const ok = firstDiff < 0 || r.margins[firstDiff] < 0.05;
  check(ok, `greedy ${b ? 'steered  ' : 'unsteered'}: ${firstDiff < 0 ? '30/30 tokens match' : `first difference at ${firstDiff} (margin ${r.margins[firstDiff].toFixed(3)})`}  "${tok.decode(outs[b])}"`);
}
console.log(`${ms.toFixed(1)} ms per step (2 rows)`);
console.log(fail ? `${fail} FAIL` : 'PASS');
process.exit(fail ? 1 : 0);
