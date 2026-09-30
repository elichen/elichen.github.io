// Build the tuning predictor's training set from the offline tuner's labels:
// for every labelled (doodle, machine) pair, the exact features the page
// computes, and the error the machine reached after 15 tuning steps.
// usage: node kempe/tools/ranker_data.mjs kempe/data/ doodles.bin tuned.jsonl OUT_PREFIX
// Writes OUT_PREFIX.x.f32 (rows x features), OUT_PREFIX.y.f32 (tuned error),
// OUT_PREFIX.q.i32 (doodle index per row), OUT_PREFIX.json (shape).
import { readFileSync, writeFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { Machine, Fitter, targetFromPoints } from '../mech.js';
import { fourierDescriptor } from '../encoder.js';
import { machineFeatures, pairFeatures } from '../ranker.js';

const [dir, binFile, tunedFile, out] = process.argv.slice(2);
const meta = JSON.parse(readFileSync(dir + 'index.json', 'utf8'));
const raw = gunzipSync(readFileSync(dir + meta.file));
const buf = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
const sec = (name, T) => new T(buf, meta.sections[name][0], meta.sections[name][1] / T.BYTES_PER_ELEMENT);
const nArr = sec('n', Uint8Array), pos = sec('pos', Int16Array), emb = sec('emb', Int8Array), struct = sec('struct', Uint8Array);
const start = new Uint32Array(meta.count + 1);
for (let i = 0; i < meta.count; i++) start[i + 1] = start[i] + nArr[i];

function specOf(i) {
  const n = nArr[i], s = start[i], ps = meta.posScale;
  const kind = [], a = [], b = [], gear = [], p = new Float64Array(2 * n);
  for (let j = 0; j < n; j++) {
    const s0 = struct[2 * (s + j)], s1 = struct[2 * (s + j) + 1];
    kind.push(s0 & 3); gear.push((s0 >> 2) - 4);
    a.push((s1 & 15) === 15 ? -1 : s1 & 15); b.push((s1 >> 4) === 15 ? -1 : s1 >> 4);
    p[2 * j] = pos[2 * (s + j)] / ps; p[2 * j + 1] = pos[2 * (s + j) + 1] / ps;
  }
  return { kind, a, b, gear, pos: p };
}

const dummy = targetFromPoints(Float64Array.from([0, 0, 1, 0, 0, 1]));
const mfCache = new Map();
function machineFeat(e) {
  if (mfCache.has(e)) return mfCache.get(e);
  const m = new Machine(specOf(e));
  const f = new Fitter(m, dummy);
  const c = f.curveOf(m.initial, 0);
  const fm = Float32Array.from(emb.subarray(e * meta.dim, (e + 1) * meta.dim), (v) => v / 127);
  const r = { struct: machineFeatures(m, f, c), fm };
  mfCache.set(e, r);
  return r;
}

const dood = readFileSync(binFile);
const D = new Float32Array(dood.buffer, dood.byteOffset, dood.byteLength / 4);
const X = [], Y = [], Qi = [];
let F = 0;
for (const line of readFileSync(tunedFile, 'utf8').split('\n')) {
  if (!line.trim()) continue;
  const r = JSON.parse(line);
  if (!r.labels) continue;
  const t = targetFromPoints(Float64Array.from(D.subarray(r.i * 512, (r.i + 1) * 512)));
  const fq = fourierDescriptor(t.re, t.im);
  for (const [e, rawErr, tuned] of r.labels) {
    if (!Number.isFinite(tuned)) continue;
    const mf = machineFeat(e);
    const x = pairFeatures(rawErr, fq, mf.fm, mf.struct);
    F = x.length;
    X.push(x); Y.push(tuned); Qi.push(r.i);
  }
}
const xs = new Float32Array(X.length * F);
X.forEach((x, k) => xs.set(x, k * F));
writeFileSync(out + '.x.f32', Buffer.from(xs.buffer));
writeFileSync(out + '.y.f32', Buffer.from(Float32Array.from(Y).buffer));
writeFileSync(out + '.q.i32', Buffer.from(Int32Array.from(Qi).buffer));
writeFileSync(out + '.json', JSON.stringify({ rows: X.length, features: F }));
console.log(`${X.length} pairs, ${F} features, ${mfCache.size} machines`);
