// Offline: run the app's own search and refinement on many training doodles
// with a generous budget, and keep the tuned machines. Shipped alongside the
// pool, they give new drawings starting points already shaped like drawings.
//
// usage: node kempe/tools/tune_doodles.mjs BASE_URL doodles.bin doodles.json OUT.jsonl [procs] [start] [count]
// doodles.bin holds float32 [N][256][2] closed paths (see train/export_doodles.py).
import { readFileSync, appendFileSync } from 'node:fs';
import { fork } from 'node:child_process';
import { Machine, Fitter, targetFromPoints, thetaGrid } from '../mech.js';

const [base, binFile, metaFile, out, procArg = '12', startArg = '0', countArg = '0'] = process.argv.slice(2);
const ROUNDS = [[32, 15], [10, 50], [3, 250]];

// Joint positions of a tuned design at crank angle 0, in its curve's frame
// (curve centroid at the origin, RMS radius 1): the index's storage format.
function machineRecord(snap) {
  const m = new Machine({ ...snap.spec, pos: Float64Array.from(snap.spec.pos) });
  const P = Float64Array.from(snap.P);
  const buf = new Float64Array(m.n * 2);
  if (m.simulate(P, Float64Array.of(0), buf) < 0) return null;
  const pts = Array.from({ length: m.n }, (_, j) => [buf[2 * j], buf[2 * j + 1]]);
  const f = new Fitter(m, targetFromPoints(Float64Array.from([0, 0, 1, 0, 0, 1])));
  const c = f.curveOf(P, 0);
  if (!c) return null;
  const pos = pts.flatMap(([x, y]) => [(x - c.mean[0]) / c.scale, (y - c.mean[1]) / c.scale]);
  return { kind: Array.from(m.kind), a: Array.from(m.a), b: Array.from(m.b), gear: Array.from(m.gear), pos };
}

if (process.env.KEMPE_CHILD) {
  let resolveDone, lastLabels = null;
  globalThis.postMessage = (m) => {
    if (m.type === 'labels') lastLabels = m.rows;
    if (m.type === 'done' || m.type === 'error' || m.type === 'ready') resolveDone(m);
  };
  await import('../worker.js');
  const send = (data) => globalThis.onmessage({ data });
  await new Promise((r) => { resolveDone = r; send({ type: 'init', base }); });
  const buf = readFileSync(binFile);
  const all = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  const idx = JSON.parse(process.env.KEMPE_IDX);
  const lines = [];
  for (const i of idx) {
    const pts = Array.from(all.subarray(i * 512, (i + 1) * 512));
    lastLabels = null;
    const done = await new Promise((r) => { resolveDone = r; send({ type: 'solve', id: i + 1, points: pts, simplicity: 0, rounds: ROUNDS, labels: 16 }); });
    if (!done.best) continue;
    const rec = machineRecord(done.best);
    if (rec) lines.push(JSON.stringify({ i, err: done.best.error, entry: done.best.entry, ...rec, labels: lastLabels }));
    if (lines.length >= 50) { process.send(lines.splice(0)); }
  }
  process.send(lines);
  process.send('end');
} else {
  const meta = JSON.parse(readFileSync(metaFile, 'utf8'));
  const start = +startArg, count = +countArg || meta.count - start, procs = +procArg;
  const slices = Array.from({ length: procs }, () => []);
  for (let i = start; i < start + count; i++) slices[(i - start) % procs].push(i);
  let done = 0, t0 = Date.now();
  await Promise.all(slices.map((idx) => new Promise((res) => {
    const c = fork(new URL(import.meta.url).pathname, process.argv.slice(2), { env: { ...process.env, KEMPE_CHILD: '1', KEMPE_IDX: JSON.stringify(idx) } });
    c.on('message', (m) => {
      if (m === 'end') { c.kill(); res(); return; }
      if (m.length) appendFileSync(out, m.join('\n') + '\n');
      done += m.length;
      if (m.length) console.log(`${done}/${count} tuned, ${((Date.now() - t0) / 1000).toFixed(0)}s`);
    });
  })));
}
