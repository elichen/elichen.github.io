// End-to-end evaluation of the web app's search + refinement on held-out
// doodles, running the real worker code in Node (in parallel processes).
// usage: node kempe/tools/eval_app.mjs http://localhost:8123/kempe/data/ heldout.json [n] [procs] [simplicity]
import { readFileSync } from 'node:fs';
import { fork } from 'node:child_process';

const [base, file, nArg = '200', procArg = '10', simpArg = '1', roundsArg = ''] = process.argv.slice(2);
const rounds = roundsArg ? JSON.parse(roundsArg) : undefined;

if (process.env.KEMPE_CHILD) {
  // child: run a slice of the doodles through the worker
  const results = [];
  let resolveDone;
  globalThis.postMessage = (m) => {
    if (m.type === 'done' || m.type === 'error' || m.type === 'ready') resolveDone(m);
  };
  await import('../worker.js');
  const send = (data) => globalThis.onmessage({ data });
  await new Promise((r) => { resolveDone = r; send({ type: 'init', base }); });
  const all = JSON.parse(readFileSync(file, 'utf8'));
  const idx = JSON.parse(process.env.KEMPE_IDX);
  for (const i of idx) {
    const d = all[i];
    const t0 = performance.now();
    const done = await new Promise((r) => { resolveDone = r; send({ type: 'solve', id: i + 1, points: d.pts, simplicity: +simpArg, rounds }); });
    results.push({ i, cat: d.cat, closed: d.closed, err: done.best ? done.best.error : null, raw: done.rawBest,
      bars: done.best ? done.best.bars : null, ms: performance.now() - t0 });
  }
  process.send(results);
  process.exit(0);
}

const all = JSON.parse(readFileSync(file, 'utf8'));
const n = Math.min(+nArg, all.length), procs = +procArg;
const slices = Array.from({ length: procs }, () => []);
for (let i = 0; i < n; i++) slices[i % procs].push(i);
const results = (await Promise.all(slices.map((idx) => new Promise((res) => {
  const c = fork(new URL(import.meta.url).pathname, process.argv.slice(2), { env: { ...process.env, KEMPE_CHILD: '1', KEMPE_IDX: JSON.stringify(idx) } });
  c.on('message', res);
})))).flat();
const errs = results.map((r) => r.err).filter((e) => e != null).sort((a, b) => a - b);
const q = (p) => errs[Math.min(errs.length - 1, Math.floor(p * errs.length))];
const mean = errs.reduce((a, b) => a + b, 0) / errs.length;
const frac = (t) => errs.filter((e) => e < t).length / errs.length;
const bars = results.reduce((a, r) => a + (r.bars || 0), 0) / results.length;
const raws = results.map((r) => r.raw).filter((e) => e != null).sort((a, b) => a - b);
const rawMedian = raws[Math.floor(raws.length / 2)];
const rawUnder5 = raws.filter((e) => e < 0.05).length / raws.length;
const ms = results.map((r) => r.ms).sort((a, b) => a - b);
console.log(JSON.stringify({
  n: errs.length, mean: +mean.toFixed(4), median: +q(0.5).toFixed(4), p90: +q(0.9).toFixed(4),
  under3: +frac(0.03).toFixed(3), under5: +frac(0.05).toFixed(3), under10: +frac(0.1).toFixed(3),
  bars: +bars.toFixed(1), msMedian: Math.round(ms[Math.floor(ms.length / 2)]),
  rawMedian: +rawMedian.toFixed(4), rawUnder5: +rawUnder5.toFixed(3),
}));
const byCat = {};
for (const r of results) (byCat[r.cat] ||= []).push(r.err);
console.log(Object.entries(byCat).map(([c, e]) => `${c} ${(e.reduce((a, b) => a + b, 0) / e.length * 100).toFixed(1)}`).join(', '));
