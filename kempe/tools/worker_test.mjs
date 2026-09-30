// Run the search worker in Node against a served copy of the site.
// usage: node kempe/tools/worker_test.mjs http://localhost:8123/kempe/data/ [shape]
import { SHAPES, densify, outAndBack } from '../shapes.js';

const base = process.argv[2];
const shape = process.argv[3] || 'heart';
const msgs = [];
globalThis.postMessage = (m) => {
  msgs.push(m);
  if (m.type === 'progress') console.log(`progress stage ${m.best.stage} iter ${m.iter} entry ${m.best.entry} err ${m.best.error.toFixed(4)} bars ${m.best.bars}`);
  else console.log(m.type, JSON.stringify(m, (k, v) => (['spec', 'P', 'placement', 'others', 'meta'].includes(k) ? undefined : v)));
};
await import('../worker.js');
const send = (data) => globalThis.onmessage({ data });
await send({ type: 'init', base });
const s = SHAPES[shape];
const pts = s.open ? outAndBack(densify(s.points)) : s.points;
const t0 = performance.now();
await send({ type: 'solve', id: 1, points: pts, simplicity: 1 });
console.log('total ms', (performance.now() - t0).toFixed(0));
