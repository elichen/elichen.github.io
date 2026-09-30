// Search and refinement off the main thread.
//
// solve: drawing -> embedding -> nearest machines in the index -> exact
// alignment of the top few hundred -> successive-halving refinement of the
// best, streaming each improvement back to the page.

import { Machine, Refiner, Fitter, targetFromPoints } from './mech.js';
import { CurveEncoder, fourierDescriptor } from './encoder.js';
import { Ranker, machineFeatures, pairFeatures } from './ranker.js';

let index = null;
let encoder = null;
let ranker = null;
let current = 0;          // id of the solve in progress; newer ids cancel older ones
const RERANK = 800;

async function fetchBytes(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Couldn't load ${url} (${res.status})`);
  // read in chunks so the page can show progress
  const total = +res.headers.get('content-length') || 0;
  const reader = res.body.getReader(), parts = [];
  let loaded = 0, lastPost = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    loaded += value.length;
    if (total && performance.now() - lastPost > 100) {
      lastPost = performance.now();
      postMessage({ type: 'loading', loaded, total });
    }
  }
  const buf = new Uint8Array(loaded);
  let o = 0;
  for (const p of parts) { buf.set(p, o); o += p.length; }
  if (buf[0] !== 0x1f || buf[1] !== 0x8b) return buf.buffer;   // already decoded by the server
  const stream = new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip'));
  return await new Response(stream).arrayBuffer();
}

async function load(base) {
  const meta = await (await fetch(base + 'index.json')).json();
  const buf = await fetchBytes(base + meta.file);
  const sec = (name, Type) => new Type(buf, meta.sections[name][0], meta.sections[name][1] / Type.BYTES_PER_ELEMENT);
  const n = sec('n', Uint8Array);
  const start = new Uint32Array(meta.count + 1);
  for (let i = 0; i < meta.count; i++) start[i + 1] = start[i] + n[i];
  index = {
    meta, n, start, count: meta.count, dim: meta.dim,
    pos: sec('pos', Int16Array), emb: sec('emb', Int8Array), struct: sec('struct', Uint8Array),
  };
  if (meta.embedding === 'encoder') encoder = await CurveEncoder.load(base + 'encoder.json');
  if (meta.ranker && meta.embedding === 'fourier') ranker = await Ranker.load(base + meta.ranker);
}

export function specOf(i) {
  const n = index.n[i], s = index.start[i], ps = index.meta.posScale;
  const kind = [], a = [], b = [], gear = [], pos = new Float64Array(2 * n);
  for (let j = 0; j < n; j++) {
    const s0 = index.struct[2 * (s + j)], s1 = index.struct[2 * (s + j) + 1];
    kind.push(s0 & 3);
    gear.push((s0 >> 2) - 4);
    a.push((s1 & 15) === 15 ? -1 : s1 & 15);
    b.push((s1 >> 4) === 15 ? -1 : s1 >> 4);
    pos[2 * j] = index.pos[2 * (s + j)] / ps;
    pos[2 * j + 1] = index.pos[2 * (s + j) + 1] / ps;
  }
  return { kind, a, b, gear, pos };
}

const barCount = (m) => m.bars.length;

function embedQuery(target) {
  return encoder ? encoder.embed(target.re, target.im) : fourierDescriptor(target.re, target.im);
}

// Indices of the k best dot products (partial selection with a min-heap).
function nearest(q, k) {
  const { emb, count, dim } = index;
  const heapV = new Float32Array(k).fill(-Infinity), heapI = new Int32Array(k).fill(-1);
  for (let i = 0; i < count; i++) {
    let s = 0;
    const o = i * dim;
    for (let d = 0; d < dim; d++) s += q[d] * emb[o + d];
    if (s <= heapV[0]) continue;
    // replace root and sift down
    let p = 0;
    heapV[0] = s; heapI[0] = i;
    for (;;) {
      const l = 2 * p + 1, r = l + 1;
      let m = p;
      if (l < k && heapV[l] < heapV[m]) m = l;
      if (r < k && heapV[r] < heapV[m]) m = r;
      if (m === p) break;
      [heapV[p], heapV[m]] = [heapV[m], heapV[p]];
      [heapI[p], heapI[m]] = [heapI[m], heapI[p]];
      p = m;
    }
  }
  return Array.from(heapI).filter((i) => i >= 0);
}

const tick = () => new Promise((r) => setTimeout(r, 0));

function snapshot(c, stage, extra = {}) {
  return {
    entry: c.entry,
    spec: c.spec,
    P: Array.from(c.ref.P),
    delta: c.ref.delta,
    placement: c.ref.placement(),
    error: c.ref.shapeError(),
    bars: barCount(c.machine),
    gears: c.machine.cranks.length - 1,
    stage,
    ...extra,
  };
}

const ROUNDS = [[40, 15], [12, 50], [3, 300]];

async function solve({ id, points, simplicity, rounds = ROUNDS, labels = 0 }) {
  current = id;
  const t0 = performance.now();
  const target = targetFromPoints(Float64Array.from(points));
  const q = embedQuery(target);
  const near = nearest(q, Math.min(RERANK, index.count));
  const tSearch = performance.now() - t0;

  // exact alignment of the shortlist; the tuning predictor (if shipped)
  // estimates where each machine will end up once tuned. Then a nudge
  // towards fewer bars.
  const fq = ranker ? (encoder ? fourierDescriptor(target.re, target.im) : q) : null;
  const scored = [];
  for (const i of near) {
    const spec = specOf(i);
    const machine = new Machine(spec);
    const f = new Fitter(machine, target);
    const c = f.curveOf(machine.initial, 0);
    if (!c) continue;
    const err = Math.sqrt(Math.max(0, target.align(f.cre, f.cim).dist));
    let expect = err;
    if (ranker) {
      const fm = Float32Array.from(index.emb.subarray(i * index.dim, (i + 1) * index.dim), (v) => v / 127);
      expect = Math.exp(ranker.score(pairFeatures(err, fq, fm, machineFeatures(machine, f, c)))) - 1e-3;
    }
    scored.push({ entry: i, spec, machine, err, expect, score: expect + simplicity * 0.002 * barCount(machine) });
  }
  scored.sort((x, y) => x.score - y.score);
  const tRerank = performance.now() - t0;
  postMessage({ type: 'searched', id, searched: index.count, shortlist: scored.length, ms: tRerank, msSearch: tSearch });

  // successive halving: many short refinements, then fewer longer ones
  let pool = scored.slice(0, rounds[0][0]).map((c) => ({ ...c, ref: new Refiner(c.machine, target) }));
  let best = null, lastPost = 0, labelRows = [], round0 = [];
  const score = (c) => c.ref.shapeError() * (1 + simplicity * 0.015 * barCount(c.machine));
  // Offline only (labels > 0): also tune some machines from deeper in the
  // shortlist, and report how every first-round machine tuned, as training
  // data for predicting which machines will tune well.
  const extra = [];
  if (labels) {
    const rest = scored.slice(rounds[0][0]);
    let seed = id * 2654435761 >>> 0;
    for (let k = 0; k < labels && rest.length; k++) {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      const c = rest.splice(seed % rest.length, 1)[0];
      extra.push({ ...c, ref: new Refiner(c.machine, target) });
    }
  }
  for (const [r, [keep, iters]] of rounds.entries()) {
    pool.sort((x, y) => score(x) - score(y));
    pool = pool.slice(0, keep);
    if (labels && r === 0) {
      for (const c of extra) for (let it = 0; it < iters; it++) c.ref.step();
    }
    if (labels && r === 1) {
      // pool was just cut after round 0; every round-0 machine has tuned for rounds[0][1] steps
      labelRows = [...round0, ...extra].map((c) => [c.entry, c.err, c.ref.shapeError()]);
    }
    if (r === 0) round0 = pool.slice();
    for (let it = 0; it < iters; it++) {
      for (const c of pool) {
        c.ref.step();
        if (!best || score(c) < score(best) - 1e-9) best = c;
      }
      const now = performance.now();
      if (now - lastPost > 40) {
        lastPost = now;
        postMessage({ type: 'progress', id, best: snapshot(best, keep), iter: it });
        await tick();
        if (current !== id) return;
      }
    }
  }
  pool.sort((x, y) => score(x) - score(y));
  if (labels) postMessage({ type: 'labels', id, rows: labelRows });
  postMessage({
    type: 'done', id, ms: performance.now() - t0,
    rawBest: scored.reduce((m, c) => Math.min(m, c.err), Infinity),
    best: snapshot(pool[0], 0),
    others: pool.slice(1).map((c) => snapshot(c, 0)),
  });
}

// Keep improving one machine: more LM steps, then small random nudges
// (basin hopping) whenever it stalls.
async function polish({ id, entry, P, points, simplicity = 0, budgetMs = 2500 }) {
  current = id;
  const target = targetFromPoints(Float64Array.from(points));
  const spec = specOf(entry);
  const machine = new Machine(spec);
  let ref = new Refiner(machine, target, Float64Array.from(P));
  let best = ref, bestErr = ref.shapeError();
  const t0 = performance.now();
  let lastPost = 0, hops = 0, seed = 12345 + id;
  const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff) * 2 - 1;
  while (performance.now() - t0 < budgetMs) {
    const before = ref.shapeError();
    for (let k = 0; k < 20 && !ref.stalled; k++) ref.step();
    const e = ref.shapeError();
    if (e < bestErr) { best = ref; bestErr = e; }
    const now = performance.now();
    if (now - lastPost > 60) {
      lastPost = now;
      postMessage({ type: 'progress', id, best: snapshot({ entry, spec, machine, ref: best }, 0, { hops }) });
      await tick();
      if (current !== id) return;
    }
    if (ref.stalled || before - e < 1e-4) {
      // hop: perturb the best design by a few percent and descend again
      const Pn = Float64Array.from(best.P, (v) => v * (1 + 0.04 * rnd()) + 0.01 * rnd());
      ref = new Refiner(machine, target, Pn);
      hops++;
    }
  }
  postMessage({ type: 'done', id, polished: true, best: snapshot({ entry, spec, machine, ref: best }, 0, { hops }), others: null });
}

globalThis.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === 'init') {
      await load(msg.base);
      postMessage({ type: 'ready', count: index.count, meta: index.meta, encoder: !!encoder });
    } else if (msg.type === 'solve') {
      await solve(msg);
    } else if (msg.type === 'polish') {
      await polish(msg);
    } else if (msg.type === 'cancel') {
      current = -1;
    }
  } catch (err) {
    postMessage({ type: 'error', id: msg.id, message: String(err && err.message || err) });
  }
};
