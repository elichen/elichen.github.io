// Runs Gemma 3 270M + the layer-12 dictionary off the main thread.
// in:  {type: 'read', text} | {type: 'write', prompt, concepts, scale, seed, tokens}
// out: progress, ready, read results, streamed continuations (row 0 steered, row 1 the same draw unsteered)
import { Gemma } from './gemma.js';
import { Tokenizer } from './tokenizer.js';

// int8 weights (~280 MB) live on Hugging Face, pinned to a commit; tools/export.py writes them to model/
const MODEL = 'https://huggingface.co/eli-aifund/gemma-3-270m-scope-steering-int8/resolve/349ab85251693013bf0bf167c705ccd2e46a04fc/';
const DATA = new URL('./data/', import.meta.url).href;
const CACHE = 'concept-steering-v1';

let model, tok, best, freq, strength, F, banned = [], job = 0;

async function fetchCached(url, onBytes) {
  let cache = null;
  try { cache = await caches.open(CACHE); } catch { /* no Cache Storage (private window) */ }
  let res = cache && await cache.match(url);
  if (!res) {
    res = await fetch(url);
    if (!res.ok) throw new Error(`${url}: ${res.status}`);
    if (cache) {
      try { await cache.put(url, res.clone()); } catch { /* quota */ }
    }
  }
  const reader = res.body.getReader();
  const parts = [];
  let n = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    n += value.length;
    onBytes(value.length);
  }
  const out = new Uint8Array(n);
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out.buffer;
}

async function load() {
  if (!navigator.gpu) throw new Error('This page needs WebGPU (Chrome, Edge or Safari 26).');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No WebGPU adapter.');
  const device = await adapter.requestDevice();
  device.lost.then(info => postMessage({ type: 'error', message: `The GPU stopped (${info.message || info.reason}). Reload the page.` }));
  const man = await (await fetch(MODEL + 'manifest.json')).json();
  const total = Object.values(man.files).reduce((a, b) => a + b, 0);
  let loaded = 0;
  const bump = n => { loaded += n; postMessage({ type: 'progress', loaded, total }); };
  const files = {};
  const [vocab, merges, stats] = await Promise.all([
    fetch(DATA + 'vocab.json').then(r => r.json()),
    fetchCached(DATA + 'merges.bin', () => {}),
    fetchCached(DATA + 'stats.bin', () => {}),
    ...Object.keys(man.files).map(async f => { files[f] = await fetchCached(MODEL + f, bump); }),
  ]);
  tok = new Tokenizer(vocab, merges);
  banned = vocab.added.filter(a => /^[<[]/.test(a.content) && a.id !== tok.eos).map(a => a.id);   // markup tokens a base model shouldn't write
  F = man.concepts;
  const s = new Float32Array(stats);
  best = s.subarray(0, F); freq = s.subarray(F, 2 * F); strength = s.subarray(2 * F, 3 * F);
  model = await Gemma.create(device, man, files);
  postMessage({ type: 'ready' });
}

const usable = j => freq[j] > 1e-6 && freq[j] < 1e-2 && strength[j] > 0;

async function read(text) {
  const ids = tok.encode(text).slice(0, 512);
  const { z, T } = await model.read(ids);
  const spread = new Float32Array(F);
  const perToken = [];
  for (let n = 0; n < T; n++) {
    const row = [];
    for (let j = 0; j < F; j++) {
      const a = z[n * F + j];
      if (a > 0 && usable(j)) {
        const r = a / best[j];
        if (n > 0) spread[j] += r;
        row.push([j, r]);
      }
    }
    row.sort((a, b) => b[1] - a[1]);
    perToken.push(row.slice(0, 8));
  }
  const top = [];
  for (let j = 0; j < F; j++) if (spread[j] > 0) top.push([j, spread[j] / Math.max(1, T - 1)]);
  top.sort((a, b) => b[1] - a[1]);
  const list = top.slice(0, 40);
  const acts = list.map(([j]) => Float32Array.from({ length: T }, (_, n) => z[n * F + j] / best[j]));
  postMessage({ type: 'read', text, tokens: ids.map(id => tok.decode([id])), perToken, spread: list, acts, truncated: ids.length === 512 });
}

// top-p 0.95 at temperature 0.8 by inverse CDF on a shared uniform, so both rows use the same draw. Exact:
// a histogram of log-probabilities finds the bin where the top_p mass is reached, and only tokens in or above
// that bin are sorted.
const BINS = 4096, SPAN = 30;                    // bins over 30 nats below the top token; beyond that p < 1e-13
const mass = new Float64Array(BINS);
function sample(logits, u, temp = 0.8, topP = 0.95) {
  for (const id of banned) logits[id] = -Infinity;
  let mx = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > mx) mx = logits[i];
  mass.fill(0);
  let Z = 0;
  const k = BINS / SPAN / temp;
  for (let i = 0; i < logits.length; i++) {
    const d = mx - logits[i];
    if (d < SPAN * temp) {
      const p = Math.exp(-d / temp);
      Z += p;
      mass[Math.floor(d * k)] += p;
    }
  }
  let bin = 0;
  for (let c = 0; bin < BINS; bin++) { c += mass[bin]; if (c >= topP * Z) break; }
  const cand = [];
  for (let i = 0; i < logits.length; i++) {
    const d = mx - logits[i];
    if (d < SPAN * temp && Math.floor(d * k) <= bin) cand.push([i, Math.exp(-d / temp)]);
  }
  cand.sort((a, b) => b[1] - a[1]);
  let keep = 0, c = 0;
  while (keep < cand.length && c < topP * Z) c += cand[keep++][1];
  let t = u * c;
  for (let j = 0; j < keep; j++) { t -= cand[j][1]; if (t <= 0) return cand[j][0]; }
  return cand[keep - 1][0];
}

function mulberry32(a) {
  return () => {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

async function write({ prompt, concepts, scale, seed, tokens = 48 }) {
  const me = ++job;
  const vec = new Float32Array(640);
  for (const j of concepts) {
    const v = model.conceptVector(j, scale * strength[j] * best[j]);
    for (let i = 0; i < 640; i++) vec[i] += v[i];
  }
  model.setSteer([concepts.length ? vec : null, null]);
  const ids = tok.encode(prompt).slice(0, 200);
  const rand = mulberry32(seed);
  const out = [[], []], done = [false, false];
  let logits = await model.forward([ids, ids], 0);
  const ms = { forward: 0, sample: 0 };
  for (let t = 0; t < tokens; t++) {
    if (me !== job) return;
    const u = rand();
    let t0 = performance.now();
    const next = logits.map((lg, b) => (done[b] ? 0 : sample(lg, u)));
    ms.sample += performance.now() - t0;
    next.forEach((id, b) => {
      if (done[b]) return;
      if (id === tok.eos) { done[b] = true; return; }
      out[b].push(id);
    });
    postMessage({ type: 'text', job: me, rows: out.map(o => tok.decode(o)) });
    if (done[0] && done[1]) break;
    t0 = performance.now();
    if (t + 1 < tokens) logits = await model.forward(next.map(id => [id]), ids.length + t);
    ms.forward += performance.now() - t0;
  }
  postMessage({ type: 'done', job: me, ms });
}

let queue = Promise.resolve();
onmessage = e => {
  const m = e.data;
  if (m.type === 'write') job++;           // a newer request stops the running one at its next token
  queue = queue.then(async () => {
    try {
      if (m.type === 'load') await load();
      else if (m.type === 'read') await read(m.text);
      else if (m.type === 'write') await write(m);
    } catch (err) {
      postMessage({ type: 'error', message: err.message || String(err) });
    }
  });
};
