// Runs the handwriting network off the main thread and streams pen points back.
importScripts("engine.js?h=2");

let model = null;
let job = 0;
const MIX_K = 5;
const POST = 64;         // moves per message

const ready = (async () => {
  const [meta, blob] = await Promise.all([
    fetch("model/hand.json").then((r) => r.json()),
    fetch("model/hand.bin").then((r) => r.arrayBuffer())
  ]);
  model = new HandEngine.HandModel(meta, blob);
  postMessage({ type: "ready", spacing: meta.spacing, vocab: meta.vocab });
})().catch((err) => postMessage({ type: "error", message: String(err) }));

// Yield to the message queue (so a newer request can cancel this one) without
// using timers, which browsers throttle in background tabs.
const yieldPort = new MessageChannel();
const waiting = [];
yieldPort.port1.onmessage = () => waiting.shift()();
const pause = () => new Promise((r) => { waiting.push(r); yieldPort.port2.postMessage(0); });

// Reading the visitor's line is the same work for every chunk of a job, except
// near its end, where the window starts to reach the new text. So read all but
// the last TAIL moves once, and finish the read per chunk with its own text.
const TAIL = 80;
const LEAD_MIN = 0.45;   // x-heights

// The network keeps a hand for about a dozen letters, then drifts toward a
// generic narrow one as it reads more of its own ink than the visitor's. So it
// writes a few words at a time, each chunk freshly primed with the visitor's
// line, and each chunk goes after the last the way training put one line of a
// writer after another.
const CHUNK = 13;        // characters

function chunks(text) {
  const out = [];
  let cur = "", start = 0;
  for (const word of text.split(" ")) {
    if (cur && cur.length + 1 + word.length > CHUNK) {
      out.push({ text: cur, offset: start });
      start += cur.length + 1;
      cur = word;
    } else cur = cur ? cur + " " + word : word;
  }
  if (cur) out.push({ text: cur, offset: start });
  return out;
}

// largest side of the bounding box of [x, y, pen, focus] entries
function extent(e) {
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (let i = 0; i < e.length; i += 4) {
    x0 = Math.min(x0, e[i]); x1 = Math.max(x1, e[i]); y0 = Math.min(y0, e[i + 1]); y1 = Math.max(y1, e[i + 1]);
  }
  return Math.max(x1 - x0, y1 - y0);
}
function rightmost(e, x) {
  for (let i = 0; i < e.length; i += 4) x = Math.max(x, e[i]);
  return x;
}
function readPrefix(prime, primeText) {
  const s = model.newState(primeText + " ");
  const end = Math.max(0, prime.length / 3 - TAIL);
  for (let i = 0; i < end; i++) model.step(s, prime[3 * i], prime[3 * i + 1], prime[3 * i + 2]);
  // how far the prime's ink reaches right of its last point, in x-heights
  let x = 0, right = 0;
  for (let i = 0; i < prime.length; i += 3) { x += prime[i] * model.spacing; right = Math.max(right, x); }
  return { s, end, right: right - x };
}

// Now and then the network loses its place and a stroke runs off across the
// page. A chunk is short, so draw it again when its ink is far taller or wider
// than any handwriting.
const TRIES = 4;
const TALL = 7;          // x-heights, top to bottom
const WIDE = 3;          // x-heights per character

function wildness(e, chars) {
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (let i = 0; i < e.length; i += 4) {
    x0 = Math.min(x0, e[i]); x1 = Math.max(x1, e[i]); y0 = Math.min(y0, e[i + 1]); y1 = Math.max(y1, e[i + 1]);
  }
  return e.length ? Math.max(0, y1 - y0 - TALL) + Math.max(0, x1 - x0 - WIDE * chars) : Infinity;
}

// Write one line, a chunk at a time. Points are in x-heights from the end of the prime.
async function writeLine(id, line, prefix, prime, primeText, text, bias, seed) {
  const rand = HandEngine.mulberry32(seed);
  let right = prefix.right;
  for (const chunk of chunks(text)) {
    // as if the ink so far were the prime: the network's jump from it sets the gap
    let best = null;
    for (let k = 0; k < TRIES && !(best && best.wild === 0); k++) {
      const c = await sampleChunk(id, prefix, prime, primeText, chunk, bias, rand, right - prefix.right);
      if (!c) return false;
      c.wild = wildness(c.pts, chunk.text.length);
      if (!best || c.wild < best.wild) best = c;
    }
    for (let i = 0; i < best.pts.length; i += 4 * POST) {
      const n = Math.min(POST, (best.pts.length - i) / 4), stride = 6 * MIX_K + 1, j = (i / 4) * stride;
      postMessage({ type: "points", id, line, pts: best.pts.slice(i, i + 4 * n), mix: best.mix.slice(j, j + n * stride), mixK: MIX_K });
      await pause();
      if (id !== job) return false;
    }
    right = rightmost(best.pts, right);
  }
  return true;
}

// Finish reading the prime as context, then sample one chunk: { pts, mix }, or
// null if a newer job took over.
async function sampleChunk(id, prefix, prime, primeText, chunk, bias, rand, shift) {
  const text = chunk.text;
  const s = model.newState(primeText + " " + text);
  for (const k of ["h1", "c1", "h2", "c2", "h3", "c3", "kappa", "w"]) s[k].set(prefix.s[k]);
  const x = prime;
  for (let i = 3 * prefix.end; i < x.length; i += 3) model.step(s, x[i], x[i + 1], x[i + 2]);
  const primeLen = primeText.length + 1;
  const maxSteps = 40 * text.length + 60;
  let px = shift, py = 0, doneAt = -1;
  const pts = [], mixes = [];
  const mix = new Float32Array(6 * MIX_K + 1);
  // Many training lines open with a quote mark, so the net sometimes starts a line
  // with a small tick. Unless the text really starts with punctuation, hold back
  // the first strokes and drop any that end up tiny.
  let leading = /^[A-Za-z0-9]/.test(text), lead = [], leadMix = [];
  const flushLead = () => { pts.push(...lead); mixes.push(...leadMix); lead = []; leadMix = []; leading = false; };
  for (let t = 0; t < maxSteps; t++) {
    model.mixture(s, bias, MIX_K, mix);   // what it considered before choosing this move
    const [dx, dy, pen] = t === 0 ? model.sampleJump(s, bias, rand) : model.sample(s, bias, rand);
    // once the window is past the last letter, stop at the next pen lift
    if (doneAt >= 0 && (pen === 1 || t - doneAt > 8)) break;
    px += dx * model.spacing; py += dy * model.spacing;
    model.step(s, dx, dy, pen);
    const entry = [px, py, pen, chunk.offset + Math.max(0, model.focus(s) - primeLen)];
    if (leading && pen === 1 && lead.length) {
      if (extent(lead) < LEAD_MIN) { lead = []; leadMix = []; } else flushLead();
    }
    if (leading) {
      lead.push(...entry); leadMix.push(...mix);
      if (extent(lead) >= LEAD_MIN) flushLead();
    } else {
      pts.push(...entry); mixes.push(...mix);
    }
    if (doneAt < 0 && model.finished(s)) doneAt = t;
    if (t % POST === POST - 1) {
      await pause();
      if (id !== job) return null;
    }
  }
  if (lead.length) flushLead();
  return { pts: new Float32Array(pts), mix: new Float32Array(mixes) };
}

onmessage = async (e) => {
  const msg = e.data;
  if (msg.type === "cancel") { job = -1; return; }
  if (msg.type !== "write") return;
  await ready;
  if (!model) return;
  job = msg.id;
  try {
    const prefix = readPrefix(msg.prime, msg.primeText);
    for (let i = 0; i < msg.lines.length; i++) {
      const ok = await writeLine(msg.id, i, prefix, msg.prime, msg.primeText, msg.lines[i], msg.bias, msg.seed + 7919 * i);
      if (!ok) return;
      postMessage({ type: "lineDone", id: msg.id, line: i });
    }
    postMessage({ type: "done", id: msg.id });
  } catch (err) {
    postMessage({ type: "error", id: msg.id, message: String(err && err.stack || err) });
  }
};
