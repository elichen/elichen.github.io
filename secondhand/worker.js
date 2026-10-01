// Runs the handwriting network off the main thread and streams pen points back.
importScripts("engine.js?h=1");

let model = null;
let job = 0;
const MIX_K = 5;

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

// Reading the visitor's line is the same work for every line of a job, except
// near its end, where the window starts to reach the new text. So read all but
// the last TAIL moves once, and finish the read per line with its own text.
const TAIL = 80;
const LEAD_MIN = 0.45;   // x-heights

// largest side of the bounding box of [x, y, pen, focus] entries
function extent(e) {
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (let i = 0; i < e.length; i += 4) {
    x0 = Math.min(x0, e[i]); x1 = Math.max(x1, e[i]); y0 = Math.min(y0, e[i + 1]); y1 = Math.max(y1, e[i + 1]);
  }
  return Math.max(x1 - x0, y1 - y0);
}
function readPrefix(prime, primeText) {
  const s = model.newState(primeText + " ");
  const end = Math.max(0, prime.length / 3 - TAIL);
  for (let i = 0; i < end; i++) model.step(s, prime[3 * i], prime[3 * i + 1], prime[3 * i + 2]);
  return { s, end };
}

// Write one line: finish reading the prime as context, then sample.
async function writeLine(id, line, prefix, prime, primeText, text, bias, seed) {
  const s = model.newState(primeText + " " + text);
  for (const k of ["h1", "c1", "h2", "c2", "h3", "c3", "kappa", "w"]) s[k].set(prefix.s[k]);
  const x = prime;
  for (let i = 3 * prefix.end; i < x.length; i += 3) model.step(s, x[i], x[i + 1], x[i + 2]);
  const primeLen = primeText.length + 1;
  const rand = HandEngine.mulberry32(seed);
  const maxSteps = 40 * text.length + 60;
  let px = 0, py = 0, doneAt = -1, batch = [], mixes = [];
  const mix = new Float32Array(6 * MIX_K + 1);
  // Many training lines open with a quote mark, so the net sometimes starts a line
  // with a small tick. Unless the text really starts with punctuation, hold back
  // the first strokes and drop any that end up tiny.
  let leading = /^[A-Za-z0-9]/.test(text), lead = [], leadMix = [];
  const flushLead = () => { batch.push(...lead); mixes.push(...leadMix); lead = []; leadMix = []; leading = false; };
  for (let t = 0; t < maxSteps; t++) {
    model.mixture(s, bias, MIX_K, mix);   // what it considered before choosing this move
    let [dx, dy, pen] = model.sample(s, bias, rand);
    if (t === 0) pen = 1; // the prime ended with the pen lifted
    // once the window is past the last letter, stop at the next pen lift
    const stop = doneAt >= 0 && (pen === 1 || t - doneAt > 8);
    if (!stop) {
      px += dx * model.spacing; py += dy * model.spacing;
      model.step(s, dx, dy, pen);
      const entry = [px, py, pen, Math.max(0, model.focus(s) - primeLen)];
      if (leading && pen === 1 && lead.length) {
        if (extent(lead) < LEAD_MIN) { lead = []; leadMix = []; } else flushLead();
      }
      if (leading) {
        lead.push(...entry); leadMix.push(...mix);
        if (extent(lead) >= LEAD_MIN) flushLead();
      } else {
        batch.push(...entry); mixes.push(...mix);
      }
      if (doneAt < 0 && model.finished(s)) doneAt = t;
    }
    if ((stop || t === maxSteps - 1) && lead.length) flushLead();
    if (batch.length >= 64 || stop || t === maxSteps - 1) {
      if (batch.length) postMessage({ type: "points", id, line, pts: new Float32Array(batch), mix: new Float32Array(mixes), mixK: MIX_K });
      batch = []; mixes = [];
      await pause();
      if (id !== job) return false;
    }
    if (stop) break;
  }
  return true;
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
