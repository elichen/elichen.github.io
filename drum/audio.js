// The drum's sound: each mode rings as a decaying sine at f0 * sqrt(λk / λ1).
// How loud a mode sounds depends on how much it moves at the strike point
// (hit near the middle and the modes with a node there stay quiet). A softer
// mallet rolls off the high modes, and high modes die away faster.

const RATE = 44100;
let ctx = null, out = null;

function context() {
  if (!ctx) {
    ctx = new AudioContext();
    const comp = ctx.createDynamicsCompressor();
    comp.threshold.value = -12;
    comp.ratio.value = 4;
    out = ctx.createGain();
    out.gain.value = 0.9;
    out.connect(comp).connect(ctx.destination);
  }
  if (ctx.state === 'suspended') ctx.resume();
  return ctx;
}

// lam: eigenvalues; weights: each mode's displacement at the strike point
// (the mode shapes are mass-normalised). f0: pitch of the lowest mode.
export function strike(lam, weights, { f0 = 110, mallet = 0.5, ring = 1.6, seconds = 3 } = {}) {
  const c = context();
  const n = Math.round(seconds * RATE);
  const buf = c.createBuffer(1, n, RATE);
  const data = buf.getChannelData(0);
  const brightness = 4 + 36 * mallet;            // modes above this many λ1 are muted
  let norm = 0;
  for (let k = 0; k < lam.length; k++) {
    const r = lam[k] / lam[0];
    const f = f0 * Math.sqrt(r);
    if (f > 0.45 * RATE) break;
    // displacement at the strike point twice: struck there, heard there
    const a = weights[k] * weights[k] * Math.exp(-r / brightness) / Math.sqrt(r);
    const tau = ring * Math.pow(r, -0.45);
    // damped sine by recurrence: y[t] = 2 cos(w) d y[t-1] - d^2 y[t-2]
    const w = (2 * Math.PI * f) / RATE, d = Math.exp(-1 / (tau * RATE));
    const c1 = 2 * Math.cos(w) * d, c2 = -d * d;
    let y1 = 0, y2 = 0;                               // y[t] = a d^t sin(w t)
    for (let t = 0; t < n; t++) {
      const y = t === 0 ? 0 : t === 1 ? a * d * Math.sin(w) : c1 * y1 + c2 * y2;
      data[t] += y;
      y2 = y1; y1 = y;
    }
    norm += Math.abs(a);
  }
  play(buf, data, norm);
}

// One mode on its own, held a little longer.
export function tone(ratio, { f0 = 110, seconds = 2.5 } = {}) {
  const c = context();
  const n = Math.round(seconds * RATE);
  const buf = c.createBuffer(1, n, RATE);
  const data = buf.getChannelData(0);
  const f = f0 * ratio;
  for (let t = 0; t < n; t++) {
    const env = Math.min(1, t / (0.01 * RATE)) * Math.exp(-t / (0.9 * RATE));
    data[t] = 0.5 * env * Math.sin((2 * Math.PI * f * t) / RATE);
  }
  play(buf, data, 0);
}

function play(buf, data, norm) {
  // peak-normalise, with a 3 ms attack and a short fade at the end
  let peak = 0;
  for (const v of data) peak = Math.max(peak, Math.abs(v));
  const g = peak > 0 ? 0.6 / peak : 0;
  const n = data.length, att = Math.round(0.003 * RATE), rel = Math.round(0.05 * RATE);
  for (let t = 0; t < n; t++) {
    let e = g;
    if (t < att) e *= t / att;
    if (t > n - rel) e *= (n - t) / rel;
    data[t] *= e;
  }
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(out);
  src.start();
  return norm;
}

export function unlock() { context(); }
