// Page: runs the simulation, draws it, reads the beads back for the trails and
// statistics, and handles the controls and dragging.
import { Sim, initialState, STATE_BYTES, REC_BYTES } from './sim.js';
import { Renderer } from './render.js';
import { MotionStats } from './analysis.js';
import { MSDChart } from './chart.js';

const DT = 0.005;              // τ per step
const DENSITY = 0.7;           // molecules per σ²
const BEAD_A = 4;              // bead radius, σ
const N_BEADS = 5;
const RECORD_EVERY = 10;       // steps between bead records (0.05 τ)
const RING = 4096;             // records kept on the GPU
const TRAIL_DT = 50;           // τ between trail dots
const TRAIL_MAX = 40;
const PS_PER_TAU = 2.15;       // τ for argon-sized molecules (σ = 0.34 nm), in picoseconds
const SETTLE = 100;            // τ to wait after a temperature change before measuring again

const $ = id => document.getElementById(id);
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const status = msg => { $('status').textContent = msg; };

const viewport = $('viewport'), canvas = $('view'), overlay = $('overlay');
const speedInput = $('speed'), tempInput = $('temp');

let device, context, renderer, chart, sim, stats;
let temperature = +tempInput.value;
let latest = null;               // last bead state read back
let recNext = 1;                 // next bead record to read
let pool = [];                   // free staging buffers for readback
let transform = { scale: 1, ox: 0, oy: 0 };
let trails = [], trailIndex = -1;
let grab = null;                 // { bead, pointerId }
let excludeUntil = [], needsForget = [];
let resetAt = -1;                // step at which to clear the statistics
let liquidT = NaN;
let stepCap = 1000, frameMs = 16.7, baseMs = 100, frames = 0, lastNow = 0, carry = 0;
let stepsPerSec = 0, lastRead = { t: 0, steps: 0 };

// Simulated time per second of real time, 0.5 to 200 τ/s on a log scale
function tauPerSec() {
  return 0.5 * 400 ** (+speedInput.value / 100);
}

function boxFor(w, h) {
  const pxPerSigma = 7;
  const area = Math.min(24000, Math.max(5000, (w / pxPerSigma) * (h / pxPerSigma)));
  const aspect = Math.min(3.5, Math.max(0.5, w / h));
  return [Math.round(Math.sqrt(area * aspect)), Math.round(Math.sqrt(area / aspect))];
}

function placeBeads(L) {
  const m = DENSITY * Math.PI * BEAD_A * BEAD_A;   // as heavy as the liquid it displaces
  const cols = Math.ceil(Math.sqrt(N_BEADS * L[0] / L[1])), rows = Math.ceil(N_BEADS / cols);
  return Array.from({ length: N_BEADS }, (_, k) => {
    const r = Math.floor(k / cols), c = k % cols, inRow = Math.min(cols, N_BEADS - r * cols);
    return {
      x: (c + 0.5 + (Math.random() - 0.5) * 0.4) * L[0] / inRow,
      y: (r + 0.5 + (Math.random() - 0.5) * 0.4) * L[1] / rows,
      a: BEAD_A, m,
    };
  });
}

function newSim() {
  const r = viewport.getBoundingClientRect();
  const L = boxFor(r.width, r.height);
  const init = initialState({ L, density: DENSITY, beads: placeBeads(L), T: temperature, seed: (Math.random() * 2 ** 31) | 0 });
  if (sim) sim.destroy();
  for (const b of pool) b.destroy();
  renderer.forget();
  // the drag spring is about critically damped for a bead in this liquid (γ ≈ 6.5, M = 35)
  sim = new Sim(device, { L, dt: DT, T: temperature, tauT: 1, recordEvery: RECORD_EVERY, ringSize: RING, kSpring: 0.5, fMax: 10 }, init);
  sim.run(3000);                 // melt the starting lattice with a strong thermostat
  sim.tauT = 20;
  sim.writeParams();
  const size = STATE_BYTES + RING * sim.nb * REC_BYTES;
  pool = [0, 1, 2].map(() => device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }));
  stats = new MotionStats(sim.nb, RECORD_EVERY * DT);
  recNext = Math.floor(sim.steps / RECORD_EVERY) + 1;
  trails = init.beads.map(() => []);
  trailIndex = -1;
  excludeUntil = init.beads.map(() => 0);
  needsForget = init.beads.map(() => false);
  resetAt = -1;
  latest = null;
  grab = null;
  for (const e of document.querySelectorAll('.n-molecules')) e.textContent = sim.n.toLocaleString('en-US');
  layout();
}

function layout() {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.round(viewport.clientWidth * dpr), h = Math.round(viewport.clientHeight * dpr);
  for (const c of [canvas, overlay]) {
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
  }
  if (!sim) return;
  const scale = Math.min(w / sim.L[0], h / sim.L[1]);
  transform = { scale, ox: (w - sim.L[0] * scale) / 2, oy: (h - sim.L[1] * scale) / 2 };
}

// Copy the state and the new bead records into a staging buffer (if one is free).
function readback(enc) {
  const buf = pool.pop();
  if (!buf) return null;
  const kmax = Math.floor(sim.steps / RECORD_EVERY);
  let from = recNext, gap = false;
  if (kmax - from + 1 > RING - 64) { from = kmax - (RING - 64) + 1; gap = true; }
  const recBytes = REC_BYTES * sim.nb;
  enc.copyBufferToBuffer(sim.stateBuffer, 0, buf, 0, STATE_BYTES);
  let off = STATE_BYTES;
  for (let k = from; k <= kmax;) {
    const slot = k % RING, len = Math.min(kmax - k + 1, RING - slot);
    enc.copyBufferToBuffer(sim.ring, slot * recBytes, buf, off, len * recBytes);
    off += len * recBytes;
    k += len;
  }
  recNext = kmax + 1;
  return { buf, from, count: Math.max(0, kmax - from + 1), gap, owner: sim };
}

function absorb({ buf, from, count, gap, owner }) {
  if (owner !== sim) return;
  const ab = buf.getMappedRange();
  latest = Sim.parseState(ab.slice(0, STATE_BYTES), sim.nb);
  liquidT = Number.isNaN(liquidT) ? latest.T : 0.97 * liquidT + 0.03 * latest.T;
  const f = new Float32Array(ab, STATE_BYTES, count * sim.nb * 8);
  const iv = new Int32Array(ab, STATE_BYTES, count * sim.nb * 8);
  if (gap) for (let b = 0; b < sim.nb; b++) stats.forget(b);
  const [Lx, Ly] = sim.L;
  for (let r = 0; r < count; r++) {
    const step = (from + r) * RECORD_EVERY;
    if (resetAt >= 0 && step >= resetAt) { stats.reset(); resetAt = -1; }
    const ti = Math.floor(step * DT / TRAIL_DT);
    const newDot = ti > trailIndex;
    if (newDot) trailIndex = ti;
    for (let b = 0; b < sim.nb; b++) {
      const o = (r * sim.nb + b) * 8;
      const x = f[o] + iv[o + 4] * Lx, y = f[o + 1] + iv[o + 5] * Ly;
      if (step < excludeUntil[b]) { needsForget[b] = true; continue; }
      if (needsForget[b]) { stats.forget(b); needsForget[b] = false; }
      if (newDot) {
        trails[b].push([x, y]);
        if (trails[b].length > TRAIL_MAX) trails[b].shift();
      }
      if (resetAt < 0) stats.add(b, x, y, f[o + 2], f[o + 3], sim.beadInfo[b].m);
    }
  }
}

function frame(now) {
  requestAnimationFrame(frame);
  const dtf = adapt(now);
  carry += tauPerSec() * Math.min(dtf, 50) / 1000 / DT;
  const steps = Math.min(stepCap, Math.floor(carry));
  carry = steps < Math.floor(carry) ? 0 : carry - steps;   // if the GPU can't keep up, drop the backlog
  if (steps > 0) step(steps); else drawOverlay();
}

function step(steps) {
  layout();
  const enc = device.createCommandEncoder();
  sim.encode(enc, steps);
  const rb = readback(enc);
  renderer.draw(enc, sim, transform, canvas.width, canvas.height);
  device.queue.submit([enc.finish()]);
  if (rb) {
    rb.buf.mapAsync(GPUMapMode.READ).then(() => {
      absorb(rb);
      rb.buf.unmap();
      if (rb.owner === sim) pool.push(rb.buf); else rb.buf.destroy();
    }, () => {});
  }
  drawOverlay();
}

// Keep each frame's GPU work inside the frame: if frames start to stretch, take fewer steps.
// Returns the time since the last frame (ms).
function adapt(now) {
  const dtf = now - lastNow;
  lastNow = now;
  if (!(dtf > 0 && dtf < 250)) return 16.7;
  baseMs = Math.min(baseMs * 1.0005, dtf);           // the display's frame interval
  frameMs = 0.9 * frameMs + 0.1 * dtf;
  if (++frames % 20 === 0) {
    const want = Math.ceil(tauPerSec() * baseMs / 1000 / DT);
    if (frameMs > 1.3 * baseMs) stepCap = Math.max(4, Math.floor(Math.min(stepCap, want) * 0.8));
    else if (frameMs < 1.1 * baseMs && stepCap < want) stepCap = Math.ceil(stepCap * 1.1);
  }
  return dtf;
}

function toSim(e) {
  const r = overlay.getBoundingClientRect(), dpr = overlay.width / r.width;
  const x = ((e.clientX - r.left) * dpr - transform.ox) / transform.scale;
  const y = ((e.clientY - r.top) * dpr - transform.oy) / transform.scale;
  return [x - sim.L[0] * Math.floor(x / sim.L[0]), y - sim.L[1] * Math.floor(y / sim.L[1])];
}

function nearestBead(p) {
  if (!latest) return -1;
  let best = -1, bestD = Infinity;
  latest.beads.forEach((b, k) => {
    let dx = p[0] - b.x[0], dy = p[1] - b.x[1];
    dx -= sim.L[0] * Math.round(dx / sim.L[0]);
    dy -= sim.L[1] * Math.round(dy / sim.L[1]);
    const d = Math.hypot(dx, dy);
    if (d < bestD) { bestD = d; best = k; }
  });
  return bestD < BEAD_A + 4 ? best : -1;
}

overlay.addEventListener('pointerdown', e => {
  if (!sim) return;
  const p = toSim(e), b = nearestBead(p);
  if (b < 0) return;
  grab = { bead: b, pointerId: e.pointerId, at: p };
  try { overlay.setPointerCapture(e.pointerId); } catch { /* pointer already gone */ }
  sim.grab = b;
  sim.anchor = p;
  sim.writeParams();
  excludeUntil[b] = Infinity;
  trails[b] = [];
  document.body.classList.add('dragged');
  e.preventDefault();
});

// On touch screens the page scrolls over the liquid, except when a touch starts on a bead.
overlay.addEventListener('touchstart', e => {
  if (sim && e.touches.length === 1 && nearestBead(toSim(e.touches[0])) >= 0) e.preventDefault();
}, { passive: false });

overlay.addEventListener('pointermove', e => {
  if (grab && e.pointerId === grab.pointerId) {
    grab.at = toSim(e);
    sim.anchor = grab.at;
    sim.writeParams();
  } else if (sim) {
    overlay.style.cursor = nearestBead(toSim(e)) >= 0 ? 'grab' : 'default';
  }
});

const release = e => {
  if (!grab || e.pointerId !== grab.pointerId) return;
  excludeUntil[grab.bead] = sim.steps + Math.round(20 / DT);   // let it settle before measuring it again
  grab = null;
  sim.grab = -1;
  sim.writeParams();
};
overlay.addEventListener('pointerup', release);
overlay.addEventListener('pointercancel', release);

function drawOverlay() {
  const g = overlay.getContext('2d');
  g.setTransform(1, 0, 0, 1, 0, 0);
  g.clearRect(0, 0, overlay.width, overlay.height);
  if (!latest) return;
  const { scale, ox, oy } = transform;
  const [Lx, Ly] = sim.L;
  const dpr = window.devicePixelRatio || 1;
  g.setTransform(scale, 0, 0, scale, ox, oy);
  g.beginPath();
  g.rect(0, 0, Lx, Ly);
  g.clip();
  const trail = css('--trail');
  latest.beads.forEach((b, k) => {
    const pts = trails[k];
    if (!pts.length) return;
    // draw the trail next to the bead's current image (and its periodic copies)
    const ux = b.x[0] + b.wrap[0] * Lx, uy = b.x[1] + b.wrap[1] * Ly;
    const sx = b.x[0] - ux, sy = b.x[1] - uy;
    for (const cx of [-Lx, 0, Lx]) for (const cy of [-Ly, 0, Ly]) {
      g.save();
      g.translate(sx + cx, sy + cy);
      g.strokeStyle = trail;
      g.fillStyle = trail;
      g.lineWidth = 1.2 * dpr / scale;
      for (let i = 1; i < pts.length; i++) {
        g.globalAlpha = 0.25 + 0.75 * i / pts.length;
        g.beginPath(); g.moveTo(pts[i - 1][0], pts[i - 1][1]); g.lineTo(pts[i][0], pts[i][1]); g.stroke();
      }
      g.globalAlpha = 1;
      g.setLineDash([2 * dpr / scale, 3 * dpr / scale]);
      g.beginPath(); g.moveTo(pts[pts.length - 1][0], pts[pts.length - 1][1]); g.lineTo(ux, uy); g.stroke();
      g.setLineDash([]);
      for (let i = 0; i < pts.length; i++) {
        g.globalAlpha = 0.25 + 0.75 * (i + 1) / pts.length;
        g.beginPath(); g.arc(pts[i][0], pts[i][1], 1.8 * dpr / scale, 0, 2 * Math.PI); g.fill();
      }
      g.restore();
    }
  });
  if (grab) {
    const b = latest.beads[grab.bead];
    let dx = grab.at[0] - b.x[0], dy = grab.at[1] - b.x[1];
    dx -= Lx * Math.round(dx / Lx);
    dy -= Ly * Math.round(dy / Ly);
    g.strokeStyle = css('--trail');
    g.lineWidth = 1.5 * dpr / scale;
    g.beginPath(); g.moveTo(b.x[0], b.x[1]); g.lineTo(b.x[0] + dx, b.x[1] + dy); g.stroke();
    g.beginPath(); g.arc(b.x[0] + dx, b.x[1] + dy, 3 * dpr / scale, 0, 2 * Math.PI); g.stroke();
  }
}

function fmt(v, digits = 2) { return Number.isFinite(v) ? v.toFixed(digits) : '—'; }

function updateReadouts() {
  if (!sim || !stats) return;
  const now = performance.now();
  if (now - lastRead.t > 1000) {
    stepsPerSec = (sim.steps - lastRead.steps) / ((now - lastRead.t) / 1000);
    lastRead = { t: now, steps: sim.steps };
  }
  const D = stats.diffusion(100);
  $('ro-liquid').textContent = fmt(liquidT);
  $('ro-bead').textContent = fmt(stats.beadT);
  $('ro-d').textContent = Number.isFinite(D) ? `${D.toFixed(3)} σ²/τ` : 'measuring…';
  $('ro-time').textContent = `${Math.round(stepsPerSec).toLocaleString('en-US')}/s`;
  const ps = tauPerSec() * PS_PER_TAU;
  $('speed-out').textContent = `${ps < 10 ? ps.toFixed(1) : Math.round(ps)} ps/s`;
  chart.draw({
    points: stats.msd(), T: temperature, M: sim.beadInfo[0].m, D,
    tMin: RECORD_EVERY * DT, tMax: 1e4,
  });
}

function applyColors() {
  renderer.setColors({
    bg: css('--viewport-bg'), cool: css('--mol-cool'), hot: css('--mol-hot'),
    bead: css('--bead'), beadHi: css('--bead-hi'), beadLo: css('--bead-lo'),
  });
  chart.colors = {
    grid: css('--grid'), text3: css('--text-3'), series: css('--series'),
    ballistic: css('--ballistic'), diffusive: css('--diffusive'),
  };
}

function setTemperature(v) {
  temperature = v;
  $('temp-out').textContent = v.toFixed(2);
  if (!sim) return;
  sim.T0 = v;
  sim.writeParams();
  resetAt = sim.steps + Math.round(SETTLE / DT);
  liquidT = NaN;
}

tempInput.addEventListener('input', () => setTemperature(+tempInput.value));
speedInput.addEventListener('input', () => { stepCap = Math.max(stepCap, 8); updateReadouts(); });
$('restart').addEventListener('click', () => newSim());
for (const b of document.querySelectorAll('button[data-speed], button[data-temp]')) {
  b.addEventListener('click', () => {
    if (b.dataset.speed) { speedInput.value = b.dataset.speed; stepCap = Math.max(stepCap, 8); updateReadouts(); }
    if (b.dataset.temp) { tempInput.value = b.dataset.temp; setTemperature(+b.dataset.temp); }
    viewport.scrollIntoView({ behavior: 'smooth', block: 'start' });
  });
}
window.addEventListener('resize', layout);

async function start() {
  if (!navigator.gpu) {
    status('This page needs WebGPU (Chrome, Edge or Safari 26 and newer).');
    return;
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) { status('No WebGPU adapter is available on this device.'); return; }
  device = await adapter.requestDevice();
  device.lost.then(info => { if (info.reason !== 'destroyed') status(`The GPU stopped (${info.message}). Reload the page to start again.`); });
  context = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });
  renderer = new Renderer(device, context, format);
  chart = new MSDChart($('msd'));
  applyColors();
  setTemperature(temperature);
  newSim();
  status('');
  document.body.classList.add('ready');
  requestAnimationFrame(frame);
  setInterval(updateReadouts, 250);

  // For testing in a background tab, where requestAnimationFrame stops:
  // brownian.advance(τ) steps the simulation and draws.
  window.brownian = {
    get sim() { return sim; }, get stats() { return stats; }, get latest() { return latest; },
    colors(vars) {                        // try another palette: { '--bead': '#…', … }
      for (const [k, v] of Object.entries(vars)) document.documentElement.style.setProperty(k, v);
      applyColors();
    },
    async advance(tau, perFrame = 300) {
      for (let s = Math.round(tau / DT), k = 1; s > 0; s -= perFrame, k++) {
        step(Math.min(s, perFrame));
        if (k % 8 === 0) {                // let the readbacks land, as they would between frames
          await device.queue.onSubmittedWorkDone();
          await new Promise(r => setTimeout(r, 0));
        }
      }
      await device.queue.onSubmittedWorkDone();
    },
  };
}

start().catch(err => { console.error(err); status(`Could not start: ${err.message}`); });
