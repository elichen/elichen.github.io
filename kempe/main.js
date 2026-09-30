import { MachineView } from './render.js';
import { targetFromPoints } from './mech.js';
import { SHAPES, iconPath, densify, outAndBack } from './shapes.js';

const $ = (id) => document.getElementById(id);
const canvas = $('view');
const view = new MachineView(canvas);
const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

const state = {
  ready: false,
  count: 0,
  poolSize: 0,
  solveId: 0,
  points: null,        // closed path sent to the worker
  raw: null,           // what was drawn (for the ghost and the share link)
  open: false,
  target: null,
  shown: null,         // entry currently on the plate
  best: null,          // latest snapshot
  running: !reduceMotion,
  theta: 0,
  busy: false,
  pendingShape: null,
};

const fmt = new Intl.NumberFormat('en-US');
const setStatus = (text) => { $('status').textContent = text; };

// --- worker messages --------------------------------------------------------

worker.onmessage = (e) => {
  const msg = e.data;
  if (msg.type === 'loading') {
    setStatus(`Loading machines… ${(msg.loaded / 1e6).toFixed(1)} of ${(msg.total / 1e6).toFixed(1)} MB`);
    return;
  }
  if (msg.type === 'ready') {
    state.ready = true;
    state.count = msg.count;
    state.poolSize = msg.meta.poolSize || msg.count;
    $('count').textContent = fmt.format(msg.count);
    setStatus(`${fmt.format(msg.count)} machines loaded. Draw a shape on the plate.`);
    const fromLink = readLink();
    if (fromLink) useDrawing(fromLink.points, fromLink.open);
    else useShape(state.pendingShape || 'heart');
    return;
  }
  if (msg.id !== state.solveId) return;
  if (msg.type === 'searched') {
    setStatus(`Searched ${fmt.format(msg.searched)} machines in ${Math.round(msg.ms)} ms. Tuning the closest ones…`);
  } else if (msg.type === 'progress') {
    show(msg.best);
    const stage = msg.best.stage;
    if (stage > 8) setStatus(`Trying the ${stage} closest machines…`);
    else if (stage > 1) setStatus(`Tuning the best ${stage}…`);
    else if (msg.best.hops !== undefined) setStatus(`Refining: ${msg.best.hops} fresh starts so far…`);
  } else if (msg.type === 'done') {
    show(msg.best, true);
    if (msg.others) showOthers([msg.best, ...msg.others]);
    state.busy = false;
    $('refine').disabled = false;
    const err = (msg.best.error * 100).toFixed(1);
    if (!msg.polished) setStatus(`Found in ${(msg.ms / 1000).toFixed(1)} s. The pen's path is within ${err}% of your drawing.`);
    else if (msg.best.error < state.errBefore - 5e-4) setStatus(`Refined from ${(state.errBefore * 100).toFixed(1)}% to ${err}%.`);
    else setStatus(`No closer fit nearby: ${err}% is as close as this machine gets.`);
  } else if (msg.type === 'error') {
    state.busy = false;
    setStatus(`Something went wrong: ${msg.message}. Try drawing again.`);
  }
};

function show(snap, final = false) {
  state.best = snap;
  if (state.shown !== snap.entry) {
    view.setMachine(snap.spec, snap.P, snap.placement, state.target, state.shown === null);
    state.shown = snap.entry;
    frameAll();
  } else {
    view.updateDesign(snap.P, snap.placement);
  }
  if (final) frameAll();
  $('err').textContent = `${(snap.error * 100).toFixed(1)}%`;
  $('bars').textContent = snap.bars;
  $('cranks').textContent = snap.gears + 1;
}

// The runners-up of the last search, so the fit/size trade-off is visible.
function showOthers(all) {
  const row = $('others');
  row.replaceChildren();
  const seen = new Set();
  const list = all.filter((s) => {
    const key = `${s.bars}:${(s.error * 100).toFixed(1)}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
  if (list.length < 2) { row.hidden = true; return; }
  const label = document.createElement('span');
  label.textContent = 'Machines found';
  row.append(label);
  for (const snap of list) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'other';
    b.textContent = `${snap.bars} bars, ${(snap.error * 100).toFixed(1)}%`;
    b.setAttribute('aria-pressed', String(snap.entry === state.best.entry));
    b.addEventListener('click', () => {
      for (const o of row.querySelectorAll('.other')) o.setAttribute('aria-pressed', String(o === b));
      show(snap, true);
      setStatus(`This machine uses ${snap.bars} bars and stays within ${(snap.error * 100).toFixed(1)}% of your drawing.`);
    });
    row.append(b);
  }
  row.hidden = false;
}

function frameAll() {
  const b = view.bounds();
  const t = state.target;
  if (!b || !t) return;
  // union of the machine's sweep and the drawing
  const r2 = t.scale * 1.6;
  const x0 = Math.min(b.cx - b.rad, t.mean[0] - r2), x1 = Math.max(b.cx + b.rad, t.mean[0] + r2);
  const y0 = Math.min(b.cy - b.rad, t.mean[1] - r2), y1 = Math.max(b.cy + b.rad, t.mean[1] + r2);
  view.frameCircle((x0 + x1) / 2, (y0 + y1) / 2, (Math.max(x1 - x0, y1 - y0) / 2) * 0.9);
}

// --- drawings ---------------------------------------------------------------

function useShape(name) {
  if (!state.ready) { state.pendingShape = name; return; }
  const s = SHAPES[name];
  view.setHome(8);
  useDrawing(s.open ? densify(s.points) : s.points, !!s.open);
  history.replaceState(null, '', location.pathname);
}

function useDrawing(pts, open) {
  state.raw = pts;
  state.open = open;
  state.points = open ? outAndBack(pts) : pts;
  state.target = targetFromPoints(Float64Array.from(state.points));
  view.clearMachine();
  view.setGhost(pts, !open);
  state.shown = null;
  $('others').hidden = true;
  state.busy = true;
  $('refine').disabled = true;
  state.solveId++;
  setStatus(`Searching ${fmt.format(state.count)} machines…`);
  worker.postMessage({ type: 'solve', id: state.solveId, points: state.points, simplicity: +$('simplicity').value });
}

// Freehand drawing on the plate: one pointer, left button or a finger. The
// old machine goes away only once the pen has actually moved, so a tap or a
// click on the plate changes nothing.
let stroke = null;
canvas.addEventListener('pointerdown', (e) => {
  if (e.button !== 0 || !e.isPrimary) {
    if (stroke && stroke.started) restorePrevious();
    stroke = null;
    return;
  }
  const p = view.pick(e.clientX, e.clientY);
  if (p) stroke = { id: e.pointerId, pts: [p[0], p[1]], started: false };
});
canvas.addEventListener('pointermove', (e) => {
  if (!stroke || e.pointerId !== stroke.id) return;
  const p = view.pick(e.clientX, e.clientY);
  if (!p) return;
  const n = stroke.pts.length;
  const step = (view.focus ? view.focus.rad : 8) * 0.004;
  if (Math.hypot(p[0] - stroke.pts[n - 2], p[1] - stroke.pts[n - 1]) < step) return;
  stroke.pts.push(p[0], p[1]);
  if (!stroke.started && stroke.pts.length >= 8) {
    // a new drawing supersedes whatever the worker is still doing
    stroke.started = true;
    state.solveId++;
    worker.postMessage({ type: 'cancel' });
    state.best = null;
    state.busy = false;
    $('refine').disabled = true;
    $('others').hidden = true;
    $('hint').hidden = true;
    view.clearMachine();
  }
  if (stroke.started) view.setGhost(stroke.pts, false);
});
window.addEventListener('pointerup', (e) => {
  if (!stroke || e.pointerId !== stroke.id) return;
  const { pts, started } = stroke;
  stroke = null;
  if (started) finishStroke(pts);
});

// Put the last drawing's machine back (after an abandoned stroke).
function restorePrevious() {
  if (state.raw) useDrawing(state.raw, state.open);
  else view.setGhost([], false);
}

function finishStroke(pts) {
  const n = pts.length / 2;
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (let i = 0; i < n; i++) {
    x0 = Math.min(x0, pts[2 * i]); x1 = Math.max(x1, pts[2 * i]);
    y0 = Math.min(y0, pts[2 * i + 1]); y1 = Math.max(y1, pts[2 * i + 1]);
  }
  const diag = Math.hypot(x1 - x0, y1 - y0);
  if (n < 8 || diag < (view.focus ? view.focus.rad : 8) * 0.05) {
    restorePrevious();
    setStatus('That stroke was too short. Draw a bigger shape in one stroke.');
    return;
  }
  const gap = Math.hypot(pts[0] - pts[2 * n - 2], pts[1] - pts[2 * n - 1]);
  const open = gap > 0.2 * diag;
  useDrawing(pts, open);
  writeLink(pts, open);
}

// --- share links: the drawing, 64 points quantised to bytes -----------------

function writeLink(pts, open) {
  const n = pts.length / 2, m = 64;
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (let i = 0; i < n; i++) {
    x0 = Math.min(x0, pts[2 * i]); x1 = Math.max(x1, pts[2 * i]);
    y0 = Math.min(y0, pts[2 * i + 1]); y1 = Math.max(y1, pts[2 * i + 1]);
  }
  const s = Math.max(x1 - x0, y1 - y0) || 1;
  const bytes = new Uint8Array(1 + 2 * m);
  bytes[0] = open ? 1 : 0;
  for (let k = 0; k < m; k++) {
    const i = Math.round((k * (n - 1)) / (m - 1));
    bytes[1 + 2 * k] = Math.round(((pts[2 * i] - x0) / s) * 255);
    bytes[2 + 2 * k] = Math.round(((pts[2 * i + 1] - y0) / s) * 255);
  }
  const b64 = btoa(String.fromCharCode(...bytes)).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
  history.replaceState(null, '', `#d=${b64}`);
}

function readLink() {
  const m = location.hash.match(/d=([A-Za-z0-9_-]+)/);
  if (!m) return null;
  try {
    const bin = atob(m[1].replace(/-/g, '+').replace(/_/g, '/'));
    const open = bin.charCodeAt(0) === 1, pts = [];
    for (let k = 1; k + 1 < bin.length; k += 2) pts.push((bin.charCodeAt(k) / 255 - 0.5) * 6, (bin.charCodeAt(k + 1) / 255 - 0.5) * 6);
    return { points: open ? densify(pts) : pts, open };
  } catch {
    return null;
  }
}

// --- controls ----------------------------------------------------------------

for (const [name, s] of Object.entries(SHAPES)) {
  const b = document.createElement('button');
  b.type = 'button';
  b.className = 'key';
  b.title = s.label;
  b.setAttribute('aria-label', s.label);
  b.innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="${iconPath(s.points, s.open)}"/></svg>`;
  b.addEventListener('click', () => useShape(name));
  $('shapes').append(b);
}

function setRunning(on) {
  state.running = on;
  $('run').setAttribute('aria-pressed', String(on));
  $('run-label').textContent = on ? 'Stop' : 'Run';
}
$('run').addEventListener('click', () => setRunning(!state.running));
setRunning(state.running);

$('plan').addEventListener('click', () => {
  const plan = view.view !== 'plan';
  view.setView(plan ? 'plan' : 'angle');
  $('plan').setAttribute('aria-pressed', String(plan));
});

$('refine').addEventListener('click', () => {
  if (!state.best || state.busy) return;
  state.busy = true;
  state.errBefore = state.best.error;
  $('refine').disabled = true;
  state.solveId++;
  setStatus('Refining…');
  worker.postMessage({ type: 'polish', id: state.solveId, entry: state.best.entry, P: state.best.P, points: state.points });
});

$('simplicity').addEventListener('change', () => { if (state.raw) useDrawing(state.raw, state.open); });

$('share').addEventListener('click', async () => {
  if (state.raw) writeLink(state.raw, state.open);
  try {
    await navigator.clipboard.writeText(location.href);
    setStatus('Link copied. It opens this drawing and finds its machine again.');
  } catch {
    setStatus('Copy the address bar to share this drawing.');
  }
});

window.addEventListener('keydown', (e) => {
  if (e.target.closest('input, textarea') || e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === ' ' && !e.target.closest('button')) { e.preventDefault(); setRunning(!state.running); }
  else if (e.key === 't' || e.key === 'T') $('plan').click();
  else if (e.key === 'r' || e.key === 'R') $('refine').click();
});

// --- start -------------------------------------------------------------------

worker.postMessage({ type: 'init', base: new URL('./data/', import.meta.url).href });
setStatus('Loading machines…');
if (!location.hash) view.setGhost(SHAPES.heart.points, true);

let last = performance.now();
function loop(now) {
  const dt = Math.min(0.05, (now - last) / 1000);
  last = now;
  if (state.running) state.theta += dt * Math.PI * 2 * 0.22;
  view.frame(state.theta);
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);

// for poking at from the console
window.kempe = { state, view };
