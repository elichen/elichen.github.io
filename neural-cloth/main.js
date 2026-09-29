import { ClothEngine } from './cloth-engine.js';
import { ClothRenderer, PALETTE, sub, add, scale, dot, cross, norm } from './cloth-render.js';

const $ = (id) => document.getElementById(id);
const REST_Y = 0.0045;          // resting height on the table (barrier offset + half the barrier gap)
const MAX_GRIP_STEP = 0.03;     // m per frame (1.8 m/s), inside the training range
const TARGET = [0, 0.08, 0];

const ui = {
  canvas: $('view'),
  status: $('status'),
  hint: $('hint'),
  stepMs: $('stat-step'),
  stepMsTable: $('stat-step-table'),
  verts: $('stat-verts'),
  contacts: $('stat-contacts'),
  stiff: $('stiffness'),
  stiffOut: $('stiffness-out'),
  friction: $('friction'),
  frictionOut: $('friction-out'),
  wind: $('wind'),
  windOut: $('wind-out'),
  pause: $('pause'),
  viewport: $('viewport'),
  solverCanvas: $('view-solver'),
  clips: $('clips'),
  clipError: $('clip-error'),
};

const state = {
  engine: null,
  renderer: null,
  device: null,
  scene: 'table',
  frame: 0,
  paused: false,
  grips: [],              // {vertex, target:[x,y,z], goal:[x,y,z], pinned, scripted}
  drag: null,             // {grip, planeN, planeD}
  orbit: null,
  cam: { theta: 0.6, phi: 0.95, radius: 1.7 },
  target: [...TARGET],
  camSaved: null,         // the scene camera while compare mode frames a clip
  positions: null,        // CPU copy [n*4], refreshed asynchronously
  obstacles: { spheres: [], capsules: [] },
  script: null,
  inflight: 0,
  kb: 3e-6,
  mu: 0.5,
  wind: 0,
  clip: null,             // compare mode: {meta, pos, n, frame}
  solverRenderer: null,
};

// ------------------------------------------------------------------ scenes

const minJerk = (t) => { t = Math.min(1, Math.max(0, t)); return 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5; };

// Waypoints relative to where the gripped vertices were when grabbed: [[seconds, [dx,dy,dz]], ...]
function scriptPath(keys) {
  return (t) => {
    let prev = [0, [0, 0, 0]];
    for (const k of keys) {
      if (t <= k[0]) {
        const s = minJerk((t - prev[0]) / (k[0] - prev[0]));
        return add(prev[1], scale(sub(k[1], prev[1]), s));
      }
      prev = k;
    }
    return prev[1];
  };
}

const SCENES = {
  table: {
    label: 'Towel on a table', size: [31, 31], height: 0, hint: 'Drag the towel to pick it up. Hold Shift when you let go to pin it in place.',
  },
  ball: {
    label: 'Drape over a ball', size: [31, 31], height: 0.34, hint: 'The towel falls onto a ball. Drag a corner to pull it off.',
    obstacles: { spheres: [{ c: [0, 0.11, 0], r: 0.11 }] },
  },
  rack: {
    label: 'Towel rack', size: [31, 21], height: 0.42, offset: [0, 0, 0.06], hint: 'Hang it on the rail, or drag it off.',
    obstacles: { capsules: [{ a: [-0.42, 0.3, 0], b: [0.42, 0.3, 0], r: 0.014 }] },
  },
  fold: {
    label: 'Robot fold', size: [31, 31], height: 0, hint: 'Two grippers fold the towel in half, then it is yours to drag.',
    script: { start: 20, corners: 'left', release: 2.2, path: scriptPath([[0.9, [0.3, 0.16, 0]], [1.8, [0.575, 0.02, 0]], [2.2, [0.575, 0.02, 0]]]) },
  },
  fling: {
    label: 'Robot fling', size: [31, 26], height: 0, hint: 'A two-handed fling, the way cloth-folding robots flatten laundry.',
    script: { start: 20, corners: 'left', release: 2.1, path: scriptPath([[1.1, [0.05, 0.55, 0]], [1.35, [0.3, 0.6, 0]], [1.9, [-0.2, 0.04, 0]], [2.1, [-0.2, 0.04, 0]]]) },
  },
};

function clothStart(scene) {
  const [nx, ny] = scene.size;
  const x = new Float32Array(nx * ny * 3);
  const cx = ((nx - 1) * 0.02) / 2;
  const cz = ((ny - 1) * 0.02) / 2;
  const off = scene.offset || [0, 0, 0];
  for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) {
    const v = i * ny + j;
    x[3 * v] = i * 0.02 - cx + off[0];
    x[3 * v + 1] = REST_Y + scene.height;
    x[3 * v + 2] = j * 0.02 - cz + off[2];
  }
  return x;
}

async function loadScene(name) {
  const scene = SCENES[name];
  state.clip = null;
  ui.viewport.classList.remove('compare');
  if (state.camSaved) {
    state.cam = state.camSaved.cam;
    state.target = state.camSaved.target;
    state.camSaved = null;
  }
  document.querySelectorAll('[data-clip]').forEach((b) => b.setAttribute('aria-pressed', 'false'));
  ui.clipError.hidden = true;
  state.scene = name;
  state.frame = 0;
  state.grips = [];
  state.drag = null;
  const { engine, renderer } = state;
  if (!engine.mesh || engine.mesh.nx !== scene.size[0] || engine.mesh.ny !== scene.size[1]) {
    await engine.setCloth(scene.size[0], scene.size[1]);
    renderer.setCloth(engine);
  }
  const x = clothStart(scene);
  engine.setState(x, x);
  state.positions = null;
  state.obstacles = { spheres: [...(scene.obstacles?.spheres || [])], capsules: [...(scene.obstacles?.capsules || [])] };
  engine.setObstacles(state.obstacles, state.obstacles);
  state.script = scene.script ? { ...scene.script, anchors: null } : null;
  ui.hint.textContent = scene.hint;
  ui.verts.textContent = engine.n.toLocaleString();
  document.querySelectorAll('[data-scene]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.scene === name)));
  applyMaterial();
}

function applyMaterial() {
  if (!state.engine?.buf) return;
  const w = state.wind;
  state.engine.setScene({ kb: state.kb, mu: state.mu, wind: [w * 0.8, 0, w * 0.6] });
}

// ------------------------------------------------------------------ grippers

function vertexPos(v) {
  const p = state.positions;
  return p ? [p[4 * v], p[4 * v + 1], p[4 * v + 2]] : null;
}

function runScript() {
  const s = state.script;
  if (!s) return;
  const t = (state.frame - s.start) / 60;
  if (t < 0) return;
  const { nx, ny } = state.engine.mesh;
  if (!s.anchors) {
    if (!state.positions) return;
    const corners = s.corners === 'left' ? [0, ny - 1] : [(nx - 1) * ny, nx * ny - 1];
    s.anchors = corners.map((v) => ({ v, p: vertexPos(v) }));
    state.grips = state.grips.filter((g) => !g.scripted);
    for (const a of s.anchors) state.grips.push({ vertex: a.v, target: a.p, goal: a.p, scripted: true, pinned: true });
  }
  if (t > s.release) {
    state.grips = state.grips.filter((g) => !g.scripted);
    state.script = null;
    return;
  }
  const d = s.path(t);
  for (const g of state.grips) {
    if (!g.scripted) continue;
    const a = s.anchors.find((x) => x.v === g.vertex);
    g.goal = add(a.p, d);
  }
}

function advanceGrips() {
  for (const g of state.grips) {
    const delta = sub(g.goal, g.target);
    const len = Math.hypot(...delta);
    const stepLen = Math.min(len, MAX_GRIP_STEP);
    g.target = len > 1e-9 ? add(g.target, scale(delta, stepLen / len)) : g.target;
    g.target[1] = Math.max(g.target[1], REST_Y);
  }
  state.engine.setGrippers(state.grips.map((g) => ({ vertex: g.vertex, target: g.target })));
}

function solids() {
  const out = [];
  for (const s of state.obstacles.spheres) out.push({ kind: 'sphere', a: s.c, r: s.r, color: PALETTE.obstacle });
  for (const c of state.obstacles.capsules) out.push({ kind: 'capsule', a: c.a, b: c.b, r: c.r, color: PALETTE.obstacle });
  for (const g of state.grips) {
    out.push({ kind: 'sphere', a: g.target, r: 0.011, color: PALETTE.gripper });
    out.push({ kind: 'capsule', a: add(g.target, [0, 0.012, 0]), b: add(g.target, [0, 0.3, 0]), r: 0.0025, color: PALETTE.ink });
  }
  return out;
}

// ------------------------------------------------------------------ picking + pointer

function rayHitCloth(ray) {
  const p = state.positions;
  if (!p) return null;
  const tris = state.engine.mesh.tris;
  let best = null;
  for (let t = 0; t < tris.length; t += 3) {
    const a = vertexPos(tris[t]), b = vertexPos(tris[t + 1]), c = vertexPos(tris[t + 2]);
    const e1 = sub(b, a), e2 = sub(c, a);
    const h = cross(ray.dir, e2);
    const det = dot(e1, h);
    if (Math.abs(det) < 1e-12) continue;
    const f = 1 / det;
    const s = sub(ray.origin, a);
    const u = f * dot(s, h);
    if (u < 0 || u > 1) continue;
    const q = cross(s, e1);
    const w = f * dot(ray.dir, q);
    if (w < 0 || u + w > 1) continue;
    const dist = f * dot(e2, q);
    if (dist > 0 && (!best || dist < best.dist)) {
      const hit = add(ray.origin, scale(ray.dir, dist));
      let vbest = tris[t], dbest = Infinity;
      for (const v of [tris[t], tris[t + 1], tris[t + 2]]) {
        const dd = Math.hypot(...sub(vertexPos(v), hit));
        if (dd < dbest) { dbest = dd; vbest = v; }
      }
      best = { dist, hit, vertex: vbest };
    }
  }
  return best;
}

function cameraEye() {
  const { theta, phi, radius } = state.cam;
  const t = state.target;
  return [t[0] + radius * Math.sin(phi) * Math.sin(theta), t[1] + radius * Math.cos(phi),
    t[2] + radius * Math.sin(phi) * Math.cos(theta)];
}

function planePoint(ray, n, d) {
  const denom = dot(n, ray.dir);
  if (Math.abs(denom) < 1e-6) return null;
  const t = (d - dot(n, ray.origin)) / denom;
  return t > 0 ? add(ray.origin, scale(ray.dir, t)) : null;
}

function bindOrbitOnly(c) {
  c.addEventListener('pointerdown', (ev) => {
    c.setPointerCapture(ev.pointerId);
    state.orbit = { x: ev.clientX, y: ev.clientY, theta: state.cam.theta, phi: state.cam.phi };
  });
  c.addEventListener('pointermove', (ev) => {
    const o = state.orbit;
    if (!o) return;
    state.cam.theta = o.theta - (ev.clientX - o.x) * 0.006;
    state.cam.phi = Math.min(1.45, Math.max(0.15, o.phi - (ev.clientY - o.y) * 0.006));
  });
  const end = () => { state.orbit = null; };
  c.addEventListener('pointerup', end);
  c.addEventListener('pointercancel', end);
}

function bindPointer() {
  bindOrbitOnly(ui.solverCanvas);
  const c = ui.canvas;
  c.addEventListener('contextmenu', (e) => e.preventDefault());
  c.addEventListener('pointerdown', async (ev) => {
    c.setPointerCapture(ev.pointerId);
    await refreshPositions();
    const ray = state.renderer.ray(ev.clientX, ev.clientY);
    // clicking an existing gripper grabs (or unpins) it
    let grip = null;
    for (const g of state.grips) {
      const toG = sub(g.target, ray.origin);
      const along = dot(toG, ray.dir);
      if (along > 0 && Math.hypot(...sub(toG, scale(ray.dir, along))) < 0.02) grip = g;
    }
    const hit = grip || state.clip ? null : rayHitCloth(ray);
    if (ev.button === 0 && !state.clip && (grip || hit)) {
      if (!grip) {
        const p = vertexPos(hit.vertex);
        grip = { vertex: hit.vertex, target: p, goal: p, pinned: false };
        state.grips.push(grip);
      }
      grip.scripted = false;
      grip.pinned = false;
      const n = norm(sub(cameraEye(), state.target));
      state.drag = { grip, n, d: dot(n, grip.target) };
      c.classList.add('grabbing');
    } else {
      state.orbit = { x: ev.clientX, y: ev.clientY, theta: state.cam.theta, phi: state.cam.phi };
    }
  });
  c.addEventListener('pointermove', (ev) => {
    if (state.drag) {
      const p = planePoint(state.renderer.ray(ev.clientX, ev.clientY), state.drag.n, state.drag.d);
      if (p) state.drag.grip.goal = [p[0], Math.max(p[1], REST_Y), p[2]];
    } else if (state.orbit) {
      const o = state.orbit;
      state.cam.theta = o.theta - (ev.clientX - o.x) * 0.006;
      state.cam.phi = Math.min(1.45, Math.max(0.15, o.phi - (ev.clientY - o.y) * 0.006));
    }
  });
  const end = (ev) => {
    if (state.drag) {
      const g = state.drag.grip;
      if (ev.shiftKey) g.pinned = true;
      else state.grips = state.grips.filter((x) => x !== g);
    }
    state.drag = null;
    state.orbit = null;
    c.classList.remove('grabbing');
  };
  c.addEventListener('pointerup', end);
  c.addEventListener('pointercancel', end);
  c.addEventListener('wheel', (ev) => {
    ev.preventDefault();
    state.cam.radius = Math.min(3, Math.max(0.5, state.cam.radius * Math.exp(ev.deltaY * 0.001)));
  }, { passive: false });
}


// ------------------------------------------------------------------ compare: solver recording vs network

async function loadClipList() {
  try {
    const list = await fetch('replays/index.json').then((r) => r.json());
    const names = { fold: 'Fold', fling: 'Fling', place: 'Pick and place', drag: 'Drag', shake: 'Shake', lift_drop: 'Lift and drop', none: 'Drop' };
    for (const c of list) {
      const b = document.createElement('button');
      b.type = 'button';
      b.dataset.clip = c.name;
      b.textContent = names[c.kind] || c.kind;
      b.addEventListener('click', () => loadClip(c.name));
      ui.clips.appendChild(b);
    }
  } catch {
    ui.clips.closest('.group').hidden = true;
  }
}

async function loadClip(name) {
  const meta = await fetch(`replays/${name}.json`).then((r) => r.json());
  const q = new Int16Array(await fetch(`replays/${name}.bin`).then((r) => r.arrayBuffer()));
  const n = meta.size[0] * meta.size[1];
  const pos = new Float32Array(q.length);
  for (let i = 0; i < q.length; i++) {
    const k = i % 3;
    pos[i] = meta.lo[k] + ((q[i] + 32768) / 65535) * meta.span[k];
  }
  frameClip(pos);
  const { engine, renderer, device } = state;
  state.scene = null;
  state.grips = [];
  state.script = null;
  state.drag = null;
  await engine.setCloth(meta.size[0], meta.size[1]);
  renderer.setCloth(engine);
  engine.setState(pos.subarray(0, n * 3), pos.subarray(n * 3, 2 * n * 3));
  if (!state.solverRenderer) state.solverRenderer = new ClothRenderer(device, ui.solverCanvas);
  const src = {
    mesh: engine.mesh,
    buf: {
      pos: device.createBuffer({ size: 3 * n * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }),
      st: device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }),
    },
  };
  device.queue.writeBuffer(src.buf.st, 0, new Uint32Array([0, 0, 0, 0]));
  state.solverRenderer.setCloth(src);
  state.clip = { meta, pos, n, frame: 1, src, errSum: 0, errN: 0, anchor: {}, waiting: null };
  writeSolverFrame(1);
  state.kb = meta.kb;
  state.mu = meta.mu;
  ui.viewport.classList.add('compare');
  document.querySelectorAll('[data-scene]').forEach((b) => b.setAttribute('aria-pressed', 'false'));
  document.querySelectorAll('[data-clip]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.clip === name)));
  ui.hint.textContent = '';
  ui.clipError.hidden = false;
  ui.clipError.textContent = 'Mean distance between the two towels: 0.0 cm';
  ui.verts.textContent = n.toLocaleString();
}

// Aim the camera at everywhere the clip's towel goes: flings travel well outside the default view,
// and each compare pane is only half as wide.
function frameClip(pos) {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < pos.length; i++) {
    const k = i % 3;
    lo[k] = Math.min(lo[k], pos[i]);
    hi[k] = Math.max(hi[k], pos[i]);
  }
  if (!state.camSaved) state.camSaved = { cam: { ...state.cam }, target: state.target };
  state.target = [(lo[0] + hi[0]) / 2, 0.5 * (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
  const r = 0.5 * Math.hypot(hi[0] - lo[0], 0.5 * (hi[1] - lo[1]), hi[2] - lo[2]);
  const aspect = Math.min(1, ui.viewport.clientWidth / 2 / ui.viewport.clientHeight);
  const half = Math.atan(Math.tan(0.31) * aspect);
  state.cam = { ...state.cam, radius: Math.min(3, Math.max(0.8, r / Math.sin(half))) };
}

function writeSolverFrame(f) {
  const c = state.clip;
  const x = c.pos.subarray(f * c.n * 3, (f + 1) * c.n * 3);
  const v4 = new Float32Array(c.n * 4);
  for (let i = 0; i < c.n; i++) v4.set([x[3 * i], x[3 * i + 1], x[3 * i + 2], 1], 4 * i);
  state.device.queue.writeBuffer(c.src.buf.pos, 0, v4);
}

function clipObstacles(t) {
  const m = state.clip.meta;
  const T = m.frames - 1;
  const at = (arr, f) => arr[Math.min(f, T)];
  const vel = (arr) => sub(at(arr, t + 1), at(arr, t)).map((d) => d * 60);
  const cur = {
    spheres: m.spheres.map((s) => ({ c: at(s.c, t), r: s.r, v: vel(s.c) })),
    capsules: m.capsules.map((c) => ({ a: at(c.a, t), b: at(c.b, t), r: c.r, va: vel(c.a), vb: vel(c.b) })),
  };
  const next = {
    spheres: m.spheres.map((s) => ({ c: at(s.c, t + 1), r: s.r })),
    capsules: m.capsules.map((c) => ({ a: at(c.a, t + 1), b: at(c.b, t + 1), r: c.r })),
  };
  return { cur, next };
}

// One network frame of the clip. Each gripper replays the robot's recorded motion: from the moment it
// closes, its vertex moves by the solver's displacement since then, starting from wherever the network's
// towel is. (Snapping it to the solver's absolute position would yank the towel across any gap between
// the two.) Returns false while waiting on the one readback a closing gripper needs.
function stepClip() {
  const c = state.clip;
  const t = c.frame;
  const m = c.meta;
  if (c.waiting) return false;
  if (t >= m.frames - 1) {
    const x = c.pos;
    state.engine.setState(x.subarray(0, c.n * 3), x.subarray(c.n * 3, 2 * c.n * 3));
    c.frame = 1;
    c.errSum = 0;
    c.errN = 0;
    c.anchor = {};
    writeSolverFrame(1);
    return true;
  }
  const solverAt = (f, v) => c.pos.subarray((f * c.n + v) * 3, (f * c.n + v) * 3 + 3);
  const closing = m.grips.filter((g) => g.t0 <= t && t < g.t1 && !c.anchor[g.vertex]);
  if (closing.length) {
    c.waiting = state.engine.readPositions().then((p) => {
      for (const g of closing) {
        const v = g.vertex;
        c.anchor[v] = { t, offset: sub([p[4 * v], p[4 * v + 1], p[4 * v + 2]], [...solverAt(t, v)]) };
      }
      c.waiting = null;
    });
    return false;
  }
  state.grips = [];
  for (const g of m.grips) {
    if (g.t0 <= t && t < g.t1) {
      const target = add([...solverAt(t + 1, g.vertex)], c.anchor[g.vertex].offset);
      state.grips.push({ vertex: g.vertex, target, goal: target });
    }
  }
  state.engine.setGrippers(state.grips.map((g) => ({ vertex: g.vertex, target: g.target })));
  const { cur, next } = clipObstacles(t);
  state.obstacles = cur;
  state.engine.setObstacles(cur, next);
  state.engine.setScene({ kb: m.kb, mu: m.mu, wind: m.wind[Math.min(t, m.wind.length - 1)] });
  state.engine.step();
  c.frame = t + 1;
  writeSolverFrame(c.frame);
  return true;
}

async function measureClipError() {
  const c = state.clip;
  if (!c) return;
  const f = c.frame;
  const p = await state.engine.readPositions();
  if (state.clip !== c || c.frame !== f) return;
  let sum = 0;
  for (let i = 0; i < c.n; i++) {
    const o = (f * c.n + i) * 3;
    sum += Math.hypot(p[4 * i] - c.pos[o], p[4 * i + 1] - c.pos[o + 1], p[4 * i + 2] - c.pos[o + 2]);
  }
  ui.clipError.textContent = `Mean distance between the two towels at ${(f / 60).toFixed(1)} s: ${((sum / c.n) * 100).toFixed(1)} cm`;
}

// ------------------------------------------------------------------ controls

const KB_MIN = Math.log10(3e-7), KB_MAX = Math.log10(3e-5);

function wireControls() {
  document.querySelectorAll('[data-scene]').forEach((b) => b.addEventListener('click', () => loadScene(b.dataset.scene)));
  const stiff = () => {
    state.kb = 10 ** (KB_MIN + (KB_MAX - KB_MIN) * Number(ui.stiff.value) / 100);
    const v = Number(ui.stiff.value);
    ui.stiffOut.textContent = v < 25 ? 'silk' : v < 50 ? 'cotton shirt' : v < 75 ? 'towel' : 'canvas';
    applyMaterial();
  };
  ui.stiff.addEventListener('input', stiff);
  ui.friction.addEventListener('input', () => {
    state.mu = Number(ui.friction.value) / 100;
    ui.frictionOut.textContent = state.mu.toFixed(2);
    applyMaterial();
  });
  ui.wind.addEventListener('input', () => {
    state.wind = Number(ui.wind.value) / 10;
    ui.windOut.textContent = state.wind === 0 ? 'still' : `${state.wind.toFixed(1)} m/s`;
    applyMaterial();
  });
  ui.pause.addEventListener('click', () => setPaused(!state.paused));
  window.addEventListener('keydown', (ev) => {
    if (ev.target.closest('input, select, textarea')) return;
    if (ev.key === ' ') { ev.preventDefault(); setPaused(!state.paused); }
    if (ev.key.toLowerCase() === 'r') loadScene(state.scene);
    const idx = Number(ev.key) - 1;
    const names = Object.keys(SCENES);
    if (idx >= 0 && idx < names.length) loadScene(names[idx]);
  });
  stiff();
}

function setPaused(p) {
  state.paused = p;
  ui.pause.textContent = p ? 'Resume' : 'Pause';
}

// ------------------------------------------------------------------ loop

let refreshing = null;
function refreshPositions() {
  if (!refreshing) {
    refreshing = state.engine.readPositions().then((p) => { state.positions = p; refreshing = null; });
  }
  return refreshing;
}

const timing = { t0: performance.now(), steps: 0, gpu: 0, frames: 0 };

function frame() {
  const { engine, renderer } = state;
  if (!state.paused && state.inflight < 2 && !(state.clip && state.clip.waiting)) {
    const t0 = performance.now();
    if (state.clip) {
      stepClip();
    } else {
      runScript();
      advanceGrips();
      engine.step();
    }
    state.inflight++;
    state.device.queue.onSubmittedWorkDone().then(() => {
      state.inflight--;
      timing.gpu += performance.now() - t0;
      timing.steps++;
    });
    state.frame++;
  }
  renderer.setSolids(solids());
  renderer.render({ eye: cameraEye(), target: state.target, fov: 0.62 });
  if (state.clip) {
    state.solverRenderer.setSolids(solids());
    state.solverRenderer.render({ eye: cameraEye(), target: state.target, fov: 0.62 });
  }
  timing.frames++;
  if (timing.frames % 6 === 0) refreshPositions();
  if (state.clip && timing.frames % 15 === 0) measureClipError();
  if (performance.now() - timing.t0 > 700 && timing.steps) {
    ui.stepMs.textContent = `${(timing.gpu / timing.steps).toFixed(1)} ms`;
    ui.stepMsTable.textContent = `${(timing.gpu / timing.steps).toFixed(1)} ms, measured live`;
    timing.t0 = performance.now();
    timing.gpu = 0;
    timing.steps = 0;
    engine.readBuffer(engine.buf.st, 16).then((b) => { ui.contacts.textContent = new Uint32Array(b)[1].toLocaleString(); });
  }
  requestAnimationFrame(frame);
}

async function main() {
  if (!navigator.gpu) {
    ui.status.textContent = 'This demo needs WebGPU. Try a recent Chrome, Edge or Safari.';
    return;
  }
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('No WebGPU adapter is available on this device.');
    const device = await adapter.requestDevice();
    device.lost.then((info) => { ui.status.textContent = `The GPU device was lost (${info.message || info.reason}). Reload to restart.`; ui.status.hidden = false; });
    state.device = device;
    ui.status.textContent = 'Loading the trained network…';
    const [meta, weights] = await Promise.all([
      fetch('model/cloth.json').then((r) => r.json()),
      fetch('model/cloth.bin').then((r) => r.arrayBuffer()).then((b) => new Float32Array(b)),
    ]);
    state.engine = await ClothEngine.create(device, meta, weights);
    state.renderer = new ClothRenderer(device, ui.canvas);
    wireControls();
    bindPointer();
    await loadClipList();
    await loadScene('table');
    ui.status.hidden = true;
    // Debug hooks for automated tests (the rAF loop does not run in hidden tabs).
    window.nc = {
      state, loadScene, loadClip, SCENES, refreshPositions,
      async advance(n) {
        for (let i = 0; i < n; i++) {
          if (state.clip) {
            while (!stepClip()) await state.clip.waiting;
          } else {
            runScript();
            advanceGrips();
            state.engine.step();
          }
          state.frame++;
          if (i % 6 === 5) await refreshPositions();
        }
        await refreshPositions();
        state.renderer.setSolids(solids());
        state.renderer.render({ eye: cameraEye(), target: state.target, fov: 0.62 });
        if (state.clip) {
          state.solverRenderer.setSolids(solids());
          state.solverRenderer.render({ eye: cameraEye(), target: state.target, fov: 0.62 });
          await measureClipError();
        }
        await state.device.queue.onSubmittedWorkDone();
      },
    };
    requestAnimationFrame(frame);
  } catch (err) {
    console.error(err);
    ui.status.textContent = `The simulator could not start: ${err.message}`;
  }
}

main();
