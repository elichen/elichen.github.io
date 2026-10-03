import { Renderer } from './renderer.js';
import { SCENES, OBJECTS, PLACES, PRESETS, material, frameCamera } from './scenes.js';
import { assemble, packMaterials } from './assemble.js';

const $ = id => document.getElementById(id);
const canvas = $('view');
const NONE = 0xffffffff;
const MAX_SAMPLES = 16384;
const TARGET_MS = 22;            // GPU time per frame while converging
const MOVING_MS = 14;            // and while the camera moves
const STALL_MS = 1000;           // a frame this slow pauses rendering rather than risk starving the display
const MAX_BOUNCES = 12;
const CLAMP = 40;                // firefly clamp on indirect light
const LOOK = [1.15, 1.15];       // AgX look: contrast power, saturation

// ---------------------------------------------------------------- workers and caches

class Pool {
  constructor(n) {
    this.jobs = new Map();
    this.id = 0;
    this.turn = 0;
    this.workers = Array.from({ length: n }, () => {
      const w = new Worker('worker.js', { type: 'module' });
      w.onmessage = e => {
        const job = this.jobs.get(e.data.id);
        this.jobs.delete(e.data.id);
        if (e.data.error) job.reject(new Error(e.data.error)); else job.resolve(e.data.result);
      };
      return w;
    });
  }
  run(msg) {
    return new Promise((resolve, reject) => {
      const id = ++this.id;
      this.jobs.set(id, { resolve, reject });
      this.workers[this.turn++ % this.workers.length].postMessage({ ...msg, id });
    });
  }
}

const pool = new Pool(2);
const geometryCache = new Map();
const envCache = new Map();

function getGeometry(key, g) {
  if (!geometryCache.has(key)) {
    let job;
    if (g.mesh) job = pool.run({ type: 'mesh', url: g.mesh });
    else if (g.triangles) job = pool.run({ type: 'triangles', geometry: g.triangles() });
    else {
      const s = g.spheres();
      job = pool.run({ type: 'spheres', spheres: s.spheres, mat: s.mat }).then(r => ({ ...r, materials: s.materials }));
    }
    geometryCache.set(key, job.catch(err => { geometryCache.delete(key); throw err; }));
  }
  return geometryCache.get(key);
}

function getEnv(url) {
  if (!envCache.has(url)) envCache.set(url, pool.run({ type: 'hdr', url }).catch(err => { envCache.delete(url); throw err; }));
  return envCache.get(url);
}

// ---------------------------------------------------------------- state

const state = {
  scene: null, sceneName: null, place: 'venice', loadToken: 0,
  cam: null, env: null, ground: null,
  samples: 0, spp: 1, movingSpp: 1, seed: 1, dirty: true, redisplay: true,
  lastMove: -1e9, inFlight: false, gpuMs: 0, sampleMs: 0, mrays: 0, measureAt: 0, paused: false,
  selected: NONE, dof: true, xray: false, exposure: 1, homeDistance: 1,
};

let renderer;
let SELECT_COLOR = [0.93, 0.84, 0.13];

function cameraVectors() {
  const c = state.cam;
  const yaw = (c.yaw * Math.PI) / 180, pitch = (c.pitch * Math.PI) / 180;
  const dir = [Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch), Math.cos(yaw) * Math.cos(pitch)];
  const pos = c.target.map((t, k) => t + dir[k] * c.distance);
  const forward = dir.map(x => -x);
  const right = norm([-forward[2], 0, forward[0]]);      // forward × world up
  const up = [
    right[1] * forward[2] - right[2] * forward[1],
    right[2] * forward[0] - right[0] * forward[2],
    right[0] * forward[1] - right[1] * forward[0],
  ];
  const lensRadius = state.dof ? c.aperture / 2 : 0;
  return { pos, forward, right, up, fov: (c.fov * Math.PI) / 180, lensRadius, focusDist: c.focus };
}

function norm(v) { const l = Math.hypot(...v) || 1; return v.map(x => x / l); }

function renderOptions() {
  const env = state.env;
  return {
    maxBounces: MAX_BOUNCES, mode: state.xray ? 1 : 0, clampMax: CLAMP,
    envIntensity: env ? env.intensity : 0, envRotation: env ? env.rotation : 0,
    envVisible: env ? env.visible !== false : false, bgColor: (env && env.background) || [0, 0, 0],
    ground: state.ground,
  };
}

function reset() { state.dirty = true; }

// ---------------------------------------------------------------- loading

async function loadScene(name, place = state.place) {
  const def = SCENES[name];
  const envDef = def.places ? { ...PLACES[place], visible: true } : def.env;
  const token = ++state.loadToken;
  const sameScene = name === state.sceneName;
  state.place = place;
  document.querySelectorAll('[data-scene]').forEach(b => b.setAttribute('aria-pressed', b.dataset.scene === name));
  document.querySelectorAll('[data-place]').forEach(b => b.setAttribute('aria-pressed', b.dataset.place === place));
  $('place-group').hidden = !def.places;
  setStatus(sameScene ? `Loading ${PLACES[place].label}…` : `Loading ${def.label} and building its BVH…`);
  try {
    const geoms = {};
    const pending = Object.entries(def.geometry).map(async ([key, g]) => { geoms[key] = await getGeometry(key, g); });
    const [env] = await Promise.all([envDef ? getEnv(envDef.url) : null, ...pending]);
    if (token !== state.loadToken) return;
    if (!sameScene) {
      const built = assemble(def, geoms);
      renderer.setScene(built.gpu);
      state.scene = built;
      state.sceneName = name;
      state.cam = { ...def.camera, target: [...def.camera.target] };
      if (def.object) {         // frame the object, whatever its shape
        const i = built.gpu.instances;
        state.cam = frameCamera(def.camera, [i[12], i[13], i[14], i[16], i[17], i[18]], canvas.clientWidth / Math.max(1, canvas.clientHeight));
      }
      state.homeDistance = state.cam.distance;
      state.selected = NONE;
      $('scene-note').textContent = def.note || '';
      showSelection();
    }
    renderer.setEnvironment(env);
    state.env = envDef ? { ...envDef } : null;
    state.ground = def.places ? [0, PLACES[place].height, 0] : null;
    state.exposure = Math.pow(2, def.exposure || 0);
    reset();
    setStatus('');
  } catch (err) {
    console.error(err);
    if (token === state.loadToken) setStatus(`Could not load ${def.label}: ${err.message}`);
  }
}

function setStatus(text) {
  const s = $('status');
  s.textContent = text;
  s.hidden = !text;
}

// ---------------------------------------------------------------- rendering loop

// One sample per CSS pixel, so phones and high-density screens aren't asked for 4x the work.
function resizeCanvas() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(1, Math.round(rect.width * dpr)), h = Math.max(1, Math.round(rect.height * dpr));
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; state.redisplay = true; }
  const rw = Math.max(1, Math.round(w / dpr)), rh = Math.max(1, Math.round(h / dpr));
  if (rw !== renderer.fullWidth || rh !== renderer.fullHeight) { renderer.resize(rw, rh); reset(); }
}

function frame() {
  requestAnimationFrame(frame);
  tick();
}

// Render one frame if the GPU is free; returns a promise that settles when it finishes.
function tick() {
  if (!state.scene || state.inFlight || state.paused) return null;
  resizeCanvas();
  const now = performance.now();
  const moving = now - state.lastMove < 120;
  const preview = moving && state.sampleMs > 12;   // slow scenes drop to half resolution in motion
  if (preview !== renderer.preview) { renderer.setPreview(preview); state.dirty = true; }
  if (state.dirty) { state.samples = 0; state.dirty = false; }
  const converged = state.samples >= MAX_SAMPLES;
  if (converged && !state.redisplay) return null;
  state.redisplay = false;

  let spp = 0;
  if (!converged) spp = Math.max(1, Math.min(32, moving ? state.movingSpp : state.spp, MAX_SAMPLES - state.samples));
  const measure = spp > 0 && now - state.measureAt > 400;
  if (measure) state.measureAt = now;
  const opts = {
    ...renderOptions(), spp, seed: state.seed++, accumulate: state.samples > 0,
    samples: state.samples + spp, exposure: state.exposure, selected: state.selected,
    accent: SELECT_COLOR, measure, look: LOOK,
  };
  state.inFlight = true;
  const { done, rays } = renderer.render(cameraVectors(), opts);
  state.samples += spp;
  return done.then(ms => {
    state.inFlight = false;
    if (ms > STALL_MS) {
      state.paused = true;
      setStatus(`Paused: one frame took ${(ms / 1000).toFixed(1)} s on this GPU.`);
      $('resume').hidden = false;
    }
    if (!spp) return;
    state.gpuMs = state.gpuMs ? state.gpuMs * 0.8 + ms * 0.2 : ms;
    if (!preview) state.sampleMs = ms / spp;
    // several samples per pixel even while moving, so a single unlucky path doesn't show as a dark speck
    if (moving) state.movingSpp = Math.max(1, Math.min(8, Math.round(spp * MOVING_MS / Math.max(ms, 1))));
    else state.spp = Math.max(1, Math.min(32, Math.round(spp * TARGET_MS / Math.max(ms, 1))));
    if (rays) rays.then(n => { state.mrays = n / ms / 1000; });
    updateStats();
  });
}

function updateStats() {
  $('hud-spp').textContent = state.samples.toLocaleString();
  $('hud-mrays').textContent = state.mrays ? state.mrays.toFixed(0) : '–';
}

// ---------------------------------------------------------------- camera input

const pointers = new Map();
let drag = null;

canvas.addEventListener('pointerdown', e => {
  canvas.setPointerCapture(e.pointerId);
  pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
  drag = { x: e.clientX, y: e.clientY, t: performance.now(), moved: 0, pan: e.button === 2 || e.shiftKey, pinch: null };
});

canvas.addEventListener('pointermove', e => {
  if (!pointers.has(e.pointerId) || !state.cam) return;
  const prev = pointers.get(e.pointerId);
  const dx = e.clientX - prev.x, dy = e.clientY - prev.y;
  pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
  drag.moved += Math.abs(dx) + Math.abs(dy);
  const c = state.cam;
  if (pointers.size === 2) {
    const [a, b] = [...pointers.values()];
    const d = Math.hypot(a.x - b.x, a.y - b.y);
    if (drag.pinch) c.distance = clampDistance(c.distance * drag.pinch / d);
    drag.pinch = d;
  } else if (drag.pan) {
    const v = cameraVectors();
    const k = (c.distance * Math.tan(v.fov / 2) * 2) / canvas.clientHeight;
    c.target = c.target.map((t, i) => t - v.right[i] * dx * k + v.up[i] * dy * k);
    if (SCENES[state.sceneName].places) c.target[1] = Math.max(0.05, c.target[1]);
  } else {
    c.yaw -= dx * 0.3;
    const lowest = SCENES[state.sceneName].places ? 1 : -5;   // stay above the photo's ground
    c.pitch = Math.max(lowest, Math.min(85, c.pitch + dy * 0.25));
  }
  moved();
});

const endPointer = e => {
  if (!pointers.has(e.pointerId)) return;
  pointers.delete(e.pointerId);
  if (drag && pointers.size === 0) {
    if (drag.moved < 5 && performance.now() - drag.t < 400 && e.type === 'pointerup') clickAt(e);
    drag = null;
  }
};
canvas.addEventListener('pointerup', endPointer);
canvas.addEventListener('pointercancel', endPointer);
canvas.addEventListener('contextmenu', e => e.preventDefault());

canvas.addEventListener('wheel', e => {
  e.preventDefault();
  if (!state.cam) return;
  state.cam.distance = clampDistance(state.cam.distance * Math.exp(e.deltaY * 0.0012));
  moved();
}, { passive: false });

function clampDistance(d) {
  return Math.max(state.homeDistance * 0.25, Math.min(state.homeDistance * 4, d));
}

function moved() {
  state.lastMove = performance.now();
  document.querySelector('.help').classList.add('faded');
  reset();
}

function canvasPoint(e) {
  const r = canvas.getBoundingClientRect();
  return [((e.clientX - r.left) / r.width) * renderer.width, ((e.clientY - r.top) / r.height) * renderer.height];
}

let lastClick = 0;
async function clickAt(e) {
  const now = performance.now();
  const double = now - lastClick < 350;
  lastClick = now;
  const [x, y] = canvasPoint(e);
  const hit = await renderer.pick(cameraVectors(), x, y, renderOptions());
  if (double) {
    if (hit) { state.cam.focus = hit.depth; state.dof = true; $('dof').checked = true; reset(); }
    return;
  }
  const pickable = hit && state.scene.owners[hit.material].pickable;
  state.selected = pickable ? hit.material : NONE;
  state.redisplay = true;
  showSelection();
}

// ---------------------------------------------------------------- panel

function linearToHex(c) {
  return '#' + c.map(v => {
    const s = v <= 0.0031308 ? 12.92 * v : 1.055 * Math.pow(v, 1 / 2.4) - 0.055;
    return Math.round(Math.max(0, Math.min(1, s)) * 255).toString(16).padStart(2, '0');
  }).join('');
}

function showSelection() {
  const m = state.selected !== NONE && state.scene ? state.scene.materials[state.selected] : null;
  $('material').hidden = !m;
  $('pick-hint').hidden = !!m;
  if (!m) return;
  $('mat-name').textContent = state.scene.owners[state.selected].name;
  $('presets').querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed', b.dataset.label === m.label));
}

function applyPreset(p) {
  const m = state.scene?.materials[state.selected];
  if (!m) return;
  const fresh = material(p);
  for (const k of Object.keys(fresh)) if (k !== 'emission') m[k] = fresh[k];
  renderer.updateMaterials(packMaterials(state.scene.materials));
  showSelection();
  reset();
}

function buildPresetButtons() {
  for (const p of Object.values(PRESETS)) {
    const b = document.createElement('button');
    b.type = 'button';
    b.dataset.label = p.label;
    const dot = document.createElement('i');
    if (p.transmission && !p.scatter && !p.density) dot.className = 'clear';
    else dot.style.background = linearToHex(p.color);
    b.append(dot, p.label);
    b.addEventListener('click', () => applyPreset(p));
    $('presets').appendChild(b);
  }
}

function bindUI() {
  for (const [key, o] of Object.entries(OBJECTS)) {
    const b = document.createElement('button');
    b.type = 'button';
    b.dataset.scene = key;
    b.textContent = o.label;
    $('objects').appendChild(b);
  }
  document.querySelectorAll('[data-scene]').forEach(b => b.addEventListener('click', () => loadScene(b.dataset.scene)));
  document.querySelectorAll('[data-place]').forEach(b => b.addEventListener('click', () => loadScene(state.sceneName, b.dataset.place)));
  buildPresetButtons();
  $('dof').addEventListener('change', e => { state.dof = e.target.checked; reset(); });
  $('xray').addEventListener('change', e => { state.xray = e.target.checked; reset(); });
  $('resume').addEventListener('click', () => { state.paused = false; $('resume').hidden = true; setStatus(''); reset(); });
  window.addEventListener('keydown', e => {
    if (e.key === 'Escape') { state.selected = NONE; state.redisplay = true; showSelection(); }
  });
}

// the selection outline uses the ColorChecker's yellow patch from the stylesheet
function cssColor(name) {
  const hex = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16) / 255);
}

// ---------------------------------------------------------------- start

async function main() {
  try {
    renderer = await Renderer.create(canvas);
  } catch (err) {
    setStatus(`${err.message} Try a recent Chrome, Edge or Safari.`);
    return;
  }
  renderer.compileInfo.then(infos => infos.forEach(info => info.messages.forEach(m => console[m.type === 'error' ? 'error' : 'log'](`[wgsl] ${m.lineNum}:${m.linePos} ${m.message}`))));
  renderer.device.lost.then(info => setStatus(`The GPU device was lost (${info.message}). Reload the page.`));
  bindUI();
  SELECT_COLOR = cssColor('--yellow-patch');
  // for the console: pathtracer.run(100) renders 100 frames even when the tab is in the background
  window.pathtracer = {
    state, loadScene, get renderer() { return renderer; },
    async run(n) { for (let i = 0; i < n; i++) await (tick() || new Promise(r => setTimeout(r, 0))); return state.samples; },
  };
  requestAnimationFrame(frame);
  await loadScene('dragon');
}

main();
