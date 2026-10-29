// BAAIWorm, live: the page. worker.js runs the brain and body (engine.wasm);
// this file draws the body, the nervous system inside it, and the panels.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const $ = (id) => document.getElementById(id);

// Palettes sampled from photographs of C. elegans on Wikimedia Commons
const PALETTES = {
  // A: Bob Goldstein's DIC micrograph (UNC Chapel Hill, CC BY-SA 3.0)
  A: { name: 'DIC', agar: '#8ea597', agar2: '#7a9387', tint: '#d4e4d8', lawn: '#eef2ea', trail: '#263832', bg: '#66847a', bg2: '#97b0a3', body: '#dfe8e0', rim: '#f0f5ef', shade: '#263832', rest: '#3e564e', active: '#f0f5ef', muscle: '#263832', ui: '#1f2f2a', uiDim: '#3e564e', panel: 'rgba(151,176,163,0.82)', accent: '#263832', additive: false },
  // B: confocal image of a PVM neuron (Guasp, CC BY 4.0): mCherry, GFP, DAPI
  B: { name: 'Confocal', agar: '#14060f', agar2: '#0b0308', tint: '#3d0b66', lawn: '#6b7a2e', trail: '#9a8f96', bg: '#0a0002', bg2: '#140210', body: '#0c023a', rim: '#2603dc', shade: '#0a0002', rest: '#71060a', active: '#a3ff46', muscle: '#fa0407', ui: '#f7f9f7', uiDim: '#9a8f96', panel: 'rgba(20,2,16,0.78)', accent: '#67f514', additive: true },
  // C: DAPI-stained worms (HoPo, CC BY-SA 3.0)
  C: { name: 'DAPI', agar: '#080b18', agar2: '#04050a', tint: '#173a80', lawn: '#46598a', trail: '#7c8db0', bg: '#04050a', bg2: '#050712', body: '#07113b', rim: '#406dcc', shade: '#030207', rest: '#12318a', active: '#a6d4ff', muscle: '#64aaed', ui: '#dfe8f7', uiDim: '#7c8db0', panel: 'rgba(5,7,18,0.8)', accent: '#64aaed', additive: true },
  // D: worms on an agar plate through a microscope (Gannu03, CC BY-SA 4.0)
  D: { name: 'Agar plate', agar: '#cfccbc', agar2: '#bab7a6', tint: '#ead9a0', lawn: '#f1e8c9', trail: '#4a4945', bg: '#c4c2b4', bg2: '#dfdfd2', body: '#a3a092', rim: '#4a4945', shade: '#030305', rest: '#757268', active: '#030305', muscle: '#030305', ui: '#0f0f11', uiDim: '#4a4945', panel: 'rgba(223,223,210,0.85)', accent: '#030305', additive: false },
};
let pal = PALETTES.D;

// ---------- data shared with the worker ----------
async function f64(url) {
  return new Float64Array(await (await fetch(url)).arrayBuffer());
}
async function f32(url) {
  return new Float32Array(await (await fetch(url)).arrayBuffer());
}

const [bodyData, neuronData, muscleEnds, brainMeta] = await Promise.all([
  f64('model/body.bin'), f32('model/neurons.bin'), f64('model/body_muscles.bin'), fetch('model/brain.json').then((r) => r.json()),
]);
const nv = bodyData[0], nt = bodyData[1], nm = bodyData[2], ntri = bodyData[3];
const rest = bodyData.subarray(5, 5 + nv * 3);
const triOffset = 5 + nv * 3 + nt * 14 + nm * 19;
const tris = new Uint32Array(ntri * 3);
for (let i = 0; i < ntri * 3; i++) tris[i] = bodyData[triOffset + i];
const headVertex = bodyData[triOffset + ntri * 3 + bodyData[4] + 2];
const bodyLength = 0.6276;

// ---------- three.js ----------
const canvas = $('view');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
canvas.addEventListener('webglcontextlost', () => {
  $('loading-text').textContent = 'The browser reset the graphics. Reload the page to continue.';
  $('loading').classList.remove('done');
});
renderer.setPixelRatio(Math.min(2, devicePixelRatio));
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(38, 1, 0.005, 400);
camera.position.set(0.25, 0.42, -0.3);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.minDistance = 0.08;
controls.maxDistance = 90;

// body surface
const bodyGeo = new THREE.BufferGeometry();
const bodyPos = new Float32Array(rest);
bodyGeo.setAttribute('position', new THREE.BufferAttribute(bodyPos, 3));
bodyGeo.setIndex(new THREE.BufferAttribute(tris, 1));
// each surface vertex takes the activation of its nearest muscle segment
const vertMuscle = new Float32Array(nv);
const vertMuscleW = new Float32Array(nv);
for (let i = 0; i < nv; i++) {
  let best = 1e9, bj = 0;
  const p = [rest[i * 3], rest[i * 3 + 1], rest[i * 3 + 2]];
  for (let j = 0; j < 96; j++) {
    const a = muscleEnds.subarray(j * 6, j * 6 + 3), b = muscleEnds.subarray(j * 6 + 3, j * 6 + 6);
    const ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    const t = Math.max(0, Math.min(1, ((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1] + (p[2] - a[2]) * ab[2]) / (ab[0] ** 2 + ab[1] ** 2 + ab[2] ** 2)));
    const d = Math.hypot(p[0] - a[0] - t * ab[0], p[1] - a[1] - t * ab[1], p[2] - a[2] - t * ab[2]);
    if (d < best) (best = d), (bj = j);
  }
  vertMuscle[i] = bj;
  vertMuscleW[i] = Math.exp(-best / 0.006);
}
const vertAct = new Float32Array(nv);
bodyGeo.setAttribute('act', new THREE.BufferAttribute(vertAct, 1));
const bodyMat = new THREE.ShaderMaterial({
  uniforms: { body: { value: new THREE.Color() }, rim: { value: new THREE.Color() }, muscle: { value: new THREE.Color() }, alpha: { value: 0.25 } },
  vertexShader: `
    attribute float act;
    varying vec3 vN; varying vec3 vV; varying float vAct;
    void main() {
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      vN = normalize(normalMatrix * normal); vV = normalize(-mv.xyz); vAct = act;
      gl_Position = projectionMatrix * mv;
    }`,
  fragmentShader: `
    uniform vec3 body; uniform vec3 rim; uniform vec3 muscle; uniform float alpha;
    varying vec3 vN; varying vec3 vV; varying float vAct;
    void main() {
      float f = 1.0 - abs(dot(normalize(vN), normalize(vV)));
      float r = pow(f, 2.2);
      vec3 c = mix(body, rim, r);
      c = mix(c, muscle, clamp(vAct, 0.0, 1.0) * 0.55);
      gl_FragColor = vec4(c, alpha + 0.55 * r + 0.25 * vAct);
    }`,
  transparent: true,
  depthWrite: false,
  side: THREE.DoubleSide,
});
const bodyMesh = new THREE.Mesh(bodyGeo, bodyMat);
bodyMesh.frustumCulled = false;
bodyMesh.renderOrder = 1;
scene.add(bodyMesh);

// nervous system: a line from every compartment to its parent
const nNodes = neuronData.length / 10;
const segs = [];
for (let i = 0; i < nNodes; i++) if (neuronData[i * 10 + 8] >= 0) segs.push(i);
const nervePos = new Float32Array(segs.length * 6);
const nerveCol = new Float32Array(segs.length * 6);
const nerveGeo = new THREE.BufferGeometry();
nerveGeo.setAttribute('position', new THREE.BufferAttribute(nervePos, 3));
nerveGeo.setAttribute('color', new THREE.BufferAttribute(nerveCol, 3));
const nerveMat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, depthWrite: false });
const nerves = new THREE.LineSegments(nerveGeo, nerveMat);
nerves.frustumCulled = false;
nerves.renderOrder = 2;
scene.add(nerves);

// somas as round dots
const somaNode = brainMeta.soma;
const somaPos = new Float32Array(136 * 3);
const somaCol = new Float32Array(136 * 3);
const somaGeo = new THREE.BufferGeometry();
somaGeo.setAttribute('position', new THREE.BufferAttribute(somaPos, 3));
somaGeo.setAttribute('color', new THREE.BufferAttribute(somaCol, 3));
const dot = (() => {
  const c = document.createElement('canvas');
  c.width = c.height = 64;
  const g = c.getContext('2d');
  const grd = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  grd.addColorStop(0, 'rgba(255,255,255,1)');
  grd.addColorStop(0.45, 'rgba(255,255,255,0.8)');
  grd.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = grd;
  g.fillRect(0, 0, 64, 64);
  return new THREE.CanvasTexture(c);
})();
const somaMat = new THREE.PointsMaterial({ size: 0.006, map: dot, vertexColors: true, transparent: true, depthWrite: false });
const somas = new THREE.Points(somaGeo, somaMat);
somas.frustumCulled = false;
somas.renderOrder = 3;
scene.add(somas);

// ---------- the agar plate ----------
// 9 cm Petri dish (0.63 units = 1 mm); the worm lies on its left side on the agar
const MM = 0.63;
const PLATE_R = 45 * MM;
const PLATE_Y = -0.0185;
const SIGMA = 6.3; // the attractant hill (behavior.rs Params::sigma)
const LAWN_R = 1.0 * MM;
const plateMat = new THREE.ShaderMaterial({
  uniforms: { agar: { value: new THREE.Color() }, agar2: { value: new THREE.Color() }, tint: { value: new THREE.Color() }, food: { value: new THREE.Vector2() }, sigma: { value: SIGMA }, radius: { value: PLATE_R } },
  vertexShader: `
    varying vec2 vP;
    void main() { vP = position.xz; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
  fragmentShader: `
    uniform vec3 agar; uniform vec3 agar2; uniform vec3 tint; uniform vec2 food; uniform float sigma; uniform float radius;
    varying vec2 vP;
    float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
    float noise(vec2 p) {
      vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
      return mix(mix(hash(i), hash(i + vec2(1, 0)), f.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), f.x), f.y);
    }
    void main() {
      float r = length(vP);
      vec3 c = mix(agar, agar2, smoothstep(0.0, radius, r) * 0.6);
      // fine grain in the agar, so the crawl reads up close
      float g = noise(vP * 60.0) * 0.6 + noise(vP * 240.0) * 0.4;
      c *= 0.94 + 0.1 * g;
      // the attractant, faintly
      float d = length(vP - food);
      c = mix(c, tint, 0.28 * exp(-d * d / (2.0 * sigma * sigma)));
      gl_FragColor = vec4(c, 1.0);
    }`,
});
const plate = new THREE.Mesh(new THREE.CircleGeometry(PLATE_R, 128).rotateX(-Math.PI / 2), plateMat);
plate.position.y = PLATE_Y;
scene.add(plate);
const rimMat = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.5, depthTest: false });
const rim = new THREE.Mesh(new THREE.RingGeometry(PLATE_R, PLATE_R + 0.6 * MM, 128).rotateX(-Math.PI / 2), rimMat);
rim.position.y = PLATE_Y + 0.001;
rim.renderOrder = 0.5;
scene.add(rim);

// the bacterial lawn: a thin cream disc, denser at its edge as real lawns are
const lawnMat = new THREE.ShaderMaterial({
  uniforms: { lawn: { value: new THREE.Color() } },
  depthTest: false,
  vertexShader: `varying vec2 vU; void main() { vU = position.xz; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
  fragmentShader: `
    uniform vec3 lawn; varying vec2 vU;
    void main() {
      float r = length(vU) / ${LAWN_R.toFixed(4)};
      float a = (0.45 + 0.4 * smoothstep(0.75, 0.97, r)) * (1.0 - smoothstep(0.97, 1.0, r));
      gl_FragColor = vec4(lawn, a);
    }`,
  transparent: true,
  depthWrite: false,
});
const lawn = new THREE.Mesh(new THREE.CircleGeometry(LAWN_R, 64).rotateX(-Math.PI / 2), lawnMat);
lawn.position.y = PLATE_Y + 0.002;
lawn.renderOrder = 0.5;
scene.add(lawn);
const food = new THREE.Vector2(3.0, -4.0); // worker.js FOOD
function placeFood(x, z) {
  food.set(x, z);
  lawn.position.x = x;
  lawn.position.z = z;
  plateMat.uniforms.food.value.copy(food);
}
placeFood(food.x, food.y);

// the head's track on the agar, colored by what the worm was doing
const TRACK = 12000; // control steps (20 min)
const trackPos = new Float32Array(TRACK * 3);
const trackCol = new Float32Array(TRACK * 3);
const trackGeo = new THREE.BufferGeometry();
trackGeo.setAttribute('position', new THREE.BufferAttribute(trackPos, 3));
trackGeo.setAttribute('color', new THREE.BufferAttribute(trackCol, 3));
trackGeo.setDrawRange(0, 0);
const trackMat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.75, depthWrite: false, depthTest: false });
const track = new THREE.Line(trackGeo, trackMat);
track.frustumCulled = false;
track.renderOrder = 0.5;
scene.add(track);
let trackN = 0;
const modeColor = [new THREE.Color(), new THREE.Color('#c0392b'), new THREE.Color('#2e6fb7')];
function addTrack(x, z, mode) {
  if (trackN === TRACK) {
    trackPos.copyWithin(0, 3);
    trackCol.copyWithin(0, 3);
    trackN--;
  }
  const o = trackN * 3, c = modeColor[mode] || modeColor[0];
  trackPos[o] = x; trackPos[o + 1] = PLATE_Y + 0.003; trackPos[o + 2] = z;
  trackCol[o] = c.r; trackCol[o + 1] = c.g; trackCol[o + 2] = c.b;
  trackN++;
  trackGeo.setDrawRange(0, trackN);
  trackGeo.attributes.position.needsUpdate = true;
  trackGeo.attributes.color.needsUpdate = true;
}

// a ring that marks the worm when the camera is far away
const markerMat = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, depthWrite: false, depthTest: false });
const marker = new THREE.Mesh(new THREE.RingGeometry(0.85, 1, 48).rotateX(-Math.PI / 2), markerMat);
marker.renderOrder = 0.5;
scene.add(marker);

const cRest = new THREE.Color(), cActive = new THREE.Color(), cTmp = new THREE.Color();
function applyPalette(key) {
  pal = PALETTES[key] || pal;
  scene.background = new THREE.Color(pal.bg);
  bodyMat.uniforms.body.value.set(pal.body);
  bodyMat.uniforms.rim.value.set(pal.rim);
  bodyMat.uniforms.muscle.value.set(pal.muscle);
  bodyMat.uniforms.alpha.value = pal.additive ? 0.18 : 0.06;
  const blend = pal.additive ? THREE.AdditiveBlending : THREE.NormalBlending;
  nerveMat.blending = blend;
  somaMat.blending = blend;
  nerveMat.needsUpdate = somaMat.needsUpdate = true;
  plateMat.uniforms.agar.value.set(pal.agar);
  plateMat.uniforms.agar2.value.set(pal.agar2);
  plateMat.uniforms.tint.value.set(pal.tint);
  lawnMat.uniforms.lawn.value.set(pal.lawn);
  rimMat.color.set(pal.uiDim);
  markerMat.color.set(pal.accent);
  modeColor[0].set(pal.trail);
  cRest.set(pal.rest);
  cActive.set(pal.active);
  const root = document.documentElement.style;
  root.setProperty('--bg', pal.bg);
  root.setProperty('--ui', pal.ui);
  root.setProperty('--dim', pal.uiDim);
  root.setProperty('--panel', pal.panel);
  root.setProperty('--accent', pal.accent);
}

// ---------- panels ----------
const muscleCanvas = $('muscles');
const rasterCanvas = $('raster');
const inputCanvas = $('input');
const HISTORY = 300; // control steps shown (30 s)
const motorHist = new Float32Array(80 * HISTORY).fill(-65);
const smellHist = new Float32Array(HISTORY).fill(NaN); // mM at the nose
const modeHist = new Uint8Array(HISTORY);
let histHead = 0;

// voltage → 0..1
const level = (v) => Math.max(0, Math.min(1, (v + 60) / 40));
// on dark palettes the two labels add like fluorescence (dim red at rest, green
// when depolarized) instead of blending through yellow
function ramp(t, out) {
  if (pal.additive) {
    const a = Math.pow(1 - t, 1.2) * 0.8, b = Math.pow(t, 2.0);
    out.setRGB(cRest.r * a + cActive.r * b, cRest.g * a + cActive.g * b, cRest.b * a + cActive.b * b);
  } else out.copy(cRest).lerp(cActive, t);
  return out;
}

function drawMuscles(m) {
  const g = muscleCanvas.getContext('2d');
  const W = muscleCanvas.width, H = muscleCanvas.height;
  g.clearRect(0, 0, W, H);
  // rows: dorsal left, dorsal right, ventral right, ventral left (DR VR DL VL in the data)
  const rows = [[2, 'DL'], [0, 'DR'], [1, 'VR'], [3, 'VL']];
  const cw = (W - 30) / 24, ch = (H - 4) / 4;
  g.font = `${11 * devicePixelRatio}px IBM Plex Mono, monospace`;
  g.textBaseline = 'middle';
  rows.forEach(([q, label], r) => {
    g.fillStyle = pal.uiDim;
    g.fillText(label, 0, r * ch + ch / 2 + 2);
    for (let s = 0; s < 24; s++) {
      g.fillStyle = '#' + ramp(Math.min(1, m[q * 24 + s] / 0.8), cTmp).getHexString();
      g.fillRect(30 + s * cw + 1, r * ch + 2, cw - 2, ch - 3);
    }
  });
}

function drawRaster() {
  const g = rasterCanvas.getContext('2d');
  const W = rasterCanvas.width, H = rasterCanvas.height;
  const img = g.createImageData(HISTORY, 80);
  for (let r = 0; r < 80; r++) {
    for (let k = 0; k < HISTORY; k++) {
      const idx = (histHead + k) % HISTORY;
      ramp(level(motorHist[r * HISTORY + idx]), cTmp);
      const o = (r * HISTORY + k) * 4;
      img.data[o] = cTmp.r * 255; img.data[o + 1] = cTmp.g * 255; img.data[o + 2] = cTmp.b * 255; img.data[o + 3] = 255;
    }
  }
  const off = drawRaster.off || (drawRaster.off = new OffscreenCanvas(HISTORY, 80));
  off.getContext('2d').putImageData(img, 0, 0);
  g.imageSmoothingEnabled = false;
  g.clearRect(0, 0, W, H);
  const lw = 44 * devicePixelRatio;
  g.drawImage(off, lw, 0, W - lw, H);
  // group labels (rows follow the readout's motor neuron order)
  const groups = [['head', 0, 22], ['DA', 22, 31], ['DB', 31, 38], ['DD', 38, 44], ['VA', 44, 56], ['VB', 56, 67], ['VD', 67, 80]];
  g.font = `${10 * devicePixelRatio}px IBM Plex Mono, monospace`;
  g.textBaseline = 'middle';
  for (const [name, a, b] of groups) {
    g.fillStyle = pal.uiDim;
    g.fillText(name, 2, ((a + b) / 2 / 80) * H);
    g.fillRect(lw - 6 * devicePixelRatio, (a / 80) * H + 1, 2, ((b - a) / 80) * H - 2);
  }
}

// the attractant at the nose over the last 30 s, with reversals and omega turns shaded
function drawInput() {
  const g = inputCanvas.getContext('2d');
  const W = inputCanvas.width, H = inputCanvas.height;
  g.clearRect(0, 0, W, H);
  let lo = Infinity, hi = -Infinity;
  for (const v of smellHist) if (v === v) (lo = Math.min(lo, v)), (hi = Math.max(hi, v));
  if (!(hi >= lo)) return;
  const pad = Math.max(0.002, (hi - lo) * 0.15);
  (lo -= pad), (hi += pad);
  const bw = W / HISTORY;
  for (let k = 0; k < HISTORY; k++) {
    const md = modeHist[(histHead + k) % HISTORY];
    if (md) {
      g.fillStyle = '#' + modeColor[md].getHexString() + '55';
      g.fillRect(k * bw, 0, bw + 1, H);
    }
  }
  g.strokeStyle = pal.accent;
  g.lineWidth = 1.5 * devicePixelRatio;
  g.beginPath();
  let started = false;
  for (let k = 0; k < HISTORY; k++) {
    const v = smellHist[(histHead + k) % HISTORY];
    if (v !== v) continue;
    const x = (k / (HISTORY - 1)) * W, y = H - ((v - lo) / (hi - lo)) * H;
    started ? g.lineTo(x, y) : g.moveTo(x, y);
    started = true;
  }
  g.stroke();
  $('smell-lo').textContent = `${lo.toFixed(3)} mM`;
  $('smell-hi').textContent = `${hi.toFixed(3)} mM`;
}

// ---------- state from the worker ----------
let latest = null;
let centroid = new THREE.Vector3();
let followed = null;
const speedTrack = [];
let epoch = 0; // bumped by Restart; states from an earlier run are dropped

const worker = new Worker('worker.js');
worker.onmessage = (e) => {
  const m = e.data;
  if (m.type === 'progress') $('loading-text').textContent = m.text + '…';
  else if (m.type === 'error') $('loading-text').textContent = 'Could not start: ' + m.text;
  else if (m.type === 'ready') $('loading').classList.add('done');
  else if (m.type === 'state' && m.epoch === epoch) {
    latest = m;
    // history for the panels: the newest step stands for all steps of this batch
    const mode = m.seek[0];
    for (let s = 0; s < Math.max(1, m.steps); s++) {
      for (let r = 0; r < 80; r++) motorHist[r * HISTORY + histHead] = m.motor[r];
      smellHist[histHead] = m.seek[1];
      modeHist[histHead] = mode;
      histHead = (histHead + 1) % HISTORY;
      addTrack(m.seek[6], m.seek[7], mode);
    }
    let cx = 0, cy = 0, cz = 0;
    for (let i = 0; i < nv; i++) (cx += m.x[i * 3]), (cy += m.x[i * 3 + 1]), (cz += m.x[i * 3 + 2]);
    centroid.set(cx / nv, cy / nv, cz / nv);
    speedTrack.push([m.time, centroid.clone()]);
    while (speedTrack.length > 2 && m.time - speedTrack[0][0] > 3) speedTrack.shift();
    // sim seconds per wall second over the last 3 s
    rateTrack.push([performance.now() / 1000, m.time]);
    while (rateTrack.length > 2 && rateTrack[rateTrack.length - 1][0] - rateTrack[0][0] > 3) rateTrack.shift();
    const [w0, s0] = rateTrack[0], [w1, s1] = rateTrack[rateTrack.length - 1];
    if (w1 - w0 > 1 && s1 > s0) rate = (s1 - s0) / (w1 - w0);
    // ?report sends the speed to a local test server (a tab under DevTools runs wasm unoptimized)
    if (location.search.includes('report') && Math.floor(m.time) % 5 === 0 && Math.floor(m.time) !== reported) {
      reported = Math.floor(m.time);
      fetch(`/report?t=${m.time.toFixed(1)}&cap=${m.capacity.toFixed(3)}&rate=${m.rate.toFixed(3)}&queue=${m.queue}`);
    }
  }
};
let rate = 1;
const rateTrack = [];
let reported = -1;

function updateGeometry(m) {
  bodyPos.set(m.x);
  bodyGeo.attributes.position.needsUpdate = true;
  bodyGeo.computeVertexNormals();
  for (let i = 0; i < nv; i++) vertAct[i] = (m.muscle[vertMuscle[i]] / 0.8) * vertMuscleW[i];
  bodyGeo.attributes.act.needsUpdate = true;
  const x = m.x, nd = neuronData, v = m.nodes;
  const place = (i, out, o) => {
    const b = i * 10;
    let px = 0, py = 0, pz = 0;
    for (let k = 0; k < 4; k++) {
      const vi = nd[b + k] * 3, wk = nd[b + 4 + k];
      px += wk * x[vi]; py += wk * x[vi + 1]; pz += wk * x[vi + 2];
    }
    out[o] = px; out[o + 1] = py; out[o + 2] = pz;
  };
  for (let s = 0; s < segs.length; s++) {
    const i = segs[s], p = nd[i * 10 + 8];
    place(i, nervePos, s * 6);
    place(p, nervePos, s * 6 + 3);
    ramp(level(v[i]), cTmp);
    nerveCol[s * 6] = cTmp.r; nerveCol[s * 6 + 1] = cTmp.g; nerveCol[s * 6 + 2] = cTmp.b;
    ramp(level(v[p]), cTmp);
    nerveCol[s * 6 + 3] = cTmp.r; nerveCol[s * 6 + 4] = cTmp.g; nerveCol[s * 6 + 5] = cTmp.b;
  }
  nerveGeo.attributes.position.needsUpdate = true;
  nerveGeo.attributes.color.needsUpdate = true;
  for (let c = 0; c < 136; c++) {
    place(somaNode[c], somaPos, c * 3);
    ramp(level(m.soma[c]), cTmp);
    somaCol[c * 3] = cTmp.r; somaCol[c * 3 + 1] = cTmp.g; somaCol[c * 3 + 2] = cTmp.b;
  }
  somaGeo.attributes.position.needsUpdate = true;
  somaGeo.attributes.color.needsUpdate = true;
}

// ---------- hover: name the nearest soma ----------
let mouse = null;
canvas.addEventListener('pointermove', (e) => (mouse = [e.clientX, e.clientY]));
canvas.addEventListener('pointerleave', () => (mouse = null));
const vtmp = new THREE.Vector3();
function hover() {
  const tip = $('tip');
  if (!mouse || !latest) return (tip.hidden = true);
  let best = 14, bc = -1;
  for (let c = 0; c < 136; c++) {
    vtmp.set(somaPos[c * 3], somaPos[c * 3 + 1], somaPos[c * 3 + 2]).project(camera);
    const sx = (vtmp.x * 0.5 + 0.5) * innerWidth, sy = (-vtmp.y * 0.5 + 0.5) * innerHeight;
    const d = Math.hypot(sx - mouse[0], sy - mouse[1]);
    if (d < best && vtmp.z < 1) (best = d), (bc = c);
  }
  if (bc < 0) return (tip.hidden = true);
  tip.hidden = false;
  tip.textContent = `${brainMeta.names[bc]}  ${latest.soma[bc].toFixed(1)} mV`;
  tip.style.left = mouse[0] + 14 + 'px';
  tip.style.top = mouse[1] + 10 + 'px';
}

// ---------- food: click the agar to move it ----------
const ray = new THREE.Raycaster();
let down = null;
canvas.addEventListener('pointerdown', (e) => (down = [e.clientX, e.clientY]));
canvas.addEventListener('pointerup', (e) => {
  if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 5) return;
  ray.setFromCamera(new THREE.Vector2((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1), camera);
  const hit = ray.intersectObject(plate)[0];
  if (!hit || Math.hypot(hit.point.x, hit.point.z) > PLATE_R - LAWN_R) return;
  placeFood(hit.point.x, hit.point.z);
  worker.postMessage({ type: 'food', x: hit.point.x, z: hit.point.z });
});

// ---------- camera: close up on the worm, or the whole plate from above ----------
let fly = null;
function flyTo(dist, elev) {
  const off = camera.position.clone().sub(controls.target);
  const az = Math.atan2(off.x, off.z);
  const to = new THREE.Vector3(Math.sin(az) * Math.cos(elev), Math.sin(elev), Math.cos(az) * Math.cos(elev)).multiplyScalar(dist);
  fly = { from: off, to, t: 0 };
}
$('overview').addEventListener('click', () => {
  const far = camera.position.distanceTo(controls.target) > 6;
  flyTo(far ? 0.55 : 2.6 * PLATE_R * (innerWidth < innerHeight ? 1.6 : 1), far ? 0.85 : 1.45);
  $('overview').textContent = far ? 'Whole plate' : 'Close up';
});

// ---------- controls ----------
function sendGain() {
  const syn = $('syn').value / 100, gap = $('gap').value / 100;
  $('syn-out').textContent = `${$('syn').value}%`;
  $('gap-out').textContent = `${$('gap').value}%`;
  worker.postMessage({ type: 'gain', syn, gap });
}
$('syn').addEventListener('input', sendGain);
$('gap').addEventListener('input', sendGain);
$('reset').addEventListener('click', () => {
  epoch++;
  worker.postMessage({ type: 'reset', epoch });
  speedTrack.length = 0;
  rateTrack.length = 0;
  motorHist.fill(-65);
  smellHist.fill(NaN);
  modeHist.fill(0);
  trackN = 0;
  trackGeo.setDrawRange(0, 0);
  followed = null;
});
$('about-open').addEventListener('click', () => $('about').showModal());
$('about-close').addEventListener('click', () => $('about').close());
document.addEventListener('visibilitychange', () => worker.postMessage({ type: 'run', on: !document.hidden }));

function resize() {
  const w = innerWidth, h = innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  // portrait screens need more distance to fit the worm's length
  const far = w < h ? 1.9 : 1;
  if (far !== resize.far) {
    const off = camera.position.clone().sub(controls.target).multiplyScalar(far / (resize.far || 1));
    camera.position.copy(controls.target).add(off);
    resize.far = far;
  }
  for (const c of [muscleCanvas, rasterCanvas, inputCanvas]) {
    c.width = Math.max(1, c.clientWidth * devicePixelRatio);
    c.height = Math.max(1, c.clientHeight * devicePixelRatio);
  }
}
addEventListener('resize', resize);

// ---------- frame ----------
let shown = null;
function frame() {
  requestAnimationFrame(frame);
  if (latest && latest !== shown) {
    shown = latest;
    updateGeometry(latest);
    drawMuscles(latest.muscle);
    drawRaster();
    drawInput();
    $('t').textContent = `${latest.time.toFixed(1)} s`;
    if (speedTrack.length > 1) {
      const [t0, p0] = speedTrack[0], [t1, p1] = speedTrack[speedTrack.length - 1];
      $('speed').textContent = `${(p1.distanceTo(p0) / Math.max(1e-6, t1 - t0) / bodyLength).toFixed(2)} lengths/s`;
    }
    $('rate').textContent = rate > 0.95 ? 'real time' : `${rate.toFixed(2)}× real time`;
    const sk = latest.seek;
    $('doing').textContent = ['crawling forward', 'reversing', 'omega turn'][sk[0]] || '';
    const d = Math.hypot(sk[6] - food.x, sk[7] - food.y) / MM;
    $('food-dist').textContent = d < LAWN_R / MM ? 'on the food' : `${d.toFixed(1)} mm to food`;
    $('turn-rate').textContent = `${(sk[3] * 60).toFixed(1)} turns/min`;
  }
  // follow the worm, keeping the visitor's view angle
  if (latest) {
    if (!followed) {
      // (re)start following: keep the camera's offset from what it was looking at
      const off = camera.position.clone().sub(controls.target);
      followed = centroid.clone();
      controls.target.copy(centroid);
      camera.position.copy(centroid).add(off);
    }
    const d = centroid.clone().sub(followed).multiplyScalar(0.08);
    followed.add(d);
    controls.target.add(d);
    camera.position.add(d);
  }
  if (fly) {
    fly.t = Math.min(1, fly.t + 0.025);
    const e = fly.t * fly.t * (3 - 2 * fly.t);
    camera.position.copy(controls.target).add(fly.from.clone().lerp(fly.to, e));
    if (fly.t === 1) fly = null;
  }
  controls.update();
  // the marker ring around the worm grows with the camera's distance, so it stays visible
  const dist = camera.position.distanceTo(controls.target);
  // keep the depth range in step with the zoom
  const near = Math.max(0.005, dist * 0.01);
  if (Math.abs(camera.near - near) > near * 0.05) (camera.near = near), camera.updateProjectionMatrix();
  marker.position.set(centroid.x, PLATE_Y + 0.004, centroid.z);
  marker.scale.setScalar(Math.max(0.5, dist * 0.03));
  markerMat.opacity = Math.min(0.9, Math.max(0, (dist - 3) / 6));
  hover();
  renderer.render(scene, camera);
}

applyPalette(new URLSearchParams(location.search).get('palette') || 'D');
resize();
frame();

window.wormSim = {
  colors: (key) => applyPalette(key),
  palettes: PALETTES,
  get state() {
    return latest;
  },
  worker,
};
