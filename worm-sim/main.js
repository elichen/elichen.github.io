// BAAIWorm, live: the page. worker.js runs the brain and body (engine.wasm);
// this file draws the body, the nervous system inside it, and the panels.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const $ = (id) => document.getElementById(id);

// Palettes sampled from photographs of C. elegans on Wikimedia Commons
const PALETTES = {
  // A: Bob Goldstein's DIC micrograph (UNC Chapel Hill, CC BY-SA 3.0)
  A: { name: 'DIC', bg: '#66847a', bg2: '#97b0a3', body: '#dfe8e0', rim: '#f0f5ef', shade: '#263832', rest: '#3e564e', active: '#f0f5ef', muscle: '#263832', speck: '#b4c7bc', ui: '#1f2f2a', uiDim: '#3e564e', panel: 'rgba(151,176,163,0.82)', accent: '#263832', additive: false },
  // B: confocal image of a PVM neuron (Guasp, CC BY 4.0): mCherry, GFP, DAPI
  B: { name: 'Confocal', bg: '#0a0002', bg2: '#140210', body: '#0c023a', rim: '#2603dc', shade: '#0a0002', rest: '#71060a', active: '#a3ff46', muscle: '#fa0407', speck: '#26036a', ui: '#f7f9f7', uiDim: '#9a8f96', panel: 'rgba(20,2,16,0.78)', accent: '#67f514', additive: true },
  // C: DAPI-stained worms (HoPo, CC BY-SA 3.0)
  C: { name: 'DAPI', bg: '#04050a', bg2: '#050712', body: '#07113b', rim: '#406dcc', shade: '#030207', rest: '#12318a', active: '#a6d4ff', muscle: '#64aaed', speck: '#0c216c', ui: '#dfe8f7', uiDim: '#7c8db0', panel: 'rgba(5,7,18,0.8)', accent: '#64aaed', additive: true },
  // D: worms on an agar plate through a microscope (Gannu03, CC BY-SA 4.0)
  D: { name: 'Agar plate', bg: '#c4c2b4', bg2: '#dfdfd2', body: '#a3a092', rim: '#4a4945', shade: '#030305', rest: '#757268', active: '#030305', muscle: '#030305', speck: '#a3a092', ui: '#0f0f11', uiDim: '#4a4945', panel: 'rgba(223,223,210,0.85)', accent: '#030305', additive: false },
};
let pal = PALETTES.B;

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
renderer.setPixelRatio(Math.min(2, devicePixelRatio));
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(38, 1, 0.005, 50);
camera.position.set(0.36, 0.24, -0.3);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.minDistance = 0.08;
controls.maxDistance = 4;

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

// specks in the water, wrapped around the view so the motion reads
const NSPECK = 1400, BOX = 2.4;
const speckBase = new Float32Array(NSPECK * 3).map(() => (Math.random() - 0.5) * BOX);
const speckPos = new Float32Array(NSPECK * 3);
const speckGeo = new THREE.BufferGeometry();
speckGeo.setAttribute('position', new THREE.BufferAttribute(speckPos, 3));
const speckMat = new THREE.PointsMaterial({ size: 0.004, map: dot, transparent: true, depthWrite: false, opacity: 0.7 });
const specks = new THREE.Points(speckGeo, speckMat);
specks.frustumCulled = false;
scene.add(specks);

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
  speckMat.color.set(pal.speck);
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
const HISTORY = 300; // control steps shown (10 s)
const motorHist = new Float32Array(80 * HISTORY).fill(-65);
const inputHist = new Float32Array(HISTORY).fill(-0.08);
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

function drawInput() {
  const g = inputCanvas.getContext('2d');
  const W = inputCanvas.width, H = inputCanvas.height;
  g.clearRect(0, 0, W, H);
  g.strokeStyle = pal.accent;
  g.lineWidth = 1.5 * devicePixelRatio;
  g.beginPath();
  for (let k = 0; k < HISTORY; k++) {
    const v = inputHist[(histHead + k) % HISTORY];
    const x = (k / (HISTORY - 1)) * W;
    const y = H - ((v + 0.09) / 0.12) * H;
    k ? g.lineTo(x, y) : g.moveTo(x, y);
  }
  g.stroke();
}

// ---------- state from the worker ----------
let latest = null;
let centroid = new THREE.Vector3();
let followed = null;
const speedTrack = [];

const worker = new Worker('worker.js');
worker.onmessage = (e) => {
  const m = e.data;
  if (m.type === 'progress') $('loading-text').textContent = m.text + '…';
  else if (m.type === 'error') $('loading-text').textContent = 'Could not start: ' + m.text;
  else if (m.type === 'ready') $('loading').classList.add('done');
  else if (m.type === 'state') {
    latest = m;
    // history for the panels: the newest step stands for all steps of this batch
    for (let s = 0; s < Math.max(1, m.steps); s++) {
      for (let r = 0; r < 80; r++) motorHist[r * HISTORY + histHead] = m.motor[r];
      inputHist[histHead] = m.stimulus;
      histHead = (histHead + 1) % HISTORY;
    }
    let cx = 0, cy = 0, cz = 0;
    for (let i = 0; i < nv; i++) (cx += m.x[i * 3]), (cy += m.x[i * 3 + 1]), (cz += m.x[i * 3 + 2]);
    centroid.set(cx / nv, cy / nv, cz / nv);
    speedTrack.push([m.time, centroid.clone()]);
    while (speedTrack.length > 2 && m.time - speedTrack[0][0] > 3) speedTrack.shift();
    rate = rate * 0.9 + m.rate * 0.1;
    // ?report sends the speed to a local test server (a tab under DevTools runs wasm unoptimized)
    if (location.search.includes('report') && Math.floor(m.time) % 5 === 0 && Math.floor(m.time) !== reported) {
      reported = Math.floor(m.time);
      fetch(`/report?t=${m.time.toFixed(1)}&cap=${m.capacity.toFixed(3)}&rate=${m.rate.toFixed(3)}`);
    }
  }
};
let rate = 1;
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
  worker.postMessage({ type: 'reset' });
  speedTrack.length = 0;
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
    $('rate').textContent = rate > 0.97 ? 'real time' : `${rate.toFixed(2)}× real time`;
  }
  // follow the worm, keeping the visitor's view angle
  if (latest) {
    if (!followed) followed = centroid.clone(), controls.target.copy(centroid), camera.position.add(centroid);
    const d = centroid.clone().sub(followed).multiplyScalar(0.08);
    followed.add(d);
    controls.target.add(d);
    camera.position.add(d);
  }
  controls.update();
  const t = controls.target;
  for (let i = 0; i < NSPECK; i++) {
    for (let k = 0; k < 3; k++) {
      const c = k === 0 ? t.x : k === 1 ? t.y : t.z;
      let p = speckBase[i * 3 + k] - c;
      p -= Math.round(p / BOX) * BOX;
      speckPos[i * 3 + k] = c + p;
    }
  }
  speckGeo.attributes.position.needsUpdate = true;
  hover();
  renderer.render(scene, camera);
}

applyPalette(new URLSearchParams(location.search).get('palette') || 'B');
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
