// Live client-side Open-Vocabulary English -> Single Orthogonal Rotation R -> Language-Free UR5e Robot demo:
//   1) Frozen pretrained English Transformer (gte-small, int8, zero robot data) -> emb [384]
//   2) ONE single orthogonal Procrustes rotation matrix R (R R^T = I) -> z_beh [112]
//   3) Language-free recurrent BehaviorPolicy pi(a_t | obs_t, z_beh, h_t) runs at 20 Hz on MuJoCo WASM
//   4) Cube position in obs_t comes purely from frozen ViT-tiny + Readout2 on the 224x224 front camera!

import * as THREE from 'three';
import loadMujoco from 'https://cdn.jsdelivr.net/npm/@mujoco/mujoco@3.14.0/mujoco.js';
import { CubeEnv, SUBSTEPS, WORKSPACE_LO, WORKSPACE_HI, makeRandom } from './cube-env.js';
import { loadWorldModel } from './model.js';
import { loadMiniLM } from './minilm.js';
import { loadArenaPolicy } from './behavior-policy.js';
import { SceneRenderer } from './render.js';
import { LatentSpaceViz } from './latent-viz.js';

// Weights live on Hugging Face, pinned to an immutable commit (model/ is the local copy, not on Pages).
const MODEL = 'https://huggingface.co/elichen77/ur5e-language-rotation-tfjs/resolve/a481480947229f4fee1b45980636bd2247675b1f';

const $ = id => document.getElementById(id);
const statusEl = $('status');
const eyeCanvas = $('eye');
const random = makeRandom();

const CYCLIC_SKILLS = new Set([
  'rotate_base', 'wave', 'circle', 'sweep', 'clap_gripper', 'nod', 'dance', 'poke_cube',
  'osc_z', 'osc_y', 'osc_x', 'circle_xy', 'circle_vert', 'base_revolve',
  'wrist_pitch', 'wrist_roll', 'multi_sway', 'cube_tap',
]);

async function loadRobotFiles() {
  const { files: list } = await fetch('robot/manifest.json').then(r => r.json());
  const files = {};
  await Promise.all(list.map(async f => {
    const r = await fetch('robot/' + f);
    files[f] = f.endsWith('.xml') ? await r.text() : new Uint8Array(await r.arrayBuffer());
  }));
  return files;
}

async function initBackend() {
  const tf = window.tf;
  if (navigator.gpu) {
    try {
      await tf.setBackend('webgpu');
      await tf.ready();
      return 'WebGPU';
    } catch (e) {
      console.warn('WebGPU unavailable, falling back to WebGL:', e);
    }
  }
  await tf.setBackend('webgl');
  await tf.ready();
  return 'WebGL';
}

let env, wm, lm, pol, view, latentViz;
let backendName = '';
let simTime = 0, substep = 0, lastFrame = 0, speed = 1;
let active = false;
let stepping = false;
let needAction = false;
let skillStep = 0;
let drag = null;

// Latest visual cube estimate strictly from 224x224 front_pixels -> ViT-tiny + Readout2!
let cubeEst = [0.42, 0.0, 0.02];
let contactEst = 0;
let visTick = 0;
let lastStepMs = 0;

// Active prompt & rotation config (no auto-cycling: runs only when the user clicks or submits!)
let rotN = '10000';
let activePrompt = '';
let lastEmb = null, lastPieces = null, lastEncMs = 0, lastRot = null;

async function init() {
  try {
    statusEl.textContent = 'Initializing GPU backend…';
    backendName = await initBackend();
    $('backend-label').textContent = backendName;

    statusEl.textContent = 'Loading MuJoCo physics, English encoder & robot policy…';
    const [mujoco, files, model, mini, arenaPol] = await Promise.all([
      loadMujoco(),
      loadRobotFiles(),
      loadWorldModel(window.tf, MODEL),
      loadMiniLM(window.tf, MODEL),
      loadArenaPolicy(window.tf, MODEL),
    ]);
    wm = model;
    lm = mini;
    pol = arenaPol;

    env = new CubeEnv(mujoco, files, random);
    env.reset();
    view = new SceneRenderer($('view'), env);
    latentViz = new LatentSpaceViz($('latent-canvas'), pol);
    setupPointer();
    populateMeasuredTable();
    view.draw();

    // Warm up visual encoder, English encoder, and policy once so first user command is instant
    statusEl.textContent = 'Compiling GPU shaders…';
    await refreshVisualReadout();
    const warm = await lm.encode('throw cube');
    const warmRot = pol.rotateEmbedding(warm.emb, 'throw cube', warm.pieces, { nKey: rotN, encMs: 0 });
    pol.setActiveLatent(warmRot.zBeh);
    await pol.step(env.arenaObs(cubeEst, contactEst), 0);
    pol.resetHidden();
    if (pol.zBehT) { pol.zBehT.dispose(); pol.zBehT = null; }

    updateIdleStatus();
    statusEl.hidden = true;
    window.demo = {
      env, wm, lm, pol, view, latentViz,
      applyPrompt, setRotationN, resetEpisode, respawnCube,
      getSkillStep: () => skillStep,
      getLastStepMs: () => lastStepMs,
    };
    requestAnimationFrame(frame);
  } catch (e) {
    statusEl.textContent = `Couldn't start demo: ${e.message}`;
    console.error(e);
  }
}

function updateIdleStatus() {
  const isAligned = rotN !== 'shuffled';
  const badge = $('rot-status-badge');
  badge.textContent = isAligned ? `Aligned · O(${pol.kT}, ${pol.kV})` : 'Shuffled (control)';
  badge.classList.toggle('warn', !isAligned);
  $('trace-ortho').textContent = `max ‖RRᵀ−I‖ = ${pol.orthoError(rotN).toExponential(1)}`;
}

function populateMeasuredTable() {
  const m = pol.meta.metrics || {};
  const lmShort = (pol.meta.lmModel || 'gte-small').split('/').pop();
  $('prose-lm-name').textContent = lmShort;
  $('trace-lm-title').textContent = `1. Pretrained ${lmShort} (zero robot data)`;
  const fill = (idPre, obj) => {
    if (!obj) return;
    const t1 = $(`${idPre}-t1`), f = $(`${idPre}-f`), u = $(`${idPre}-u`), p = $(`${idPre}-p`);
    if (t1) t1.textContent = `${(obj.top1 * 100).toFixed(1)}%`;
    if (f) f.textContent = obj.foscttm.toFixed(3);
    if (u) {
      u.textContent = obj.zs_fam !== undefined
        ? `${obj.zs_fam}/37 (${obj.zs_full} full)`
        : `${obj.unseen40}/40 (${(obj.unseen40 * 2.5).toFixed(0)}%)`;
    }
    if (p) p.textContent = `${obj.place_cm.toFixed(1)} cm`;
  };
  fill('r-10000', m['10000']);
  fill('r-1000', m['1000']);
  fill('r-300', m['300']);
  fill('r-100', m['100']);
  fill('r-30', m['30']);
  fill('r-rand', m['random_lm']);
  fill('r-shuf', m['shuffled']);
  if (m.closed_loop) {
    $('prose-cl-summary').innerHTML =
      `Closed-loop MuJoCo physics verification across 24 strictly zero-shot English commands (3 trials each from randomized arm poses, with all 80+ user/metaphor words banned from training captions by automated <code>assert</code>, driven end-to-end by quantized <code>${escapeHtml(lmShort)}</code> &rarr; single orthogonal <code>R</code> &rarr; language-free <code>BehaviorPolicy</code>): <strong>${escapeHtml(m.closed_loop.aligned)}</strong> succeeded physically with aligned <code>R</code> vs. <strong>${escapeHtml(m.closed_loop.shuffled)}</strong> for the shuffled-pair rotation control.`;
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function fmtSkill(s) {
  return (s || '').replace(/_/g, ' ');
}

function updateRotationUI(res) {
  const isAligned = res.nKey !== 'shuffled';
  const m = pol.meta.metrics?.[res.nKey];
  const isComp = Boolean(m && m.zs_fam !== undefined);

  const badge = $('rot-status-badge');
  badge.textContent = isAligned ? `Aligned · O(${pol.kT}, ${pol.kV})` : 'Shuffled (control)';
  badge.classList.toggle('warn', !isAligned);

  $('trace-enc-ms').textContent = `${res.encMs.toFixed(1)} ms`;
  const tokStr = res.pieces.length ? res.pieces.map(p => escapeHtml(p)).join(' ') : '&lt;empty&gt;';
  $('trace-tokens').innerHTML = `${tokStr} &rarr; <span class="${isAligned ? 'hi' : 'warn'}">e<sub>text</sub> &isin; &Ropf;<sup>384</sup></span>`;

  $('trace-ortho').textContent = `max ‖RRᵀ−I‖ = ${res.orthoErr.toExponential(1)}`;
  const nLabel = res.nKey === 'shuffled'
    ? (isComp ? '23,503 shuffled pairs' : '10,000 shuffled pairs')
    : (res.nKey === '10000' && isComp ? '23,503 paired clips' : `${Number(res.nKey).toLocaleString()} paired clips`);
  const statTxt = m ? ` · val ${(m.top1 * 100).toFixed(1)}%, FOSCTTM ${m.foscttm.toFixed(3)}` : '';
  const nSub = res.nKey === 'shuffled' ? 'shuf' : (res.nKey === '10000' && isComp ? '23k' : res.nKey);
  $('trace-rot').innerHTML = `R<sub>${nSub}</sub> (${pol.kT}&times;${pol.kV}, ${nLabel})${statTxt}`;

  $('trace-nn').textContent = `cos = ${res.nearestCos.toFixed(2)}`;
  const placeTxt = res.placeGoal
    ? ` &middot; target (${Math.round(res.placeGoal[0] * 100)}, ${res.placeGoal[1] >= 0 ? '+' : ''}${Math.round(res.placeGoal[1] * 100)}) cm`
    : '';
  $('trace-goal').innerHTML =
    `<span class="${isAligned ? 'hi' : 'warn'}">${escapeHtml(fmtSkill(res.topSkill))}</span> (nearest: ${escapeHtml(res.nearestLabel)})${placeTxt}`;

  const hud = $('active-cmd');
  const tag = res.nKey === 'shuffled' ? 'Shuf' : `N=${res.nKey === '10000' ? (isComp ? '23k' : '10k') : res.nKey === '1000' ? '1k' : res.nKey}`;
  hud.innerHTML = `<span class="arrow">R(${tag}) &rarr;</span> <strong>${escapeHtml(res.nearestLabel)}</strong> (cos ${res.nearestCos.toFixed(2)})`;
  hud.classList.toggle('unaligned', !isAligned);

  $('matched-skill').textContent = fmtSkill(res.topSkill);

  document.querySelectorAll('#prompt-chips .chip').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.prompt.toLowerCase() === res.text.trim().toLowerCase());
  });
}

async function applyPrompt(text, { silent = false, reencode = true } = {}) {
  const clean = (text || '').trim();
  if (!clean || !lm || !pol || !env) return null;
  activePrompt = clean;
  $('prompt-input').value = clean;

  // If a previous throw launched the cube outside the tabletop camera view, place a fresh cube back on the table
  const cp = env.cubePos();
  const outOfView = cp[0] < 0.25 || cp[0] > 0.59 || Math.abs(cp[1]) > 0.29 || cp[2] > 0.38;
  if (outOfView) {
    env.respawnCube();
    view.clearTrail();
    if (!silent) note('Respawned cube on table');
  }

  if (reencode || !lastEmb) {
    const t0 = performance.now();
    const { emb, pieces } = await lm.encode(clean);
    lastEmb = emb;
    lastPieces = pieces;
    lastEncMs = performance.now() - t0;
  }

  const res = pol.rotateEmbedding(lastEmb, clean, lastPieces, { nKey: rotN, encMs: lastEncMs });
  lastRot = res;
  pol.setActiveLatent(res.zBeh);
  skillStep = 0;
  substep = 0;
  visTick = 0;
  needAction = true;
  active = true;

  if (res.placeGoal) {
    env.setGoal(res.placeGoal);
    view.showGoalMarker = true;
  } else {
    view.showGoalMarker = false;
  }
  view.clearTrail();
  latentViz.setRotation(res);
  updateRotationUI(res);
  await refreshVisualReadout();
  if (!silent && !outOfView) {
    $('note').hidden = true;
    clearTimeout(noteTimer);
  }
  return res;
}

async function setRotationN(nKey) {
  rotN = nKey;
  document.querySelectorAll('#n-toggle .seg').forEach(b => b.classList.toggle('active', b.dataset.n === rotN));
  if (activePrompt && pol && env) {
    await applyPrompt(activePrompt, { reencode: false });
  } else if (pol) {
    updateIdleStatus();
  }
}

async function refreshVisualReadout() {
  // Purely visual cube perception from the 224x224 front_pixels camera (ZERO privileged cube coordinates!)
  const pixels = view.captureEyePixels(eyeCanvas, view.showGoalMarker);
  const { s0 } = await wm.encodeFrame(pixels, 3);
  cubeEst = [s0[0], s0[1], s0[2]];
  contactEst = s0[6];
  const cp = env.cubePos();
  const inView = Math.hypot(cp[0] - 0.42, cp[1]) < 0.45 && cp[2] < 0.45;
  $('eye-phase').textContent = inView ? '20 Hz' : 'launched';
  view.updateVisualReadout(inView ? cubeEst : null, cp);
}

async function runPolicyStep() {
  if (stepping || drag || !active || !pol || !pol.zBehT) return;
  stepping = true;
  try {
    const t0 = performance.now();
    // 10 Hz visual Readout2 during early grasp/approach (steps 0..31), 5 Hz after (steps 32+) so 20 Hz control runs at 1.0x real time
    const visPeriod = skillStep < 32 ? 2 : 4;
    if (visTick % visPeriod === 0) {
      await refreshVisualReadout();
    } else {
      view.updateVisualReadout(null, env.cubePos());
    }
    visTick++;

    const obs = env.arenaObs(cubeEst, contactEst);
    const tFrac = Math.min(1, skillStep / Math.max(1, pol.horizon - 1));
    const act = await pol.step(obs, tFrac);
    env.setJointControl(act);
    needAction = false;
    lastStepMs = performance.now() - t0;
    $('step-ms').textContent = `${lastStepMs.toFixed(1)} ms`;
    skillStep++;

    if (skillStep >= pol.horizon) {
      if (lastRot && CYCLIC_SKILLS.has(lastRot.topSkill)) {
        // Keep cyclic motions (spinning base, pumping, circling, dancing) rolling smoothly
        skillStep = 24;
      } else {
        skillStep = pol.horizon - 1;
      }
    }
  } catch (err) {
    console.error('Policy step error:', err);
  } finally {
    stepping = false;
  }
}

function frame(now) {
  const dt = Math.min(0.05, (now - (lastFrame || now)) / 1000);
  lastFrame = now;

  if (drag || !active) {
    advanceSubsteps(dt * (drag ? 1 : speed));
  } else if (!stepping) {
    advanceControl(dt * speed);
  } else {
    simTime = Math.min(simTime + dt * speed, SUBSTEPS * env.model.opt.timestep);
  }

  view.draw();
  requestAnimationFrame(frame);
}

function advanceSubsteps(seconds) {
  simTime += seconds;
  const h = env.model.opt.timestep;
  while (simTime >= h) {
    simTime -= h;
    env.substep();
  }
}

function advanceControl(seconds) {
  const h = env.model.opt.timestep;
  simTime = Math.min(simTime + seconds, 2 * SUBSTEPS * h);
  while (simTime >= h) {
    if (needAction) {
      runPolicyStep();
      return;
    }
    simTime -= h;
    env.substep();
    substep++;
    if (substep % SUBSTEPS === 0) {
      needAction = true;
    }
  }
}

function respawnCube() {
  if (!env) return;
  env.respawnCube();
  view.clearTrail();
  refreshVisualReadout();
}

function resetEpisode() {
  if (!env) return;
  env.reset();
  view.showGoalMarker = false;
  view.clearTrail();
  active = false;
  skillStep = 0;
  substep = 0;
  simTime = 0;
  needAction = false;
  activePrompt = '';
  lastRot = null;
  $('prompt-input').value = '';
  $('active-cmd').textContent = 'Ready · click a prompt or type a command';
  $('active-cmd').classList.remove('unaligned');
  $('step-ms').textContent = '';
  document.querySelectorAll('#prompt-chips .chip').forEach(b => b.classList.remove('active'));
  pol?.resetHidden();
  refreshVisualReadout();
  note('Arm & cube reset');
}

let noteTimer;
function note(text) {
  const el = $('note');
  el.textContent = text;
  el.hidden = false;
  clearTimeout(noteTimer);
  noteTimer = setTimeout(() => { el.hidden = true; }, 2200);
}

// --- Pointer interaction: drag cube, click table to generate natural placement command, or orbit
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const UP = new THREE.Vector3(0, 0, 1);
const TABLE_PLANE = new THREE.Plane(UP, 0);
const LIFT = 0.10;

function rayAt(e) {
  const r = view.canvas.getBoundingClientRect();
  pointer.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  raycaster.setFromCamera(pointer, view.camera);
  return raycaster.ray;
}

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const onCube = () => raycaster.intersectObject(view.cubeMesh, false).length > 0;

function coordsToEnglish(x, y) {
  const cmY = Math.round(Math.abs(y) * 100 / 5) * 5;
  const side = y < 0 ? 'left' : 'right';
  const depth = x > 0.48 ? 'near the front edge' : (x < 0.36 ? 'near the robot' : 'halfway back');
  if (cmY < 5) return `put the cube centered, ${depth}`;
  return `put the cube ${cmY} cm to the ${side}, ${depth}`;
}

function setupPointer() {
  const canvas = view.canvas;
  let downPos = null;

  canvas.addEventListener('wheel', e => { if (!e.ctrlKey) e.stopImmediatePropagation(); }, { capture: true });

  canvas.addEventListener('pointerdown', e => {
    downPos = [e.clientX, e.clientY];
    rayAt(e);
    if (!onCube()) return;
    e.stopImmediatePropagation();
    try { canvas.setPointerCapture(e.pointerId); } catch {}
    const pos = new THREE.Vector3(...env.cubePos());
    drag = { goal: pos, height: Math.max(pos.z, 0.02) + LIFT };
    env.pull = pullCube;
    moveDraggedCube(raycaster.ray);
    canvas.classList.add('grabbing');
  }, { capture: true });

  canvas.addEventListener('pointermove', e => {
    const ray = rayAt(e);
    if (drag) moveDraggedCube(ray);
    else canvas.classList.toggle('can-grab', onCube());
  });

  const release = e => {
    if (drag) {
      env.pull = null;
      for (let i = 0; i < 6; i++) env.data.xfrc_applied[6 * env.cubeBody + i] = 0;
      drag = null;
      canvas.classList.remove('grabbing');
      downPos = null;
      refreshVisualReadout();
      return;
    }
    if (downPos && e.type === 'pointerup') {
      const moved = Math.hypot(e.clientX - downPos[0], e.clientY - downPos[1]);
      if (moved < 6) {
        const ray = rayAt(e);
        const hit = new THREE.Vector3();
        if (ray.intersectPlane(TABLE_PLANE, hit)) {
          if (hit.x >= WORKSPACE_LO[0] - 0.04 && hit.x <= WORKSPACE_HI[0] + 0.04 &&
              hit.y >= WORKSPACE_LO[1] - 0.04 && hit.y <= WORKSPACE_HI[1] + 0.04) {
            const cmd = coordsToEnglish(clamp(hit.x, 0.29, 0.55), clamp(hit.y, -0.26, 0.26));
            applyPrompt(cmd);
          }
        }
      }
    }
    downPos = null;
  };
  canvas.addEventListener('pointerup', release);
  canvas.addEventListener('pointercancel', release);
}

function moveDraggedCube(ray) {
  const p = new THREE.Vector3();
  const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(UP, new THREE.Vector3(0, 0, drag.height));
  if (!ray.intersectPlane(plane, p)) return;
  drag.goal.set(clamp(p.x, 0.20, 0.65), clamp(p.y, -0.45, 0.45), drag.height);
}

function pullCube() {
  const d = env.data, b = env.cubeBody, mass = env.model.body_mass[b], v = env.cubeDofAdr;
  const k = 600, c = 2 * Math.sqrt(k), x = env.cubePos();
  for (let i = 0; i < 3; i++) {
    let f = mass * (k * (drag.goal.getComponent(i) - x[i]) - c * d.qvel[v + i]);
    if (i === 2) f += mass * 9.81;
    d.xfrc_applied[6 * b + i] = clamp(f, -20, 20);
  }
  const R = d.xmat;
  for (let i = 0; i < 3; i++) {
    let w = 0;
    for (let j = 0; j < 3; j++) w += R[9 * b + 3 * i + j] * d.qvel[v + 3 + j];
    d.xfrc_applied[6 * b + 3 + i] = -0.002 * w;
  }
}

// --- UI Listeners
$('prompt-form').addEventListener('submit', e => {
  e.preventDefault();
  applyPrompt($('prompt-input').value);
});

$('prompt-chips').addEventListener('click', e => {
  const btn = e.target.closest('.chip');
  if (!btn) return;
  applyPrompt(btn.dataset.prompt);
});

$('n-toggle').addEventListener('click', e => {
  const btn = e.target.closest('.seg');
  if (!btn) return;
  setRotationN(btn.dataset.n);
});

$('reset-btn').addEventListener('click', () => resetEpisode());
$('slow').addEventListener('change', e => { speed = e.target.checked ? 0.35 : 1; });

// Redraw 2D behavior manifold cleanly when the user expands the inspector
$('inspector').addEventListener('toggle', e => {
  if (e.target.open && latentViz) latentViz.draw(1);
});

init();
