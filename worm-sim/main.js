/**
 * BAAIWorm browser locomotion player
 *
 * The original browser port loaded WetNet muscle data but then ignored it in
 * favor of a synthetic sine wave. This version replays the downloaded muscle
 * drive directly, solves a lightweight hydrodynamic locomotion step, and skins
 * the existing tetrahedral mesh onto a moving centerline so the worm actually
 * swims instead of wriggling in place.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { ReducedMotorNet } from './reduced_motor_net.js';

const NUM_MUSCLES = 96;
const MUSCLE_SEGMENTS = 24;
const CENTERLINE_SAMPLES = 128;
const MIN_DELTA_TIME = 1 / 120;
const MAX_DELTA_TIME = 0.05;
const REDUCED_SOLVER_STEP = 1 / 180;

let meshData;
let muscleData;
let neuralTraces;
let sampleMuscle;

let scene;
let camera;
let renderer;
let controls;
let wormMesh;
let wormGeometry;

let positions;
let restPositions;

let restCenterlineX;
let restCenterlineY;
let restCenterlineZ;
let centerlineX;
let centerlineY;
let centerlineZ;
let previousCenterlineX;
let previousCenterlineZ;
let centerlineAngles;
let smoothedCurvature;

let vertexRig;
let bodyLength = 0;

let simTime = 0;
let driveTime = 0;
let playbackSpeed = 1.0;
let bodyStiffness = 0.55;
let fluidAnisotropy = 4.4;
let muscleGain = 1.9;
let hydroDamping = 0.72;

let lastTime = 0;
let lastFrameTime = 0;
let frameCount = 0;
let fps = 0;

let currentMuscleActivations = new Float32Array(NUM_MUSCLES);
let shapeVelocityX = new Float32Array(CENTERLINE_SAMPLES);
let shapeVelocityZ = new Float32Array(CENTERLINE_SAMPLES);
let dorsalSegments = new Float32Array(MUSCLE_SEGMENTS);
let ventralSegments = new Float32Array(MUSCLE_SEGMENTS);

let currentFrameFloat = 0;
let currentFrameIndex = 0;
let currentReplayTime = 0;
let currentDominantSide = 'Balanced';
let currentActivationCentroid = 11.5;
let currentDriftSpeed = 0;
let currentNeuronVoltages = new Float32Array(0);
let driveSourceLabel = 'WetNet replay';
let modelMode = 'replay';
let reducedMotorNet;
let reducedSolverAccumulator = 0;

let muscleCells = { dorsal: [], ventral: [] };
let neuronRows = [];
let lastUserZoomAtMs = -Infinity;
let uiRefs = {};

let bodyPose = { x: 0, z: 0, yaw: 0 };
let bodyVelocity = { x: 0, z: 0, omega: 0 };

function updateLoading(text, progress = '') {
  document.getElementById('loading-text').textContent = text;
  document.getElementById('loading-progress').textContent = progress;
}

async function loadJSON(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Failed to load ${path}: ${response.status}`);
  }
  return response.json();
}

async function loadAllData() {
  const basePath = './data';

  updateLoading('Loading mesh data...', '1/4');
  meshData = await loadJSON(`${basePath}/mesh_data.json`);

  updateLoading('Loading muscle centerlines...', '2/4');
  muscleData = await loadJSON(`${basePath}/muscle_data.json`);

  updateLoading('Loading neural traces...', '3/4');
  neuralTraces = await loadJSON(`${basePath}/neural_traces.json`);

  updateLoading('Loading WetNet muscle drive...', '4/4');
  sampleMuscle = await loadJSON(`${basePath}/sample_muscle.json`);
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function formatSigned(value, digits = 2) {
  return `${value >= 0 ? '+' : ''}${value.toFixed(digits)}`;
}

function syncReducedControls() {
  if (!uiRefs.forwardDrive) {
    return;
  }

  uiRefs.forwardDrive.value = reducedMotorNet.params.forwardDrive.toFixed(2);
  uiRefs.forwardDriveValue.textContent = reducedMotorNet.params.forwardDrive.toFixed(2);
  uiRefs.turnBias.value = reducedMotorNet.params.turnBias.toFixed(2);
  uiRefs.turnBiasValue.textContent = formatSigned(reducedMotorNet.params.turnBias);
  uiRefs.wavePeriod.value = reducedMotorNet.params.wavePeriod.toFixed(2);
  uiRefs.wavePeriodValue.textContent = `${reducedMotorNet.params.wavePeriod.toFixed(2)} s`;
}

function applyReducedControlState() {
  const reducedActive = modelMode === 'reduced';
  if (!uiRefs.forwardDrive) {
    return;
  }

  uiRefs.forwardDrive.disabled = !reducedActive;
  uiRefs.turnBias.disabled = !reducedActive;
  uiRefs.wavePeriod.disabled = !reducedActive;

  for (const button of uiRefs.modeButtons) {
    button.classList.toggle('active', button.dataset.modelMode === modelMode);
  }
}

function updateSourcePanel() {
  const replayMode = modelMode === 'replay';

  document.getElementById('mode-subtitle').textContent = replayMode
    ? 'Downloaded WetNet replay'
    : 'Live reduced motor net';
  document.getElementById('hero-copy').textContent = replayMode
    ? 'What is real here: the neural traces and muscle activations come from a downloaded WetNet run. What is approximated here: the browser uses a reduced-order body replay instead of the original FEM stack or a live neural solve.'
    : 'What is real here: the browser still uses the downloaded mesh and muscle layout. What is approximated here: a reduced c302-style motor net now generates the drive live in the browser instead of replaying the downloaded WetNet trace.';
  document.getElementById('truth-pill').textContent = replayMode
    ? 'Real WetNet output, simplified browser biomechanics'
    : 'Live browser motor net, simplified browser biomechanics';
  document.getElementById('source-tag-a').textContent = replayMode
    ? 'data/neural_traces.json'
    : 'browser reduced motor net';
  document.getElementById('source-tag-b').textContent = replayMode
    ? 'data/sample_muscle.json'
    : 'live neuromuscular drive';
  document.getElementById('mode-copy').textContent = replayMode
    ? 'Switch between the downloaded WetNet replay and a live reduced motor net that runs directly in the browser.'
    : 'The reduced motor net is a browser-side approximation: descending drive, segmental wave propagation, and named neuron snapshots are synthesized live.';
  document.getElementById('muscle-copy').textContent = replayMode
    ? 'Current WetNet activation packet across the 24 body segments. Dorsal is the top row, ventral is the bottom row.'
    : 'Current output of the reduced browser motor net across the 24 body segments. Dorsal is the top row, ventral is the bottom row.';
  document.getElementById('neuron-copy').textContent = replayMode
    ? 'Most depolarized motor neurons from the same downloaded frame. This gives provenance for the replay: the motion comes from recorded WetNet output, not a browser-side sine wave.'
    : 'Most depolarized named neurons from the live reduced motor net. These voltages are synthetic browser-side states, not recorded WetNet traces.';
  document.getElementById('frame-label').textContent = replayMode ? 'Dataset Frame' : 'Solver Step';
  document.getElementById('time-label').textContent = replayMode ? 'Replay Time' : 'Model Time';
  document.getElementById('drive-source').textContent = driveSourceLabel;
}

function setModelMode(nextMode) {
  if (nextMode === modelMode) {
    return;
  }

  modelMode = nextMode;
  simTime = 0;
  driveTime = 0;
  currentFrameFloat = 0;
  currentFrameIndex = 0;
  currentReplayTime = 0;
  reducedSolverAccumulator = 0;
  bodyVelocity = { x: 0, z: 0, omega: 0 };
  bodyPose = { x: 0, z: 0, yaw: 0 };

  if (modelMode === 'replay') {
    driveSourceLabel = 'WetNet replay';
    updateMuscleActivations(0);
  } else {
    driveSourceLabel = 'Reduced motor net';
    reducedMotorNet.reset();
    syncReducedControls();
    stepReducedMotorNet(REDUCED_SOLVER_STEP);
  }

  updateCenterlineShape(MIN_DELTA_TIME, true);
  updateSkinnedVertices();
  applyReducedControlState();
  updateSourcePanel();
}

function createCenterlineArrays() {
  return {
    x: new Float32Array(CENTERLINE_SAMPLES),
    y: new Float32Array(CENTERLINE_SAMPLES),
    z: new Float32Array(CENTERLINE_SAMPLES)
  };
}

function buildRestCenterline() {
  const muscleNames = muscleData.muscle_names || Object.keys(muscleData.muscles);
  const pointCount = muscleData.muscles[muscleNames[0]].length;
  const averaged = new Array(pointCount);

  for (let i = 0; i < pointCount; i++) {
    let x = 0;
    let y = 0;
    let z = 0;

    for (const name of muscleNames) {
      const point = muscleData.muscles[name][i];
      x += point[0];
      y += point[1];
      z += point[2];
    }

    const scale = 1 / muscleNames.length;
    averaged[i] = [x * scale, y * scale, z * scale];
  }

  const cumulative = new Float32Array(pointCount);
  let totalLength = 0;
  for (let i = 1; i < pointCount; i++) {
    const dx = averaged[i][0] - averaged[i - 1][0];
    const dy = averaged[i][1] - averaged[i - 1][1];
    const dz = averaged[i][2] - averaged[i - 1][2];
    totalLength += Math.hypot(dx, dy, dz);
    cumulative[i] = totalLength;
  }
  bodyLength = totalLength;

  const resampled = createCenterlineArrays();
  for (let sample = 0; sample < CENTERLINE_SAMPLES; sample++) {
    const targetLength = (sample / (CENTERLINE_SAMPLES - 1)) * totalLength;
    let seg = 0;
    while (seg < pointCount - 2 && cumulative[seg + 1] < targetLength) {
      seg++;
    }

    const segStart = cumulative[seg];
    const segEnd = cumulative[seg + 1];
    const span = Math.max(segEnd - segStart, 1e-6);
    const t = clamp((targetLength - segStart) / span, 0, 1);
    const a = averaged[seg];
    const b = averaged[seg + 1];

    resampled.x[sample] = lerp(a[0], b[0], t);
    resampled.y[sample] = lerp(a[1], b[1], t);
    resampled.z[sample] = lerp(a[2], b[2], t);
  }

  return resampled;
}

function projectPointToCenterline(x, z) {
  let bestDistSq = Infinity;
  let bestS = 0;
  let bestPointX = 0;
  let bestPointZ = 0;
  let bestNormalX = 1;
  let bestNormalZ = 0;
  let bestTangentX = 0;
  let bestTangentZ = 1;
  let bestCenterY = 0;

  const segmentLength = bodyLength / (CENTERLINE_SAMPLES - 1);

  for (let i = 0; i < CENTERLINE_SAMPLES - 1; i++) {
    const ax = restCenterlineX[i];
    const az = restCenterlineZ[i];
    const bx = restCenterlineX[i + 1];
    const bz = restCenterlineZ[i + 1];
    const dx = bx - ax;
    const dz = bz - az;
    const lenSq = dx * dx + dz * dz;
    if (lenSq < 1e-12) {
      continue;
    }

    const t = clamp(((x - ax) * dx + (z - az) * dz) / lenSq, 0, 1);
    const px = ax + dx * t;
    const pz = az + dz * t;
    const distX = x - px;
    const distZ = z - pz;
    const distSq = distX * distX + distZ * distZ;

    if (distSq < bestDistSq) {
      const invLen = 1 / Math.sqrt(lenSq);
      bestDistSq = distSq;
      bestPointX = px;
      bestPointZ = pz;
      bestTangentX = dx * invLen;
      bestTangentZ = dz * invLen;
      bestNormalX = bestTangentZ;
      bestNormalZ = -bestTangentX;
      bestS = (i + t) * segmentLength;
      bestCenterY = lerp(restCenterlineY[i], restCenterlineY[i + 1], t);
    }
  }

  const offsetX = x - bestPointX;
  const offsetZ = z - bestPointZ;

  return {
    s: bestS / bodyLength,
    lateral: offsetX * bestNormalX + offsetZ * bestNormalZ,
    axial: offsetX * bestTangentX + offsetZ * bestTangentZ,
    centerY: bestCenterY
  };
}

function initVertexRig() {
  const numVerts = meshData.num_vertices;
  const rig = {
    s: new Float32Array(numVerts),
    lateral: new Float32Array(numVerts),
    axial: new Float32Array(numVerts),
    vertical: new Float32Array(numVerts)
  };

  for (let i = 0; i < numVerts; i++) {
    const vertex = meshData.vertices[i];
    const projection = projectPointToCenterline(vertex[0], vertex[2]);
    rig.s[i] = projection.s;
    rig.lateral[i] = projection.lateral;
    rig.axial[i] = projection.axial;
    rig.vertical[i] = vertex[1] - projection.centerY;
  }

  vertexRig = rig;
}

function buildWormColorAttribute() {
  const numVerts = meshData.num_vertices;
  const colors = new Float32Array(numVerts * 3);

  let maxAbsLateral = 1e-6;
  let maxAbsVertical = 1e-6;
  for (let i = 0; i < numVerts; i++) {
    maxAbsLateral = Math.max(maxAbsLateral, Math.abs(vertexRig.lateral[i]));
    maxAbsVertical = Math.max(maxAbsVertical, Math.abs(vertexRig.vertical[i]));
  }

  const dorsalTone = new THREE.Color(0x93715f);
  const ventralTone = new THREE.Color(0xe0c0ad);
  const headTone = new THREE.Color(0xd7a488);
  const tailTone = new THREE.Color(0x7b5f4f);
  const midbodyTone = new THREE.Color(0xbc8d72);
  const rimTone = new THREE.Color(0xf2ddcf);
  const color = new THREE.Color();

  for (let i = 0; i < numVerts; i++) {
    const s = vertexRig.s[i];
    const vertical = clamp(vertexRig.vertical[i] / maxAbsVertical, -1, 1);
    const lateral = clamp(Math.abs(vertexRig.lateral[i]) / maxAbsLateral, 0, 1);
    const dorsalMix = clamp(vertical * 0.5 + 0.5, 0, 1);
    const headMix = Math.exp(-Math.pow(s / 0.17, 2));
    const tailMix = Math.exp(-Math.pow((1 - s) / 0.16, 2));
    const midbodyMix = Math.exp(-Math.pow((s - 0.5) / 0.34, 2));
    const rimMix = Math.pow(lateral, 1.35);

    color.copy(ventralTone).lerp(dorsalTone, dorsalMix);
    color.lerp(headTone, headMix * 0.45);
    color.lerp(tailTone, tailMix * 0.55);
    color.lerp(midbodyTone, midbodyMix * 0.16);
    color.lerp(rimTone, rimMix * 0.12);

    colors[i * 3] = color.r;
    colors[i * 3 + 1] = color.g;
    colors[i * 3 + 2] = color.b;
  }

  return new THREE.BufferAttribute(colors, 3);
}

function initSimulation() {
  const numVerts = meshData.num_vertices;

  currentNeuronVoltages = new Float32Array(neuralTraces.neuron_names.length);
  reducedMotorNet = new ReducedMotorNet(neuralTraces.neuron_names);
  syncReducedControls();

  positions = new Float32Array(numVerts * 3);
  restPositions = new Float32Array(numVerts * 3);

  for (let i = 0; i < numVerts; i++) {
    const vertex = meshData.vertices[i];
    positions[i * 3] = vertex[0];
    positions[i * 3 + 1] = vertex[1];
    positions[i * 3 + 2] = vertex[2];
    restPositions[i * 3] = vertex[0];
    restPositions[i * 3 + 1] = vertex[1];
    restPositions[i * 3 + 2] = vertex[2];
  }

  const restCenterline = buildRestCenterline();
  restCenterlineX = restCenterline.x;
  restCenterlineY = restCenterline.y;
  restCenterlineZ = restCenterline.z;

  centerlineX = new Float32Array(restCenterlineX);
  centerlineY = new Float32Array(restCenterlineY);
  centerlineZ = new Float32Array(restCenterlineZ);
  previousCenterlineX = new Float32Array(restCenterlineX);
  previousCenterlineZ = new Float32Array(restCenterlineZ);
  centerlineAngles = new Float32Array(CENTERLINE_SAMPLES);
  smoothedCurvature = new Float32Array(CENTERLINE_SAMPLES);

  initVertexRig();
  updateMuscleActivations(0);
  updateCenterlineShape(MIN_DELTA_TIME, true);
  updateSkinnedVertices();

  window.positions = positions;
  window.restPositions = restPositions;
  window.currentMuscleActivations = currentMuscleActivations;
  window.meshData = meshData;
}

function updateMuscleActivations(timeSeconds) {
  const frameCount = sampleMuscle.frames.length;
  const frameDuration = sampleMuscle.dt_ms / 1000;
  const frameFloat = ((timeSeconds / frameDuration) % frameCount + frameCount) % frameCount;
  const frame0 = Math.floor(frameFloat);
  const frame1 = (frame0 + 1) % frameCount;
  const alpha = frameFloat - frame0;
  const a = sampleMuscle.frames[frame0];
  const b = sampleMuscle.frames[frame1];

  currentFrameFloat = frameFloat;
  currentFrameIndex = frame0;
  currentReplayTime = frameFloat * frameDuration;
  driveSourceLabel = 'WetNet replay';

  for (let i = 0; i < NUM_MUSCLES; i++) {
    currentMuscleActivations[i] = lerp(a[i], b[i], alpha);
  }

  for (let i = 0; i < currentNeuronVoltages.length; i++) {
    const values = neuralTraces.motor_neuron_voltages[i];
    currentNeuronVoltages[i] = lerp(values[frame0], values[frame1], alpha);
  }
}

function stepReducedMotorNet(deltaTime) {
  reducedMotorNet.step(deltaTime);
  const outputs = reducedMotorNet.getOutputs();

  currentMuscleActivations.set(outputs.muscles);
  currentNeuronVoltages.set(outputs.neuronVoltages);
  currentFrameFloat = outputs.stepCount;
  currentFrameIndex = outputs.stepCount;
  currentReplayTime = outputs.time;
  driveSourceLabel = 'Reduced motor net';
}

function smoothSegments(values) {
  const current = new Float32Array(values);
  const next = new Float32Array(values.length);
  const passes = 1 + Math.round(bodyStiffness * 2);

  for (let pass = 0; pass < passes; pass++) {
    for (let i = 0; i < current.length; i++) {
      const prev = current[Math.max(0, i - 1)];
      const value = current[i];
      const upcoming = current[Math.min(current.length - 1, i + 1)];
      next[i] = (prev + value * (1.8 + bodyStiffness) + upcoming) / (3.8 + bodyStiffness);
    }
    current.set(next);
  }

  return current;
}

function sampleArrayLinear(values, u) {
  const clamped = clamp(u, 0, values.length - 1);
  const i0 = Math.floor(clamped);
  const i1 = Math.min(values.length - 1, i0 + 1);
  const t = clamped - i0;
  return lerp(values[i0], values[i1], t);
}

function updateCenterlineShape(deltaTime, forceInstant = false) {
  const segmentDrive = new Float32Array(MUSCLE_SEGMENTS);
  let driveBias = 0;
  let dorsalSum = 0;
  let ventralSum = 0;
  let centroid = 0;
  let centroidWeight = 0;

  for (let seg = 0; seg < MUSCLE_SEGMENTS; seg++) {
    const dorsal = 0.5 * (currentMuscleActivations[seg] + currentMuscleActivations[seg + 48]);
    const ventral = 0.5 * (currentMuscleActivations[seg + 24] + currentMuscleActivations[seg + 72]);
    dorsalSegments[seg] = dorsal;
    ventralSegments[seg] = ventral;
    segmentDrive[seg] = ventral - dorsal;
    driveBias += segmentDrive[seg];
    dorsalSum += dorsal;
    ventralSum += ventral;
    const total = dorsal + ventral;
    centroid += seg * total;
    centroidWeight += total;
  }

  driveBias /= MUSCLE_SEGMENTS;
  for (let seg = 0; seg < MUSCLE_SEGMENTS; seg++) {
    segmentDrive[seg] -= driveBias;
  }

  currentActivationCentroid = centroidWeight > 1e-5 ? centroid / centroidWeight : 11.5;
  if (dorsalSum > ventralSum + 0.08) {
    currentDominantSide = 'Dorsal lead';
  } else if (ventralSum > dorsalSum + 0.08) {
    currentDominantSide = 'Ventral lead';
  } else {
    currentDominantSide = 'Balanced';
  }

  const smoothedSegments = smoothSegments(segmentDrive);
  const curvatureGain = 6.5 * muscleGain / Math.max(0.35, bodyStiffness);
  const response = forceInstant ? 1 : 1 - Math.pow(hydroDamping, deltaTime * 60);

  for (let i = 0; i < CENTERLINE_SAMPLES; i++) {
    const bodyU = i / (CENTERLINE_SAMPLES - 1);
    const segmentU = bodyU * (MUSCLE_SEGMENTS - 1);
    const taper = 0.25 + 0.75 * Math.sin(Math.PI * bodyU);
    const target = sampleArrayLinear(smoothedSegments, segmentU) * curvatureGain * taper;
    smoothedCurvature[i] = lerp(smoothedCurvature[i], target, response);
  }

  const ds = bodyLength / (CENTERLINE_SAMPLES - 1);
  centerlineAngles[0] = 0;
  for (let i = 1; i < CENTERLINE_SAMPLES; i++) {
    const curvatureMid = 0.5 * (smoothedCurvature[i - 1] + smoothedCurvature[i]);
    centerlineAngles[i] = centerlineAngles[i - 1] + curvatureMid * ds;
  }

  let meanAngle = 0;
  for (let i = 0; i < CENTERLINE_SAMPLES; i++) {
    meanAngle += centerlineAngles[i];
  }
  meanAngle /= CENTERLINE_SAMPLES;
  for (let i = 0; i < CENTERLINE_SAMPLES; i++) {
    centerlineAngles[i] -= meanAngle;
  }

  centerlineX[0] = 0;
  centerlineZ[0] = 0;
  centerlineY[0] = restCenterlineY[0];

  for (let i = 1; i < CENTERLINE_SAMPLES; i++) {
    const heading = 0.5 * (centerlineAngles[i - 1] + centerlineAngles[i]);
    centerlineX[i] = centerlineX[i - 1] + Math.sin(heading) * ds;
    centerlineZ[i] = centerlineZ[i - 1] + Math.cos(heading) * ds;
    centerlineY[i] = restCenterlineY[i];
  }

  let centerX = 0;
  let centerZ = 0;
  for (let i = 0; i < CENTERLINE_SAMPLES; i++) {
    centerX += centerlineX[i];
    centerZ += centerlineZ[i];
  }
  centerX /= CENTERLINE_SAMPLES;
  centerZ /= CENTERLINE_SAMPLES;

  for (let i = 0; i < CENTERLINE_SAMPLES; i++) {
    centerlineX[i] -= centerX;
    centerlineZ[i] -= centerZ;
  }
}

function computeDragTotals(vx, vz, omega, includeShapeVelocity) {
  let forceX = 0;
  let forceZ = 0;
  let torque = 0;

  const xiParallel = 1;
  const xiPerpendicular = fluidAnisotropy;

  for (let i = 0; i < CENTERLINE_SAMPLES - 1; i++) {
    const x0 = centerlineX[i];
    const z0 = centerlineZ[i];
    const x1 = centerlineX[i + 1];
    const z1 = centerlineZ[i + 1];
    const dx = x1 - x0;
    const dz = z1 - z0;
    const ds = Math.hypot(dx, dz);
    if (ds < 1e-8) {
      continue;
    }

    const tx = dx / ds;
    const tz = dz / ds;
    const nx = tz;
    const nz = -tx;

    const mx = 0.5 * (x0 + x1);
    const mz = 0.5 * (z0 + z1);
    const shapeVXMid = includeShapeVelocity ? 0.5 * (shapeVelocityX[i] + shapeVelocityX[i + 1]) : 0;
    const shapeVZMid = includeShapeVelocity ? 0.5 * (shapeVelocityZ[i] + shapeVelocityZ[i + 1]) : 0;
    const totalVX = vx + omega * mz + shapeVXMid;
    const totalVZ = vz - omega * mx + shapeVZMid;

    const vParallel = totalVX * tx + totalVZ * tz;
    const vPerpendicular = totalVX * nx + totalVZ * nz;

    const dragX = -(xiParallel * vParallel * tx + xiPerpendicular * vPerpendicular * nx) * ds;
    const dragZ = -(xiParallel * vParallel * tz + xiPerpendicular * vPerpendicular * nz) * ds;

    forceX += dragX;
    forceZ += dragZ;
    torque += mx * dragZ - mz * dragX;
  }

  return { forceX, forceZ, torque };
}

function solve3x3(matrix, vector) {
  const a00 = matrix[0][0];
  const a01 = matrix[0][1];
  const a02 = matrix[0][2];
  const a10 = matrix[1][0];
  const a11 = matrix[1][1];
  const a12 = matrix[1][2];
  const a20 = matrix[2][0];
  const a21 = matrix[2][1];
  const a22 = matrix[2][2];

  const det =
    a00 * (a11 * a22 - a12 * a21) -
    a01 * (a10 * a22 - a12 * a20) +
    a02 * (a10 * a21 - a11 * a20);

  if (Math.abs(det) < 1e-10) {
    return null;
  }

  const invDet = 1 / det;
  const inv = [
    [
      (a11 * a22 - a12 * a21) * invDet,
      (a02 * a21 - a01 * a22) * invDet,
      (a01 * a12 - a02 * a11) * invDet
    ],
    [
      (a12 * a20 - a10 * a22) * invDet,
      (a00 * a22 - a02 * a20) * invDet,
      (a02 * a10 - a00 * a12) * invDet
    ],
    [
      (a10 * a21 - a11 * a20) * invDet,
      (a01 * a20 - a00 * a21) * invDet,
      (a00 * a11 - a01 * a10) * invDet
    ]
  ];

  return [
    inv[0][0] * vector[0] + inv[0][1] * vector[1] + inv[0][2] * vector[2],
    inv[1][0] * vector[0] + inv[1][1] * vector[1] + inv[1][2] * vector[2],
    inv[2][0] * vector[0] + inv[2][1] * vector[1] + inv[2][2] * vector[2]
  ];
}

function updateHydrodynamics(deltaTime) {
  for (let i = 0; i < CENTERLINE_SAMPLES; i++) {
    shapeVelocityX[i] = (centerlineX[i] - previousCenterlineX[i]) / deltaTime;
    shapeVelocityZ[i] = (centerlineZ[i] - previousCenterlineZ[i]) / deltaTime;
  }

  const basisX = computeDragTotals(1, 0, 0, false);
  const basisZ = computeDragTotals(0, 1, 0, false);
  const basisOmega = computeDragTotals(0, 0, 1, false);
  const shapeDrag = computeDragTotals(0, 0, 0, true);

  const matrix = [
    [basisX.forceX, basisZ.forceX, basisOmega.forceX],
    [basisX.forceZ, basisZ.forceZ, basisOmega.forceZ],
    [basisX.torque, basisZ.torque, basisOmega.torque]
  ];
  const rhs = [-shapeDrag.forceX, -shapeDrag.forceZ, -shapeDrag.torque];
  const solution = solve3x3(matrix, rhs);

  if (!solution) {
    bodyVelocity.x *= 0.9;
    bodyVelocity.z *= 0.9;
    bodyVelocity.omega *= 0.9;
    return;
  }

  const blend = 1 - Math.pow(0.25, deltaTime * 8);
  bodyVelocity.x = lerp(bodyVelocity.x, solution[0], blend);
  bodyVelocity.z = lerp(bodyVelocity.z, solution[1], blend);
  bodyVelocity.omega = lerp(bodyVelocity.omega, solution[2], blend);

  bodyPose.yaw += bodyVelocity.omega * deltaTime;

  const cosYaw = Math.cos(bodyPose.yaw);
  const sinYaw = Math.sin(bodyPose.yaw);
  const worldVX = cosYaw * bodyVelocity.x + sinYaw * bodyVelocity.z;
  const worldVZ = -sinYaw * bodyVelocity.x + cosYaw * bodyVelocity.z;

  bodyPose.x += worldVX * deltaTime;
  bodyPose.z += worldVZ * deltaTime;
  currentDriftSpeed = Math.hypot(worldVX, worldVZ) / bodyLength;
}

function sampleCenterlinePoint(sampleU) {
  const scaled = clamp(sampleU, 0, 1) * (CENTERLINE_SAMPLES - 1);
  const i0 = Math.floor(scaled);
  const i1 = Math.min(CENTERLINE_SAMPLES - 1, i0 + 1);
  const t = scaled - i0;
  return {
    x: lerp(centerlineX[i0], centerlineX[i1], t),
    y: lerp(centerlineY[i0], centerlineY[i1], t),
    z: lerp(centerlineZ[i0], centerlineZ[i1], t),
    angle: lerp(centerlineAngles[i0], centerlineAngles[i1], t)
  };
}

function updateSkinnedVertices() {
  const cosYaw = Math.cos(bodyPose.yaw);
  const sinYaw = Math.sin(bodyPose.yaw);

  for (let i = 0; i < meshData.num_vertices; i++) {
    const center = sampleCenterlinePoint(vertexRig.s[i]);
    const tangentX = Math.sin(center.angle);
    const tangentZ = Math.cos(center.angle);
    const normalX = tangentZ;
    const normalZ = -tangentX;

    const localX = center.x + vertexRig.lateral[i] * normalX + vertexRig.axial[i] * tangentX;
    const localY = center.y + vertexRig.vertical[i];
    const localZ = center.z + vertexRig.lateral[i] * normalZ + vertexRig.axial[i] * tangentZ;

    positions[i * 3] = bodyPose.x + cosYaw * localX + sinYaw * localZ;
    positions[i * 3 + 1] = localY;
    positions[i * 3 + 2] = bodyPose.z - sinYaw * localX + cosYaw * localZ;
  }
}

function advanceSimulation(deltaTime) {
  const step = clamp(deltaTime, MIN_DELTA_TIME, MAX_DELTA_TIME);
  simTime += step;

  if (modelMode === 'replay') {
    driveTime += step * playbackSpeed;
    updateMuscleActivations(driveTime);
  } else {
    reducedSolverAccumulator += step * playbackSpeed;
    while (reducedSolverAccumulator >= REDUCED_SOLVER_STEP) {
      stepReducedMotorNet(REDUCED_SOLVER_STEP);
      reducedSolverAccumulator -= REDUCED_SOLVER_STEP;
    }
    if (reducedSolverAccumulator > 1e-6) {
      stepReducedMotorNet(reducedSolverAccumulator);
      reducedSolverAccumulator = 0;
    }
  }

  updateCenterlineShape(step);
  updateHydrodynamics(step);
  updateSkinnedVertices();

  previousCenterlineX.set(centerlineX);
  previousCenterlineZ.set(centerlineZ);
}

function initThreeJS() {
  const container = document.getElementById('canvas-container');
  const width = container.clientWidth;
  const height = container.clientHeight;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x090807);

  camera = new THREE.PerspectiveCamera(46, width / height, 0.001, 10);
  camera.position.set(0.18, 0.6, 0.22);
  camera.lookAt(0, 0, 0);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  container.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.enableZoom = true;
  controls.zoomSpeed = 0.9;
  controls.minDistance = 0.35;
  controls.maxDistance = 1.4;
  controls.target.set(0, -0.005, 0);
  renderer.domElement.addEventListener('wheel', () => {
    lastUserZoomAtMs = performance.now();
  }, { passive: true });

  const ambientLight = new THREE.AmbientLight(0x5f564a, 0.95);
  scene.add(ambientLight);

  const directionalLight = new THREE.DirectionalLight(0xfff2e3, 1.25);
  directionalLight.position.set(0.18, 0.34, 0.12);
  scene.add(directionalLight);

  const fillLight = new THREE.DirectionalLight(0xbec7b5, 0.24);
  fillLight.position.set(-0.16, 0.08, -0.18);
  scene.add(fillLight);

  const gridHelper = new THREE.GridHelper(0.55, 44, 0x312a23, 0x171411);
  gridHelper.position.y = -0.02;
  scene.add(gridHelper);

  const axesHelper = new THREE.AxesHelper(0.05);
  scene.add(axesHelper);

  createWormMesh();

  window.camera = camera;
  window.controls = controls;

  window.addEventListener('resize', () => {
    const w = container.clientWidth;
    const h = container.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  });
}

function createWormMesh() {
  wormGeometry = new THREE.BufferGeometry();

  const positionArray = new Float32Array(meshData.num_vertices * 3);
  positionArray.set(positions);
  wormGeometry.setAttribute('position', new THREE.BufferAttribute(positionArray, 3));
  wormGeometry.setAttribute('color', buildWormColorAttribute());

  const indices = [];
  for (const tri of meshData.surface_triangles) {
    indices.push(tri[0], tri[1], tri[2]);
  }
  wormGeometry.setIndex(indices);
  wormGeometry.computeVertexNormals();

  const material = new THREE.MeshPhysicalMaterial({
    color: 0xf4d7c7,
    vertexColors: true,
    roughness: 0.88,
    metalness: 0,
    clearcoat: 0.18,
    clearcoatRoughness: 0.72,
    sheen: 0.3,
    sheenColor: 0xf6d2c0,
    emissive: 0x20140f,
    emissiveIntensity: 0.1,
    side: THREE.DoubleSide
  });

  wormMesh = new THREE.Mesh(wormGeometry, material);
  // The worm vertices are rewritten directly in geometry space each frame,
  // so default frustum culling can drop it using stale bounds.
  wormMesh.frustumCulled = false;
  scene.add(wormMesh);

  const wireframeMaterial = new THREE.MeshBasicMaterial({
    color: 0x5f4638,
    wireframe: true,
    transparent: true,
    opacity: 0.035
  });
  const wireframeMesh = new THREE.Mesh(wormGeometry, wireframeMaterial);
  wireframeMesh.frustumCulled = false;
  wormMesh.add(wireframeMesh);
}

function updateWormMesh() {
  const positionAttr = wormGeometry.getAttribute('position');
  positionAttr.array.set(positions);
  positionAttr.needsUpdate = true;
  wormGeometry.computeVertexNormals();
}

function initUI() {
  uiRefs = {
    modeButtons: [...document.querySelectorAll('[data-model-mode]')],
    forwardDrive: document.getElementById('forward-drive'),
    forwardDriveValue: document.getElementById('forward-drive-value'),
    turnBias: document.getElementById('turn-bias'),
    turnBiasValue: document.getElementById('turn-bias-value'),
    wavePeriod: document.getElementById('wave-period'),
    wavePeriodValue: document.getElementById('wave-period-value')
  };

  const strip = document.getElementById('muscle-strip');
  strip.innerHTML = '';
  muscleCells = { dorsal: [], ventral: [] };

  for (let row = 0; row < 2; row++) {
    for (let seg = 0; seg < MUSCLE_SEGMENTS; seg++) {
      const cell = document.createElement('div');
      cell.className = 'muscle-cell';
      cell.title = `${row === 0 ? 'Dorsal' : 'Ventral'} segment ${seg}`;
      strip.appendChild(cell);
      if (row === 0) {
        muscleCells.dorsal.push(cell);
      } else {
        muscleCells.ventral.push(cell);
      }
    }
  }

  const neuronList = document.getElementById('neuron-list');
  neuronList.innerHTML = '';
  neuronRows = [];
  for (let i = 0; i < 5; i++) {
    const row = document.createElement('div');
    row.className = 'neuron-row';

    const name = document.createElement('span');
    name.className = 'neuron-name';
    name.textContent = '--';

    const bar = document.createElement('div');
    bar.className = 'neuron-bar';

    const fill = document.createElement('div');
    fill.className = 'neuron-fill';
    bar.appendChild(fill);

    const value = document.createElement('span');
    value.className = 'neuron-value';
    value.textContent = '--';

    row.append(name, bar, value);
    neuronList.appendChild(row);
    neuronRows.push({ name, fill, value });
  }

  document.getElementById('info-vertices').textContent = meshData.num_vertices;
  document.getElementById('info-tets').textContent = meshData.num_tetrahedra;
  document.getElementById('info-tris').textContent = meshData.surface_triangles.length;

  for (const button of uiRefs.modeButtons) {
    button.addEventListener('click', () => {
      setModelMode(button.dataset.modelMode);
    });
  }

  uiRefs.forwardDrive.addEventListener('input', (event) => {
    reducedMotorNet.setParams({ forwardDrive: Number(event.target.value) });
    syncReducedControls();
  });

  uiRefs.turnBias.addEventListener('input', (event) => {
    reducedMotorNet.setParams({ turnBias: Number(event.target.value) });
    syncReducedControls();
  });

  uiRefs.wavePeriod.addEventListener('input', (event) => {
    reducedMotorNet.setParams({ wavePeriod: Number(event.target.value) });
    syncReducedControls();
  });

  syncReducedControls();
  applyReducedControlState();
  updateSourcePanel();
}

function colorizeMuscleCell(cell, intensity, hue) {
  const normalized = clamp(intensity / 0.8, 0, 1);
  const saturation = 18 + normalized * 72;
  const lightness = 11 + normalized * 55;
  cell.style.backgroundColor = `hsl(${hue}, ${saturation}%, ${lightness}%)`;
}

function updateNeuronSnapshot() {
  const ranked = [];

  for (let i = 0; i < neuralTraces.neuron_names.length; i++) {
    ranked.push({
      name: neuralTraces.neuron_names[i],
      value: currentNeuronVoltages[i]
    });
  }

  ranked.sort((a, b) => b.value - a.value);

  for (let i = 0; i < neuronRows.length; i++) {
    const row = neuronRows[i];
    const neuron = ranked[i];
    const normalized = clamp((neuron.value + 80) / 100, 0, 1);
    row.name.textContent = neuron.name;
    row.fill.style.width = `${(normalized * 100).toFixed(1)}%`;
    row.value.textContent = `${neuron.value >= 0 ? '+' : ''}${neuron.value.toFixed(1)} mV`;
  }
}

function updateDashboard() {
  const loopDuration = sampleMuscle.frames.length * (sampleMuscle.dt_ms / 1000);
  const replayMode = modelMode === 'replay';

  document.getElementById('live-frame').textContent = replayMode
    ? `${currentFrameIndex + 1} / ${sampleMuscle.frames.length}`
    : `${currentFrameIndex.toLocaleString()} steps`;
  document.getElementById('sim-time').textContent = replayMode
    ? `${currentReplayTime.toFixed(2)} / ${loopDuration.toFixed(1)} s`
    : `${currentReplayTime.toFixed(2)} s`;
  document.getElementById('sim-fps').textContent = fps;
  document.getElementById('live-speed').textContent = `${currentDriftSpeed.toFixed(2)} body/s`;
  document.getElementById('live-side').textContent = currentDominantSide;
  document.getElementById('live-centroid').textContent = `seg ${currentActivationCentroid.toFixed(1)}`;
  document.getElementById('drive-source').textContent = driveSourceLabel;

  for (let seg = 0; seg < MUSCLE_SEGMENTS; seg++) {
    colorizeMuscleCell(muscleCells.dorsal[seg], dorsalSegments[seg], 194);
    colorizeMuscleCell(muscleCells.ventral[seg], ventralSegments[seg], 24);
  }

  updateNeuronSnapshot();
}

function measureWormFrame() {
  const numVerts = meshData.num_vertices;
  let centerX = 0;
  let centerY = 0;
  let centerZ = 0;

  for (let i = 0; i < numVerts; i++) {
    const x = positions[i * 3];
    const y = positions[i * 3 + 1];
    const z = positions[i * 3 + 2];
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
      return {
        centerX: NaN,
        centerY: NaN,
        centerZ: NaN,
        wormRadius: NaN,
        valid: false
      };
    }
    centerX += x;
    centerY += y;
    centerZ += z;
  }

  centerX /= numVerts;
  centerY /= numVerts;
  centerZ /= numVerts;

  let maxRadiusSq = 0;
  for (let i = 0; i < numVerts; i++) {
    const dxVertex = positions[i * 3] - centerX;
    const dyVertex = positions[i * 3 + 1] - centerY;
    const dzVertex = positions[i * 3 + 2] - centerZ;
    const radiusSq = dxVertex * dxVertex + dyVertex * dyVertex + dzVertex * dzVertex;
    if (radiusSq > maxRadiusSq) {
      maxRadiusSq = radiusSq;
    }
  }

  return {
    centerX,
    centerY,
    centerZ,
    wormRadius: Math.sqrt(maxRadiusSq),
    valid: true
  };
}

function updateCameraTarget() {
  const { centerX, centerY, centerZ, wormRadius, valid } = measureWormFrame();

  if (!valid) {
    return;
  }

  const dx = centerX - controls.target.x;
  const dy = centerY - controls.target.y;
  const dz = centerZ - controls.target.z;
  const planarOffset = Math.hypot(dx, dz);
  const projected = new THREE.Vector3(centerX, centerY, centerZ).project(camera);
  const offscreen =
    Math.abs(projected.x) > 0.82 ||
    Math.abs(projected.y) > 0.82 ||
    projected.z < -1 ||
    projected.z > 1;
  let smoothing = modelMode === 'reduced' ? 0.02 : 0.01;

  if (offscreen) {
    smoothing = 1;
  } else if (planarOffset > bodyLength * 0.45) {
    smoothing = 0.24;
  } else if (planarOffset > bodyLength * 0.24) {
    smoothing = 0.09;
  } else if (planarOffset > bodyLength * 0.18) {
    smoothing = 0.05;
  }

  const targetShiftX = dx * smoothing;
  const targetShiftY = dy * (offscreen ? 0.2 : 0.04);
  const targetShiftZ = dz * smoothing;

  controls.target.x += targetShiftX;
  controls.target.y += targetShiftY;
  controls.target.z += targetShiftZ;

  // Keep the orbit radius stable while recentering. Moving only the target
  // makes the worm appear to shrink because the camera is left behind.
  camera.position.x += targetShiftX;
  camera.position.y += targetShiftY;
  camera.position.z += targetShiftZ;

  const offset = camera.position.clone().sub(controls.target);
  const currentDistance = offset.length();
  const desiredDistance = clamp(wormRadius * 2.25, controls.minDistance, controls.maxDistance);

  const msSinceUserZoom = performance.now() - lastUserZoomAtMs;
  if (msSinceUserZoom < 2500) {
    return;
  }

  const zoomBlend = offscreen ? 0.2 : currentDistance > desiredDistance ? 0.08 : 0.03;
  const nextDistance = lerp(currentDistance, desiredDistance, zoomBlend);

  if (Math.abs(nextDistance - currentDistance) < 1e-4 || currentDistance < 1e-6) {
    return;
  }

  offset.setLength(nextDistance);
  camera.position.copy(controls.target).add(offset);
}

function animate(time) {
  requestAnimationFrame(animate);

  const deltaTime = lastFrameTime > 0 ? (time - lastFrameTime) / 1000 : 1 / 60;
  lastFrameTime = time;

  frameCount++;
  if (time - lastTime >= 1000) {
    fps = frameCount;
    frameCount = 0;
    lastTime = time;
  }

  advanceSimulation(deltaTime);
  updateWormMesh();
  updateDashboard();
  updateCameraTarget();

  controls.update();
  renderer.render(scene, camera);
}

async function main() {
  try {
    await loadAllData();

    updateLoading('Rigging the worm mesh...');
    initSimulation();

    updateLoading('Initializing controls...');
    initUI();

    document.getElementById('loading').style.display = 'none';
    document.getElementById('container').style.display = 'flex';

    initThreeJS();
    requestAnimationFrame(animate);
  } catch (error) {
    console.error('Initialization failed:', error);
    document.getElementById('loading-text').textContent = 'Initialization failed';
    document.getElementById('loading-progress').textContent = error.message;
  }
}

main();
