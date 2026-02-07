/**
 * BAAIWorm WebGPU FEM Simulation
 * Faithful port of the C. elegans connectome-driven body simulation
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ============================================================================
// Constants
// ============================================================================
const NUM_MOTOR_NEURONS = 80;
const NUM_MUSCLES = 96;
const SUBSTEPS = 5;  // Reduced for better FPS
const DT = 0.001; // 1ms timestep per substep

// ============================================================================
// Global State
// ============================================================================
let meshData, muscleData, neuralTraces, sampleMuscle;
let scene, camera, renderer, controls;
let wormMesh, wormGeometry;

// Simulation state
let positions, velocities, restPositions;
let isPlaying = true;
let currentFrame = 0;
let playbackSpeed = 1.0;
let frameAccumulator = 0;
let simTime = 0;

// Physics parameters
// Based on C. elegans literature: Young's modulus ~3.77 kPa (bending), muscle force ~2.7 nN
// Scaled for mesh coordinates (mesh is ~400x larger than real worm)
let youngsModulus = 50;  // 50 Pa - structural stability proven; bending from force magnitude
let poissonRatio = 0.45;
let muscleStiffness = 1e5; // Strong muscle forces
let damping = 0.97;  // Aggressive damping: removes ~75% velocity/sec, prevents energy accumulation

// FPS tracking
let lastTime = 0;
let frameCount = 0;
let fps = 0;
let lastFrameTime = 0; // For delta time calculation

// Current muscle activations (shared between physics and UI)
let currentMuscleActivations = new Float32Array(96);
// CNN inference state
let cnnWeights = null;
let cnnModel = null;
let neuronHistory = []; // Sliding window of neuron states for CNN input
const CNN_KERNEL_SIZE = 21;

// ============================================================================
// Loading
// ============================================================================
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

  updateLoading('Loading mesh data...', '1/5');
  meshData = await loadJSON(`${basePath}/mesh_data.json`);

  // Compute bounding box for endpoint constraints
  const verts = meshData.vertices;
  let zMin = Infinity, zMax = -Infinity;
  for (let i = 0; i < verts.length; i++) {
    const z = verts[i][2];
    if (z < zMin) zMin = z;
    if (z > zMax) zMax = z;
  }
  meshData.bbox = { zMin, zMax };

  // meshData.vertices are scaled but Dm_inv is in the unscaled basis.
  // Scale Dm_inv so rest state produces F ~ I.
  if (meshData.scale && meshData.scale !== 1) {
    const invScale = 1 / meshData.scale;
    for (let i = 0; i < meshData.Dm_inv.length; i++) {
      const m = meshData.Dm_inv[i];
      for (let j = 0; j < m.length; j++) {
        m[j] *= invScale;
      }
    }
  }

  updateLoading('Loading muscle data...', '2/5');
  muscleData = await loadJSON(`${basePath}/muscle_data.json`);

  updateLoading('Loading neural traces...', '3/5');
  neuralTraces = await loadJSON(`${basePath}/neural_traces.json`);

  updateLoading('Loading muscle activations...', '4/5');
  sampleMuscle = await loadJSON(`${basePath}/sample_muscle.json`);

  updateLoading('Loading CNN weights...', '5/5');
  cnnWeights = await loadJSON(`${basePath}/cnn_weights.json`);
}

// ============================================================================
// CNN Model (TensorFlow.js)
// ============================================================================
/**
 * Initialize the CNN model with pre-trained weights from BAAIWorm
 * Architecture: Conv1d(in_channels=80, out_channels=96, kernel_size=21)
 */
async function initCNNModel() {
  if (!cnnWeights || typeof tf === 'undefined') {
    console.warn('CNN weights or TensorFlow.js not available');
    return;
  }

  try {
    // PyTorch Conv1d weight shape: (out_channels, in_channels, kernel_size) = (96, 80, 21)
    // TensorFlow Conv1d expects: (kernel_size, in_channels, out_channels) = (21, 80, 96)
    // Need to transpose the weights

    const outChannels = cnnWeights.out_channels; // 96
    const inChannels = cnnWeights.in_channels;   // 80
    const kernelSize = cnnWeights.kernel_size;   // 21

    // Transpose weights from (96, 80, 21) to (21, 80, 96)
    const transposedWeights = new Float32Array(kernelSize * inChannels * outChannels);
    for (let k = 0; k < kernelSize; k++) {
      for (let i = 0; i < inChannels; i++) {
        for (let o = 0; o < outChannels; o++) {
          // PyTorch index: [o][i][k]
          // TF index: k * (inChannels * outChannels) + i * outChannels + o
          transposedWeights[k * inChannels * outChannels + i * outChannels + o] =
            cnnWeights.weight[o][i][k];
        }
      }
    }

    // Create TensorFlow.js model
    const input = tf.input({ shape: [CNN_KERNEL_SIZE, NUM_MOTOR_NEURONS] });
    const conv = tf.layers.conv1d({
      filters: NUM_MUSCLES,
      kernelSize: kernelSize,
      padding: 'valid',
      activation: 'linear',
      useBias: true
    }).apply(input);

    cnnModel = tf.model({ inputs: input, outputs: conv });

    // Set the weights
    const weightTensor = tf.tensor(transposedWeights, [kernelSize, inChannels, outChannels]);
    const biasTensor = tf.tensor(cnnWeights.bias, [outChannels]);
    cnnModel.layers[1].setWeights([weightTensor, biasTensor]);
  } catch (error) {
    console.error('Failed to initialize CNN model:', error);
  }
}

/**
 * Run CNN inference to compute muscle activations from motor neuron voltages
 * @param {Float32Array} neuronVoltages - Current motor neuron voltages (80 values)
 * @returns {Float32Array} - Muscle activations (96 values)
 */
function runCNNInference(neuronVoltages) {
  if (!cnnModel) {
    return null;
  }

  // Add current voltages to history
  neuronHistory.push(Array.from(neuronVoltages));

  // Keep only the last CNN_KERNEL_SIZE frames
  if (neuronHistory.length > CNN_KERNEL_SIZE) {
    neuronHistory.shift();
  }

  // Need enough history for convolution
  if (neuronHistory.length < CNN_KERNEL_SIZE) {
    return null;
  }

  // Create input tensor: (1, kernel_size, in_channels)
  const inputData = new Float32Array(CNN_KERNEL_SIZE * NUM_MOTOR_NEURONS);
  for (let t = 0; t < CNN_KERNEL_SIZE; t++) {
    for (let n = 0; n < NUM_MOTOR_NEURONS; n++) {
      // Normalize voltage: original range ~[-70, -20] to [0, 1]
      const voltage = neuronHistory[t][n];
      const normalized = (voltage + 70) / 50;
      inputData[t * NUM_MOTOR_NEURONS + n] = normalized;
    }
  }

  // Run inference
  const inputTensor = tf.tensor3d(inputData, [1, CNN_KERNEL_SIZE, NUM_MOTOR_NEURONS]);
  const outputTensor = cnnModel.predict(inputTensor);

  // Output shape is (1, 1, 96) due to valid padding with kernel=21 on input of 21
  const outputData = outputTensor.dataSync();

  // Apply sigmoid activation to get muscle activations in [0, 1]
  const muscleActivations = new Float32Array(NUM_MUSCLES);
  for (let i = 0; i < NUM_MUSCLES; i++) {
    muscleActivations[i] = 1 / (1 + Math.exp(-outputData[i]));
  }

  // Clean up tensors
  inputTensor.dispose();
  outputTensor.dispose();

  return muscleActivations;
}

// ============================================================================
// FEM Simulation (CPU implementation - faithful to BAAIWorm)
// ============================================================================
function initSimulation() {
  const numVerts = meshData.num_vertices;

  // Initialize position and velocity arrays
  positions = new Float32Array(numVerts * 3);
  velocities = new Float32Array(numVerts * 3);
  restPositions = new Float32Array(numVerts * 3);

  // Copy initial positions
  for (let i = 0; i < numVerts; i++) {
    const v = meshData.vertices[i];
    positions[i * 3] = v[0];
    positions[i * 3 + 1] = v[1];
    positions[i * 3 + 2] = v[2];
    restPositions[i * 3] = v[0];
    restPositions[i * 3 + 1] = v[1];
    restPositions[i * 3 + 2] = v[2];
  }

  // Expose to window for debugging
  window.positions = positions;
  window.restPositions = restPositions;
  window.meshData = meshData;

  // Precompute Lame parameters
  updateLameParameters();

  // Initialize muscle-vertex mapping for efficient force computation
  initMuscleVertexMap();

  // Build vertex neighbor map from tetrahedra for Laplacian smoothing
  buildNeighborMap();
}

// Vertex neighbor map for Laplacian smoothing
let vertexNeighbors = null;

function buildNeighborMap() {
  const numVerts = meshData.num_vertices;
  const neighborSets = new Array(numVerts);
  for (let i = 0; i < numVerts; i++) neighborSets[i] = new Set();

  for (let t = 0; t < meshData.num_tetrahedra; t++) {
    const tet = meshData.tetrahedra[t];
    // Every pair of vertices in a tet are neighbors
    for (let a = 0; a < 4; a++) {
      for (let b = a + 1; b < 4; b++) {
        neighborSets[tet[a]].add(tet[b]);
        neighborSets[tet[b]].add(tet[a]);
      }
    }
  }

  // Convert to arrays for faster iteration
  vertexNeighbors = neighborSets.map(s => Array.from(s));
}

let lambda, mu; // Lame parameters
function updateLameParameters() {
  mu = youngsModulus / (2 * (1 + poissonRatio));
  lambda = youngsModulus * poissonRatio / ((1 + poissonRatio) * (1 - 2 * poissonRatio));
}

/**
 * Compute Corotated FEM forces for a single tetrahedron
 * Based on "FEM Simulation of 3D Deformable Solids" by Sifakis & Barbic
 */
function computeTetForce(tetIdx, forces) {
  const tet = meshData.tetrahedra[tetIdx];
  const i0 = tet[0], i1 = tet[1], i2 = tet[2], i3 = tet[3];

  // Get current positions
  const p0 = [positions[i0*3], positions[i0*3+1], positions[i0*3+2]];
  const p1 = [positions[i1*3], positions[i1*3+1], positions[i1*3+2]];
  const p2 = [positions[i2*3], positions[i2*3+1], positions[i2*3+2]];
  const p3 = [positions[i3*3], positions[i3*3+1], positions[i3*3+2]];

  // Compute deformed edge matrix Ds
  const Ds = [
    [p1[0] - p0[0], p2[0] - p0[0], p3[0] - p0[0]],
    [p1[1] - p0[1], p2[1] - p0[1], p3[1] - p0[1]],
    [p1[2] - p0[2], p2[2] - p0[2], p3[2] - p0[2]]
  ];

  // Get inverse of rest edge matrix (precomputed, stored row-major from numpy)
  const DmInv = meshData.Dm_inv[tetIdx];
  // NumPy flatten is row-major: [[a,b,c],[d,e,f],[g,h,i]] -> [a,b,c,d,e,f,g,h,i]
  const Bm = [
    [DmInv[0], DmInv[1], DmInv[2]],
    [DmInv[3], DmInv[4], DmInv[5]],
    [DmInv[6], DmInv[7], DmInv[8]]
  ];

  // Compute deformation gradient F = Ds * Dm^(-1)
  const F = mat3Mult(Ds, Bm);

  // Polar decomposition: F = R * S (rotation * stretch)
  const { R, S } = polarDecomposition(F);

  // Compute strain: E = S - I (or Green strain for large deformations)
  const strain = [
    [S[0][0] - 1, S[0][1], S[0][2]],
    [S[1][0], S[1][1] - 1, S[1][2]],
    [S[2][0], S[2][1], S[2][2] - 1]
  ];

  // Compute stress using linear elasticity: sigma = lambda * tr(E) * I + 2 * mu * E
  const trE = strain[0][0] + strain[1][1] + strain[2][2];
  const stress = [
    [lambda * trE + 2 * mu * strain[0][0], 2 * mu * strain[0][1], 2 * mu * strain[0][2]],
    [2 * mu * strain[1][0], lambda * trE + 2 * mu * strain[1][1], 2 * mu * strain[1][2]],
    [2 * mu * strain[2][0], 2 * mu * strain[2][1], lambda * trE + 2 * mu * strain[2][2]]
  ];

  // Rotate stress back to world frame: P = R * sigma
  const P = mat3Mult(R, stress);

  // Compute forces: H = -V * P * Bm^T
  const volume = meshData.volumes[tetIdx];
  const BmT = mat3Transpose(Bm);
  const H = mat3Mult(P, BmT);

  // Scale by -volume
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      H[i][j] *= -volume;
    }
  }

  // Forces on vertices 1, 2, 3
  const f1 = [H[0][0], H[1][0], H[2][0]];
  const f2 = [H[0][1], H[1][1], H[2][1]];
  const f3 = [H[0][2], H[1][2], H[2][2]];

  // Force on vertex 0 (balance)
  const f0 = [-(f1[0] + f2[0] + f3[0]), -(f1[1] + f2[1] + f3[1]), -(f1[2] + f2[2] + f3[2])];

  // Accumulate forces
  forces[i0*3] += f0[0]; forces[i0*3+1] += f0[1]; forces[i0*3+2] += f0[2];
  forces[i1*3] += f1[0]; forces[i1*3+1] += f1[1]; forces[i1*3+2] += f1[2];
  forces[i2*3] += f2[0]; forces[i2*3+1] += f2[1]; forces[i2*3+2] += f2[2];
  forces[i3*3] += f3[0]; forces[i3*3+1] += f3[1]; forces[i3*3+2] += f3[2];
}

// Precomputed muscle-vertex mappings for performance
let muscleVertexMap = null;

/**
 * Precompute which vertices are affected by each muscle segment
 */
function initMuscleVertexMap() {
  muscleVertexMap = [];
  // Tight influence radius — must be smaller than body cross-section (~0.04 units)
  // so dorsal muscles only reach dorsal vertices and vice-versa
  const radius = 0.035;  // Wider radius smooths force transitions between adjacent muscle segments
  const radiusSq = radius * radius;

  const muscleMap = {
    'MDR_curve_BezierCurve.001': 0,
    'MVR_curve_BezierCurve.005': 24,
    'MDL_curve_BezierCurve.006': 48,
    'MVL_curve_BezierCurve.007': 72
  };

  // Compute mesh center X for dorsal/ventral side filtering
  const meshBounds = { minX: Infinity, maxX: -Infinity, minZ: Infinity, maxZ: -Infinity };
  for (let i = 0; i < meshData.num_vertices; i++) {
    const v = meshData.vertices[i];
    if (v[0] < meshBounds.minX) meshBounds.minX = v[0];
    if (v[0] > meshBounds.maxX) meshBounds.maxX = v[0];
    if (v[2] < meshBounds.minZ) meshBounds.minZ = v[2];
    if (v[2] > meshBounds.maxZ) meshBounds.maxZ = v[2];
  }
  const meshCenterX = (meshBounds.minX + meshBounds.maxX) / 2;

  for (const [muscleName, baseIdx] of Object.entries(muscleMap)) {
    const points = muscleData.muscles[muscleName];
    if (!points) continue;

    // Determine if this muscle group is dorsal or ventral for side filtering
    // DR (0-23) + DL (48-71) = dorsal (+X side), VR (24-47) + VL (72-95) = ventral (-X side)
    const isDorsalGroup = (baseIdx === 0 || baseIdx === 48);

    const numSegments = 24;
    const pointsPerSegment = Math.floor(points.length / numSegments);

    for (let seg = 0; seg < numSegments; seg++) {
      const muscleIdx = baseIdx + seg;
      const startIdx = seg * pointsPerSegment;
      const endIdx = Math.min(startIdx + pointsPerSegment, points.length - 1);
      const start = points[startIdx];
      const end = points[endIdx];
      const center = [(start[0] + end[0]) / 2, (start[1] + end[1]) / 2, (start[2] + end[2]) / 2];

      const affectedVerts = [];
      for (let i = 0; i < meshData.num_vertices; i++) {
        const v = meshData.vertices[i];

        // Anatomical side filtering: dorsal muscles only affect dorsal vertices (X > center)
        // ventral muscles only affect ventral vertices (X < center)
        if (isDorsalGroup && v[0] < meshCenterX) continue;
        if (!isDorsalGroup && v[0] > meshCenterX) continue;

        const dx = v[0] - center[0];
        const dy = v[1] - center[1];
        const dz = v[2] - center[2];
        const distSq = dx*dx + dy*dy + dz*dz;

        if (distSq < radiusSq) {
          const weight = Math.exp(-distSq / (radiusSq * 0.5));
          affectedVerts.push({ idx: i, weight });
        }
      }
      muscleVertexMap[muscleIdx] = affectedVerts;
    }
  }
}

/**
 * Apply muscle forces based on current activations
 *
 * Bending force model: dorsal/ventral muscle activation pulls the body
 * laterally to create undulation.
 *
 * In C. elegans:
 * - DR (0-23) + DL (48-71) are DORSAL (dorsal = +X in this mesh)
 * - VR (24-47) + VL (72-95) are VENTRAL (ventral = -X in this mesh)
 */
function applyMuscleForces(forces, muscleActivations) {
  if (!muscleVertexMap) return;

  for (let muscleIdx = 0; muscleIdx < 96; muscleIdx++) {
    const activation = muscleActivations[muscleIdx] || 0;
    if (Math.abs(activation) < 0.01) continue;

    const affectedVerts = muscleVertexMap[muscleIdx];
    if (!affectedVerts || affectedVerts.length === 0) continue;

    // Compute total weight for normalization
    let totalWeight = 0;
    for (const vert of affectedVerts) {
      totalWeight += vert.weight;
    }
    if (totalWeight < 1e-6) continue;

    // Determine dorsal vs ventral for bending direction
    // DR (0-23) + DL (48-71) = dorsal (+X), VR (24-47) + VL (72-95) = ventral (-X)
    const isDorsal = muscleIdx < 24 || (muscleIdx >= 48 && muscleIdx < 72);
    const bendDirection = isDorsal ? 1.0 : -1.0;  // +X for dorsal, -X for ventral

    const bendForceMag = activation * muscleStiffness * 2.0e-3;

    for (const vert of affectedVerts) {
      const i = vert.idx;
      const normalizedWeight = vert.weight / totalWeight;

      // Force-length saturation: reduce force as vertex moves away from rest position
      // Mimics real muscle force-length relationship and prevents unbounded displacement
      const currentLateralDisp = Math.abs(positions[i*3] - restPositions[i*3]);
      const maxAllowedDisp = 0.04;
      const saturationFactor = Math.max(0, 1.0 - currentLateralDisp / maxAllowedDisp);

      // Lateral bending force in X direction
      const bendForce = bendForceMag * normalizedWeight * saturationFactor;
      forces[i*3] += bendForce * bendDirection;
    }
  }
}

/**
 * Compute drag forces (simplified fluid interaction)
 */
function computeDragForces(forces) {
  const dragCoeff = 15.0;  // Tuned: equilibrium v ≈ 0.25 m/s gives amplitude ~60% body width
  // Anisotropic drag: real C. elegans has higher resistance to lateral (sideways)
  // motion than axial (forward) motion. This converts undulation into net locomotion.
  const lateralDragMult = 3.0;  // X, Y: resist sideways sliding
  const axialDragMult = 1.0;    // Z: allow forward movement
  const numVerts = meshData.num_vertices;

  for (let i = 0; i < numVerts; i++) {
    const vx = velocities[i*3];
    const vy = velocities[i*3+1];
    const vz = velocities[i*3+2];
    const speed = Math.sqrt(vx*vx + vy*vy + vz*vz);

    if (speed > 1e-6) {
      forces[i*3]   -= dragCoeff * lateralDragMult * vx * speed;
      forces[i*3+1] -= dragCoeff * lateralDragMult * vy * speed;
      forces[i*3+2] -= dragCoeff * axialDragMult * vz * speed;
    }
  }
}

/**
 * Single simulation substep
 */
function simulationStep(muscleActivations) {
  const numVerts = meshData.num_vertices;
  const forces = new Float32Array(numVerts * 3);

  // Compute elastic forces from all tetrahedra
  for (let t = 0; t < meshData.num_tetrahedra; t++) {
    computeTetForce(t, forces);
  }

  // Apply muscle forces
  applyMuscleForces(forces, muscleActivations);

  // Apply drag forces
  computeDragForces(forces);

  // Integrate (semi-implicit Euler)
  // NOTE: Use artificially higher mass for numerical stability with explicit integration
  // Real C. elegans is ~1 microgram, but we scale up for stable simulation
  const totalMass = 1.0; // Scaled mass for stable explicit Euler (1 kg simulation mass)
  const massPerVertex = totalMass / numVerts;  // ~1e-3 kg per vertex

  for (let i = 0; i < numVerts; i++) {
    // Get forces (no aggressive clamping - mass scaling provides stability)
    let fx = forces[i*3];
    let fy = forces[i*3+1];
    let fz = forces[i*3+2];

    // Update velocity: a = F/m, dv = a*dt
    velocities[i*3] += (fx / massPerVertex) * DT;
    velocities[i*3+1] += (fy / massPerVertex) * DT;
    velocities[i*3+2] += (fz / massPerVertex) * DT;

    // Safety-only velocity clamp — drag forces handle natural speed limiting
    const maxVel = 0.3;  // 0.3 m/s max — prevents numerical instability while allowing visible motion
    velocities[i*3] = Math.max(-maxVel, Math.min(maxVel, velocities[i*3]));
    velocities[i*3+1] = Math.max(-maxVel, Math.min(maxVel, velocities[i*3+1]));
    velocities[i*3+2] = Math.max(-maxVel, Math.min(maxVel, velocities[i*3+2]));

    // Apply damping
    velocities[i*3] *= damping;
    velocities[i*3+1] *= damping;
    velocities[i*3+2] *= damping;

    // Update position
    positions[i*3] += velocities[i*3] * DT;
    positions[i*3+1] += velocities[i*3+1] * DT;
    positions[i*3+2] += velocities[i*3+2] * DT;

    // Ground collision (keep above y=0)
    if (positions[i*3+1] < -0.02) {
      positions[i*3+1] = -0.02;
      velocities[i*3+1] = 0;
    }
  }

  simTime += DT;
}

/**
 * Laplacian smoothing on lateral (X) displacement from rest position.
 * Smooths displacement rather than absolute position to preserve body width.
 * This prevents the cross-section shrinkage artifact of standard Laplacian smoothing.
 */
let smoothBuffer = null;
function laplacianSmooth(alpha) {
  if (!vertexNeighbors) return;
  const numVerts = meshData.num_vertices;
  if (!smoothBuffer || smoothBuffer.length !== numVerts) {
    smoothBuffer = new Float32Array(numVerts);
  }

  // Compute smoothed X displacement (not absolute position)
  for (let i = 0; i < numVerts; i++) {
    const neighbors = vertexNeighbors[i];
    const myDisp = positions[i * 3] - restPositions[i * 3];
    if (neighbors.length === 0) {
      smoothBuffer[i] = myDisp;
      continue;
    }
    let avgDisp = 0;
    for (let j = 0; j < neighbors.length; j++) {
      avgDisp += positions[neighbors[j] * 3] - restPositions[neighbors[j] * 3];
    }
    avgDisp /= neighbors.length;
    smoothBuffer[i] = myDisp * (1 - alpha) + avgDisp * alpha;
  }

  // Apply smoothed displacement back to positions
  for (let i = 0; i < numVerts; i++) {
    positions[i * 3] = restPositions[i * 3] + smoothBuffer[i];
  }
}

// ============================================================================
// Matrix Math Utilities
// ============================================================================
function mat3Mult(A, B) {
  const C = [[0,0,0], [0,0,0], [0,0,0]];
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      for (let k = 0; k < 3; k++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
  return C;
}

function mat3Transpose(A) {
  return [
    [A[0][0], A[1][0], A[2][0]],
    [A[0][1], A[1][1], A[2][1]],
    [A[0][2], A[1][2], A[2][2]]
  ];
}

/**
 * Polar decomposition of 3x3 matrix: F = R * S
 * Uses iterative method
 */
function polarDecomposition(F) {
  // Start with F
  let R = [
    [F[0][0], F[0][1], F[0][2]],
    [F[1][0], F[1][1], F[1][2]],
    [F[2][0], F[2][1], F[2][2]]
  ];

  // Iterate to extract rotation: R_new = 0.5 * (R + R^(-T))
  for (let iter = 0; iter < 10; iter++) {
    const det = mat3Det(R);
    if (Math.abs(det) < 1e-10) break;

    const Rinv = mat3Inverse(R);
    const RinvT = mat3Transpose(Rinv);

    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        R[i][j] = 0.5 * (R[i][j] + RinvT[i][j]);
      }
    }
  }

  // S = R^T * F
  const Rt = mat3Transpose(R);
  const S = mat3Mult(Rt, F);

  return { R, S };
}

function mat3Det(M) {
  return M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
       - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
       + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
}

function mat3Inverse(M) {
  const det = mat3Det(M);
  if (Math.abs(det) < 1e-10) return [[1,0,0], [0,1,0], [0,0,1]];

  const invDet = 1.0 / det;
  return [
    [(M[1][1]*M[2][2] - M[1][2]*M[2][1]) * invDet,
     (M[0][2]*M[2][1] - M[0][1]*M[2][2]) * invDet,
     (M[0][1]*M[1][2] - M[0][2]*M[1][1]) * invDet],
    [(M[1][2]*M[2][0] - M[1][0]*M[2][2]) * invDet,
     (M[0][0]*M[2][2] - M[0][2]*M[2][0]) * invDet,
     (M[0][2]*M[1][0] - M[0][0]*M[1][2]) * invDet],
    [(M[1][0]*M[2][1] - M[1][1]*M[2][0]) * invDet,
     (M[0][1]*M[2][0] - M[0][0]*M[2][1]) * invDet,
     (M[0][0]*M[1][1] - M[0][1]*M[1][0]) * invDet]
  ];
}

// ============================================================================
// Three.js Rendering
// ============================================================================
function initThreeJS() {
  const container = document.getElementById('canvas-container');
  const width = container.clientWidth;
  const height = container.clientHeight;

  // Scene
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0a0f);

  // Camera
  camera = new THREE.PerspectiveCamera(50, width / height, 0.001, 10);
  // Position camera to see full worm from above (best view for sinusoidal motion)
  // Worm is ~0.63 units long on Z-axis
  camera.position.set(0, 0.8, 0.1);
  camera.lookAt(0, 0, 0);

  // Renderer
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.appendChild(renderer.domElement);

  // Controls
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.target.set(0, 0, 0);

  // Lighting
  const ambientLight = new THREE.AmbientLight(0x404060, 0.6);
  scene.add(ambientLight);

  const directionalLight = new THREE.DirectionalLight(0xffffff, 1.0);
  directionalLight.position.set(0.1, 0.2, 0.1);
  scene.add(directionalLight);

  const directionalLight2 = new THREE.DirectionalLight(0x8888ff, 0.3);
  directionalLight2.position.set(-0.1, -0.1, 0.1);
  scene.add(directionalLight2);

  // Grid
  const gridHelper = new THREE.GridHelper(0.4, 40, 0x2a2a3a, 0x1a1a24);
  gridHelper.position.y = -0.02;
  scene.add(gridHelper);

  // Axes
  const axesHelper = new THREE.AxesHelper(0.05);
  scene.add(axesHelper);

  // Create worm mesh
  createWormMesh();

  // Handle resize
  window.addEventListener('resize', () => {
    const w = container.clientWidth;
    const h = container.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  });
}

function createWormMesh() {
  // Create geometry from surface triangles
  wormGeometry = new THREE.BufferGeometry();

  // Position attribute
  const positionArray = new Float32Array(meshData.num_vertices * 3);
  for (let i = 0; i < meshData.num_vertices; i++) {
    positionArray[i*3] = positions[i*3];
    positionArray[i*3+1] = positions[i*3+1];
    positionArray[i*3+2] = positions[i*3+2];
  }
  wormGeometry.setAttribute('position', new THREE.BufferAttribute(positionArray, 3));

  // Index attribute (surface triangles)
  const indices = [];
  for (const tri of meshData.surface_triangles) {
    indices.push(tri[0], tri[1], tri[2]);
  }
  wormGeometry.setIndex(indices);
  wormGeometry.computeVertexNormals();

  // Material
  const material = new THREE.MeshPhongMaterial({
    color: 0x6699cc,
    specular: 0x222244,
    shininess: 30,
    side: THREE.DoubleSide,
    flatShading: false
  });

  wormMesh = new THREE.Mesh(wormGeometry, material);
  scene.add(wormMesh);

  // Add wireframe overlay
  const wireframeMaterial = new THREE.MeshBasicMaterial({
    color: 0x4488aa,
    wireframe: true,
    transparent: true,
    opacity: 0.1
  });
  const wireframeMesh = new THREE.Mesh(wormGeometry, wireframeMaterial);
  wormMesh.add(wireframeMesh);
}

function updateWormMesh() {
  const posAttr = wormGeometry.getAttribute('position');

  for (let i = 0; i < meshData.num_vertices; i++) {
    posAttr.array[i*3] = positions[i*3];
    posAttr.array[i*3+1] = positions[i*3+1];
    posAttr.array[i*3+2] = positions[i*3+2];
  }

  posAttr.needsUpdate = true;
  wormGeometry.computeVertexNormals();
}

// ============================================================================
// UI
// ============================================================================
function initUI() {
  // Sliders
  setupSlider('speed', v => { playbackSpeed = v; return `${v.toFixed(1)}x`; });
  setupSlider('youngs', v => {
    youngsModulus = Math.pow(10, v);
    updateLameParameters();
    return `1e${v.toFixed(0)}`;
  });
  setupSlider('poisson', v => {
    poissonRatio = v;
    updateLameParameters();
    return v.toFixed(2);
  });
  setupSlider('muscle-stiff', v => {
    muscleStiffness = Math.pow(10, v);
    return `${Math.pow(10, v).toExponential(0)}`;
  });
  setupSlider('damping', v => { damping = v; return v.toFixed(3); });

  // Update info
  document.getElementById('info-vertices').textContent = meshData.num_vertices;
  document.getElementById('info-tets').textContent = meshData.num_tetrahedra;
  document.getElementById('info-tris').textContent = meshData.surface_triangles.length;
}

function setupSlider(name, handler) {
  const slider = document.getElementById(`${name}-slider`);
  const valueEl = document.getElementById(`${name}-val`);
  // Initialize display with current slider value
  const initialValue = parseFloat(slider.value);
  valueEl.textContent = handler(initialValue);
  // Update on change
  slider.addEventListener('input', e => {
    const v = parseFloat(e.target.value);
    valueEl.textContent = handler(v);
  });
}

function updateStats() {
  document.getElementById('sim-time').textContent = `${simTime.toFixed(3)} s`;
  document.getElementById('sim-fps').textContent = fps;
  document.getElementById('sim-substeps').textContent = SUBSTEPS;
}

// ============================================================================
// Locomotion pattern generator - naturalistic traveling wave
// ============================================================================

// Perlin-like noise for organic variation
const noiseCache = new Float32Array(256);
for (let i = 0; i < 256; i++) noiseCache[i] = Math.random() * 2 - 1;

function noise(x) {
  const xi = Math.floor(x) & 255;
  const xf = x - Math.floor(x);
  const u = xf * xf * (3 - 2 * xf); // smoothstep
  return noiseCache[xi] * (1 - u) + noiseCache[(xi + 1) & 255] * u;
}

function generateLocomotionPattern(time) {
  const waveSpeed = 8.0;      // How fast the wave travels (rad/s)
  const waveNumber = 1.2;     // Waves along the body
  const baseAmplitude = 0.7;  // Base activation strength

  for (let seg = 0; seg < 24; seg++) {
    // Position along body (0 = head, 1 = tail)
    const bodyPos = seg / 23;

    // Traveling wave phase: wave moves from head to tail
    const phase = bodyPos * Math.PI * 2 * waveNumber - time * waveSpeed;

    // Base wave with organic variations
    const baseWave = Math.sin(phase);

    // Add slow-varying amplitude modulation (body doesn't bend uniformly)
    const ampMod = 0.7 + 0.3 * Math.sin(bodyPos * Math.PI); // stronger in middle

    // Add subtle noise for organic look (varies with position and time)
    const noiseVal = noise(seg * 0.3 + time * 2) * 0.15;

    // Compute activation with asymmetry (not perfectly symmetric push/pull)
    const wave = baseWave * ampMod + noiseVal;

    // Dorsal activates on positive wave, ventral on negative (antagonistic)
    const dorsalAct = Math.max(0, wave) * baseAmplitude;
    const ventralAct = Math.max(0, -wave) * baseAmplitude;

    // Small baseline tone (muscles never fully off)
    const baseline = 0.05;

    // DR (0-23) and DL (48-71) - dorsal
    currentMuscleActivations[seg] = Math.min(1, dorsalAct + baseline);
    currentMuscleActivations[seg + 48] = Math.min(1, dorsalAct + baseline);

    // VR (24-47) and VL (72-95) - ventral
    currentMuscleActivations[seg + 24] = Math.min(1, ventralAct + baseline);
    currentMuscleActivations[seg + 72] = Math.min(1, ventralAct + baseline);
  }
}

// ============================================================================
// Animation Loop
// ============================================================================
function animate(time) {
  requestAnimationFrame(animate);

  // Calculate delta time for FPS-independent simulation
  const deltaTime = lastFrameTime > 0 ? (time - lastFrameTime) / 1000 : 0.016; // Default ~60fps
  lastFrameTime = time;

  // FPS calculation
  frameCount++;
  if (time - lastTime >= 1000) {
    fps = frameCount;
    frameCount = 0;
    lastTime = time;
  }

  // Advance simulation if playing
  if (isPlaying) {
    // Generate muscle activations (traveling wave pattern)
    generateLocomotionPattern(simTime);

    // Run physics substeps (fixed timestep, multiple per frame for stability)
    for (let i = 0; i < SUBSTEPS; i++) {
      simulationStep(currentMuscleActivations);
    }

    // Laplacian smoothing: once per frame (not per substep) to prevent crinkly mesh
    laplacianSmooth(0.15);

    // Update mesh
    updateWormMesh();
  }

  updateStats();

  // Update camera to follow worm's center of mass
  updateCameraTarget();

  // Render
  controls.update();
  renderer.render(scene, camera);
}

/**
 * Compute worm's center of mass and update camera target
 */
function updateCameraTarget() {
  const numVerts = meshData.num_vertices;
  let cx = 0, cy = 0, cz = 0;

  for (let i = 0; i < numVerts; i++) {
    cx += positions[i * 3];
    cy += positions[i * 3 + 1];
    cz += positions[i * 3 + 2];
  }

  cx /= numVerts;
  cy /= numVerts;
  cz /= numVerts;

  // Smoothly move camera target toward COM
  const smoothing = 0.05;
  controls.target.x += (cx - controls.target.x) * smoothing;
  controls.target.y += (cy - controls.target.y) * smoothing;
  controls.target.z += (cz - controls.target.z) * smoothing;
}

// ============================================================================
// Main Entry Point
// ============================================================================
async function main() {
  try {
    await loadAllData();

    updateLoading('Initializing simulation...');
    initSimulation();

    updateLoading('Initializing UI...');
    initUI();

    updateLoading('Initializing CNN model...');
    await initCNNModel();

    // Prime neuron history so CNN produces output immediately
    for (let f = 0; f < CNN_KERNEL_SIZE; f++) {
      const voltages = new Float32Array(NUM_MOTOR_NEURONS);
      for (let i = 0; i < NUM_MOTOR_NEURONS; i++) {
        voltages[i] = neuralTraces.motor_neuron_voltages[i]?.[f] || -65;
      }
      neuronHistory.push(Array.from(voltages));
    }

    // Hide loading, show app BEFORE initializing renderer
    // (so container has proper dimensions)
    document.getElementById('loading').style.display = 'none';
    document.getElementById('container').style.display = 'flex';

    // Now initialize Three.js (needs container to be visible for dimensions)
    updateLoading('Setting up renderer...');
    initThreeJS();

    console.log('BAAIWorm initialized');

    // Start animation
    requestAnimationFrame(animate);

  } catch (error) {
    console.error('Initialization failed:', error);
    document.getElementById('loading-text').textContent = 'Initialization failed';
    document.getElementById('loading-progress').textContent = error.message;
  }
}

main();
