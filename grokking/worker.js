// Web Worker running the Rust WASM SIMD128 1-layer Transformer trainer.
// Keeps live weights, AdamW state, and periodic weight snapshots in worker memory
// so the UI can train smoothly at ~100-180 epochs/sec AND time-travel scrub to any checkpoint.

const P = 59;
const NFREQ = 29;
const FULL_MASK = (1 << NFREQ) - 1;
const SNAPSHOT_EVERY = 50;
const MAX_SNAPSHOTS = 121; // epochs 0..6000 inclusive

let wasm = null;
let wasmLoading = false;
let pendingMsgs = [];
let nParams = 0;
let evalLen = 0;
let probeLen = 0;

let running = false;
let loopTimer = null;
let targetEpoch = 3000;

let config = {
  op: 0,
  trainFrac: 0.35,
  lr: 3e-3,
  wd: 2.2,
  seed: 1,
};

let keepMask = FULL_MASK;
let probeA = 17;
let probeB = 45;
let selectedFreq = 0; // 0 = auto-follow #1 dominant frequency

// Checkpoints for time-travel scrubbing: { epoch, params: Float32Array, evalObj }
let snapshots = [];
let scrubbedIndex = -1; // -1 = live head
let liveParamsBackup = null;
let isTrainFlags = null; // Uint8Array[P*P]

// Throughput tracking
let rateStart = 0;
let rateEpochs = 0;
let epochsPerSec = 0;

function memF32() {
  return new Float32Array(wasm.memory.buffer);
}

function memU8() {
  return new Uint8Array(wasm.memory.buffer);
}

async function loadWasm() {
  const res = await fetch('./grok.wasm');
  const bytes = await res.arrayBuffer();
  const { instance } = await WebAssembly.instantiate(bytes);
  wasm = instance.exports;
  nParams = wasm.get_n_params();
  evalLen = wasm.get_eval_len();
  probeLen = wasm.get_probe_len();
}

function copyParamsOut() {
  const ptr = wasm.get_params_ptr() >> 2;
  return new Float32Array(memF32().subarray(ptr, ptr + nParams));
}

function copyParamsIn(arr) {
  const ptr = wasm.get_params_ptr() >> 2;
  memF32().set(arr, ptr);
}

function readEval() {
  const ptr = wasm.evaluate() >> 2;
  const raw = memF32().subarray(ptr, ptr + evalLen);
  const top5K = [
    Math.round(raw[9]),
    Math.round(raw[10]),
    Math.round(raw[11]),
    Math.round(raw[12]),
    Math.round(raw[13]),
  ];
  const power = new Float32Array(NFREQ);
  for (let i = 0; i < NFREQ; i++) power[i] = raw[14 + i];
  return {
    trLoss: raw[0],
    trAcc: raw[1],
    teLoss: raw[2],
    teAcc: raw[3],
    wnorm: raw[4],
    resLoss: raw[5],
    excLoss: raw[6],
    top5Share: raw[7],
    gini: raw[8],
    top5K,
    power,
  };
}

function readAblated(mask) {
  if ((mask & FULL_MASK) === FULL_MASK) return null;
  const ptr = wasm.eval_ablated(mask) >> 2;
  const raw = memF32().subarray(ptr, ptr + 4);
  return {
    trAcc: raw[0],
    teAcc: raw[1],
    trLoss: raw[2],
    teLoss: raw[3],
  };
}

function readCircle(k) {
  const ptr = wasm.compute_circle(k) >> 2;
  return new Float32Array(memF32().subarray(ptr, ptr + P * 2));
}

function readProbe(a, b, mask) {
  const ptr = wasm.probe_pair(a, b, mask) >> 2;
  const raw = memF32().subarray(ptr, ptr + probeLen);
  const logits = new Float32Array(raw.subarray(0, P));
  const probs = new Float32Array(raw.subarray(P, 2 * P));
  const attn = new Float32Array(raw.subarray(2 * P, 2 * P + 12));
  const baseK = 2 * P + 12;
  const top5K = [
    Math.round(raw[baseK]),
    Math.round(raw[baseK + 1]),
    Math.round(raw[baseK + 2]),
    Math.round(raw[baseK + 3]),
    Math.round(raw[baseK + 4]),
  ];
  const baseW = baseK + 5;
  const waves = [];
  for (let f = 0; f < 5; f++) {
    waves.push(new Float32Array(raw.subarray(baseW + f * P, baseW + (f + 1) * P)));
  }
  const resid = new Float32Array(raw.subarray(baseW + 5 * P, baseW + 6 * P));
  const pairIdx = (a % P) * P + (b % P);
  const inTrain = isTrainFlags ? isTrainFlags[pairIdx] !== 0 : false;
  return { a, b, inTrain, logits, probs, attn, top5K, waves, resid };
}

function inspectCurrent(epoch, ev) {
  const circleK = selectedFreq > 0 ? selectedFreq : ev.top5K[0];
  const circle = readCircle(circleK);
  const ablated = readAblated(keepMask);
  const probe = readProbe(probeA, probeB, keepMask);
  return {
    epoch,
    ev,
    circleK,
    circle,
    ablated,
    probe,
    keepMask,
    scrubbed: scrubbedIndex >= 0,
    liveEpoch: wasm.get_step(),
    targetEpoch,
    epochsPerSec,
    running,
  };
}

function emitInspection(type = 'inspect') {
  if (scrubbedIndex >= 0 && scrubbedIndex < snapshots.length) {
    const snap = snapshots[scrubbedIndex];
    if (!liveParamsBackup) {
      liveParamsBackup = copyParamsOut();
    }
    copyParamsIn(snap.params);
    const ev = snap.ev;
    // Call evaluate once so last_top5 in WASM matches this snapshot for probe_pair
    const evFresh = readEval();
    const payload = inspectCurrent(snap.epoch, evFresh);
    // Restore live params immediately so AdamW state & live weights stay untouched
    copyParamsIn(liveParamsBackup);
    liveParamsBackup = null;
    postMessage({ type, ...payload, ev });
  } else {
    const epoch = wasm.get_step();
    const ev = readEval();
    const payload = inspectCurrent(epoch, ev);
    postMessage({ type, ...payload });
  }
}

function recordSnapshot(epoch, ev) {
  if (snapshots.length >= MAX_SNAPSHOTS) return;
  snapshots.push({
    epoch,
    params: copyParamsOut(),
    ev,
  });
}

function resetModel(newCfg) {
  stopLoop();
  config = { ...config, ...newCfg };
  scrubbedIndex = -1;
  liveParamsBackup = null;
  snapshots = [];
  keepMask = FULL_MASK;
  selectedFreq = 0;
  targetEpoch = 3000;
  epochsPerSec = 0;
  rateStart = 0;
  rateEpochs = 0;

  wasm.init_model(config.op, config.trainFrac, config.lr, config.wd, config.seed);
  const flagPtr = wasm.get_is_train_ptr();
  isTrainFlags = new Uint8Array(memU8().subarray(flagPtr, flagPtr + P * P));

  const ev0 = readEval();
  recordSnapshot(0, ev0);
  const payload = inspectCurrent(0, ev0);
  postMessage({
    type: 'reset',
    config,
    nParams,
    nTrain: wasm.get_n_train(),
    ...payload,
  });
}

function stopLoop() {
  running = false;
  if (loopTimer !== null) {
    clearTimeout(loopTimer);
    loopTimer = null;
  }
}

function runSlice() {
  if (!running) return;
  try {
    const curEp = wasm.get_step();
    if (curEp >= targetEpoch) {
      stopLoop();
      const ev = readEval();
      if (scrubbedIndex >= 0) {
        postMessage({
          type: 'history',
          epoch: curEp,
          ev,
          epochsPerSec,
          running: false,
          liveEpoch: curEp,
          targetEpoch,
        });
      } else {
        const payload = inspectCurrent(curEp, ev);
        postMessage({ type: 'done', ...payload });
      }
      return;
    }

    // Finer early cadence (every 10 epochs up to 200), then 25 epochs per UI update
    const stepBatch = curEp < 200 ? 10 : 25;
    const remaining = targetEpoch - curEp;
    const n = Math.min(stepBatch, remaining);

    const t0 = performance.now();
    wasm.step_n(n);
    const dt = performance.now() - t0;

    if (rateStart === 0) {
      rateStart = t0;
      rateEpochs = n;
    } else {
      rateEpochs += n;
      const elapsed = (performance.now() - rateStart) / 1000;
      if (elapsed > 0.25) {
        const inst = rateEpochs / elapsed;
        epochsPerSec = epochsPerSec === 0 ? inst : epochsPerSec * 0.7 + inst * 0.3;
        rateStart = performance.now();
        rateEpochs = 0;
      }
    }

    const ep = wasm.get_step();
    const ev = readEval();
    if (ep % SNAPSHOT_EVERY === 0 || ep === targetEpoch) {
      recordSnapshot(ep, ev);
    }

    // If user is not scrubbed, send full live inspection; if scrubbed, only append point to history
    if (scrubbedIndex < 0) {
      const payload = inspectCurrent(ep, ev);
      postMessage({ type: 'tick', ...payload });
    } else {
      postMessage({
        type: 'history',
        epoch: ep,
        ev,
        epochsPerSec,
        running,
        liveEpoch: ep,
        targetEpoch,
      });
    }

    // Yield briefly so incoming messages (pause, slider drags, ablations) are processed promptly
    loopTimer = setTimeout(runSlice, dt < 16 ? 4 : 0);
  } catch (err) {
    stopLoop();
    postMessage({ type: 'error', message: err.message || String(err) });
  }
}

function pickRandomTestPair() {
  if (!isTrainFlags) return { a: 17, b: 45 };
  for (let tries = 0; tries < 200; tries++) {
    const a = Math.floor(Math.random() * P);
    const b = Math.floor(Math.random() * P);
    if (isTrainFlags[a * P + b] === 0) return { a, b };
  }
  return { a: 17, b: 45 };
}

function handleMessage(msg) {
  switch (msg.type) {
    case 'reset':
      resetModel(msg.config || {});
      if (msg.autoStart) {
        running = true;
        runSlice();
      }
      break;

    case 'start':
      if (wasm.get_step() >= targetEpoch) {
        targetEpoch = wasm.get_step() + 2000;
      }
      scrubbedIndex = -1;
      if (!running) {
        running = true;
        rateStart = 0;
        rateEpochs = 0;
        runSlice();
      }
      break;

    case 'pause':
      stopLoop();
      emitInspection('paused');
      break;

    case 'setHparams':
      if (typeof msg.lr === 'number') config.lr = msg.lr;
      if (typeof msg.wd === 'number') config.wd = msg.wd;
      wasm.set_hparams(config.lr, config.wd);
      break;

    case 'setKeepMask':
      keepMask = msg.keepMask & FULL_MASK;
      emitInspection('inspect');
      break;

    case 'setSelectedFreq':
      selectedFreq = msg.k | 0;
      emitInspection('inspect');
      break;

    case 'setProbe':
      probeA = ((msg.a | 0) % P + P) % P;
      probeB = ((msg.b | 0) % P + P) % P;
      emitInspection('inspect');
      break;

    case 'randomTestPair': {
      const pair = pickRandomTestPair();
      probeA = pair.a;
      probeB = pair.b;
      emitInspection('inspect');
      break;
    }

    case 'scrubEpoch': {
      if (msg.epoch === null || msg.epoch === undefined || snapshots.length === 0) {
        scrubbedIndex = -1;
        emitInspection('inspect');
        break;
      }
      let bestIdx = 0;
      let bestDist = Infinity;
      for (let i = 0; i < snapshots.length; i++) {
        const d = Math.abs(snapshots[i].epoch - msg.epoch);
        if (d < bestDist) {
          bestDist = d;
          bestIdx = i;
        }
      }
      scrubbedIndex = bestIdx;
      emitInspection('inspect');
      break;
    }
  }
}

self.onmessage = async (ev) => {
  const msg = ev.data;
  try {
    if (msg.type === 'init') {
      if (!wasm && !wasmLoading) {
        wasmLoading = true;
        await loadWasm();
        wasmLoading = false;
      }
      if (!wasm) return;
      resetModel(msg.config || {});
      if (msg.autoStart) {
        running = true;
        runSlice();
      }
      const queued = pendingMsgs.splice(0);
      for (const q of queued) handleMessage(q);
      return;
    }
    if (!wasm) {
      pendingMsgs.push(msg);
      return;
    }
    handleMessage(msg);
  } catch (err) {
    postMessage({ type: 'error', message: err.message || String(err) });
  }
};
