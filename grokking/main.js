import {
  COLORS,
  drawTrainingChart,
  drawSpectrumChart,
  drawCircleChart,
  drawProbeChart,
  drawComparisonChart,
} from './charts.js';

const P = 59;
const NFREQ = 29;
const FULL_MASK = (1 << NFREQ) - 1;

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const worker = new Worker('./worker.js', { type: 'module' });

let state = {
  config: { op: 0, trainFrac: 0.35, lr: 3e-3, wd: 2.2, seed: 1 },
  running: false,
  epoch: 0,
  liveEpoch: 0,
  targetEpoch: 3000,
  scrubbed: false,
  epochsPerSec: 0,
  chartMode: 'acc', // 'acc' | 'loss'
  history: [],
  wdChanges: [],
  ev: null,
  circleK: 1,
  manualFreq: 0,
  circle: null,
  ablated: null,
  probe: null,
  keepMask: FULL_MASK,
  hoveredK: 0,
  expData: null,
};

// DOM refs
const btnToggle = $('#btn-toggle');
const btnReset = $('#btn-reset');
const btnSeed = $('#btn-seed');
const btnBackLive = $('#btn-back-live');
const sliderFrac = $('#train-frac');
const outFrac = $('#train-frac-out');
const sliderWd = $('#wd');
const outWd = $('#wd-out');
const phasePill = $('#phase-pill');
const ratePill = $('#rate-pill');
const trainLegend = $('#train-legend');

const canvasTrain = $('#chart-train');
const canvasSpectrum = $('#chart-spectrum');
const canvasCircle = $('#chart-circle');
const canvasProbe = $('#chart-probe');
const canvasCompare = $('#chart-compare');
const spectrumTip = $('#spectrum-tip');

const roEpoch = $('#ro-epoch');
const roAcc = $('#ro-acc');
const roWnorm = $('#ro-wnorm');
const roTop5 = $('#ro-top5');
const roRes = $('#ro-res');
const roExc = $('#ro-exc');

const ablateBanner = $('#ablate-bar');
const circleChips = $('#circle-chips');
const circleSub = $('#circle-subtitle');

const inputA = $('#probe-a');
const rangeA = $('#probe-a-range');
const inputB = $('#probe-b');
const rangeB = $('#probe-b-range');
const btnRandTest = $('#btn-rand-test');
const probeEq = $('#probe-eq');
const probeSplit = $('#probe-split-badge');
const probePred = $('#probe-pred');
const probeAttn = $('#probe-attn');

function targetFor(a, b, op) {
  return op === 0 ? (a + b) % P : (a + P - b) % P;
}

function fmtPct(v) {
  return `${(v * 100).toFixed(v >= 0.9995 || v < 0.001 ? 0 : 1)}%`;
}

function fmtLoss(v) {
  if (v < 0.001) return v.toExponential(1);
  if (v < 10) return v.toFixed(3);
  return v.toFixed(1);
}

function countBits(mask) {
  let c = 0;
  for (let i = 0; i < NFREQ; i++) if (mask & (1 << i)) c++;
  return c;
}

function top5Mask(top5K) {
  let m = 0;
  for (const k of top5K) m |= (1 << (k - 1));
  return m;
}

function classifyPhase(ev, epoch) {
  if (!ev || epoch === 0) return { text: 'Initialized', cls: '' };
  if (ev.teAcc >= 0.98) {
    if (ev.top5Share >= 0.82) return { text: 'Generalized (cleanup)', cls: 'phase-clean' };
    return { text: 'Grokking!', cls: 'phase-grok' };
  }
  if (ev.teAcc >= 0.18) return { text: 'Grokking!', cls: 'phase-grok' };
  if (ev.trAcc >= 0.95) {
    if (ev.resLoss < 1.8) return { text: 'Circuit forming (plateau)', cls: 'phase-mem' };
    return { text: 'Memorized (plateau)', cls: 'phase-mem' };
  }
  return { text: 'Memorizing…', cls: '' };
}

function updateLegend() {
  if (state.chartMode === 'acc') {
    trainLegend.innerHTML = `
      <span class="legend-item"><span class="swatch" style="background:${COLORS.train}"></span>Train accuracy</span>
      <span class="legend-item"><span class="swatch" style="background:${COLORS.test}"></span>Test accuracy</span>
      <span class="legend-item"><span class="swatch dashed" style="color:${COLORS.wnorm}"></span>Weight norm ‖W‖₂</span>
    `;
  } else {
    trainLegend.innerHTML = `
      <span class="legend-item"><span class="swatch" style="background:${COLORS.train}"></span>Train loss</span>
      <span class="legend-item"><span class="swatch" style="background:${COLORS.test}"></span>Test loss</span>
      <span class="legend-item"><span class="swatch" style="background:${COLORS.restricted}"></span>Restricted test loss (top 5 freqs)</span>
      <span class="legend-item"><span class="swatch dashed" style="color:${COLORS.excluded}"></span>Excluded train loss</span>
    `;
  }
}

function renderAll() {
  const { ev } = state;

  // Toggle button text
  if (state.running) {
    btnToggle.textContent = 'Pause';
    btnToggle.classList.remove('primary');
  } else {
    btnToggle.textContent = state.liveEpoch >= state.targetEpoch ? 'Continue (+2000)' : state.liveEpoch > 0 ? 'Resume' : 'Train';
    btnToggle.classList.add('primary');
  }

  // Phase pill
  if (state.scrubbed) {
    phasePill.textContent = `Viewing epoch ${state.epoch.toLocaleString()} (scrubbed)`;
    phasePill.className = 'pill phase-scrub';
    btnBackLive.hidden = false;
  } else {
    const ph = classifyPhase(ev, state.epoch);
    phasePill.textContent = ph.text;
    phasePill.className = `pill ${ph.cls}`;
    btnBackLive.hidden = true;
  }

  // Rate pill
  if (state.running && state.epochsPerSec > 0) {
    ratePill.textContent = `${Math.round(state.epochsPerSec)} epochs/s · WASM SIMD`;
  } else if (state.liveEpoch > 0) {
    ratePill.textContent = `Paused at epoch ${state.liveEpoch.toLocaleString()} · 49,024 params`;
  } else {
    ratePill.textContent = '49,024 params · WASM SIMD';
  }

  // Readouts
  if (ev) {
    roEpoch.textContent = state.epoch.toLocaleString();
    roAcc.textContent = `${fmtPct(ev.trAcc)} / ${fmtPct(ev.teAcc)}`;
    roAcc.className = ev.teAcc >= 0.5 ? 'accent-test' : '';
    roWnorm.textContent = ev.wnorm.toFixed(1);
    roTop5.textContent = `${(ev.top5Share * 100).toFixed(1)}%`;
    roRes.textContent = fmtLoss(ev.resLoss);
    roExc.textContent = fmtLoss(ev.excLoss);
  }

  // 1. Main Training Chart
  drawTrainingChart(canvasTrain, state.history, {
    mode: state.chartMode,
    inspectedEpoch: state.scrubbed ? state.epoch : null,
    wdChanges: state.wdChanges,
  });

  // 2. Fourier Spectrum + Ablation Banner
  drawSpectrumChart(canvasSpectrum, ev ? ev.power : null, {
    top5K: ev ? ev.top5K : [],
    keepMask: state.keepMask,
    selectedFreq: state.circleK,
    hoveredK: state.hoveredK,
  });

  const keptCount = countBits(state.keepMask);
  // Highlight active ablation preset button
  const t5m = ev ? top5Mask(ev.top5K) : 0;
  $('#btn-abl-all').setAttribute('aria-pressed', String(state.keepMask === FULL_MASK));
  $('#btn-abl-solo').setAttribute('aria-pressed', String(ev !== null && state.keepMask === t5m));
  $('#btn-abl-mute').setAttribute('aria-pressed', String(ev !== null && state.keepMask === (FULL_MASK ^ t5m)));

  if (keptCount === NFREQ || !state.ablated || !ev) {
    ablateBanner.className = 'ablate-banner';
    ablateBanner.innerHTML = `<span>All <strong>29</strong> frequencies active. Click any bar to mute it, or test <em>Solo top 5</em> / <em>Mute top 5</em>.</span>`;
  } else {
    const delta = state.ablated.teAcc - ev.teAcc;
    const dropCls = state.ablated.teAcc < ev.teAcc - 0.15 ? 'ablate-drop' : 'ablate-hold';
    ablateBanner.className = 'ablate-banner active';
    ablateBanner.innerHTML = `
      <span>Keeping <strong class="mono">${keptCount}/29</strong> frequencies in W<sub>E</sub></span>
      <span class="mono">Test acc: <span class="${dropCls}">${fmtPct(state.ablated.teAcc)}</span> (full: ${fmtPct(ev.teAcc)}${Math.abs(delta) >= 0.005 ? `, ${delta >= 0 ? '+' : ''}${(delta * 100).toFixed(1)}pp` : ''}) · Train: ${fmtPct(state.ablated.trAcc)}</span>
    `;
  }

  // 3. Circle chips & Clock Circle
  if (ev) {
    const chipsHtml = [
      `<button type="button" class="chip" data-freq="0" aria-pressed="${state.manualFreq === 0}">Auto</button>`,
      ...ev.top5K.map(
        (k) =>
          `<button type="button" class="chip" data-freq="${k}" aria-pressed="${state.manualFreq === k}">k=${k}</button>`
      ),
    ].join('');
    if (circleChips.dataset.sig !== `${state.manualFreq}:${ev.top5K.join(',')}`) {
      circleChips.dataset.sig = `${state.manualFreq}:${ev.top5K.join(',')}`;
      circleChips.innerHTML = chipsHtml;
    }
    const shareK = ev.power[state.circleK - 1] * 100;
    circleSub.innerHTML = `Frequency k = ${state.circleK} · ${shareK.toFixed(1)}% of W<sub>E</sub> energy`;
  }

  const aVal = state.probe ? state.probe.a : 17;
  const bVal = state.probe ? state.probe.b : 45;
  const targetC = targetFor(aVal, bVal, state.config.op);
  drawCircleChart(canvasCircle, state.circle, state.circleK, {
    highlightA: aVal,
    highlightB: bVal,
    highlightC: targetC,
  });

  // 4. Interactive Probe
  if (state.probe) {
    const pr = state.probe;
    const opSym = state.config.op === 0 ? '+' : '−';
    probeEq.textContent = `${pr.a} ${opSym} ${pr.b} ≡ ${targetC} (mod ${P})`;
    probeSplit.textContent = pr.inTrain ? 'In training set' : 'Held-out test pair';
    probeSplit.className = pr.inTrain ? 'split-tag' : 'split-tag heldout';

    let predC = 0;
    for (let c = 1; c < P; c++) if (pr.probs[c] > pr.probs[predC]) predC = c;
    const conf = pr.probs[predC] * 100;
    const ok = predC === targetC;
    probePred.textContent = `Predicts ${predC} (${conf.toFixed(1)}%)`;
    probePred.className = `probe-pred ${ok ? 'correct' : 'wrong'}`;

    // Attention strip across 4 heads
    const heads = [];
    for (let h = 0; h < 4; h++) {
      const wa = Math.round(pr.attn[h * 3] * 100);
      const wb = Math.round(pr.attn[h * 3 + 1] * 100);
      const weq = Math.round(pr.attn[h * 3 + 2] * 100);
      heads.push(`<span>H${h + 1}: a ${wa}% b ${wb}% = ${weq}%</span>`);
    }
    probeAttn.innerHTML = heads.join('');

    drawProbeChart(canvasProbe, pr, targetC, state.keepMask);
  }
}

function pushHistory(epoch, ev) {
  if (state.history.length && state.history[state.history.length - 1].epoch === epoch) {
    return;
  }
  state.history.push({
    epoch,
    trLoss: ev.trLoss,
    trAcc: ev.trAcc,
    teLoss: ev.teLoss,
    teAcc: ev.teAcc,
    wnorm: ev.wnorm,
    resLoss: ev.resLoss,
    excLoss: ev.excLoss,
    top5Share: ev.top5Share,
    gini: ev.gini,
  });
}

// Worker messages
worker.onmessage = (e) => {
  const msg = e.data;
  if (msg.type === 'error') {
    phasePill.textContent = `Error: ${msg.message}`;
    phasePill.className = 'pill phase-grok';
    return;
  }
  if (msg.type === 'reset') {
    state.config = msg.config;
    state.history = [];
    state.wdChanges = [];
    state.keepMask = msg.keepMask;
    state.manualFreq = 0;
    syncControlValues();
    pushHistory(msg.epoch, msg.ev);
  } else if (msg.type === 'tick' || msg.type === 'done') {
    pushHistory(msg.epoch, msg.ev);
  } else if (msg.type === 'history') {
    pushHistory(msg.epoch, msg.ev);
    state.liveEpoch = msg.liveEpoch;
    if (typeof msg.targetEpoch === 'number') state.targetEpoch = msg.targetEpoch;
    state.epochsPerSec = msg.epochsPerSec;
    state.running = msg.running;
    renderAll();
    return;
  }

  state.epoch = msg.epoch;
  state.liveEpoch = msg.liveEpoch;
  if (typeof msg.targetEpoch === 'number') state.targetEpoch = msg.targetEpoch;
  state.scrubbed = msg.scrubbed;
  state.running = msg.running;
  state.epochsPerSec = msg.epochsPerSec;
  state.ev = msg.ev;
  state.circleK = msg.circleK;
  state.circle = msg.circle;
  state.ablated = msg.ablated;
  state.probe = msg.probe;
  state.keepMask = msg.keepMask;
  if (msg.probe) {
    if (document.activeElement !== inputA) inputA.value = msg.probe.a;
    rangeA.value = msg.probe.a;
    if (document.activeElement !== inputB) inputB.value = msg.probe.b;
    rangeB.value = msg.probe.b;
  }
  renderAll();
};

function syncControlValues() {
  $$('button[data-op]').forEach((b) => {
    b.setAttribute('aria-pressed', String(Number(b.dataset.op) === state.config.op));
  });
  sliderFrac.value = state.config.trainFrac.toFixed(2);
  outFrac.textContent = `${Math.round(state.config.trainFrac * 100)}%`;
  sliderWd.value = state.config.wd.toFixed(1);
  outWd.textContent = state.config.wd.toFixed(1);
}

// --- Control bindings ---
btnToggle.addEventListener('click', () => {
  if (state.running) {
    worker.postMessage({ type: 'pause' });
  } else {
    worker.postMessage({ type: 'start' });
  }
});

btnReset.addEventListener('click', () => {
  worker.postMessage({ type: 'reset', config: state.config, autoStart: true });
});

btnSeed.addEventListener('click', () => {
  const nextSeed = ((state.config.seed * 7 + 13) % 97) + 1;
  state.config.seed = nextSeed;
  worker.postMessage({ type: 'reset', config: state.config, autoStart: true });
});

btnBackLive.addEventListener('click', () => {
  worker.postMessage({ type: 'scrubEpoch', epoch: null });
});

$$('button[data-op]').forEach((btn) => {
  btn.addEventListener('click', () => {
    const op = Number(btn.dataset.op);
    if (op === state.config.op) return;
    // Sensible defaults per task (subtraction has no symmetry shortcut so 40% train & seed 42 grok crisply)
    const nextCfg =
      op === 1
        ? { op: 1, trainFrac: 0.40, wd: 2.0, seed: 42 }
        : { op: 0, trainFrac: 0.35, wd: 2.2, seed: 1 };
    state.config = { ...state.config, ...nextCfg };
    syncControlValues();
    worker.postMessage({ type: 'reset', config: state.config, autoStart: true });
  });
});

sliderFrac.addEventListener('input', () => {
  outFrac.textContent = `${Math.round(Number(sliderFrac.value) * 100)}%`;
});
sliderFrac.addEventListener('change', () => {
  state.config.trainFrac = Number(sliderFrac.value);
  worker.postMessage({ type: 'reset', config: state.config, autoStart: true });
});

sliderWd.addEventListener('input', () => {
  const wd = Number(sliderWd.value);
  outWd.textContent = wd.toFixed(1);
  state.config.wd = wd;
  worker.postMessage({ type: 'setHparams', wd });
});
sliderWd.addEventListener('change', () => {
  const wd = Number(sliderWd.value);
  if (state.liveEpoch > 0) {
    state.wdChanges.push({ epoch: state.liveEpoch, wd });
    renderAll();
  }
});

// Chart mode tabs
$$('button[data-chart-mode]').forEach((btn) => {
  btn.addEventListener('click', () => {
    state.chartMode = btn.dataset.chartMode;
    $$('button[data-chart-mode]').forEach((b) =>
      b.setAttribute('aria-pressed', String(b.dataset.chartMode === state.chartMode))
    );
    updateLegend();
    renderAll();
  });
});

// Time-travel scrubbing on training chart
let scrubbing = false;
function scrubAtClientX(clientX) {
  const plot = canvasTrain._plot;
  if (!plot || state.history.length < 2) return;
  const rect = canvasTrain.getBoundingClientRect();
  const x = clientX - rect.left;
  const frac = Math.max(0, Math.min(1, (x - plot.pad.l) / plot.pw));
  const ep = frac * plot.maxEp;
  const clamped = Math.min(state.liveEpoch, Math.max(0, ep));
  if (clamped >= state.liveEpoch - 15 && !state.scrubbed) return;
  worker.postMessage({ type: 'scrubEpoch', epoch: clamped });
}
canvasTrain.addEventListener('pointerdown', (e) => {
  scrubbing = true;
  canvasTrain.setPointerCapture(e.pointerId);
  scrubAtClientX(e.clientX);
});
canvasTrain.addEventListener('pointermove', (e) => {
  if (scrubbing) scrubAtClientX(e.clientX);
});
canvasTrain.addEventListener('pointerup', () => { scrubbing = false; });
canvasTrain.addEventListener('pointercancel', () => { scrubbing = false; });

// Interactive Fourier spectrum (hover tooltip + click to mute/unmute)
function freqFromEvent(e) {
  const bars = canvasSpectrum._bars;
  if (!bars) return 0;
  const rect = canvasSpectrum.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const idx = Math.floor((x - bars.pad.l) / bars.slotW);
  if (idx < 0 || idx >= NFREQ) return 0;
  return idx + 1;
}

canvasSpectrum.addEventListener('pointermove', (e) => {
  const k = freqFromEvent(e);
  if (k !== state.hoveredK) {
    state.hoveredK = k;
    drawSpectrumChart(canvasSpectrum, state.ev ? state.ev.power : null, {
      top5K: state.ev ? state.ev.top5K : [],
      keepMask: state.keepMask,
      selectedFreq: state.circleK,
      hoveredK: state.hoveredK,
    });
  }
  if (k > 0 && state.ev) {
    const pct = (state.ev.power[k - 1] * 100).toFixed(1);
    const kept = (state.keepMask & (1 << (k - 1))) !== 0;
    const rect = canvasSpectrum.getBoundingClientRect();
    spectrumTip.hidden = false;
    spectrumTip.innerHTML = `<strong>k = ${k}</strong> · ${pct}% energy · ${kept ? 'click to mute' : 'muted (click to restore)'}`;
    const tipX = Math.min(rect.width - 170, Math.max(8, e.clientX - rect.left + 10));
    const tipY = Math.max(4, e.clientY - rect.top - 32);
    spectrumTip.style.left = `${tipX}px`;
    spectrumTip.style.top = `${tipY}px`;
  } else {
    spectrumTip.hidden = true;
  }
});

canvasSpectrum.addEventListener('pointerleave', () => {
  state.hoveredK = 0;
  spectrumTip.hidden = true;
  drawSpectrumChart(canvasSpectrum, state.ev ? state.ev.power : null, {
    top5K: state.ev ? state.ev.top5K : [],
    keepMask: state.keepMask,
    selectedFreq: state.circleK,
    hoveredK: 0,
  });
});

canvasSpectrum.addEventListener('click', (e) => {
  const k = freqFromEvent(e);
  if (k <= 0) return;
  const nextMask = state.keepMask ^ (1 << (k - 1));
  worker.postMessage({ type: 'setKeepMask', keepMask: nextMask });
});

$('#btn-abl-all').addEventListener('click', () => {
  worker.postMessage({ type: 'setKeepMask', keepMask: FULL_MASK });
});
$('#btn-abl-solo').addEventListener('click', () => {
  if (!state.ev) return;
  worker.postMessage({ type: 'setKeepMask', keepMask: top5Mask(state.ev.top5K) });
});
$('#btn-abl-mute').addEventListener('click', () => {
  if (!state.ev) return;
  worker.postMessage({ type: 'setKeepMask', keepMask: FULL_MASK ^ top5Mask(state.ev.top5K) });
});

// Circle frequency chips
circleChips.addEventListener('click', (e) => {
  const btn = e.target.closest('button[data-freq]');
  if (!btn) return;
  const k = Number(btn.dataset.freq);
  state.manualFreq = k;
  worker.postMessage({ type: 'setSelectedFreq', k });
});

// Probe controls
function sendProbe(a, b, syncInputs = true) {
  const ca = ((a | 0) % P + P) % P;
  const cb = ((b | 0) % P + P) % P;
  if (syncInputs) {
    inputA.value = ca;
    inputB.value = cb;
  }
  rangeA.value = ca;
  rangeB.value = cb;
  worker.postMessage({ type: 'setProbe', a: ca, b: cb });
}

inputA.addEventListener('input', () => {
  if (inputA.value === '' || Number.isNaN(Number(inputA.value))) return;
  const bVal = inputB.value === '' ? (state.probe ? state.probe.b : 45) : Number(inputB.value);
  sendProbe(Number(inputA.value), bVal, false);
});
inputA.addEventListener('blur', () => {
  if (state.probe) inputA.value = state.probe.a;
});
rangeA.addEventListener('input', () => sendProbe(Number(rangeA.value), Number(rangeB.value)));
inputB.addEventListener('input', () => {
  if (inputB.value === '' || Number.isNaN(Number(inputB.value))) return;
  const aVal = inputA.value === '' ? (state.probe ? state.probe.a : 17) : Number(inputA.value);
  sendProbe(aVal, Number(inputB.value), false);
});
inputB.addEventListener('blur', () => {
  if (state.probe) inputB.value = state.probe.b;
});
rangeB.addEventListener('input', () => sendProbe(Number(rangeA.value), Number(rangeB.value)));
btnRandTest.addEventListener('click', () => worker.postMessage({ type: 'randomTestPair' }));

// Inline article action buttons
$$('button.inline[data-action]').forEach((btn) => {
  btn.addEventListener('click', () => {
    const act = btn.dataset.action;
    const demoEl = $('#demo');
    demoEl.scrollIntoView({ behavior: 'smooth', block: 'start' });

    if (act === 'no-decay') {
      state.config = { op: 0, trainFrac: 0.35, lr: 3e-3, wd: 0.0, seed: 1 };
      syncControlValues();
      worker.postMessage({ type: 'reset', config: state.config, autoStart: true });
    } else if (act === 'turn-on-decay') {
      state.config.wd = 2.2;
      syncControlValues();
      worker.postMessage({ type: 'setHparams', wd: 2.2 });
      if (state.liveEpoch > 0) state.wdChanges.push({ epoch: state.liveEpoch, wd: 2.2 });
      worker.postMessage({ type: 'start' });
    } else if (act === 'solo-top5' && state.ev) {
      worker.postMessage({ type: 'setKeepMask', keepMask: top5Mask(state.ev.top5K) });
    } else if (act === 'mute-top5' && state.ev) {
      worker.postMessage({ type: 'setKeepMask', keepMask: FULL_MASK ^ top5Mask(state.ev.top5K) });
    } else if (act === 'show-loss') {
      state.chartMode = 'loss';
      $$('button[data-chart-mode]').forEach((b) =>
        b.setAttribute('aria-pressed', String(b.dataset.chartMode === 'loss'))
      );
      updateLegend();
      renderAll();
    } else if (act === 'task-sub') {
      state.config = { op: 1, trainFrac: 0.40, lr: 3e-3, wd: 2.0, seed: 42 };
      syncControlValues();
      worker.postMessage({ type: 'reset', config: state.config, autoStart: true });
    }
  });
});

// Load static article comparison figure
fetch('./data/experiments.json')
  .then((r) => r.json())
  .then((data) => {
    state.expData = data;
    drawComparisonChart(canvasCompare, data);
  })
  .catch(() => {});

// Resize handler
let resizeRaf = 0;
window.addEventListener('resize', () => {
  cancelAnimationFrame(resizeRaf);
  resizeRaf = requestAnimationFrame(() => {
    renderAll();
    if (state.expData) drawComparisonChart(canvasCompare, state.expData);
  });
});

// Boot
updateLegend();
syncControlValues();
worker.postMessage({ type: 'init', config: state.config, autoStart: true });
