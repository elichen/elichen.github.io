(function () {
  "use strict";

  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => Array.from(document.querySelectorAll(selector));

  const elements = {
    canvas: $("#field-canvas"),
    fieldFrame: $(".field-frame"),
    workbench: $("#workbench"),
    toolPanel: $("#tool-panel"),
    fieldPanel: $("#field-panel"),
    teachNumber: $("#teach-number"),
    fieldNumber: $("#field-number"),
    firstUseHint: $("#first-use-hint"),
    coordinateX: $("#coordinate-x"),
    coordinateY: $("#coordinate-y"),
    colorReadout: $("#color-readout"),
    customColor: $("#custom-color"),
    pauseButton: $("#pause-button"),
    undoButton: $("#undo-button"),
    clearButton: $("#clear-button"),
    examplesButton: $("#examples-button"),
    reseedButton: $("#reseed-button"),
    saveButton: $("#save-button"),
    headerStatus: $("#header-status"),
    localStatus: $(".local-status"),
    fieldStatus: $(".field-status"),
    trainingLabel: $("#training-label"),
    lossCanvas: $("#loss-canvas"),
    lossValue: $("#loss-value"),
    fitBadge: $("#fit-badge"),
    stepValue: $("#step-value"),
    sampleValue: $("#sample-value"),
    parameterValue: $("#parameter-value"),
    parameterBadge: $("#parameter-badge"),
    gridValue: $("#grid-value"),
    probePosition: $("#probe-position"),
    predictedChip: $("#predicted-chip"),
    predictedValue: $("#predicted-value"),
    exampleChip: $("#example-chip"),
    exampleValue: $("#example-value"),
    liveRegion: $("#live-region")
  };

  const fieldContext = elements.canvas.getContext("2d", { alpha: false });
  const lossContext = elements.lossCanvas.getContext("2d");
  const offscreen = document.createElement("canvas");
  const offscreenContext = offscreen.getContext("2d", { alpha: false });
  const exportBuffer = document.createElement("canvas");
  const exportBufferContext = exportBuffer.getContext("2d", { alpha: false });
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const mobileLayout = window.matchMedia("(max-width: 720px)");

  const detailSettings = {
    soft: { scale: 0.58, label: "Soft" },
    balanced: { scale: 1, label: "Balanced" },
    intricate: { scale: 1.55, label: "Intricate" }
  };

  const brushSettings = {
    fine: { radius: 0, points: 1 },
    medium: { radius: 0.034, points: 4 },
    broad: { radius: 0.078, points: 9 }
  };

  const presets = {
    "night-garden": {
      name: "Night garden",
      anchors: [
        [0.02, 0.02, "#101426"], [0.98, 0.03, "#17152c"],
        [0.02, 0.98, "#122b31"], [0.98, 0.98, "#27152e"],
        [0.17, 0.22, "#2664ff"], [0.80, 0.20, "#ff553e"],
        [0.48, 0.50, "#f4eea9"], [0.22, 0.78, "#13c4a3"],
        [0.81, 0.76, "#d845ff"], [0.53, 0.94, "#334f2f"]
      ]
    },
    tidepool: {
      name: "Tidepool",
      anchors: [
        [0.02, 0.02, "#102730"], [0.98, 0.02, "#183642"],
        [0.02, 0.98, "#0d2934"], [0.98, 0.98, "#1b2236"],
        [0.17, 0.28, "#13c4a3"], [0.62, 0.18, "#9ce6d4"],
        [0.86, 0.37, "#e8d2a0"], [0.35, 0.58, "#1e82a5"],
        [0.66, 0.74, "#f4eea9"], [0.20, 0.87, "#2664ff"]
      ]
    },
    "solar-print": {
      name: "Solar print",
      anchors: [
        [0.02, 0.02, "#171218"], [0.98, 0.02, "#231315"],
        [0.02, 0.98, "#11131d"], [0.98, 0.98, "#1a101d"],
        [0.23, 0.25, "#ff553e"], [0.76, 0.22, "#ff8b3d"],
        [0.51, 0.48, "#e9ff43"], [0.26, 0.77, "#d845ff"],
        [0.76, 0.78, "#ff553e"], [0.51, 0.93, "#3b1732"]
      ]
    }
  };

  const state = {
    model: null,
    modelSeed: 2049,
    detail: "balanced",
    brush: "medium",
    selectedColor: "#ff553e",
    selectedName: "Vermilion",
    samples: [],
    sampleBins: new Map(),
    recentSamples: [],
    serial: 0,
    stroke: 0,
    activePointer: null,
    lastPaintPoint: null,
    paused: false,
    showExamples: true,
    renderWidth: 180,
    renderHeight: 112,
    featureGrid: null,
    imageData: null,
    cssWidth: 0,
    cssHeight: 0,
    dpr: 1,
    lossHistory: [],
    lastLoss: 0,
    lastRenderAt: 0,
    lastUiAt: 0,
    lastPaintAt: 0,
    frameCount: 0,
    isSettled: false,
    renderDirty: true,
    slowRenderCount: 0,
    keyboardCursor: { x: 0, y: 0 },
    probe: { x: 0, y: 0 },
    hasPainted: false,
    activePreset: null,
    replacementSnapshot: null,
    strokeSnapshot: null
  };

  function createModel(pretrainSteps = 0) {
    state.model = new window.DreamfieldMLP({
      hiddenSize: 20,
      bands: 3,
      frequencyScale: detailSettings[state.detail].scale,
      learningRate: 0.007,
      seed: state.modelSeed
    });

    reencodeSamples();
    state.lossHistory = [];
    state.lastLoss = state.samples.length ? state.model.lossForBatch(state.samples.slice(0, 24)) : 0;

    const allowedPretrainSteps = state.paused ? 0 : pretrainSteps;
    for (let step = 0; step < allowedPretrainSteps && state.samples.length; step += 1) {
      const batchLoss = state.model.trainBatch(buildBatch(18));
      state.lastLoss = step === 0 ? batchLoss : state.lastLoss * 0.88 + batchLoss * 0.12;
      if (step % 7 === 0) pushLoss(state.lastLoss);
    }

    const parameterText = state.model.parameterCount.toLocaleString();
    elements.parameterValue.textContent = parameterText;
    elements.parameterBadge.textContent = `${parameterText} parameters`;
    buildRenderGrid();
    state.renderDirty = true;
    updateTelemetry(true);
  }

  function reencodeSamples() {
    if (!state.model) return;
    for (const sample of state.samples) {
      sample.features = state.model.encode(sample.x, sample.y);
    }
  }

  function setActivePreset(key) {
    state.activePreset = key;
    $$("[data-preset]").forEach((button) => {
      const active = button.dataset.preset === key;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });
  }

  function cloneSamples(samples) {
    return samples.map((sample) => ({
      x: sample.x,
      y: sample.y,
      rgb: sample.rgb.slice(),
      source: sample.source,
      serial: sample.serial,
      features: null,
      binKey: sample.binKey
    }));
  }

  function saveReplacementSnapshot() {
    if (!state.samples.length) return;
    state.replacementSnapshot = {
      samples: cloneSamples(state.samples),
      serial: state.serial,
      stroke: state.stroke,
      hasPainted: state.hasPainted,
      activePreset: state.activePreset,
      modelSeed: state.modelSeed,
      detail: state.detail
    };
  }

  function beginStroke() {
    state.strokeSnapshot = {
      samples: cloneSamples(state.samples),
      serial: state.serial,
      stroke: state.stroke,
      hasPainted: state.hasPainted
    };
    state.stroke += 1;
  }

  function restoreStrokeSnapshot() {
    const snapshot = state.strokeSnapshot;
    if (!snapshot) return false;
    state.strokeSnapshot = null;
    state.samples = cloneSamples(snapshot.samples);
    state.serial = snapshot.serial;
    state.stroke = snapshot.stroke;
    state.hasPainted = snapshot.hasPainted;
    rebuildSampleIndex();
    createModel(state.samples.length ? 110 : 0);
    elements.firstUseHint.classList.toggle("is-dismissed", state.hasPainted);
    announce(state.paused
      ? "Last paint stroke removed. Learning remains paused."
      : "Last paint stroke removed. The remaining examples are being relearned.");
    return true;
  }

  function restoreReplacementSnapshot() {
    const snapshot = state.replacementSnapshot;
    if (!snapshot) return false;
    state.replacementSnapshot = null;
    state.samples = cloneSamples(snapshot.samples);
    state.serial = snapshot.serial;
    state.stroke = snapshot.stroke;
    state.hasPainted = snapshot.hasPainted;
    state.modelSeed = snapshot.modelSeed;
    state.detail = snapshot.detail;
    state.strokeSnapshot = null;
    setActivePreset(snapshot.activePreset);
    $$("[data-detail]").forEach((button) => {
      const active = button.dataset.detail === state.detail;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    rebuildSampleIndex();
    createModel(state.samples.length ? 110 : 0);
    elements.firstUseHint.classList.toggle("is-dismissed", state.hasPainted);
    announce(state.paused
      ? "Previous study restored. Learning remains paused."
      : "Previous study restored. The model is relearning its examples.");
    return true;
  }

  function loadPreset(key, announceChange = true) {
    const preset = presets[key];
    if (!preset) return;

    if (announceChange) saveReplacementSnapshot();
    state.samples = [];
    state.sampleBins.clear();
    state.recentSamples = [];
    state.strokeSnapshot = null;
    state.serial = 0;
    state.stroke = 0;
    state.hasPainted = false;
    elements.firstUseHint.classList.remove("is-dismissed");

    const offsets = [
      [0, 0], [0.021, 0.012], [-0.018, 0.017], [0.008, -0.023]
    ];

    for (const [ux, uy, hex] of preset.anchors) {
      const rgb = hexToRgb(hex);
      for (const [offsetX, offsetY] of offsets) {
        insertSample(
          clamp(ux * 2 - 1 + offsetX, -1, 1),
          clamp(uy * 2 - 1 + offsetY, -1, 1),
          rgb,
          "preset",
          false
        );
      }
    }

    setActivePreset(key);

    state.modelSeed += 17;
    createModel(reducedMotion ? 110 : 180);
    if (announceChange) {
      announce(state.paused
        ? `${preset.name} preset loaded with forty color examples. Learning remains paused.`
        : `${preset.name} preset loaded. The network is learning forty color examples.`);
    }
  }

  function insertSample(x, y, rgb, source = "user", markRecent = true) {
    const key = sampleBinKey(x, y);
    const existing = state.sampleBins.get(key);
    let sample;

    if (existing) {
      existing.x = x;
      existing.y = y;
      existing.rgb = rgb.slice();
      existing.source = source;
      existing.serial = ++state.serial;
      existing.features = state.model ? state.model.encode(x, y) : null;
      const previousIndex = state.samples.indexOf(existing);
      if (previousIndex >= 0) state.samples.splice(previousIndex, 1);
      state.samples.push(existing);
      sample = existing;
    } else {
      sample = {
        x,
        y,
        rgb: rgb.slice(),
        source,
        serial: ++state.serial,
        features: state.model ? state.model.encode(x, y) : null,
        binKey: key
      };
      state.samples.push(sample);
      state.sampleBins.set(key, sample);

      if (state.samples.length > 640) {
        const removed = state.samples.shift();
        if (state.sampleBins.get(removed.binKey) === removed) state.sampleBins.delete(removed.binKey);
        state.recentSamples = state.recentSamples.filter((recent) => recent !== removed);
      }
    }

    if (markRecent) {
      state.recentSamples.push(sample);
      if (state.recentSamples.length > 120) state.recentSamples.shift();
    }
    return sample;
  }

  function sampleBinKey(x, y) {
    return `${Math.round((x + 1) * 78)},${Math.round((y + 1) * 78)}`;
  }

  function rebuildSampleIndex() {
    state.sampleBins.clear();
    for (const sample of state.samples) {
      sample.binKey = sampleBinKey(sample.x, sample.y);
      state.sampleBins.set(sample.binKey, sample);
    }
    state.recentSamples = [];
  }

  function stamp(x, y) {
    const setting = brushSettings[state.brush];
    const rgb = hexToRgb(state.selectedColor);

    insertSample(x, y, rgb, "user");
    for (let index = 1; index < setting.points; index += 1) {
      const angle = index * 2.399963 + state.stroke * 0.71;
      const distance = setting.radius * Math.sqrt(index / Math.max(1, setting.points - 1));
      insertSample(
        clamp(x + Math.cos(angle) * distance, -1, 1),
        clamp(y + Math.sin(angle) * distance, -1, 1),
        rgb,
        "user"
      );
    }

    state.hasPainted = true;
    state.lastPaintAt = performance.now();
    state.isSettled = false;
    elements.firstUseHint.classList.add("is-dismissed");
    state.renderDirty = true;
  }

  function buildBatch(size = 16) {
    if (!state.samples.length) return [];
    const batch = [];
    const recentCount = Math.min(Math.ceil(size * 0.55), state.recentSamples.length);

    for (let index = 0; index < recentCount; index += 1) {
      batch.push(state.recentSamples[Math.floor(Math.random() * state.recentSamples.length)]);
    }
    while (batch.length < size) {
      batch.push(state.samples[Math.floor(Math.random() * state.samples.length)]);
    }
    return batch;
  }

  function resizeFieldCanvas() {
    const rect = elements.fieldFrame.getBoundingClientRect();
    if (!rect.width || !rect.height) return;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const cssWidth = Math.round(rect.width);
    const cssHeight = Math.round(rect.height);
    const changed = cssWidth !== state.cssWidth || cssHeight !== state.cssHeight || dpr !== state.dpr;
    if (!changed) return;

    state.cssWidth = cssWidth;
    state.cssHeight = cssHeight;
    state.dpr = dpr;
    elements.canvas.width = Math.max(1, Math.round(cssWidth * dpr));
    elements.canvas.height = Math.max(1, Math.round(cssHeight * dpr));
    state.renderWidth = window.innerWidth < 720 ? 90 : 108;
    buildRenderGrid();
    state.renderDirty = true;
  }

  function buildRenderGrid() {
    if (!state.model || !state.cssWidth || !state.cssHeight) return;
    const aspectHeight = Math.round(state.renderWidth * state.cssHeight / state.cssWidth);
    state.renderHeight = clamp(aspectHeight, 72, window.innerWidth < 720 ? 146 : 166);

    offscreen.width = state.renderWidth;
    offscreen.height = state.renderHeight;
    state.imageData = offscreenContext.createImageData(state.renderWidth, state.renderHeight);
    state.featureGrid = new Float32Array(state.renderWidth * state.renderHeight * state.model.inputSize);

    let pixel = 0;
    for (let row = 0; row < state.renderHeight; row += 1) {
      const y = state.renderHeight === 1 ? 0 : row / (state.renderHeight - 1) * 2 - 1;
      for (let column = 0; column < state.renderWidth; column += 1) {
        const x = state.renderWidth === 1 ? 0 : column / (state.renderWidth - 1) * 2 - 1;
        state.model.encode(x, y, state.featureGrid, pixel * state.model.inputSize);
        pixel += 1;
      }
    }

    elements.gridValue.textContent = `${state.renderWidth}×${state.renderHeight}`;
  }

  function renderField() {
    if (!state.imageData || !state.featureGrid || !state.cssWidth || !state.cssHeight) return;
    const started = performance.now();
    const pixels = state.imageData.data;
    const prediction = new Float32Array(3);
    const inputSize = state.model.inputSize;
    const pixelCount = state.renderWidth * state.renderHeight;

    for (let pixel = 0; pixel < pixelCount; pixel += 1) {
      state.model.forwardEncoded(state.featureGrid, pixel * inputSize, prediction);
      const outputOffset = pixel * 4;
      pixels[outputOffset] = clamp(Math.round(prediction[0] * 255), 0, 255);
      pixels[outputOffset + 1] = clamp(Math.round(prediction[1] * 255), 0, 255);
      pixels[outputOffset + 2] = clamp(Math.round(prediction[2] * 255), 0, 255);
      pixels[outputOffset + 3] = 255;
    }

    offscreenContext.putImageData(state.imageData, 0, 0);
    exportBuffer.width = offscreen.width;
    exportBuffer.height = offscreen.height;
    exportBufferContext.drawImage(offscreen, 0, 0);
    fieldContext.setTransform(state.dpr, 0, 0, state.dpr, 0, 0);
    fieldContext.imageSmoothingEnabled = true;
    fieldContext.imageSmoothingQuality = "high";
    fieldContext.clearRect(0, 0, state.cssWidth, state.cssHeight);
    fieldContext.drawImage(offscreen, 0, 0, state.cssWidth, state.cssHeight);

    if (state.showExamples) drawExamples();
    if (document.activeElement === elements.canvas) drawKeyboardCursor();

    const renderTime = performance.now() - started;
    if (renderTime > 34 && state.renderWidth > 72) {
      state.slowRenderCount += 1;
      if (state.slowRenderCount >= 2) {
        state.renderWidth = Math.max(72, Math.floor(state.renderWidth * 0.82));
        state.slowRenderCount = 0;
        buildRenderGrid();
        state.renderDirty = true;
        return;
      }
    } else {
      state.slowRenderCount = 0;
    }
    state.renderDirty = false;
  }

  function drawExamples() {
    const start = Math.max(0, state.samples.length - 210);
    for (let index = start; index < state.samples.length; index += 1) {
      const sample = state.samples[index];
      const x = (sample.x + 1) * 0.5 * state.cssWidth;
      const y = (sample.y + 1) * 0.5 * state.cssHeight;
      const color = rgbToHex(sample.rgb);
      const isPreset = sample.source === "preset";

      fieldContext.save();
      fieldContext.translate(x, y);
      fieldContext.fillStyle = color;
      fieldContext.strokeStyle = "rgba(251, 247, 235, 0.92)";
      fieldContext.lineWidth = 1.5;
      fieldContext.shadowColor = "rgba(23, 24, 19, 0.46)";
      fieldContext.shadowBlur = 4;

      if (isPreset && sample.serial % 4 !== 1) {
        fieldContext.restore();
        continue;
      }
      if (!isPreset && sample.serial % 3 !== 0 && sample.serial !== state.serial) {
        fieldContext.restore();
        continue;
      }

      if (isPreset) {
        fieldContext.rotate(Math.PI / 4);
        fieldContext.fillRect(-3, -3, 6, 6);
        fieldContext.strokeRect(-3, -3, 6, 6);
      } else {
        fieldContext.beginPath();
        fieldContext.arc(0, 0, 4.4, 0, Math.PI * 2);
        fieldContext.fill();
        fieldContext.stroke();
      }
      fieldContext.restore();
    }
  }

  function drawKeyboardCursor() {
    const x = (state.keyboardCursor.x + 1) * 0.5 * state.cssWidth;
    const y = (state.keyboardCursor.y + 1) * 0.5 * state.cssHeight;
    fieldContext.save();
    fieldContext.translate(x, y);
    fieldContext.strokeStyle = "#fbf7eb";
    fieldContext.lineWidth = 2;
    fieldContext.shadowColor = "#171813";
    fieldContext.shadowBlur = 5;
    fieldContext.beginPath();
    fieldContext.arc(0, 0, 11, 0, Math.PI * 2);
    fieldContext.moveTo(-17, 0);
    fieldContext.lineTo(-6, 0);
    fieldContext.moveTo(6, 0);
    fieldContext.lineTo(17, 0);
    fieldContext.moveTo(0, -17);
    fieldContext.lineTo(0, -6);
    fieldContext.moveTo(0, 6);
    fieldContext.lineTo(0, 17);
    fieldContext.stroke();
    fieldContext.restore();
  }

  function drawLossChart() {
    const canvas = elements.lossCanvas;
    const context = lossContext;
    const width = canvas.width;
    const height = canvas.height;
    context.clearRect(0, 0, width, height);

    context.strokeStyle = "rgba(242, 237, 223, 0.12)";
    context.lineWidth = 1;
    for (let row = 1; row < 4; row += 1) {
      const y = row * height / 4;
      context.beginPath();
      context.moveTo(0, y);
      context.lineTo(width, y);
      context.stroke();
    }

    if (state.lossHistory.length < 2) return;
    const values = state.lossHistory.map((value) => Math.log10(Math.max(value, 1e-6)));
    let minimum = Math.min(...values);
    let maximum = Math.max(...values);
    if (maximum - minimum < 0.12) {
      minimum -= 0.06;
      maximum += 0.06;
    }
    const padding = 4;
    const points = values.map((value, index) => ({
      x: padding + index / (values.length - 1) * (width - padding * 2),
      y: padding + (maximum - value) / (maximum - minimum) * (height - padding * 2)
    }));

    const fill = context.createLinearGradient(0, 0, 0, height);
    fill.addColorStop(0, "rgba(233, 255, 67, 0.34)");
    fill.addColorStop(1, "rgba(233, 255, 67, 0)");
    context.beginPath();
    context.moveTo(points[0].x, height);
    for (const point of points) context.lineTo(point.x, point.y);
    context.lineTo(points[points.length - 1].x, height);
    context.closePath();
    context.fillStyle = fill;
    context.fill();

    context.beginPath();
    context.moveTo(points[0].x, points[0].y);
    for (let index = 1; index < points.length; index += 1) {
      context.lineTo(points[index].x, points[index].y);
    }
    context.strokeStyle = "#e9ff43";
    context.lineWidth = 3;
    context.lineJoin = "round";
    context.stroke();

    const latest = points[points.length - 1];
    context.beginPath();
    context.arc(latest.x, latest.y, 4, 0, Math.PI * 2);
    context.fillStyle = "#ff553e";
    context.fill();
  }

  function updateTelemetry(force = false) {
    if (!state.model) return;
    elements.stepValue.textContent = state.model.step.toLocaleString();
    elements.sampleValue.textContent = state.samples.length.toLocaleString();
    const hasPaintStroke = Boolean(state.strokeSnapshot);
    elements.undoButton.disabled = !hasPaintStroke && !state.replacementSnapshot;
    elements.undoButton.textContent = hasPaintStroke ? "Undo stroke" : state.replacementSnapshot ? "Restore study" : "Undo stroke";
    elements.clearButton.disabled = state.samples.length === 0;

    if (!state.samples.length) {
      elements.lossValue.textContent = "—";
      elements.fitBadge.textContent = "No data";
      elements.trainingLabel.textContent = "Waiting for data";
    } else {
      elements.lossValue.textContent = formatLoss(state.lastLoss);
      if (state.paused) {
        elements.fitBadge.textContent = "Paused";
      } else if (state.lastLoss < 0.0025) {
        elements.fitBadge.textContent = "Fine fit";
      } else if (state.lastLoss < 0.012) {
        elements.fitBadge.textContent = "Learning";
      } else {
        elements.fitBadge.textContent = "Searching";
      }
      if (!state.paused) elements.trainingLabel.textContent = state.isSettled ? "Refining" : "Training";
      elements.lossCanvas.setAttribute("aria-label", `Recent training loss chart. Current mean squared error ${formatLoss(state.lastLoss)}.`);
    }

    if (force || state.model.step % 12 === 0) drawLossChart();
    updateProbe(state.probe.x, state.probe.y);
  }

  function updateProbe(x, y) {
    if (!state.model) return;
    state.probe.x = clamp(x, -1, 1);
    state.probe.y = clamp(y, -1, 1);
    const prediction = state.model.predict(state.probe.x, state.probe.y);
    const predictionHex = rgbToHex(prediction);
    elements.predictedChip.style.backgroundColor = predictionHex;
    elements.predictedValue.textContent = predictionHex;

    let nearest = null;
    let nearestDistance = Infinity;
    for (const sample of state.samples) {
      const dx = sample.x - state.probe.x;
      const dy = sample.y - state.probe.y;
      const distance = dx * dx + dy * dy;
      if (distance < nearestDistance) {
        nearestDistance = distance;
        nearest = sample;
      }
    }

    if (nearest) {
      const nearestHex = rgbToHex(nearest.rgb);
      elements.exampleChip.style.backgroundColor = nearestHex;
      elements.exampleValue.textContent = nearestHex;
    } else {
      elements.exampleChip.style.backgroundColor = "#ded8c9";
      elements.exampleValue.textContent = "No example";
    }

    const xText = signed(state.probe.x);
    const yText = signed(state.probe.y);
    elements.probePosition.textContent = `x ${xText} · y ${yText}`;
    elements.coordinateX.textContent = `x ${xText}`;
    elements.coordinateY.textContent = `y ${yText}`;
  }

  function setPaused(paused, shouldAnnounce = true) {
    state.paused = paused;
    elements.pauseButton.classList.toggle("is-paused", paused);
    elements.pauseButton.querySelector("span:last-child").textContent = paused ? "Resume learning" : "Pause learning";
    elements.localStatus.classList.toggle("is-paused", paused);
    elements.fieldStatus.classList.toggle("is-paused", paused);
    elements.headerStatus.textContent = paused ? "Learning paused" : "Learning locally";
    elements.trainingLabel.textContent = paused ? "Paused" : state.samples.length ? "Training" : "Waiting for data";
    if (shouldAnnounce) announce(paused ? "Learning paused." : "Learning resumed.");
    updateTelemetry(true);
  }

  function selectColor(hex, name = "Custom") {
    state.selectedColor = hex.toLowerCase();
    state.selectedName = name;
    elements.customColor.value = state.selectedColor;
    elements.colorReadout.textContent = `${name} · ${state.selectedColor}`;
    $$(".swatch").forEach((button) => {
      button.setAttribute("aria-pressed", String(button.dataset.color.toLowerCase() === state.selectedColor));
    });
  }

  function selectBrush(name) {
    if (!brushSettings[name]) return;
    state.brush = name;
    $$("[data-brush]").forEach((button) => {
      const active = button.dataset.brush === name;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });
  }

  function selectDetail(name) {
    if (!detailSettings[name] || state.detail === name) return;
    state.detail = name;
    $$("[data-detail]").forEach((button) => {
      const active = button.dataset.detail === name;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    createModel(state.samples.length ? 120 : 0);
    announce(`${detailSettings[name].label} model bias selected. The network has been reinitialized with the same examples.`);
  }

  function undoLastStroke() {
    if (!restoreStrokeSnapshot()) restoreReplacementSnapshot();
  }

  function clearField() {
    saveReplacementSnapshot();
    state.samples = [];
    state.sampleBins.clear();
    state.recentSamples = [];
    state.strokeSnapshot = null;
    state.stroke = 0;
    state.hasPainted = false;
    setActivePreset(null);
    state.modelSeed += 19;
    createModel(0);
    elements.firstUseHint.classList.add("is-dismissed");
    elements.trainingLabel.textContent = "Waiting for data";
    announce("All training examples cleared. Paint on the field to begin learning.");
  }

  function reseedModel() {
    state.modelSeed = (Date.now() ^ Math.floor(Math.random() * 0xffffffff)) >>> 0;
    createModel(state.samples.length ? 95 : 0);
    announce("New random weights initialized. The same examples may now interpolate differently.");
  }

  function toggleExamples() {
    state.showExamples = !state.showExamples;
    elements.examplesButton.textContent = state.showExamples ? "Hide examples" : "Show examples";
    state.renderDirty = true;
  }

  function saveField() {
    if (!state.imageData) return;
    const exportCanvas = document.createElement("canvas");
    const exportWidth = 1600;
    const exportHeight = Math.round(exportWidth * state.cssHeight / state.cssWidth);
    const footerHeight = 64;
    exportCanvas.width = exportWidth;
    exportCanvas.height = exportHeight + footerHeight;
    const context = exportCanvas.getContext("2d");
    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = "high";
    context.drawImage(exportBuffer, 0, 0, exportWidth, exportHeight);
    context.fillStyle = "#171813";
    context.fillRect(0, exportHeight, exportWidth, footerHeight);
    context.fillStyle = "#e9ff43";
    context.font = "600 18px monospace";
    context.fillText("DREAMFIELD", 25, exportHeight + 40);
    context.fillStyle = "rgba(251,247,235,.72)";
    context.font = "15px monospace";
    context.textAlign = "right";
    context.fillText(`${state.samples.length} examples · ${state.model.step} optimizer steps`, exportWidth - 25, exportHeight + 39);

    exportCanvas.toBlob((blob) => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `dreamfield-${Date.now()}.png`;
      link.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    }, "image/png");
  }

  function eventToModelPoint(event) {
    const rect = elements.canvas.getBoundingClientRect();
    return {
      x: clamp((event.clientX - rect.left) / rect.width * 2 - 1, -1, 1),
      y: clamp((event.clientY - rect.top) / rect.height * 2 - 1, -1, 1)
    };
  }

  function handlePointerDown(event) {
    if (event.pointerType === "mouse" && event.button !== 0) return;
    if (state.activePointer !== null) return;
    state.activePointer = event.pointerId;
    beginStroke();
    elements.canvas.setPointerCapture(event.pointerId);
    elements.canvas.classList.add("is-painting");
    const point = eventToModelPoint(event);
    state.lastPaintPoint = point;
    stamp(point.x, point.y);
    updateProbe(point.x, point.y);
    event.preventDefault();
  }

  function handlePointerMove(event) {
    const point = eventToModelPoint(event);
    updateProbe(point.x, point.y);
    if (state.activePointer !== event.pointerId || !state.lastPaintPoint) return;

    const dxPixels = (point.x - state.lastPaintPoint.x) * state.cssWidth * 0.5;
    const dyPixels = (point.y - state.lastPaintPoint.y) * state.cssHeight * 0.5;
    const distance = Math.hypot(dxPixels, dyPixels);
    const steps = Math.max(1, Math.ceil(distance / 11));

    for (let index = 1; index <= steps; index += 1) {
      const progress = index / steps;
      stamp(
        state.lastPaintPoint.x + (point.x - state.lastPaintPoint.x) * progress,
        state.lastPaintPoint.y + (point.y - state.lastPaintPoint.y) * progress
      );
    }
    state.lastPaintPoint = point;
    event.preventDefault();
  }

  function endPointer(event) {
    if (state.activePointer !== event.pointerId) return;
    if (elements.canvas.hasPointerCapture(event.pointerId)) elements.canvas.releasePointerCapture(event.pointerId);
    state.activePointer = null;
    state.lastPaintPoint = null;
    elements.canvas.classList.remove("is-painting");
  }

  function handleCanvasKeydown(event) {
    const movement = event.shiftKey ? 0.12 : 0.035;
    let handled = true;
    switch (event.key) {
      case "ArrowLeft": state.keyboardCursor.x -= movement; break;
      case "ArrowRight": state.keyboardCursor.x += movement; break;
      case "ArrowUp": state.keyboardCursor.y -= movement; break;
      case "ArrowDown": state.keyboardCursor.y += movement; break;
      case " ":
      case "Enter":
        beginStroke();
        stamp(state.keyboardCursor.x, state.keyboardCursor.y);
        announce(`${state.selectedName} example added at x ${signed(state.keyboardCursor.x)}, y ${signed(state.keyboardCursor.y)}.`);
        break;
      default:
        handled = false;
    }
    if (!handled) return;
    event.preventDefault();
    state.keyboardCursor.x = clamp(state.keyboardCursor.x, -1, 1);
    state.keyboardCursor.y = clamp(state.keyboardCursor.y, -1, 1);
    updateProbe(state.keyboardCursor.x, state.keyboardCursor.y);
    state.renderDirty = true;
  }

  function syncResponsiveLayout() {
    const mobile = mobileLayout.matches;
    const firstPanel = mobile ? elements.fieldPanel : elements.toolPanel;
    if (elements.workbench.firstElementChild !== firstPanel) {
      elements.workbench.insertBefore(firstPanel, elements.workbench.firstElementChild);
    }
    elements.fieldNumber.textContent = `${mobile ? "01" : "02"} · Model prediction`;
    elements.teachNumber.textContent = mobile ? "02" : "01";
    window.requestAnimationFrame(resizeFieldCanvas);
  }

  function bindEvents() {
    syncResponsiveLayout();
    $$(".swatch").forEach((button) => {
      button.addEventListener("click", () => selectColor(button.dataset.color, button.dataset.name));
    });
    elements.customColor.addEventListener("input", (event) => selectColor(event.target.value, "Custom"));
    $$("[data-brush]").forEach((button) => button.addEventListener("click", () => selectBrush(button.dataset.brush)));
    $$("[data-preset]").forEach((button) => button.addEventListener("click", () => loadPreset(button.dataset.preset)));
    $$("[data-detail]").forEach((button) => button.addEventListener("click", () => selectDetail(button.dataset.detail)));

    elements.pauseButton.addEventListener("click", () => setPaused(!state.paused));
    elements.undoButton.addEventListener("click", undoLastStroke);
    elements.clearButton.addEventListener("click", clearField);
    elements.examplesButton.addEventListener("click", toggleExamples);
    elements.reseedButton.addEventListener("click", reseedModel);
    elements.saveButton.addEventListener("click", saveField);

    elements.canvas.addEventListener("pointerdown", handlePointerDown);
    elements.canvas.addEventListener("pointermove", handlePointerMove);
    elements.canvas.addEventListener("pointerup", endPointer);
    elements.canvas.addEventListener("pointercancel", endPointer);
    elements.canvas.addEventListener("keydown", handleCanvasKeydown);
    elements.canvas.addEventListener("focus", () => {
      updateProbe(state.keyboardCursor.x, state.keyboardCursor.y);
      state.renderDirty = true;
    });
    elements.canvas.addEventListener("blur", () => { state.renderDirty = true; });

    const resizeObserver = new ResizeObserver(resizeFieldCanvas);
    resizeObserver.observe(elements.fieldFrame);
    window.addEventListener("resize", resizeFieldCanvas, { passive: true });
    if (mobileLayout.addEventListener) {
      mobileLayout.addEventListener("change", syncResponsiveLayout);
    } else {
      mobileLayout.addListener(syncResponsiveLayout);
    }
    document.addEventListener("visibilitychange", () => {
      if (!document.hidden) {
        state.lastRenderAt = 0;
        state.renderDirty = true;
      }
    });
  }

  function animationLoop(timestamp) {
    state.frameCount += 1;
    state.isSettled = state.samples.length > 0 && state.lastLoss < 0.0025 && performance.now() - state.lastPaintAt > 1800;
    const trainThisFrame = !state.isSettled || state.frameCount % 8 === 0;

    if (!document.hidden && !state.paused && state.samples.length && trainThisFrame) {
      const budget = state.activePointer === null ? (reducedMotion ? 1.5 : 2.6) : 2;
      const started = performance.now();
      let batches = 0;
      const maximumBatches = state.isSettled ? 1 : 4;
      while (performance.now() - started < budget && batches < maximumBatches) {
        const batchLoss = state.model.trainBatch(buildBatch(16));
        state.lastLoss = state.lastLoss * 0.9 + batchLoss * 0.1;
        if (state.model.step % 12 === 0) pushLoss(state.lastLoss);
        batches += 1;
      }
      state.renderDirty = true;
    }

    const renderInterval = state.activePointer === null ? (state.isSettled ? 240 : 145) : 190;
    if (state.renderDirty && timestamp - state.lastRenderAt >= renderInterval) {
      renderField();
      state.lastRenderAt = timestamp;
    }
    if (timestamp - state.lastUiAt >= 180) {
      updateTelemetry();
      state.lastUiAt = timestamp;
    }

    window.requestAnimationFrame(animationLoop);
  }

  function pushLoss(loss) {
    if (!Number.isFinite(loss)) return;
    state.lossHistory.push(loss);
    if (state.lossHistory.length > 120) state.lossHistory.shift();
  }

  function announce(message) {
    elements.liveRegion.textContent = "";
    window.setTimeout(() => { elements.liveRegion.textContent = message; }, 20);
  }

  function hexToRgb(hex) {
    const value = hex.replace("#", "");
    return [
      parseInt(value.slice(0, 2), 16) / 255,
      parseInt(value.slice(2, 4), 16) / 255,
      parseInt(value.slice(4, 6), 16) / 255
    ];
  }

  function rgbToHex(rgb) {
    const channels = Array.from(rgb, (value) => clamp(Math.round(value * 255), 0, 255));
    return `#${channels.map((value) => value.toString(16).padStart(2, "0")).join("")}`.toUpperCase();
  }

  function clamp(value, minimum, maximum) {
    return Math.max(minimum, Math.min(maximum, value));
  }

  function signed(value) {
    const rounded = Math.abs(value) < 0.005 ? 0 : value;
    return `${rounded >= 0 ? "+" : ""}${rounded.toFixed(2)}`;
  }

  function formatLoss(value) {
    if (!Number.isFinite(value)) return "—";
    return value < 0.0001 ? value.toExponential(2) : value.toFixed(4);
  }

  function initialize() {
    bindEvents();
    state.model = new window.DreamfieldMLP({ hiddenSize: 20, bands: 3, seed: state.modelSeed });
    resizeFieldCanvas();
    loadPreset("night-garden", false);
    selectColor("#ff553e", "Vermilion");
    selectBrush("medium");
    setPaused(false, false);
    updateProbe(0, 0);
    renderField();
    drawLossChart();
    window.requestAnimationFrame(animationLoop);
  }

  initialize();
})();
