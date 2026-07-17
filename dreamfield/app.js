(function () {
  "use strict";

  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => Array.from(document.querySelectorAll(selector));
  const setText = (element, value) => {
    if (element.textContent !== value) element.textContent = value;
  };
  const PAPER_HEX = "#ede7d6";
  const PAPER_RGB = [237 / 255, 231 / 255, 214 / 255];
  const TARGET_WIDTH = 72;
  const TARGET_HEIGHT = 54;

  const elements = {
    canvas: $("#field-canvas"),
    teachingCanvas: $("#teaching-canvas"),
    fieldFrame: $("#field-frame"),
    teachingFrame: $("#teaching-frame"),
    workbench: $("#workbench"),
    toolPanel: $("#tool-panel"),
    fieldPanel: $("#field-panel"),
    teachNumber: $("#teach-number"),
    fieldNumber: $("#field-number"),
    drawingPrompt: $("#drawing-prompt"),
    sourceHidden: $("#source-hidden"),
    lessonCard: $("#lesson-card"),
    lessonStep: $("#lesson-step"),
    lessonTitle: $("#lesson-title"),
    lessonCopy: $("#lesson-copy"),
    loopExamples: $("#loop-examples"),
    loopLearning: $("#loop-learning"),
    loopLearningCopy: $("#loop-learning-copy"),
    loopPrediction: $("#loop-prediction"),
    causalArrowLabel: $("#causal-arrow-label"),
    predictionEmpty: $("#prediction-empty"),
    predictionStamp: $("#prediction-stamp"),
    predictionStampValue: $("#prediction-stamp-value"),
    exampleCountBadge: $("#example-count-badge"),
    predictionCountBadge: $("#prediction-count-badge"),
    coordinateX: $("#coordinate-x"),
    coordinateY: $("#coordinate-y"),
    mobileToolSummary: $("#mobile-tool-summary"),
    colorReadout: $("#color-readout"),
    customColor: $("#custom-color"),
    pauseButton: $("#pause-button"),
    undoButton: $("#undo-button"),
    clearButton: $("#clear-button"),
    hideSourceButton: $("#hide-source-button"),
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
  const teachingContext = elements.teachingCanvas.getContext("2d", { alpha: false });
  const lossContext = elements.lossCanvas.getContext("2d");
  const targetCanvas = document.createElement("canvas");
  const targetContext = targetCanvas.getContext("2d", { alpha: false, willReadFrequently: true });
  const maskCanvas = document.createElement("canvas");
  const maskContext = maskCanvas.getContext("2d", { alpha: false, willReadFrequently: true });
  const offscreen = document.createElement("canvas");
  const offscreenContext = offscreen.getContext("2d", { alpha: false });
  const exportBuffer = document.createElement("canvas");
  const exportContext = exportBuffer.getContext("2d", { alpha: false });
  const motionQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
  const mobileLayout = window.matchMedia("(max-width: 720px)");

  targetCanvas.width = TARGET_WIDTH;
  targetCanvas.height = TARGET_HEIGHT;
  maskCanvas.width = TARGET_WIDTH;
  maskCanvas.height = TARGET_HEIGHT;

  const detailSettings = {
    soft: { scale: 0.66, label: "Dreamy" },
    balanced: { scale: 1, label: "Faithful" },
    intricate: { scale: 1.42, label: "Wiggly" }
  };

  const brushSettings = {
    fine: { targetWidth: 3.2, label: "Fine" },
    medium: { targetWidth: 6.2, label: "Medium" },
    broad: { targetWidth: 10.2, label: "Bold" }
  };

  const state = {
    model: null,
    modelSeed: 2049,
    detail: "balanced",
    brush: "broad",
    selectedColor: "#161823",
    selectedName: "Midnight",
    strokes: [],
    currentStroke: null,
    nextStrokeId: 1,
    activePointer: null,
    foregroundSamples: [],
    edgeSamples: [],
    backgroundSamples: [],
    validationBatch: [],
    targetImageData: null,
    activePreset: null,
    replacementSnapshot: null,
    canRestoreReplacement: false,
    paused: false,
    sourceHidden: false,
    wasPausedBeforeHide: false,
    renderWidth: 84,
    renderHeight: 63,
    featureGrid: null,
    imageData: null,
    cssWidth: 0,
    cssHeight: 0,
    teachingCssWidth: 0,
    teachingCssHeight: 0,
    dpr: 1,
    renderDirty: true,
    teachingDirty: true,
    lossHistory: [],
    lastLoss: 0,
    practiceUntilStep: 0,
    lastRenderAt: 0,
    lastUiAt: 0,
    frameCount: 0,
    isSettled: false,
    lastUiSettledState: false,
    slowRenderCount: 0,
    keyboardCursor: { x: 0, y: 0 },
    probe: { x: 0, y: 0 }
  };

  function createModel(pretrainSteps = 0) {
    state.model = new window.DreamfieldMLP({
      hiddenSize: 28,
      bands: 4,
      frequencyScale: detailSettings[state.detail].scale,
      learningRate: 0.0065,
      seed: state.modelSeed
    });
    biasModelTowardPaper();
    reencodeDataset();
    state.lossHistory = [];
    state.lastLoss = state.validationBatch.length ? state.model.lossForBatch(state.validationBatch) : 0;

    if (!state.paused && state.foregroundSamples.length) practiceSynchronously(pretrainSteps);

    const count = state.model.parameterCount.toLocaleString();
    elements.parameterValue.textContent = count;
    elements.parameterBadge.textContent = `${count} parameters learn`;
    elements.predictionStampValue.textContent = `${count} learned numbers`;
    buildRenderGrid();
    state.renderDirty = true;
    updateTelemetry(true);
  }

  function biasModelTowardPaper() {
    const weights = state.model.params.w3;
    for (let index = 0; index < weights.length; index += 1) weights[index] *= 0.12;
    for (let channel = 0; channel < 3; channel += 1) {
      const value = clamp(PAPER_RGB[channel], 0.001, 0.999);
      state.model.params.b3[channel] = Math.log(value / (1 - value));
    }
  }

  function reencodeDataset() {
    if (!state.model) return;
    for (const pool of [state.foregroundSamples, state.edgeSamples, state.backgroundSamples]) {
      for (const sample of pool) sample.features = state.model.encode(sample.x, sample.y);
    }
    state.validationBatch = makeValidationBatch();
  }

  function drawStroke(context, stroke, width, height, colorOverride = null) {
    if (!stroke.points.length) return;
    const toX = (x) => (x + 1) * 0.5 * width;
    const toY = (y) => (y + 1) * 0.5 * height;
    const lineWidth = brushSettings[stroke.brush].targetWidth * height / TARGET_HEIGHT;
    context.save();
    context.strokeStyle = colorOverride || stroke.color;
    context.fillStyle = colorOverride || stroke.color;
    context.lineWidth = lineWidth;
    context.lineCap = "round";
    context.lineJoin = "round";
    if (stroke.points.length === 1) {
      context.beginPath();
      context.arc(toX(stroke.points[0].x), toY(stroke.points[0].y), lineWidth * 0.5, 0, Math.PI * 2);
      context.fill();
    } else {
      context.beginPath();
      context.moveTo(toX(stroke.points[0].x), toY(stroke.points[0].y));
      for (let index = 1; index < stroke.points.length - 1; index += 1) {
        const point = stroke.points[index];
        const next = stroke.points[index + 1];
        context.quadraticCurveTo(toX(point.x), toY(point.y), (toX(point.x) + toX(next.x)) * 0.5, (toY(point.y) + toY(next.y)) * 0.5);
      }
      const last = stroke.points[stroke.points.length - 1];
      context.lineTo(toX(last.x), toY(last.y));
      context.stroke();
    }
    context.restore();
  }

  function rebuildDataset() {
    state.foregroundSamples = [];
    state.edgeSamples = [];
    state.backgroundSamples = [];
    state.validationBatch = [];
    state.targetImageData = null;

    targetContext.fillStyle = PAPER_HEX;
    targetContext.fillRect(0, 0, TARGET_WIDTH, TARGET_HEIGHT);
    for (const stroke of state.strokes) drawStroke(targetContext, stroke, TARGET_WIDTH, TARGET_HEIGHT);
    maskContext.fillStyle = "#000000";
    maskContext.fillRect(0, 0, TARGET_WIDTH, TARGET_HEIGHT);
    for (const stroke of state.strokes) drawStroke(maskContext, stroke, TARGET_WIDTH, TARGET_HEIGHT, "#ffffff");

    if (!state.strokes.length) {
      state.lastLoss = 0;
      state.practiceUntilStep = 0;
      state.isSettled = false;
      state.renderDirty = true;
      return;
    }

    state.targetImageData = targetContext.getImageData(0, 0, TARGET_WIDTH, TARGET_HEIGHT);
    const pixels = state.targetImageData.data;
    const maskPixels = maskContext.getImageData(0, 0, TARGET_WIDTH, TARGET_HEIGHT).data;
    const foregroundMask = new Uint8Array(TARGET_WIDTH * TARGET_HEIGHT);

    for (let index = 0; index < foregroundMask.length; index += 1) {
      if (maskPixels[index * 4] > 18) foregroundMask[index] = 1;
    }

    for (let row = 0; row < TARGET_HEIGHT; row += 1) {
      for (let column = 0; column < TARGET_WIDTH; column += 1) {
        const index = row * TARGET_WIDTH + column;
        const offset = index * 4;
        const sample = {
          x: column / (TARGET_WIDTH - 1) * 2 - 1,
          y: row / (TARGET_HEIGHT - 1) * 2 - 1,
          rgb: [pixels[offset] / 255, pixels[offset + 1] / 255, pixels[offset + 2] / 255],
          features: null
        };

        if (foregroundMask[index]) {
          state.foregroundSamples.push(sample);
          continue;
        }

        let nearInk = false;
        for (let dy = -3; dy <= 3 && !nearInk; dy += 1) {
          const checkRow = row + dy;
          if (checkRow < 0 || checkRow >= TARGET_HEIGHT) continue;
          for (let dx = -3; dx <= 3; dx += 1) {
            if (dx * dx + dy * dy > 10) continue;
            const checkColumn = column + dx;
            if (checkColumn < 0 || checkColumn >= TARGET_WIDTH) continue;
            if (foregroundMask[checkRow * TARGET_WIDTH + checkColumn]) {
              nearInk = true;
              break;
            }
          }
        }
        (nearInk ? state.edgeSamples : state.backgroundSamples).push(sample);
      }
    }

    reencodeDataset();
    state.lastLoss = state.model.lossForBatch(state.validationBatch);
    state.practiceUntilStep = state.model.step + (motionQuery.matches ? 520 : 900);
    state.isSettled = false;
    state.renderDirty = true;
  }

  function makeValidationBatch() {
    if (!state.foregroundSamples.length) return [];
    return [
      ...pickEvenly(state.foregroundSamples, 28),
      ...pickEvenly(state.edgeSamples, 20),
      ...pickEvenly(state.backgroundSamples, 16)
    ];
  }

  function pickEvenly(pool, count) {
    if (!pool.length || count <= 0) return [];
    const result = [];
    for (let index = 0; index < count; index += 1) {
      result.push(pool[Math.floor(index * pool.length / count)]);
    }
    return result;
  }

  function takeRandom(pool) {
    return pool[Math.floor(Math.random() * pool.length)];
  }

  function buildBatch(size = 32) {
    if (!state.foregroundSamples.length) return [];
    const batch = [];
    const inkCount = Math.round(size * 0.44);
    const edgeCount = Math.round(size * 0.31);
    const edgePool = state.edgeSamples.length
      ? state.edgeSamples
      : state.backgroundSamples.length ? state.backgroundSamples : state.foregroundSamples;
    const backgroundPool = state.backgroundSamples.length
      ? state.backgroundSamples
      : state.edgeSamples.length ? state.edgeSamples : state.foregroundSamples;
    for (let index = 0; index < inkCount; index += 1) batch.push(takeRandom(state.foregroundSamples));
    for (let index = 0; index < edgeCount; index += 1) batch.push(takeRandom(edgePool));
    while (batch.length < size) batch.push(takeRandom(backgroundPool));
    return batch;
  }

  function practiceSynchronously(steps) {
    if (state.paused) return;
    for (let step = 0; step < steps && state.foregroundSamples.length; step += 1) {
      const loss = state.model.trainBatch(buildBatch(32));
      state.lastLoss = step ? state.lastLoss * 0.92 + loss * 0.08 : loss;
      if (step % 16 === 0) pushLoss(state.lastLoss);
    }
    if (state.validationBatch.length) state.lastLoss = state.model.lossForBatch(state.validationBatch);
  }

  function resizeCanvases() {
    const outputRect = elements.fieldFrame.getBoundingClientRect();
    const teachingRect = elements.teachingFrame.getBoundingClientRect();
    if (!outputRect.width || !outputRect.height || !teachingRect.width || !teachingRect.height) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const outputWidth = Math.round(outputRect.width);
    const outputHeight = Math.round(outputRect.height);
    const teachingWidth = Math.round(teachingRect.width);
    const teachingHeight = Math.round(teachingRect.height);

    if (outputWidth !== state.cssWidth || outputHeight !== state.cssHeight || dpr !== state.dpr) {
      state.cssWidth = outputWidth;
      state.cssHeight = outputHeight;
      elements.canvas.width = Math.max(1, Math.round(outputWidth * dpr));
      elements.canvas.height = Math.max(1, Math.round(outputHeight * dpr));
      state.renderWidth = window.innerWidth < 720 ? 72 : 84;
      buildRenderGrid();
      state.renderDirty = true;
    }

    if (teachingWidth !== state.teachingCssWidth || teachingHeight !== state.teachingCssHeight || dpr !== state.dpr) {
      state.teachingCssWidth = teachingWidth;
      state.teachingCssHeight = teachingHeight;
      elements.teachingCanvas.width = Math.max(1, Math.round(teachingWidth * dpr));
      elements.teachingCanvas.height = Math.max(1, Math.round(teachingHeight * dpr));
      state.teachingDirty = true;
    }
    state.dpr = dpr;
  }

  function buildRenderGrid() {
    if (!state.model || !state.cssWidth || !state.cssHeight) return;
    state.renderHeight = Math.max(48, Math.round(state.renderWidth * state.cssHeight / state.cssWidth));
    offscreen.width = state.renderWidth;
    offscreen.height = state.renderHeight;
    state.imageData = offscreenContext.createImageData(state.renderWidth, state.renderHeight);
    state.featureGrid = new Float32Array(state.renderWidth * state.renderHeight * state.model.inputSize);

    let pixel = 0;
    for (let row = 0; row < state.renderHeight; row += 1) {
      const y = row / Math.max(1, state.renderHeight - 1) * 2 - 1;
      for (let column = 0; column < state.renderWidth; column += 1) {
        const x = column / Math.max(1, state.renderWidth - 1) * 2 - 1;
        state.model.encode(x, y, state.featureGrid, pixel * state.model.inputSize);
        pixel += 1;
      }
    }
    const pixelCount = state.renderWidth * state.renderHeight;
    elements.gridValue.textContent = pixelCount.toLocaleString();
  }

  function drawPaper(context, width, height, darkGrid = false) {
    context.fillStyle = PAPER_HEX;
    context.fillRect(0, 0, width, height);
    context.save();
    context.strokeStyle = darkGrid ? "rgba(23,24,19,.07)" : "rgba(23,24,19,.045)";
    context.lineWidth = 1;
    const gap = Math.max(22, Math.round(width / 12));
    for (let x = gap; x < width; x += gap) {
      context.beginPath();
      context.moveTo(x, 0);
      context.lineTo(x, height);
      context.stroke();
    }
    for (let y = gap; y < height; y += gap) {
      context.beginPath();
      context.moveTo(0, y);
      context.lineTo(width, y);
      context.stroke();
    }
    context.restore();
  }

  function renderTeachingCanvas() {
    if (!state.teachingCssWidth || !state.teachingCssHeight) return;
    teachingContext.setTransform(state.dpr, 0, 0, state.dpr, 0, 0);
    drawPaper(teachingContext, state.teachingCssWidth, state.teachingCssHeight, true);
    if (!state.sourceHidden) {
      for (const stroke of state.strokes) drawStroke(teachingContext, stroke, state.teachingCssWidth, state.teachingCssHeight);
      if (state.currentStroke) drawStroke(teachingContext, state.currentStroke, state.teachingCssWidth, state.teachingCssHeight);
      if (document.activeElement === elements.teachingCanvas && state.activePointer === null) drawKeyboardCursor();
    }
    elements.drawingPrompt.hidden = Boolean(state.strokes.length || state.currentStroke || state.sourceHidden);
    elements.sourceHidden.hidden = !state.sourceHidden;
    const count = state.strokes.length;
    elements.exampleCountBadge.textContent = `${count} ${count === 1 ? "stroke" : "strokes"}`;
    state.teachingDirty = false;
  }

  function drawKeyboardCursor() {
    const x = (state.keyboardCursor.x + 1) * 0.5 * state.teachingCssWidth;
    const y = (state.keyboardCursor.y + 1) * 0.5 * state.teachingCssHeight;
    teachingContext.save();
    teachingContext.translate(x, y);
    teachingContext.strokeStyle = state.selectedColor;
    teachingContext.lineWidth = 2;
    teachingContext.shadowColor = PAPER_HEX;
    teachingContext.shadowBlur = 5;
    teachingContext.beginPath();
    teachingContext.arc(0, 0, 11, 0, Math.PI * 2);
    teachingContext.moveTo(-17, 0);
    teachingContext.lineTo(-6, 0);
    teachingContext.moveTo(6, 0);
    teachingContext.lineTo(17, 0);
    teachingContext.moveTo(0, -17);
    teachingContext.lineTo(0, -6);
    teachingContext.moveTo(0, 6);
    teachingContext.lineTo(0, 17);
    teachingContext.stroke();
    teachingContext.restore();
  }

  function renderField() {
    if (!state.model || !state.cssWidth || !state.cssHeight) return;
    fieldContext.setTransform(state.dpr, 0, 0, state.dpr, 0, 0);
    drawPaper(fieldContext, state.cssWidth, state.cssHeight);

    if (!state.foregroundSamples.length) {
      elements.predictionEmpty.hidden = false;
      elements.predictionStamp.hidden = true;
      elements.predictionCountBadge.textContent = "Waiting";
      elements.canvas.setAttribute("aria-hidden", "true");
      elements.canvas.setAttribute("aria-label", "No neural redraw yet. Add a stroke on the left to begin.");
      state.renderDirty = false;
      return;
    }
    if (!state.imageData || !state.featureGrid) return;

    const started = performance.now();
    const pixels = state.imageData.data;
    const prediction = new Float32Array(3);
    const pixelCount = state.renderWidth * state.renderHeight;
    for (let pixel = 0; pixel < pixelCount; pixel += 1) {
      state.model.forwardEncoded(state.featureGrid, pixel * state.model.inputSize, prediction);
      const offset = pixel * 4;
      pixels[offset] = clamp(Math.round(prediction[0] * 255), 0, 255);
      pixels[offset + 1] = clamp(Math.round(prediction[1] * 255), 0, 255);
      pixels[offset + 2] = clamp(Math.round(prediction[2] * 255), 0, 255);
      pixels[offset + 3] = 255;
    }

    offscreenContext.putImageData(state.imageData, 0, 0);
    fieldContext.imageSmoothingEnabled = true;
    fieldContext.imageSmoothingQuality = "high";
    fieldContext.drawImage(offscreen, 0, 0, state.cssWidth, state.cssHeight);
    elements.predictionEmpty.hidden = true;
    elements.predictionStamp.hidden = state.model.step < 60;
    elements.predictionCountBadge.textContent = `${pixelCount.toLocaleString()} predictions`;
    elements.canvas.removeAttribute("aria-hidden");
    elements.canvas.setAttribute("aria-label", "The neural network's redraw, generated from its learned parameters. This canvas is view-only.");

    const renderTime = performance.now() - started;
    if (renderTime > 42 && state.renderWidth > 66) {
      state.slowRenderCount += 1;
      if (state.slowRenderCount >= 2) {
        state.renderWidth = Math.max(66, Math.floor(state.renderWidth * 0.86));
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

  function drawLossChart() {
    const context = lossContext;
    const width = elements.lossCanvas.width;
    const height = elements.lossCanvas.height;
    context.clearRect(0, 0, width, height);
    context.strokeStyle = "rgba(242,237,223,.12)";
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
    fill.addColorStop(0, "rgba(233,255,67,.34)");
    fill.addColorStop(1, "rgba(233,255,67,0)");
    context.beginPath();
    context.moveTo(points[0].x, height);
    for (const point of points) context.lineTo(point.x, point.y);
    context.lineTo(points[points.length - 1].x, height);
    context.closePath();
    context.fillStyle = fill;
    context.fill();
    context.beginPath();
    context.moveTo(points[0].x, points[0].y);
    for (let index = 1; index < points.length; index += 1) context.lineTo(points[index].x, points[index].y);
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

  function updateLessonState() {
    const hasDoodle = state.strokes.length > 0;
    const practicing = hasDoodle && !state.paused && !state.isSettled;
    elements.lessonCard.classList.toggle("is-complete", hasDoodle && state.isSettled);

    if (!hasDoodle && !state.currentStroke) {
      elements.lessonStep.textContent = "Your turn";
      elements.lessonTitle.textContent = "Draw one big, simple shape.";
      elements.lessonCopy.textContent = "Try a letter, face, or heart. Thick strokes give this tiny apprentice the best chance.";
    } else if (state.currentStroke) {
      elements.lessonStep.textContent = "Collecting a lesson";
      elements.lessonTitle.textContent = "Your stroke is becoming training pixels.";
      elements.lessonCopy.textContent = "When you lift the brush, the apprentice will study your ink, its edges, and the blank paper.";
    } else if (state.paused) {
      elements.lessonStep.textContent = state.sourceHidden ? "Memory test" : "Practice paused";
      elements.lessonTitle.textContent = state.sourceHidden ? "The original is hidden. The redraw remains." : "The current redraw is frozen.";
      elements.lessonCopy.textContent = "The right canvas is rendered from learned parameters. Resume whenever you want it to keep improving.";
    } else if (practicing) {
      elements.lessonStep.textContent = `Practicing stroke ${state.strokes.length}`;
      elements.lessonTitle.textContent = "The apprentice is comparing pixels.";
      elements.lessonCopy.textContent = "Backpropagation nudges the parameters toward your ink and away from the sampled blank paper.";
    } else {
      elements.lessonStep.textContent = "Now compare them";
      elements.lessonTitle.textContent = "The whole redraw comes from predictions.";
      elements.lessonCopy.textContent = "It may simplify fine details or wobble at the edges. That imperfection is the tiny network showing its work.";
    }

    setLoopState(elements.loopExamples, !hasDoodle ? "active" : "complete");
    setLoopState(elements.loopLearning, practicing ? "active" : hasDoodle ? "complete" : "idle");
    setLoopState(elements.loopPrediction, hasDoodle ? (state.isSettled ? "complete" : "active") : "idle");
  }

  function setLoopState(element, status) {
    element.classList.toggle("is-active", status === "active");
    element.classList.toggle("is-complete", status === "complete");
    if (status === "active") element.setAttribute("aria-current", "step");
    else element.removeAttribute("aria-current");
  }

  function updateTelemetry(force = false) {
    if (!state.model) return;
    const hasDoodle = state.foregroundSamples.length > 0;
    const trainingPixelCount = state.foregroundSamples.length + state.edgeSamples.length + state.backgroundSamples.length;
    elements.stepValue.textContent = state.model.step.toLocaleString();
    elements.sampleValue.textContent = hasDoodle ? trainingPixelCount.toLocaleString() : "0";
    elements.pauseButton.disabled = !state.strokes.length || state.sourceHidden;
    elements.undoButton.disabled = !state.strokes.length && !state.canRestoreReplacement;
    elements.undoButton.textContent = state.canRestoreReplacement ? "Restore doodle" : "Undo stroke";
    elements.clearButton.disabled = !state.strokes.length;
    elements.hideSourceButton.disabled = !state.strokes.length;
    elements.reseedButton.disabled = !state.strokes.length;
    elements.saveButton.disabled = !state.strokes.length;

    if (!hasDoodle) {
      elements.lossValue.textContent = "—";
      elements.fitBadge.textContent = "No doodle";
      setText(elements.trainingLabel, state.currentStroke ? "Collecting your stroke" : "Waiting for a doodle");
      elements.loopLearningCopy.textContent = state.currentStroke ? "collecting pixels" : "waiting for data";
      elements.causalArrowLabel.textContent = "teach";
      elements.headerStatus.textContent = "Ready to learn";
    } else {
      elements.lossValue.textContent = formatLoss(state.lastLoss);
      if (state.paused) {
        elements.fitBadge.textContent = "Paused";
        setText(elements.trainingLabel, state.sourceHidden ? "Memory test" : "Learning paused");
        elements.loopLearningCopy.textContent = "paused";
        elements.causalArrowLabel.textContent = "remembers";
        elements.headerStatus.textContent = "Parameters frozen";
      } else if (state.isSettled) {
        elements.fitBadge.textContent = "Round complete";
        setText(elements.trainingLabel, "Practice round complete");
        elements.loopLearningCopy.textContent = `${state.model.step.toLocaleString()} updates`;
        elements.causalArrowLabel.textContent = "redraws";
        elements.headerStatus.textContent = "Practice complete";
      } else {
        elements.fitBadge.textContent = "Practicing";
        setText(elements.trainingLabel, "Apprentice practicing");
        elements.loopLearningCopy.textContent = "adjusting parameters";
        elements.causalArrowLabel.textContent = "learning";
        elements.headerStatus.textContent = "Learning locally";
      }
      elements.lossCanvas.setAttribute("aria-label", `Recent balanced training error chart. Current error ${formatLoss(state.lastLoss)}.`);
    }

    elements.fieldStatus.classList.toggle("is-paused", state.paused || !hasDoodle);
    elements.localStatus.classList.toggle("is-paused", state.paused);
    updateLessonState();
    updateProbe(state.probe.x, state.probe.y);
    if (force || state.frameCount % 3 === 0) drawLossChart();
    state.lastUiSettledState = state.isSettled;
  }

  function updateProbe(x, y) {
    state.probe.x = clamp(x, -1, 1);
    state.probe.y = clamp(y, -1, 1);
    elements.coordinateX.textContent = `x ${signed(state.probe.x)}`;
    elements.coordinateY.textContent = `y ${signed(state.probe.y)}`;
    elements.probePosition.textContent = `x ${signed(state.probe.x)} · y ${signed(state.probe.y)}`;

    const targetRgb = targetColorAt(state.probe.x, state.probe.y);
    elements.exampleChip.style.background = rgbToHex(targetRgb);
    elements.exampleValue.textContent = colorDistance(targetRgb, PAPER_RGB) < 0.04 ? "PAPER" : rgbToHex(targetRgb);
    if (!state.foregroundSamples.length) {
      elements.predictedChip.style.background = PAPER_HEX;
      elements.predictedValue.textContent = "WAITING";
      return;
    }
    const prediction = state.model.predict(state.probe.x, state.probe.y);
    const predictionHex = rgbToHex(prediction);
    elements.predictedChip.style.background = predictionHex;
    elements.predictedValue.textContent = predictionHex;
  }

  function targetColorAt(x, y) {
    if (!state.targetImageData) return PAPER_RGB;
    const column = clamp(Math.round((x + 1) * 0.5 * (TARGET_WIDTH - 1)), 0, TARGET_WIDTH - 1);
    const row = clamp(Math.round((y + 1) * 0.5 * (TARGET_HEIGHT - 1)), 0, TARGET_HEIGHT - 1);
    const offset = (row * TARGET_WIDTH + column) * 4;
    return [
      state.targetImageData.data[offset] / 255,
      state.targetImageData.data[offset + 1] / 255,
      state.targetImageData.data[offset + 2] / 255
    ];
  }

  function selectColor(color, name) {
    state.selectedColor = color.toLowerCase();
    state.selectedName = name;
    elements.customColor.value = state.selectedColor;
    elements.colorReadout.textContent = `${name} · ${state.selectedColor}`;
    elements.mobileToolSummary.textContent = `${name} · ${brushSettings[state.brush].label}`;
    $$('[data-color]').forEach((button) => {
      const active = button.dataset.color.toLowerCase() === state.selectedColor;
      button.setAttribute("aria-pressed", String(active));
    });
    updateTeachingCanvasLabel();
    state.teachingDirty = true;
  }

  function selectBrush(brush) {
    state.brush = brush;
    $$('[data-brush]').forEach((button) => {
      const active = button.dataset.brush === brush;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    elements.mobileToolSummary.textContent = `${state.selectedName} · ${brushSettings[brush].label}`;
    updateTeachingCanvasLabel();
    state.teachingDirty = true;
  }

  function updateTeachingCanvasLabel() {
    if (state.sourceHidden) {
      elements.teachingCanvas.setAttribute("aria-label", "Your doodle is hidden for the memory test. Use Show my doodle to make the drawing canvas available again.");
      return;
    }
    elements.teachingCanvas.setAttribute("aria-label", `Your doodle canvas. ${state.selectedName} ${brushSettings[state.brush].label.toLowerCase()} brush selected.`);
  }

  function selectDetail(detail) {
    if (!detailSettings[detail] || detail === state.detail) return;
    state.detail = detail;
    $$('[data-detail]').forEach((button) => {
      const active = button.dataset.detail === detail;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    if (state.strokes.length) {
      showSource(false);
      setPaused(false, false);
    }
    state.modelSeed += 31;
    createModel(state.strokes.length ? 90 : 0);
    if (state.strokes.length) {
      state.practiceUntilStep = state.model.step + 800;
      state.isSettled = false;
    }
    announce(`${detailSettings[detail].label} apprentice selected. The network is retraining from scratch.`);
  }

  function setPaused(paused, shouldAnnounce = true) {
    state.paused = paused;
    elements.pauseButton.classList.toggle("is-paused", paused);
    elements.pauseButton.querySelector("span:last-child").textContent = paused ? "Resume learning" : "Pause learning";
    updateTelemetry(true);
    if (shouldAnnounce) announce(paused ? "Learning paused. The current redraw is frozen." : "Learning resumed. The apprentice is practicing again.");
  }

  function cloneStrokes(strokes) {
    return strokes.map((stroke) => ({
      id: stroke.id,
      color: stroke.color,
      name: stroke.name,
      brush: stroke.brush,
      points: stroke.points.map((point) => ({ x: point.x, y: point.y }))
    }));
  }

  function saveReplacementSnapshot() {
    if (!state.strokes.length) return;
    state.replacementSnapshot = {
      strokes: cloneStrokes(state.strokes),
      nextStrokeId: state.nextStrokeId,
      modelSeed: state.modelSeed,
      activePreset: state.activePreset
    };
    state.canRestoreReplacement = true;
  }

  function restoreReplacement() {
    if (!state.replacementSnapshot) return;
    const snapshot = state.replacementSnapshot;
    state.strokes = cloneStrokes(snapshot.strokes);
    state.nextStrokeId = snapshot.nextStrokeId;
    state.modelSeed = snapshot.modelSeed;
    state.activePreset = snapshot.activePreset;
    state.replacementSnapshot = null;
    state.canRestoreReplacement = false;
    showSource(false);
    setPaused(false, false);
    syncPresetButtons();
    rebuildDataset();
    createModel(110);
    state.practiceUntilStep = state.model.step + 750;
    state.teachingDirty = true;
    announce("Your previous doodle is back. The apprentice is relearning it.");
  }

  function beginStroke(point) {
    state.canRestoreReplacement = false;
    state.replacementSnapshot = null;
    setActivePreset(null);
    state.currentStroke = {
      id: state.nextStrokeId++,
      color: state.selectedColor,
      name: state.selectedName,
      brush: state.brush,
      points: [point]
    };
    state.teachingDirty = true;
    updateLessonState();
  }

  function appendStrokePoint(point) {
    if (!state.currentStroke) return;
    const previous = state.currentStroke.points[state.currentStroke.points.length - 1];
    const dxPixels = (point.x - previous.x) * state.teachingCssWidth * 0.5;
    const dyPixels = (point.y - previous.y) * state.teachingCssHeight * 0.5;
    const distance = Math.hypot(dxPixels, dyPixels);
    const steps = Math.max(1, Math.ceil(distance / 5));
    for (let index = 1; index <= steps; index += 1) {
      const progress = index / steps;
      state.currentStroke.points.push({
        x: previous.x + (point.x - previous.x) * progress,
        y: previous.y + (point.y - previous.y) * progress
      });
    }
    if (state.currentStroke.points.length > 900) state.currentStroke.points.splice(1, state.currentStroke.points.length - 900);
    state.teachingDirty = true;
  }

  function finishStroke() {
    if (!state.currentStroke) return;
    const stroke = state.currentStroke;
    state.currentStroke = null;
    state.strokes.push(stroke);
    rebuildDataset();
    practiceSynchronously(state.strokes.length === 1 ? 85 : 42);
    state.renderDirty = true;
    state.teachingDirty = true;
    updateTelemetry(true);
    const examples = state.foregroundSamples.length;
    const trainingMessage = state.paused
      ? "are ready to train when you resume learning"
      : "are training the neural redraw";
    announce(`Added ${stroke.name} stroke ${state.strokes.length}. ${examples.toLocaleString()} ink pixels and sampled blank paper ${trainingMessage}.`);
  }

  function undoLastStroke() {
    if (state.canRestoreReplacement) {
      restoreReplacement();
      return;
    }
    if (!state.strokes.length) return;
    const removed = state.strokes.pop();
    setActivePreset(null);
    showSource(false);
    setPaused(false, false);
    rebuildDataset();
    state.modelSeed += 7;
    createModel(state.strokes.length ? 110 : 0);
    if (state.strokes.length) state.practiceUntilStep = state.model.step + 750;
    state.teachingDirty = true;
    announce(`${removed.name} stroke removed. ${state.strokes.length ? "The remaining doodle is being relearned." : "The canvas is empty."}`);
  }

  function clearDoodle() {
    if (!state.strokes.length) return;
    saveReplacementSnapshot();
    state.strokes = [];
    state.currentStroke = null;
    state.activePreset = null;
    state.nextStrokeId = 1;
    showSource(false);
    setPaused(false, false);
    rebuildDataset();
    state.modelSeed += 13;
    createModel(0);
    syncPresetButtons();
    state.teachingDirty = true;
    updateTelemetry(true);
    announce("Doodle cleared. Add a mark on the left to begin, or restore your previous doodle.");
  }

  function showSource(hidden) {
    state.sourceHidden = hidden;
    elements.sourceHidden.hidden = !hidden;
    elements.drawingPrompt.hidden = Boolean(state.strokes.length || state.currentStroke || hidden);
    elements.hideSourceButton.setAttribute("aria-pressed", String(hidden));
    elements.hideSourceButton.textContent = hidden ? "Show my doodle" : "Hide my doodle";
    if (hidden) {
      elements.teachingCanvas.setAttribute("tabindex", "-1");
      elements.teachingCanvas.setAttribute("aria-disabled", "true");
      if (document.activeElement === elements.teachingCanvas) elements.teachingCanvas.blur();
    } else {
      elements.teachingCanvas.setAttribute("tabindex", "0");
      elements.teachingCanvas.removeAttribute("aria-disabled");
    }
    updateTeachingCanvasLabel();
    state.teachingDirty = true;
  }

  function toggleSource() {
    if (!state.strokes.length) return;
    if (!state.sourceHidden) {
      state.wasPausedBeforeHide = state.paused;
      showSource(true);
      setPaused(true, false);
      announce("Your doodle is hidden and learning is paused. The redraw on the right now comes only from the learned parameters.");
    } else {
      showSource(false);
      setPaused(state.wasPausedBeforeHide, false);
      announce("Your doodle is visible again.");
    }
  }

  function reseedModel() {
    if (!state.strokes.length) return;
    state.modelSeed += 997;
    showSource(false);
    setPaused(false, false);
    createModel(90);
    state.practiceUntilStep = state.model.step + 900;
    state.isSettled = false;
    announce("A new apprentice started from different random parameters and is learning the same doodle.");
  }

  function saveRedraw() {
    if (!state.strokes.length) return;
    if (state.renderDirty) renderField();
    exportBuffer.width = 1200;
    exportBuffer.height = 900;
    exportContext.fillStyle = PAPER_HEX;
    exportContext.fillRect(0, 0, exportBuffer.width, exportBuffer.height);
    exportContext.imageSmoothingEnabled = true;
    exportContext.imageSmoothingQuality = "high";
    exportContext.drawImage(offscreen, 0, 0, exportBuffer.width, exportBuffer.height);
    const link = document.createElement("a");
    link.download = `doodle-apprentice-${Date.now()}.png`;
    link.href = exportBuffer.toDataURL("image/png");
    link.click();
    announce("The apprentice redraw was saved as a PNG.");
  }

  function setActivePreset(key) {
    state.activePreset = key;
    syncPresetButtons();
  }

  function syncPresetButtons() {
    $$('[data-preset]').forEach((button) => {
      const active = button.dataset.preset === state.activePreset;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });
  }

  function makeStroke(points, color, name = "Example", brush = "broad") {
    return { id: state.nextStrokeId++, color, name, brush, points };
  }

  function loadPreset(key) {
    if (state.strokes.length) saveReplacementSnapshot();
    state.nextStrokeId = 1;
    const strokes = [];
    if (key === "heart") {
      const points = [];
      for (let index = 0; index <= 100; index += 1) {
        const t = index / 100 * Math.PI * 2;
        points.push({
          x: 0.74 * Math.pow(Math.sin(t), 3),
          y: -0.06 - 0.055 * (13 * Math.cos(t) - 5 * Math.cos(2 * t) - 2 * Math.cos(3 * t) - Math.cos(4 * t))
        });
      }
      strokes.push(makeStroke(points, "#ff553e", "Vermilion", "broad"));
    } else if (key === "smiley") {
      const circle = [];
      for (let index = 0; index <= 90; index += 1) {
        const t = index / 90 * Math.PI * 2;
        circle.push({ x: Math.cos(t) * 0.72, y: Math.sin(t) * 0.72 });
      }
      const mouth = [];
      for (let index = 0; index <= 34; index += 1) {
        const t = index / 34 * Math.PI;
        mouth.push({ x: -0.43 + index / 34 * 0.86, y: 0.17 + Math.sin(t) * 0.31 });
      }
      strokes.push(makeStroke(circle, "#161823", "Midnight", "medium"));
      strokes.push(makeStroke([{ x: -0.27, y: -0.20 }], "#2664ff", "Cobalt", "broad"));
      strokes.push(makeStroke([{ x: 0.27, y: -0.20 }], "#2664ff", "Cobalt", "broad"));
      strokes.push(makeStroke(mouth, "#ff553e", "Vermilion", "medium"));
    } else if (key === "star") {
      const points = [];
      for (let index = 0; index <= 10; index += 1) {
        const radius = index % 2 === 0 ? 0.78 : 0.34;
        const t = -Math.PI / 2 + index * Math.PI / 5;
        points.push({ x: Math.cos(t) * radius, y: Math.sin(t) * radius });
      }
      strokes.push(makeStroke(points, "#2664ff", "Cobalt", "broad"));
    } else {
      return;
    }

    state.strokes = strokes;
    state.canRestoreReplacement = Boolean(state.replacementSnapshot);
    setActivePreset(key);
    showSource(false);
    setPaused(false, false);
    rebuildDataset();
    state.modelSeed += 53;
    createModel(140);
    state.practiceUntilStep = state.model.step + 1000;
    state.isSettled = false;
    state.teachingDirty = true;
    announce(`${key[0].toUpperCase() + key.slice(1)} example loaded. It replaced the current doodle, and the apprentice is practicing now.`);
  }

  function eventToModelPoint(event) {
    const rect = elements.teachingCanvas.getBoundingClientRect();
    return {
      x: clamp((event.clientX - rect.left) / rect.width * 2 - 1, -1, 1),
      y: clamp((event.clientY - rect.top) / rect.height * 2 - 1, -1, 1)
    };
  }

  function handlePointerDown(event) {
    if (state.sourceHidden || (event.pointerType === "mouse" && event.button !== 0) || state.activePointer !== null) return;
    state.activePointer = event.pointerId;
    elements.teachingCanvas.setPointerCapture(event.pointerId);
    elements.teachingCanvas.classList.add("is-painting");
    const point = eventToModelPoint(event);
    beginStroke(point);
    updateProbe(point.x, point.y);
    event.preventDefault();
  }

  function handlePointerMove(event) {
    const point = eventToModelPoint(event);
    updateProbe(point.x, point.y);
    if (state.activePointer !== event.pointerId || !state.currentStroke) return;
    appendStrokePoint(point);
    event.preventDefault();
  }

  function endPointer(event) {
    if (state.activePointer !== event.pointerId) return;
    if (elements.teachingCanvas.hasPointerCapture(event.pointerId)) elements.teachingCanvas.releasePointerCapture(event.pointerId);
    state.activePointer = null;
    elements.teachingCanvas.classList.remove("is-painting");
    finishStroke();
  }

  function handleCanvasKeydown(event) {
    if (state.sourceHidden) return;
    const movement = event.shiftKey ? 0.12 : 0.035;
    let handled = true;
    switch (event.key) {
      case "ArrowLeft": state.keyboardCursor.x -= movement; break;
      case "ArrowRight": state.keyboardCursor.x += movement; break;
      case "ArrowUp": state.keyboardCursor.y -= movement; break;
      case "ArrowDown": state.keyboardCursor.y += movement; break;
      case " ":
      case "Enter":
        beginStroke({ x: state.keyboardCursor.x, y: state.keyboardCursor.y });
        finishStroke();
        break;
      default: handled = false;
    }
    if (!handled) return;
    event.preventDefault();
    state.keyboardCursor.x = clamp(state.keyboardCursor.x, -1, 1);
    state.keyboardCursor.y = clamp(state.keyboardCursor.y, -1, 1);
    updateProbe(state.keyboardCursor.x, state.keyboardCursor.y);
    state.teachingDirty = true;
  }

  function syncResponsiveLayout() {
    const mobile = mobileLayout.matches;
    const firstPanel = mobile ? elements.fieldPanel : elements.toolPanel;
    if (elements.workbench.firstElementChild !== firstPanel) elements.workbench.insertBefore(firstPanel, elements.workbench.firstElementChild);
    elements.fieldNumber.textContent = `${mobile ? "01" : "02"} · The imitation game`;
    elements.teachNumber.textContent = mobile ? "02" : "01";
    window.requestAnimationFrame(resizeCanvases);
  }

  function bindEvents() {
    $$('[data-color]').forEach((button) => button.addEventListener("click", () => selectColor(button.dataset.color, button.dataset.name)));
    elements.customColor.addEventListener("input", (event) => selectColor(event.target.value, "Custom"));
    $$('[data-brush]').forEach((button) => button.addEventListener("click", () => selectBrush(button.dataset.brush)));
    $$('[data-preset]').forEach((button) => button.addEventListener("click", () => loadPreset(button.dataset.preset)));
    $$('[data-detail]').forEach((button) => button.addEventListener("click", () => selectDetail(button.dataset.detail)));
    elements.pauseButton.addEventListener("click", () => setPaused(!state.paused));
    elements.undoButton.addEventListener("click", undoLastStroke);
    elements.clearButton.addEventListener("click", clearDoodle);
    elements.hideSourceButton.addEventListener("click", toggleSource);
    elements.reseedButton.addEventListener("click", reseedModel);
    elements.saveButton.addEventListener("click", saveRedraw);

    elements.teachingCanvas.addEventListener("pointerdown", handlePointerDown);
    elements.teachingCanvas.addEventListener("pointermove", handlePointerMove);
    elements.teachingCanvas.addEventListener("pointerup", endPointer);
    elements.teachingCanvas.addEventListener("pointercancel", endPointer);
    elements.teachingCanvas.addEventListener("keydown", handleCanvasKeydown);
    elements.teachingCanvas.addEventListener("focus", () => { state.teachingDirty = true; });
    elements.teachingCanvas.addEventListener("blur", () => { state.teachingDirty = true; });

    const resizeObserver = new ResizeObserver(resizeCanvases);
    resizeObserver.observe(elements.fieldFrame);
    resizeObserver.observe(elements.teachingFrame);
    window.addEventListener("resize", resizeCanvases, { passive: true });
    if (mobileLayout.addEventListener) mobileLayout.addEventListener("change", syncResponsiveLayout);
    else mobileLayout.addListener(syncResponsiveLayout);
    const handleMotionChange = () => {
      if (motionQuery.matches && state.foregroundSamples.length) state.practiceUntilStep = Math.min(state.practiceUntilStep, state.model.step + 520);
    };
    if (motionQuery.addEventListener) motionQuery.addEventListener("change", handleMotionChange);
    else motionQuery.addListener(handleMotionChange);
    document.addEventListener("visibilitychange", () => {
      if (!document.hidden) {
        state.lastRenderAt = 0;
        state.renderDirty = true;
        state.teachingDirty = true;
      }
    });
  }

  function animationLoop(timestamp) {
    state.frameCount += 1;
    const hasDoodle = state.foregroundSamples.length > 0;
    const hasReachedGoodFit = state.model.step >= 450 && state.lastLoss < 0.004;
    state.isSettled = hasDoodle && (state.model.step >= state.practiceUntilStep || hasReachedGoodFit);
    const shouldRefineSettled = !motionQuery.matches && state.frameCount % 18 === 0;
    const shouldPractice = hasDoodle && !state.paused && (!state.isSettled || shouldRefineSettled);

    if (!document.hidden && shouldPractice) {
      const budget = state.isSettled ? 1.2 : motionQuery.matches ? 5 : 7;
      const maximumBatches = state.isSettled ? 1 : 18;
      const started = performance.now();
      let batches = 0;
      while (performance.now() - started < budget && batches < maximumBatches) {
        const loss = state.model.trainBatch(buildBatch(32));
        state.lastLoss = state.lastLoss * 0.92 + loss * 0.08;
        batches += 1;
      }
      if (state.model.step % 18 < batches) {
        if (state.validationBatch.length) state.lastLoss = state.model.lossForBatch(state.validationBatch);
        pushLoss(state.lastLoss);
      }
      state.renderDirty = true;
    }

    if (state.teachingDirty) renderTeachingCanvas();
    const mayRenderMotion = !motionQuery.matches || state.isSettled || state.model.step < 100;
    const renderInterval = state.isSettled ? 520 : 145;
    if (state.renderDirty && mayRenderMotion && timestamp - state.lastRenderAt >= renderInterval) {
      renderField();
      state.lastRenderAt = timestamp;
    }
    const uiMilestoneChanged = state.isSettled !== state.lastUiSettledState;
    const shouldUpdateUi = motionQuery.matches
      ? uiMilestoneChanged
      : timestamp - state.lastUiAt >= 180;
    if (shouldUpdateUi) {
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

  function rgbToHex(rgb) {
    const channels = Array.from(rgb, (value) => clamp(Math.round(value * 255), 0, 255));
    return `#${channels.map((value) => value.toString(16).padStart(2, "0")).join("")}`.toUpperCase();
  }

  function colorDistance(a, b) {
    return Math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2);
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
    syncResponsiveLayout();
    createModel(0);
    selectColor("#161823", "Midnight");
    selectBrush("broad");
    resizeCanvases();
    updateProbe(0, 0);
    renderTeachingCanvas();
    renderField();
    drawLossChart();
    updateTelemetry(true);
    window.requestAnimationFrame(animationLoop);
  }

  initialize();
})();
