(function () {
  "use strict";

  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => Array.from(document.querySelectorAll(selector));
  const PAPER = "#f1ead8";
  const INK = "#171813";

  const elements = {
    teachingCanvas: $("#teaching-canvas"),
    fieldCanvas: $("#field-canvas"),
    teachingFrame: $("#teaching-frame"),
    fieldFrame: $("#field-frame"),
    drawingPrompt: $("#drawing-prompt"),
    predictionEmpty: $("#prediction-empty"),
    trainingLabel: $("#training-label"),
    headerStatus: $("#header-status"),
    localStatus: $(".local-status"),
    parameterBadge: $("#parameter-badge"),
    parameterValue: $("#parameter-value"),
    stepValue: $("#step-value"),
    sampleValue: $("#sample-value"),
    generatedValue: $("#generated-value"),
    lossValue: $("#loss-value"),
    fitBadge: $("#fit-badge"),
    lossCanvas: $("#loss-canvas"),
    exampleCountBadge: $("#example-count-badge"),
    predictionCountBadge: $("#prediction-count-badge"),
    subjectName: $("#subject-name"),
    subjectNote: $("#subject-note"),
    variationValue: $("#variation-value"),
    lessonStep: $("#lesson-step"),
    lessonTitle: $("#lesson-title"),
    lessonCopy: $("#lesson-copy"),
    colorReadout: $("#color-readout"),
    customColor: $("#custom-color"),
    mobileToolSummary: $("#mobile-tool-summary"),
    pauseButton: $("#pause-button"),
    undoButton: $("#undo-button"),
    clearButton: $("#clear-button"),
    variationButton: $("#variation-button"),
    reseedButton: $("#reseed-button"),
    saveButton: $("#save-button"),
    wobbleValue: $("#wobble-value"),
    wobbleBar: $("#wobble-bar"),
    curveValue: $("#curve-value"),
    curveBar: $("#curve-bar"),
    brushValue: $("#brush-value"),
    brushBar: $("#brush-bar"),
    paletteList: $("#learned-palette"),
    liveRegion: $("#live-region")
  };

  const teachingContext = elements.teachingCanvas.getContext("2d", { alpha: false });
  const fieldContext = elements.fieldCanvas.getContext("2d", { alpha: false });
  const lossContext = elements.lossCanvas.getContext("2d");
  const exportCanvas = document.createElement("canvas");
  const exportContext = exportCanvas.getContext("2d", { alpha: false });
  const motionQuery = window.matchMedia("(prefers-reduced-motion: reduce)");

  const brushSettings = {
    fine: { strength: 0.25, pixels: 4.5, label: "Pencil" },
    medium: { strength: 0.55, pixels: 8.5, label: "Crayon" },
    broad: { strength: 0.9, pixels: 14, label: "Marker" }
  };

  const state = {
    model: null,
    modelSeed: 4109,
    variationSeed: 731,
    variationNumber: 1,
    subject: "bloom",
    selectedColor: "#171813",
    selectedName: "Graphite",
    brush: "medium",
    strokes: [],
    currentStroke: null,
    nextStrokeId: 1,
    activePointer: null,
    dataset: [],
    trainSamples: [],
    validationSamples: [],
    styleProfile: { palette: [], metrics: { wobble: 0, curvature: 0, width: 0, pointCount: 0 } },
    paused: false,
    practiceUntilStep: 0,
    isSettled: false,
    lastLoss: 0,
    lossHistory: [],
    teachingWidth: 0,
    teachingHeight: 0,
    fieldWidth: 0,
    fieldHeight: 0,
    dpr: 1,
    teachingDirty: true,
    fieldDirty: true,
    lastRenderAt: 0,
    lastUiAt: 0,
    frameCount: 0,
    keyboardCursor: { x: 0, y: 0 }
  };

  function clamp(value, minimum, maximum) {
    return Math.max(minimum, Math.min(maximum, value));
  }

  function cubic(start, controlA, controlB, end, count = 34) {
    return Array.from({ length: count }, (_, index) => {
      const t = index / (count - 1);
      const inverse = 1 - t;
      return {
        x: inverse ** 3 * start.x + 3 * inverse ** 2 * t * controlA.x + 3 * inverse * t ** 2 * controlB.x + t ** 3 * end.x,
        y: inverse ** 3 * start.y + 3 * inverse ** 2 * t * controlA.y + 3 * inverse * t ** 2 * controlB.y + t ** 3 * end.y
      };
    });
  }

  function ellipse(cx, cy, rx, ry, count = 54, phase = 0) {
    return Array.from({ length: count }, (_, index) => {
      const angle = phase + index / count * Math.PI * 2;
      return { x: cx + Math.cos(angle) * rx, y: cy + Math.sin(angle) * ry };
    });
  }

  function polyline(vertices, detail = 12) {
    const points = [];
    for (let segment = 1; segment < vertices.length; segment += 1) {
      const start = vertices[segment - 1];
      const end = vertices[segment];
      for (let index = 0; index < detail; index += 1) {
        const t = index / detail;
        points.push({ x: start.x + (end.x - start.x) * t, y: start.y + (end.y - start.y) * t });
      }
    }
    points.push({ ...vertices[vertices.length - 1] });
    return points;
  }

  const subjects = {
    bloom: {
      name: "Wildflower",
      note: "A fresh five-petal sketch, drawn with your learned marks.",
      paths() {
        const petals = Array.from({ length: 76 }, (_, index) => {
          const angle = -Math.PI / 2 + index / 76 * Math.PI * 2;
          const radius = 0.25 + 0.105 * Math.cos(angle * 5 + Math.PI / 2);
          return { x: 0.08 + Math.cos(angle) * radius, y: -0.32 + Math.sin(angle) * radius * 0.9 };
        });
        return [
          { points: cubic({ x: -0.12, y: 0.84 }, { x: -0.26, y: 0.48 }, { x: 0.14, y: 0.12 }, { x: 0.08, y: -0.16 }), closed: false },
          { points: petals, closed: true },
          { points: ellipse(0.08, -0.32, 0.075, 0.065, 28), closed: true },
          { points: [...cubic({ x: -0.08, y: 0.45 }, { x: -0.36, y: 0.28 }, { x: -0.55, y: 0.39 }, { x: -0.1, y: 0.55 }, 26), ...cubic({ x: -0.1, y: 0.55 }, { x: -0.38, y: 0.58 }, { x: -0.43, y: 0.37 }, { x: -0.08, y: 0.45 }, 26)], closed: true },
          { points: [...cubic({ x: 0.01, y: 0.24 }, { x: 0.33, y: 0.07 }, { x: 0.49, y: 0.23 }, { x: 0.03, y: 0.35 }, 26), ...cubic({ x: 0.03, y: 0.35 }, { x: 0.31, y: 0.43 }, { x: 0.42, y: 0.2 }, { x: 0.01, y: 0.24 }, 26)], closed: true }
        ];
      }
    },
    bird: {
      name: "Small bird",
      note: "A fresh little bird, drawn with your learned marks.",
      paths() {
        return [
          { points: ellipse(-0.03, 0.03, 0.48, 0.3, 64, -0.18), closed: true },
          { points: cubic({ x: -0.25, y: -0.02 }, { x: -0.08, y: -0.28 }, { x: 0.29, y: -0.15 }, { x: 0.18, y: 0.14 }, 38), closed: false },
          { points: polyline([{ x: 0.42, y: -0.08 }, { x: 0.72, y: 0.01 }, { x: 0.43, y: 0.09 }], 13), closed: false },
          { points: ellipse(0.22, -0.08, 0.025, 0.025, 18), closed: true },
          { points: polyline([{ x: -0.12, y: 0.31 }, { x: -0.15, y: 0.59 }, { x: -0.3, y: 0.66 }, { x: -0.15, y: 0.59 }, { x: -0.02, y: 0.67 }], 9), closed: false },
          { points: polyline([{ x: 0.14, y: 0.31 }, { x: 0.18, y: 0.58 }, { x: 0.05, y: 0.65 }, { x: 0.18, y: 0.58 }, { x: 0.32, y: 0.64 }], 9), closed: false }
        ];
      }
    },
    sail: {
      name: "Moon boat",
      note: "A fresh moonlit boat, drawn with your learned marks.",
      paths() {
        return [
          { points: polyline([{ x: -0.58, y: 0.25 }, { x: 0.56, y: 0.25 }, { x: 0.35, y: 0.52 }, { x: -0.36, y: 0.52 }, { x: -0.58, y: 0.25 }], 16), closed: true },
          { points: polyline([{ x: -0.02, y: 0.25 }, { x: -0.02, y: -0.62 }], 30), closed: false },
          { points: polyline([{ x: -0.04, y: -0.56 }, { x: -0.48, y: 0.14 }, { x: -0.04, y: 0.14 }], 18), closed: true },
          { points: polyline([{ x: 0.02, y: -0.5 }, { x: 0.42, y: 0.14 }, { x: 0.02, y: 0.14 }], 18), closed: true },
          { points: cubic({ x: -0.72, y: 0.64 }, { x: -0.35, y: 0.48 }, { x: -0.02, y: 0.8 }, { x: 0.72, y: 0.6 }, 52), closed: false },
          { points: ellipse(0.55, -0.5, 0.13, 0.13, 34), closed: true }
        ];
      }
    }
  };

  function drawPaper(context, width, height) {
    context.fillStyle = PAPER;
    context.fillRect(0, 0, width, height);
    context.save();
    context.strokeStyle = "rgba(23,24,19,.055)";
    context.lineWidth = 1;
    const gap = Math.max(24, Math.round(width / 13));
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

  function toCanvas(point, width, height) {
    return { x: (point.x + 1) * width * 0.5, y: (point.y + 1) * height * 0.5 };
  }

  function drawTeachingStroke(context, stroke, width, height) {
    if (!stroke.points.length) return;
    const brush = brushSettings[stroke.brush] || brushSettings.medium;
    if (stroke.points.length === 1) {
      const point = toCanvas(stroke.points[0], width, height);
      context.beginPath();
      context.arc(point.x, point.y, brush.pixels * 0.55, 0, Math.PI * 2);
      context.fillStyle = stroke.color;
      context.fill();
      return;
    }
    context.save();
    context.lineCap = "round";
    context.lineJoin = "round";
    context.strokeStyle = stroke.color;
    for (let index = 1; index < stroke.points.length; index += 1) {
      const previous = toCanvas(stroke.points[index - 1], width, height);
      const current = toCanvas(stroke.points[index], width, height);
      const t = index / (stroke.points.length - 1);
      const taper = 0.7 + 0.3 * Math.sin(Math.PI * t);
      const pressure = clamp(stroke.points[index].pressure || 0.5, 0, 1);
      context.lineWidth = brush.pixels * taper * (0.82 + pressure * 0.36);
      context.beginPath();
      context.moveTo(previous.x, previous.y);
      context.lineTo(current.x, current.y);
      context.stroke();
    }
    context.restore();
  }

  function rgbCss(rgb, alpha = 1) {
    const channels = rgb.map((value) => clamp(Math.round(value * 255), 0, 255));
    return `rgba(${channels[0]},${channels[1]},${channels[2]},${alpha})`;
  }

  function drawGeneratedStroke(context, points, width, height, closed) {
    if (!points.length) return;
    context.save();
    context.lineCap = "round";
    context.lineJoin = "round";
    const segmentCount = closed ? points.length : points.length - 1;
    for (let index = 0; index < segmentCount; index += 1) {
      const startPoint = points[index];
      const endPoint = points[(index + 1) % points.length];
      const start = toCanvas(startPoint, width, height);
      const end = toCanvas(endPoint, width, height);
      context.strokeStyle = rgbCss(endPoint.color, 0.96);
      context.lineWidth = 2.2 + endPoint.width * Math.min(width, height) * 0.026;
      context.beginPath();
      context.moveTo(start.x, start.y);
      context.lineTo(end.x, end.y);
      context.stroke();
    }
    context.restore();
  }

  function renderTeaching() {
    if (!state.teachingWidth || !state.teachingHeight) return;
    teachingContext.setTransform(state.dpr, 0, 0, state.dpr, 0, 0);
    drawPaper(teachingContext, state.teachingWidth, state.teachingHeight);
    for (const stroke of state.strokes) drawTeachingStroke(teachingContext, stroke, state.teachingWidth, state.teachingHeight);
    if (state.currentStroke) drawTeachingStroke(teachingContext, state.currentStroke, state.teachingWidth, state.teachingHeight);
    if (document.activeElement === elements.teachingCanvas && state.activePointer === null) drawKeyboardCursor();
    elements.drawingPrompt.hidden = Boolean(state.strokes.length || state.currentStroke);
    const count = state.strokes.length;
    elements.exampleCountBadge.textContent = `${count} ${count === 1 ? "lesson" : "lessons"}`;
    state.teachingDirty = false;
  }

  function drawKeyboardCursor() {
    const point = toCanvas(state.keyboardCursor, state.teachingWidth, state.teachingHeight);
    teachingContext.save();
    teachingContext.translate(point.x, point.y);
    teachingContext.strokeStyle = state.selectedColor;
    teachingContext.lineWidth = 2;
    teachingContext.beginPath();
    teachingContext.arc(0, 0, 10, 0, Math.PI * 2);
    teachingContext.moveTo(-16, 0);
    teachingContext.lineTo(-6, 0);
    teachingContext.moveTo(6, 0);
    teachingContext.lineTo(16, 0);
    teachingContext.moveTo(0, -16);
    teachingContext.lineTo(0, -6);
    teachingContext.moveTo(0, 6);
    teachingContext.lineTo(0, 16);
    teachingContext.stroke();
    teachingContext.restore();
  }

  function renderApprenticeTo(context, width, height) {
    drawPaper(context, width, height);
    if (!state.dataset.length) return 0;
    const paths = subjects[state.subject].paths();
    let pointCount = 0;
    paths.forEach((path, index) => {
      const styled = window.DreamfieldStyle.stylizePath(state.model, path.points, {
        seed: state.variationSeed + index * 1009,
        closed: path.closed
      });
      pointCount += styled.length;
      drawGeneratedStroke(context, styled, width, height, path.closed);
    });
    return pointCount;
  }

  function renderApprentice() {
    if (!state.fieldWidth || !state.fieldHeight) return;
    fieldContext.setTransform(state.dpr, 0, 0, state.dpr, 0, 0);
    const pointCount = renderApprenticeTo(fieldContext, state.fieldWidth, state.fieldHeight);
    const hasLessons = state.dataset.length > 0;
    elements.predictionEmpty.hidden = hasLessons;
    elements.predictionCountBadge.textContent = hasLessons ? `${pointCount} new points` : "Waiting";
    elements.generatedValue.textContent = hasLessons ? pointCount.toLocaleString() : "0";
    elements.fieldCanvas.setAttribute("aria-label", hasLessons
      ? `A new ${subjects[state.subject].name} drawn by the neural apprentice in the learned stroke style.`
      : "No apprentice drawing yet. Add a style lesson on the left to begin.");
    state.fieldDirty = false;
  }

  function resizeCanvases() {
    const teachingRect = elements.teachingFrame.getBoundingClientRect();
    const fieldRect = elements.fieldFrame.getBoundingClientRect();
    if (!teachingRect.width || !fieldRect.width) return;
    state.dpr = Math.min(window.devicePixelRatio || 1, 2);
    state.teachingWidth = Math.round(teachingRect.width);
    state.teachingHeight = Math.round(teachingRect.height);
    state.fieldWidth = Math.round(fieldRect.width);
    state.fieldHeight = Math.round(fieldRect.height);
    elements.teachingCanvas.width = Math.round(state.teachingWidth * state.dpr);
    elements.teachingCanvas.height = Math.round(state.teachingHeight * state.dpr);
    elements.fieldCanvas.width = Math.round(state.fieldWidth * state.dpr);
    elements.fieldCanvas.height = Math.round(state.fieldHeight * state.dpr);
    state.teachingDirty = true;
    state.fieldDirty = true;
  }

  function createModel(pretrainSteps = 0) {
    state.model = new window.DreamfieldStyleModel({ hiddenSize: 18, learningRate: 0.009, seed: state.modelSeed });
    state.lossHistory = [];
    state.lastLoss = state.validationSamples.length ? state.model.lossForBatch(state.validationSamples) : 0;
    state.practiceUntilStep = state.dataset.length ? (motionQuery.matches ? 520 : 900) : 0;
    state.isSettled = false;
    if (pretrainSteps && state.dataset.length) practiceSynchronously(pretrainSteps);
    const count = state.model.parameterCount.toLocaleString();
    elements.parameterBadge.textContent = `${count} parameters learn your hand`;
    elements.parameterValue.textContent = count;
    state.fieldDirty = true;
  }

  function rebuildStyleLesson() {
    state.styleProfile = window.DreamfieldStyle.extractStyleDataset(state.strokes);
    state.dataset = state.styleProfile.samples;
    state.validationSamples = state.dataset.filter((sample, index) => index % 6 === 0);
    state.trainSamples = state.dataset.filter((sample, index) => index % 6 !== 0);
    if (!state.trainSamples.length) state.trainSamples = state.dataset;
    state.modelSeed += 17;
    createModel(state.dataset.length ? 90 : 0);
    updateStyleFingerprint();
    updateInterface(true);
  }

  function buildBatch(size = 32) {
    if (!state.trainSamples.length) return [];
    const batch = [];
    for (let index = 0; index < size; index += 1) batch.push(state.trainSamples[Math.floor(Math.random() * state.trainSamples.length)]);
    return batch;
  }

  function practiceSynchronously(steps) {
    for (let index = 0; index < steps && state.trainSamples.length; index += 1) {
      const loss = state.model.trainBatch(buildBatch());
      state.lastLoss = index ? state.lastLoss * 0.9 + loss * 0.1 : loss;
      if (index % 12 === 0) pushLoss(state.lastLoss);
    }
    if (state.validationSamples.length) state.lastLoss = state.model.lossForBatch(state.validationSamples);
  }

  function pushLoss(loss) {
    if (!Number.isFinite(loss)) return;
    state.lossHistory.push(loss);
    if (state.lossHistory.length > 100) state.lossHistory.shift();
  }

  function drawLossChart() {
    const width = elements.lossCanvas.width;
    const height = elements.lossCanvas.height;
    lossContext.clearRect(0, 0, width, height);
    lossContext.strokeStyle = "rgba(241,234,216,.12)";
    lossContext.lineWidth = 1;
    for (let index = 1; index < 4; index += 1) {
      lossContext.beginPath();
      lossContext.moveTo(0, index * height / 4);
      lossContext.lineTo(width, index * height / 4);
      lossContext.stroke();
    }
    if (state.lossHistory.length < 2) return;
    const values = state.lossHistory.map((value) => Math.log10(Math.max(1e-6, value)));
    let minimum = Math.min(...values);
    let maximum = Math.max(...values);
    if (maximum - minimum < 0.1) { minimum -= 0.05; maximum += 0.05; }
    const points = values.map((value, index) => ({
      x: 5 + index / (values.length - 1) * (width - 10),
      y: 5 + (maximum - value) / (maximum - minimum) * (height - 10)
    }));
    const gradient = lossContext.createLinearGradient(0, 0, 0, height);
    gradient.addColorStop(0, "rgba(224,255,71,.35)");
    gradient.addColorStop(1, "rgba(224,255,71,0)");
    lossContext.beginPath();
    lossContext.moveTo(points[0].x, height);
    points.forEach((point) => lossContext.lineTo(point.x, point.y));
    lossContext.lineTo(points[points.length - 1].x, height);
    lossContext.fillStyle = gradient;
    lossContext.fill();
    lossContext.beginPath();
    lossContext.moveTo(points[0].x, points[0].y);
    points.slice(1).forEach((point) => lossContext.lineTo(point.x, point.y));
    lossContext.strokeStyle = "#e0ff47";
    lossContext.lineWidth = 3;
    lossContext.stroke();
    const last = points[points.length - 1];
    lossContext.beginPath();
    lossContext.arc(last.x, last.y, 4, 0, Math.PI * 2);
    lossContext.fillStyle = "#ff5b45";
    lossContext.fill();
  }

  function updateStyleFingerprint() {
    const metrics = state.styleProfile.metrics;
    const wobbleLevel = clamp(metrics.wobble / 0.045, 0, 1);
    const curveLevel = clamp(metrics.curvature / 0.28, 0, 1);
    const widthLevel = clamp(metrics.width, 0, 1);
    elements.wobbleBar.style.setProperty("--level", `${Math.round(wobbleLevel * 100)}%`);
    elements.curveBar.style.setProperty("--level", `${Math.round(curveLevel * 100)}%`);
    elements.brushBar.style.setProperty("--level", `${Math.round(widthLevel * 100)}%`);
    elements.wobbleValue.textContent = !state.dataset.length ? "—" : wobbleLevel < 0.25 ? "Steady" : wobbleLevel < 0.62 ? "Loose" : "Electric";
    elements.curveValue.textContent = !state.dataset.length ? "—" : curveLevel < 0.28 ? "Gentle" : curveLevel < 0.64 ? "Curvy" : "Kinky";
    elements.brushValue.textContent = !state.dataset.length ? "—" : widthLevel < 0.38 ? "Light" : widthLevel < 0.7 ? "Medium" : "Heavy";
    elements.paletteList.replaceChildren();
    if (!state.styleProfile.palette.length) {
      const empty = document.createElement("span");
      empty.className = "palette-empty";
      empty.textContent = "Waiting for ink";
      elements.paletteList.append(empty);
    } else {
      state.styleProfile.palette.slice(0, 5).forEach((entry) => {
        const chip = document.createElement("span");
        chip.className = "learned-color";
        chip.style.background = entry.color;
        chip.title = entry.color;
        chip.setAttribute("aria-label", `Learned palette color ${entry.color}`);
        elements.paletteList.append(chip);
      });
    }
  }

  function updateLessonCard() {
    const count = state.strokes.length;
    if (!count && !state.currentStroke) {
      elements.lessonStep.textContent = "0 of 3 strokes";
      elements.lessonTitle.textContent = "Start with a curve, a corner, or a loop.";
      elements.lessonCopy.textContent = "Three different gestures give it a good feel for your hand.";
    } else if (state.currentStroke) {
      elements.lessonStep.textContent = "Drawing";
      elements.lessonTitle.textContent = "It’s watching how you move.";
      elements.lessonCopy.textContent = "Release to add this stroke.";
    } else if (count < 3) {
      elements.lessonStep.textContent = `${count} of 3 strokes`;
      elements.lessonTitle.textContent = count === 1 ? "It can draw now. Add two more for character." : "One more different gesture will do it.";
      elements.lessonCopy.textContent = "The apprentice is already responding on the right.";
    } else if (state.paused) {
      elements.lessonStep.textContent = "Paused";
      elements.lessonTitle.textContent = "Your current style is frozen.";
      elements.lessonCopy.textContent = "Resume whenever you want it to keep learning.";
    } else if (!state.isSettled) {
      elements.lessonStep.textContent = `${count} strokes · learning`;
      elements.lessonTitle.textContent = "The apprentice is catching your style.";
      elements.lessonCopy.textContent = "You can keep drawing while it practices.";
    } else {
      elements.lessonStep.textContent = `${count} strokes · ready`;
      elements.lessonTitle.textContent = "Your handprint is ready to play with.";
      elements.lessonCopy.textContent = "Try another subject or make a new variation.";
    }
  }

  function updateInterface(force = false) {
    const hasLessons = state.dataset.length > 0;
    elements.stepValue.textContent = state.model.step.toLocaleString();
    elements.sampleValue.textContent = state.dataset.length.toLocaleString();
    elements.lossValue.textContent = hasLessons ? formatLoss(state.lastLoss) : "—";
    elements.pauseButton.disabled = !hasLessons;
    elements.undoButton.disabled = !state.strokes.length;
    elements.clearButton.disabled = !state.strokes.length;
    elements.variationButton.disabled = !hasLessons;
    elements.reseedButton.disabled = !hasLessons;
    elements.saveButton.disabled = !hasLessons;
    elements.pauseButton.classList.toggle("is-paused", state.paused);
    elements.pauseButton.querySelector("span:last-child").textContent = state.paused ? "Resume practice" : "Pause practice";
    elements.localStatus.classList.toggle("is-paused", state.paused || !hasLessons);

    if (!hasLessons) {
      elements.trainingLabel.textContent = "Waiting for your hand";
      elements.headerStatus.textContent = "Ready for a lesson";
      elements.fitBadge.textContent = "No style yet";
    } else if (state.paused) {
      elements.trainingLabel.textContent = "Style model frozen";
      elements.headerStatus.textContent = "Practice paused";
      elements.fitBadge.textContent = "Paused";
    } else if (state.isSettled) {
      elements.trainingLabel.textContent = "Style caught";
      elements.headerStatus.textContent = "Drawing locally";
      elements.fitBadge.textContent = "Style caught";
    } else {
      elements.trainingLabel.textContent = "Learning your strokes";
      elements.headerStatus.textContent = "Training locally";
      elements.fitBadge.textContent = "Practicing";
    }
    updateLessonCard();
    if (force || state.frameCount % 3 === 0) drawLossChart();
  }

  function formatLoss(value) {
    if (!Number.isFinite(value)) return "—";
    return value < 0.0001 ? value.toExponential(2) : value.toFixed(4);
  }

  function selectColor(color, name) {
    state.selectedColor = color.toLowerCase();
    state.selectedName = name;
    elements.customColor.value = state.selectedColor;
    elements.colorReadout.textContent = `${name} · ${state.selectedColor}`;
    elements.mobileToolSummary.textContent = `${name} · ${brushSettings[state.brush].label}`;
    $$('[data-color]').forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.color.toLowerCase() === state.selectedColor)));
  }

  function selectBrush(brush) {
    state.brush = brush;
    $$('[data-brush]').forEach((button) => {
      const active = button.dataset.brush === brush;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    elements.mobileToolSummary.textContent = `${state.selectedName} · ${brushSettings[brush].label}`;
  }

  function selectSubject(subject) {
    if (!subjects[subject]) return;
    state.subject = subject;
    state.variationNumber = 1;
    state.variationSeed += 211;
    $$('[data-subject]').forEach((button) => {
      const active = button.dataset.subject === subject;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    elements.subjectName.textContent = subjects[subject].name;
    elements.subjectNote.textContent = subjects[subject].note;
    elements.variationValue.textContent = `Variation ${state.variationNumber}`;
    state.fieldDirty = true;
    announce(`${subjects[subject].name} prompt selected. The apprentice is making a new drawing in your learned style.`);
  }

  function beginStroke(point) {
    state.currentStroke = {
      id: state.nextStrokeId++,
      color: state.selectedColor,
      name: state.selectedName,
      brush: state.brush,
      width: brushSettings[state.brush].strength,
      points: [point]
    };
    state.teachingDirty = true;
    updateLessonCard();
  }

  function appendStrokePoint(point) {
    if (!state.currentStroke) return;
    const previous = state.currentStroke.points[state.currentStroke.points.length - 1];
    const pixelDistance = Math.hypot((point.x - previous.x) * state.teachingWidth, (point.y - previous.y) * state.teachingHeight) * 0.5;
    if (pixelDistance < 2) return;
    const steps = Math.max(1, Math.ceil(pixelDistance / 5));
    for (let index = 1; index <= steps; index += 1) {
      const t = index / steps;
      state.currentStroke.points.push({
        x: previous.x + (point.x - previous.x) * t,
        y: previous.y + (point.y - previous.y) * t,
        pressure: previous.pressure + (point.pressure - previous.pressure) * t
      });
    }
    state.teachingDirty = true;
  }

  function finishStroke() {
    if (!state.currentStroke) return;
    const stroke = state.currentStroke;
    state.currentStroke = null;
    state.strokes.push(stroke);
    rebuildStyleLesson();
    state.teachingDirty = true;
    state.fieldDirty = true;
    announce(`Style lesson ${state.strokes.length} added. The apprentice is learning its wobble, turns, width, and color, then drawing a new ${subjects[state.subject].name}.`);
  }

  function eventToPoint(event) {
    const rect = elements.teachingCanvas.getBoundingClientRect();
    return {
      x: clamp((event.clientX - rect.left) / rect.width * 2 - 1, -1, 1),
      y: clamp((event.clientY - rect.top) / rect.height * 2 - 1, -1, 1),
      pressure: event.pressure > 0 ? event.pressure : 0.5
    };
  }

  function handlePointerDown(event) {
    if ((event.pointerType === "mouse" && event.button !== 0) || state.activePointer !== null) return;
    state.activePointer = event.pointerId;
    elements.teachingCanvas.setPointerCapture(event.pointerId);
    elements.teachingCanvas.classList.add("is-painting");
    beginStroke(eventToPoint(event));
    event.preventDefault();
  }

  function handlePointerMove(event) {
    if (state.activePointer !== event.pointerId) return;
    appendStrokePoint(eventToPoint(event));
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
    const movement = event.shiftKey ? 0.12 : 0.035;
    let handled = true;
    if (event.key === "ArrowLeft") state.keyboardCursor.x -= movement;
    else if (event.key === "ArrowRight") state.keyboardCursor.x += movement;
    else if (event.key === "ArrowUp") state.keyboardCursor.y -= movement;
    else if (event.key === "ArrowDown") state.keyboardCursor.y += movement;
    else if (event.key === " " || event.key === "Enter") {
      beginStroke({ ...state.keyboardCursor, pressure: 0.5 });
      finishStroke();
    } else handled = false;
    if (!handled) return;
    event.preventDefault();
    state.keyboardCursor.x = clamp(state.keyboardCursor.x, -1, 1);
    state.keyboardCursor.y = clamp(state.keyboardCursor.y, -1, 1);
    state.teachingDirty = true;
  }

  function undoStroke() {
    if (!state.strokes.length) return;
    state.strokes.pop();
    rebuildStyleLesson();
    state.teachingDirty = true;
    announce(state.strokes.length ? "Last style lesson removed. The apprentice is relearning the remaining marks." : "All style lessons removed.");
  }

  function clearLessons() {
    if (!state.strokes.length) return;
    state.strokes = [];
    state.currentStroke = null;
    state.nextStrokeId = 1;
    rebuildStyleLesson();
    state.teachingDirty = true;
    announce("The style sheet is clear. Add a new gesture when you are ready.");
  }

  function togglePaused() {
    state.paused = !state.paused;
    updateInterface(true);
    announce(state.paused ? "Practice paused. The current learned hand is frozen." : "Practice resumed.");
  }

  function newVariation() {
    if (!state.dataset.length) return;
    state.variationSeed += 7919;
    state.variationNumber += 1;
    elements.variationValue.textContent = `Variation ${state.variationNumber}`;
    state.fieldDirty = true;
    announce(`Variation ${state.variationNumber}. Same subject and learned style, new generative grain.`);
  }

  function reseedApprentice() {
    if (!state.dataset.length) return;
    state.modelSeed += 997;
    createModel(110);
    updateInterface(true);
    announce("A new tiny network started from fresh random weights and is learning the same style sheet.");
  }

  function saveDrawing() {
    if (!state.dataset.length) return;
    exportCanvas.width = 1400;
    exportCanvas.height = 1050;
    exportContext.setTransform(1, 0, 0, 1, 0, 0);
    renderApprenticeTo(exportContext, exportCanvas.width, exportCanvas.height);
    const link = document.createElement("a");
    link.download = `dreamfield-${state.subject}-variation-${state.variationNumber}.png`;
    link.href = exportCanvas.toDataURL("image/png");
    link.click();
    announce("The apprentice drawing was saved as a PNG.");
  }

  function loadDemo(kind) {
    const random = new window.DreamfieldStyle.SeededRandom(kind === "restless" ? 81 : 42);
    const colors = kind === "restless" ? ["#2664ff", "#ff5b45", "#171813"] : ["#171813", "#0c9b83", "#171813"];
    const amplitudes = kind === "restless" ? [0.065, 0.075, 0.055] : [0.01, 0.014, 0.008];
    const bases = [
      (t) => ({ x: -0.78 + 1.56 * t, y: -0.5 + 0.12 * Math.sin(t * Math.PI) }),
      (t) => ({ x: -0.72 + 1.44 * t, y: 0.02 + (t < 0.5 ? -0.5 * t : -0.25 + 0.75 * (t - 0.5)) }),
      (t) => ({ x: Math.cos(t * Math.PI * 2) * 0.34, y: 0.5 + Math.sin(t * Math.PI * 2) * 0.22 })
    ];
    state.strokes = bases.map((base, strokeIndex) => ({
      id: strokeIndex + 1,
      color: colors[strokeIndex],
      name: "Demo ink",
      brush: kind === "restless" ? "broad" : strokeIndex === 1 ? "fine" : "medium",
      width: kind === "restless" ? 0.9 : strokeIndex === 1 ? 0.25 : 0.55,
      closed: strokeIndex === 2,
      points: Array.from({ length: 60 }, (_, index) => {
        const t = index / 59;
        const point = base(t);
        const noise = (random.next() * 2 - 1) * amplitudes[strokeIndex];
        return { x: point.x, y: point.y + noise, pressure: 0.35 + random.next() * 0.45 };
      })
    }));
    state.nextStrokeId = 4;
    rebuildStyleLesson();
    state.teachingDirty = true;
    announce(`${kind === "restless" ? "Restless marker" : "Quiet pencil"} demo hand loaded. Its source gestures train the network; the drawing on the right remains a separate subject.`);
  }

  function bindEvents() {
    $$('[data-color]').forEach((button) => button.addEventListener("click", () => selectColor(button.dataset.color, button.dataset.name)));
    elements.customColor.addEventListener("input", (event) => selectColor(event.target.value, "Custom"));
    $$('[data-brush]').forEach((button) => button.addEventListener("click", () => selectBrush(button.dataset.brush)));
    $$('[data-subject]').forEach((button) => button.addEventListener("click", () => selectSubject(button.dataset.subject)));
    $$('[data-demo]').forEach((button) => button.addEventListener("click", () => loadDemo(button.dataset.demo)));
    elements.pauseButton.addEventListener("click", togglePaused);
    elements.undoButton.addEventListener("click", undoStroke);
    elements.clearButton.addEventListener("click", clearLessons);
    elements.variationButton.addEventListener("click", newVariation);
    elements.reseedButton.addEventListener("click", reseedApprentice);
    elements.saveButton.addEventListener("click", saveDrawing);
    elements.teachingCanvas.addEventListener("pointerdown", handlePointerDown);
    elements.teachingCanvas.addEventListener("pointermove", handlePointerMove);
    elements.teachingCanvas.addEventListener("pointerup", endPointer);
    elements.teachingCanvas.addEventListener("pointercancel", endPointer);
    elements.teachingCanvas.addEventListener("keydown", handleCanvasKeydown);
    elements.teachingCanvas.addEventListener("focus", () => { state.teachingDirty = true; });
    elements.teachingCanvas.addEventListener("blur", () => { state.teachingDirty = true; });
    const observer = new ResizeObserver(resizeCanvases);
    observer.observe(elements.teachingFrame);
    observer.observe(elements.fieldFrame);
    window.addEventListener("resize", resizeCanvases, { passive: true });
  }

  function animationLoop(timestamp) {
    state.frameCount += 1;
    const hasLessons = state.trainSamples.length > 0;
    state.isSettled = hasLessons && state.model.step >= state.practiceUntilStep;
    const occasionalRefinement = state.isSettled && !motionQuery.matches && state.frameCount % 24 === 0;
    if (!document.hidden && hasLessons && !state.paused && (!state.isSettled || occasionalRefinement)) {
      const started = performance.now();
      const budget = state.isSettled ? 1.1 : motionQuery.matches ? 4 : 6;
      let batches = 0;
      while (performance.now() - started < budget && batches < (state.isSettled ? 1 : 16)) {
        const loss = state.model.trainBatch(buildBatch());
        state.lastLoss = state.lastLoss * 0.91 + loss * 0.09;
        batches += 1;
      }
      if (state.model.step % 15 < batches) {
        if (state.validationSamples.length) state.lastLoss = state.model.lossForBatch(state.validationSamples);
        pushLoss(state.lastLoss);
      }
      state.fieldDirty = true;
    }
    if (state.teachingDirty) renderTeaching();
    const renderInterval = state.isSettled ? 500 : 130;
    if (state.fieldDirty && timestamp - state.lastRenderAt > renderInterval) {
      renderApprentice();
      state.lastRenderAt = timestamp;
    }
    if (timestamp - state.lastUiAt > 180) {
      updateInterface();
      state.lastUiAt = timestamp;
    }
    window.requestAnimationFrame(animationLoop);
  }

  function announce(message) {
    elements.liveRegion.textContent = "";
    window.setTimeout(() => { elements.liveRegion.textContent = message; }, 20);
  }

  function initialize() {
    bindEvents();
    createModel();
    selectColor("#171813", "Graphite");
    selectBrush("medium");
    selectSubject("bloom");
    updateStyleFingerprint();
    resizeCanvases();
    renderTeaching();
    renderApprentice();
    updateInterface(true);
    window.requestAnimationFrame(animationLoop);
  }

  initialize();
})();
