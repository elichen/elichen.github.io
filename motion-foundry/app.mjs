import {
  presetTarget,
  presetDesign,
  validateDesign,
  sampleMechanism,
  poseAt,
  resampleClosed,
} from "./kinematics.mjs";
import { fitDesignToTarget, measureDesign } from "./optimizer.mjs";
import { MechanismView } from "./render.mjs";
import { PathSketch } from "./sketch.mjs";

const $ = (id) => document.getElementById(id);
const clone = (value) => JSON.parse(JSON.stringify(value));
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const families = { fourbar: "Four-bar linkage", slider: "Crank-slider" };
const presets = ["stride", "oval", "eight", "petal"];
const savedKey = "motion-foundry-study-v1";
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
let target = presetTarget("stride", 96),
  preset = "stride",
  family = "all";
let design = null,
  best = null,
  candidates = [],
  selectedCandidate = 0,
  pinned = false;
let playing = !reducedMotion.matches,
  speed = 0.45,
  showTarget = true,
  showDimensions = false;
let searchWorker = null,
  searchActive = false,
  searchRevision = 0,
  searchPass = 0,
  study = 1;
let chartHistory = [],
  drawingHistory = [],
  toastTimer,
  editingTimer,
  initialShared = false,
  initialRestored = false;

function validatedStudy(value) {
  if (
    !value ||
    !Array.isArray(value.target) ||
    value.target.length < 3 ||
    value.target.length > 512
  )
    return null;
  if (
    !value.target.every(
      (p) =>
        p &&
        Number.isFinite(p.x) &&
        Number.isFinite(p.y) &&
        Math.abs(p.x) <= 2 &&
        Math.abs(p.y) <= 2,
    )
  )
    return null;
  const points = resampleClosed(value.target, 96);
  if (points.length < 3) return null;
  const checked = validateDesign(value.design);
  const safeDesign =
    checked &&
    Math.abs(checked.transform.tx) < 10000 &&
    Math.abs(checked.transform.ty) < 10000
      ? checked
      : null;
  return {
    target: points,
    preset: presets.includes(value.preset) ? value.preset : "custom",
    family: ["all", "fourbar", "slider"].includes(value.family)
      ? value.family
      : "all",
    design: safeDesign,
  };
}

try {
  const saved = validatedStudy(
    JSON.parse(localStorage.getItem(savedKey) || "null"),
  );
  if (saved) {
    ({ target, preset, family, design } = saved);
    initialRestored = Boolean(design);
  }
} catch {
  /* Storage is optional. */
}
if (location.hash.startsWith("#v1=")) {
  try {
    const encoded = location.hash.slice(4);
    if (encoded.length > 18000) throw new Error("Oversized study");
    const shared = validatedStudy(
      JSON.parse(atob(encoded.replace(/-/g, "+").replace(/_/g, "/"))),
    );
    if (!shared) throw new Error("Invalid study");
    ({ target, preset, family, design } = shared);
    initialShared = true;
  } catch {
    toast("This study could not be read. Start with a fresh gesture.");
  }
}

function persist() {
  try {
    localStorage.setItem(
      savedKey,
      JSON.stringify({ target, preset, family, design }),
    );
  } catch {
    /* No account or storage is needed to use the workbench. */
  }
}
function toast(message) {
  $("toast").textContent = message;
  $("toast").classList.add("visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => $("toast").classList.remove("visible"), 3700);
}

const sketch = new PathSketch(
  $("sketch"),
  (points) => {
    drawingHistory.push({ target: clone(target), preset });
    if (drawingHistory.length > 20) drawingHistory.shift();
    target = resampleClosed(points, 96);
    preset = "custom";
    if (target.length < 3) return;
    $("undo-button").disabled = false;
    study++;
    syncTarget();
    launchSearch();
  },
  (count, message) => {
    $("close-path").disabled = count < 3;
    $("sketch-label").textContent = count
      ? `${count} POINTS · ENTER TO CLOSE`
      : "THE DESIRED MOTION";
    if (message) toast(message);
  },
);

const view = new MechanismView($("machine"), {
  onJointDrag: (name, x, y) => {
    if (!design) return;
    stopSearch(false);
    pinned = true;
    selectedCandidate = -1;
    const pose = poseAt(design, view.theta);
    if (!pose.valid) return;
    if (name === "P") {
      const dx = pose.B.x - pose.A.x,
        dy = pose.B.y - pose.A.y,
        length2 = dx * dx + dy * dy;
      if (length2 < 1e-12) return;
      design.params.traceAlong = clamp(
        ((x - pose.A.x) * dx + (y - pose.A.y) * dy) / length2,
        -2,
        3,
      );
      design.params.traceOffset = clamp(
        ((y - pose.A.y) * dx - (x - pose.A.x) * dy) / length2,
        -2,
        2,
      );
    } else if (name === "O") {
      design.transform.tx += x - pose.O.x;
      design.transform.ty += y - pose.O.y;
    } else if (name === "G" && design.family === "fourbar") {
      const dx = x - pose.O.x,
        dy = y - pose.O.y;
      if (Math.hypot(dx, dy) < 0.05 || Math.hypot(dx, dy) > 20) return;
      design.transform.a = dx;
      design.transform.b = dy;
    } else return;
    updateEditedDesign(false);
  },
  onJointEnd: () => {
    updateEditedDesign(true);
    persist();
  },
});
view.setState({
  playing,
  speed,
  showTarget,
  showTrace: true,
  showDimensions,
  reducedMotion: reducedMotion.matches,
});

function syncTarget() {
  sketch.setPath(target);
  document.querySelectorAll("[data-preset]").forEach((button) => {
    const active = button.dataset.preset === preset;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  $("family").value = family;
  $("point-count").textContent = `${target.length} POINTS`;
  $("study-number").textContent = String(study).padStart(3, "0");
  view.setState({ target });
}

function drawHistory() {
  const canvas = $("history"),
    rect = canvas.getBoundingClientRect(),
    ctx = canvas.getContext("2d");
  const width = Math.max(1, rect.width),
    height = Math.max(1, rect.height),
    dpr = Math.min(devicePixelRatio || 1, 2);
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = "#c6d4be";
  ctx.lineWidth = 0.7;
  for (let i = 1; i < 4; i++) {
    const y = (height * i) / 4;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
  if (!chartHistory.length) return;
  const values = chartHistory
    .map((p) => (typeof p === "number" ? p : p.error))
    .filter(Number.isFinite);
  if (!values.length) return;
  const max = Math.max(...values, 0.001),
    min = Math.min(...values, 0),
    range = max - min || 1;
  const position = (v, i) => [
    (i / Math.max(1, values.length - 1)) * (width - 4) + 2,
    7 + (1 - (v - min) / range) * (height - 14),
  ];
  ctx.beginPath();
  values.forEach((v, i) => {
    const [x, y] = position(v, i);
    if (!i) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = "#426b64";
  ctx.lineWidth = 1.6;
  ctx.stroke();
  const [lastX, lastY] = position(values.at(-1), values.length - 1);
  ctx.lineTo(lastX, height);
  ctx.lineTo(2, height);
  ctx.closePath();
  const gradient = ctx.createLinearGradient(0, 0, 0, height);
  gradient.addColorStop(0, "#5f8a6b2b");
  gradient.addColorStop(1, "#5f8a6b00");
  ctx.fillStyle = gradient;
  ctx.fill();
  ctx.beginPath();
  ctx.arc(lastX, lastY, 2.3, 0, Math.PI * 2);
  ctx.fillStyle = "#cc4d29";
  ctx.fill();
}
new ResizeObserver(drawHistory).observe($("history"));

function formatError(error) {
  return Number.isFinite(error)
    ? (error * 100).toFixed(error < 0.01 ? 2 : 1)
    : "—";
}
function updateFit(error, edited = false) {
  $("error-output").textContent = formatError(error);
  const label =
    error < 0.015
      ? "A very close fit"
      : error < 0.04
        ? "A convincing match"
        : error < 0.08
          ? "A useful approximation"
          : "Room to explore";
  $("fit-grade").replaceChildren();
  const dot = document.createElement("span");
  $("fit-grade").append(dot, document.createTextNode(label));
  $("fit-grade").classList.toggle("good", error < 0.04);
  $("fit-description").textContent = edited
    ? "Your edits change the trace. Restore the best fit to compare."
    : error < 0.02
      ? "The orange line closely follows your gesture. Try watching it for a full turn."
      : "A lower number means a closer trace of your drawing.";
  $("machine").dataset.pathError = error;
}

function syncTuning() {
  if (!design) return;
  for (const [id, key] of [
    ["tracer-u", "traceAlong"],
    ["tracer-v", "traceOffset"],
  ]) {
    const input = $(id);
    input.disabled = false;
    // Preserve a searched tracing point even if it lies beyond the standard edit range.
    input.min = Math.min(
      Number(id === "tracer-u" ? -2 : -2),
      Math.floor(design.params[key] - 0.5),
    );
    input.max = Math.max(
      Number(id === "tracer-u" ? 3 : 2),
      Math.ceil(design.params[key] + 0.5),
    );
    input.value = design.params[key];
    $(`${id}-output`).textContent = design.params[key].toFixed(2);
    input.style.setProperty(
      "--fill",
      `${((Number(input.value) - Number(input.min)) / (Number(input.max) - Number(input.min))) * 100}%`,
    );
  }
  const scale = Math.hypot(design.transform.a, design.transform.b),
    p = design.params;
  $("assembly-name").textContent = families[design.family].toUpperCase();
  const measurements = [
    ["CRANK", p.crank],
    ["COUPLER", p.coupler],
    [
      design.family === "fourbar" ? "FOLLOWER" : "RAIL OFFSET",
      design.family === "fourbar" ? p.rocker : p.railOffset,
    ],
    ["GROUND", design.family === "fourbar" ? 1 : null],
  ];
  $("dimensions-readout").replaceChildren();
  for (const [label, value] of measurements) {
    if (value === null) continue;
    const block = document.createElement("div"),
      name = document.createElement("span"),
      number = document.createElement("b");
    name.className = "dimension-label";
    name.textContent = label;
    number.textContent = (value * scale).toFixed(3);
    block.append(name, number);
    $("dimensions-readout").append(block);
  }
  $("export-button").disabled = false;
  $("share-button").disabled = false;
  $("restore-fit").disabled = !best;
}

function applyCandidate(result, index = 0) {
  if (!result?.design || !Number.isFinite(result.error)) return;
  design = clone(result.design);
  selectedCandidate = index;
  view.setState({ design, target, curve: sampleMechanism(design, 256) });
  $("machine-title").textContent = families[design.family];
  $("machine").dataset.family = design.family;
  updateFit(result.error);
  syncTuning();
  renderCandidates();
  persist();
}

function renderCandidates() {
  const container = $("candidates");
  const focused = container.contains(document.activeElement)
    ? Number(document.activeElement.dataset.candidate)
    : -1;
  container.replaceChildren();
  if (!candidates.length) {
    const p = document.createElement("p");
    p.className = "candidate-empty";
    p.textContent = "The search will collect its best mechanisms here.";
    container.append(p);
    return;
  }
  candidates.slice(0, 3).forEach((result, index) => {
    const button = document.createElement("button");
    button.className = `candidate${selectedCandidate === index ? " selected" : ""}`;
    button.dataset.candidate = index;
    button.setAttribute("aria-pressed", String(selectedCandidate === index));
    button.setAttribute(
      "aria-label",
      `Candidate ${String.fromCharCode(65 + index)}: ${families[result.design.family]}, ${formatError(result.error)} percent path deviation`,
    );
    const letter = document.createElement("span");
    letter.className = "candidate-letter";
    letter.textContent = String.fromCharCode(65 + index);
    const info = document.createElement("div"),
      name = document.createElement("strong"),
      detail = document.createElement("small");
    name.textContent =
      result.design.family === "fourbar" ? "Four-bar" : "Crank-slider";
    detail.textContent = index === 0 ? "CLOSEST TRACE" : "ANOTHER POSSIBILITY";
    info.append(name, detail);
    const error = document.createElement("span");
    error.className = "candidate-error";
    error.textContent = `${formatError(result.error)}%`;
    button.append(letter, info, error);
    container.append(button);
    button.addEventListener("click", () => {
      stopSearch();
      pinned = true;
      applyCandidate(result, index);
      container.children[index]?.focus({ preventScroll: true });
      $("search-detail").textContent =
        `Candidate ${String.fromCharCode(65 + index)} is on the drawing board.`;
    });
  });
  if (focused >= 0)
    container.children[Math.min(focused, container.children.length - 1)]?.focus(
      { preventScroll: true },
    );
}

function stopSearch(announce = true) {
  searchRevision++;
  searchWorker?.terminate();
  searchWorker = null;
  searchActive = false;
  $("search-button").innerHTML =
    'Find a mechanism <span aria-hidden="true">→</span>';
  $("search-button").setAttribute("aria-label", "Find a mechanism");
  $("search-state").textContent = best
    ? announce
      ? "SEARCH PAUSED · BEST FIT KEPT"
      : "EDITING THE MECHANISM"
    : "READY TO EXPLORE";
}

function launchSearch() {
  if (target.length < 3) {
    toast("Draw a loop or choose an example path first.");
    return;
  }
  stopSearch(false);
  const revision = searchRevision,
    snapshot = clone(target);
  pinned = false;
  selectedCandidate = 0;
  candidates = [];
  best = null;
  chartHistory = [];
  searchActive = true;
  $("search-button").innerHTML =
    'Pause search <span aria-hidden="true">Ⅱ</span>';
  $("search-button").setAttribute("aria-label", "Pause mechanism search");
  $("search-state").textContent = "TRYING THE FIRST MECHANISMS";
  $("search-detail").textContent = "Testing ideas. Keeping the promising ones.";
  $("error-output").textContent = "—";
  $("fit-grade").textContent = "Looking for a fit";
  $("fit-description").textContent =
    "Measuring candidate paths against this drawing.";
  $("export-button").disabled = true;
  $("share-button").disabled = true;
  $("search-fill").style.width = "0%";
  $("evaluation-count").textContent = "0 TRIALS";
  renderCandidates();
  drawHistory();
  $("restore-fit").disabled = true;
  try {
    searchWorker = new Worker(new URL("./worker.mjs", import.meta.url), {
      type: "module",
    });
    searchWorker.onmessage = ({ data }) => {
      if (revision !== searchRevision) return;
      if (data.type === "error") {
        stopSearch();
        toast(`The search stopped: ${data.message}`);
        return;
      }
      const state = data.state;
      if (!state?.best) return;
      best = state.best;
      candidates = state.candidates?.length ? state.candidates : [best];
      chartHistory = state.history || [];
      $("search-fill").style.width =
        `${Math.min(100, (state.generation / 150) * 100)}%`;
      $("search-state").textContent =
        `REFINING · GENERATION ${String(state.generation).padStart(3, "0")}`;
      $("evaluation-count").textContent =
        `${state.evaluations.toLocaleString()} TRIALS`;
      $("machine").dataset.generation = state.generation;
      $("machine").dataset.evaluations = state.evaluations;
      $("machine").dataset.bestError = best.error;
      if (!pinned) applyCandidate(best, 0);
      else renderCandidates();
      drawHistory();
      if (data.type === "done" || state.done) {
        searchActive = false;
        searchWorker.terminate();
        searchWorker = null;
        $("search-button").innerHTML =
          'Explore another search <span aria-hidden="true">↗</span>';
        $("search-button").setAttribute(
          "aria-label",
          "Explore another mechanism search",
        );
        $("search-state").textContent = "SEARCH COMPLETE · YOUR TURN";
        $("search-fill").style.width = "100%";
        $("search-detail").textContent = "One motor. A small choreography.";
        $("restore-fit").disabled = false;
        persist();
      }
    };
    searchWorker.onerror = (event) => {
      if (revision !== searchRevision) return;
      stopSearch();
      $("search-state").textContent = "SEARCH COULD NOT START";
      toast(
        "The background search could not start. Reload this page to try again.",
      );
      console.error("Mechanism search failed:", event.message);
    };
    let checksum = 2166136261;
    for (const p of snapshot)
      checksum =
        Math.imul(checksum ^ Math.round((p.x + p.y * 2) * 10000), 16777619) >>>
        0;
    searchWorker.postMessage({
      target: snapshot,
      family,
      seed: (checksum + searchPass++ * 9973) >>> 0,
    });
  } catch {
    stopSearch();
    toast(
      "This browser could not open the background search. Try this page in Chrome.",
    );
  }
}

function updateEditedDesign(final = false) {
  if (!design) return;
  view.setState({
    design: clone(design),
    target,
    curve: sampleMechanism(design, 256),
  });
  syncTuning();
  $("machine-title").textContent = `${families[design.family]} / edited`;
  const measure = () => {
    if (!design || target.length < 3) return;
    const result = measureDesign(design, target);
    updateFit(result.error, true);
    renderCandidates();
  };
  clearTimeout(editingTimer);
  if (final) measure();
  else editingTimer = setTimeout(measure, 70);
  $("search-state").textContent = "YOUR GEOMETRY · YOUR MOTION";
}

$("search-button").addEventListener("click", () =>
  searchActive ? stopSearch() : launchSearch(),
);
$("clear-path").addEventListener("click", () => {
  if (target.length) drawingHistory.push({ target: clone(target), preset });
  stopSearch();
  target = [];
  preset = "custom";
  sketch.clear();
  view.setState({ target: [] });
  best = null;
  candidates = [];
  chartHistory = [];
  renderCandidates();
  drawHistory();
  $("error-output").textContent = "—";
  $("fit-grade").textContent = "Waiting for a path";
  $("restore-fit").disabled = true;
  $("export-button").disabled = true;
  $("share-button").disabled = true;
  $("point-count").textContent = "DRAW A NEW LOOP";
  $("undo-button").disabled = !drawingHistory.length;
  document.querySelectorAll("[data-preset]").forEach((button) => {
    button.classList.remove("active");
    button.setAttribute("aria-pressed", "false");
  });
  $("sketch").focus({ preventScroll: true });
});
$("close-path").addEventListener("click", () => sketch.finish());
$("undo-button").addEventListener("click", () => {
  const previous = drawingHistory.pop();
  if (!previous) return;
  ({ target, preset } = previous);
  $("undo-button").disabled = !drawingHistory.length;
  syncTarget();
  launchSearch();
});
document.querySelectorAll("[data-preset]").forEach((button) =>
  button.addEventListener("click", () => {
    drawingHistory.push({ target: clone(target), preset });
    if (drawingHistory.length > 20) drawingHistory.shift();
    preset = button.dataset.preset;
    target = presetTarget(preset, 96);
    study++;
    $("undo-button").disabled = false;
    syncTarget();
    launchSearch();
  }),
);
$("family").addEventListener("change", () => {
  family = $("family").value;
  launchSearch();
});

function syncPlaying() {
  $("play-button").innerHTML =
    `<span aria-hidden="true">${playing ? "Ⅱ" : "▶"}</span>`;
  $("play-button").setAttribute(
    "aria-label",
    playing ? "Pause animation" : "Play animation",
  );
  $("play-button").setAttribute("aria-pressed", String(playing));
  $("export-button").innerHTML =
    `Export ${playing ? "animated" : "construction"} drawing <span>↓ SVG</span>`;
  view.setState({
    playing,
    reducedMotion: playing ? false : reducedMotion.matches,
  });
}
$("play-button").addEventListener("click", () => {
  playing = !playing;
  syncPlaying();
});
$("speed").addEventListener("input", () => {
  speed = Number($("speed").value);
  $("speed-output").textContent = `${speed.toFixed(2)} rev/s`;
  $("speed").setAttribute(
    "aria-valuetext",
    `${speed.toFixed(2)} revolutions per second`,
  );
  $("speed").style.setProperty("--fill", `${((speed - 0.15) / 1.35) * 100}%`);
  view.setState({ speed });
});
for (const [id, key] of [
  ["tracer-u", "traceAlong"],
  ["tracer-v", "traceOffset"],
]) {
  $(id).addEventListener("input", () => {
    if (!design) return;
    stopSearch(false);
    pinned = true;
    selectedCandidate = -1;
    design.params[key] = Number($(id).value);
    updateEditedDesign(false);
  });
  $(id).addEventListener("change", () => {
    updateEditedDesign(true);
    persist();
  });
}
$("restore-fit").addEventListener("click", () => {
  if (best) {
    pinned = false;
    applyCandidate(best, 0);
    $("search-state").textContent = searchActive
      ? "REFINING THE BEST FIT"
      : "BEST FIT RESTORED";
  }
});
$("target-toggle").addEventListener("click", () => {
  showTarget = !showTarget;
  $("target-toggle").classList.toggle("active", showTarget);
  $("target-toggle").setAttribute("aria-pressed", String(showTarget));
  view.setState({ showTarget });
});
$("dimensions-toggle").addEventListener("click", () => {
  showDimensions = !showDimensions;
  $("dimensions-toggle").classList.toggle("active", showDimensions);
  $("dimensions-toggle").setAttribute("aria-pressed", String(showDimensions));
  view.setState({ showDimensions });
});

const dialog = $("about-dialog");
$("about-button").addEventListener("click", () => dialog.showModal());
$("error-help").addEventListener("click", () => dialog.showModal());
dialog.addEventListener("click", (event) => {
  if (event.target !== dialog) return;
  const r = dialog.getBoundingClientRect();
  if (
    event.clientX < r.left ||
    event.clientX > r.right ||
    event.clientY < r.top ||
    event.clientY > r.bottom
  )
    dialog.close();
});
$("export-button").addEventListener("click", () => {
  if (!design) return;
  try {
    const svg = view.snapshotSVG({
      title: `Motion Foundry — ${families[design.family]}`,
      animated: playing,
    });
    const url = URL.createObjectURL(new Blob([svg], { type: "image/svg+xml" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = `motion-foundry-${design.family}.svg`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 15000);
    toast(
      playing
        ? "Your animated assembly is ready. Open the SVG in a browser to play it."
        : "Your construction drawing is ready. A still SVG has been downloaded.",
    );
  } catch (error) {
    toast("This drawing could not be exported. Please try again.");
    console.error(error);
  }
});
$("share-button").addEventListener("click", async () => {
  if (!design) return;
  const state = {
    target: target.map((p) => ({
      x: Number(p.x.toFixed(4)),
      y: Number(p.y.toFixed(4)),
    })),
    preset,
    family,
    design,
  };
  const encoded = btoa(JSON.stringify(state))
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/, "");
  const hash = `#v1=${encoded}`,
    url = `${location.origin}${location.pathname}${hash}`;
  window.history.replaceState(null, "", hash);
  try {
    await navigator.clipboard.writeText(url);
    toast("Study copied. Your drawing and mechanism travel together.");
  } catch {
    toast("Your study is in the address bar. Copy the URL to share it.");
  }
});
document.addEventListener("keydown", (event) => {
  if (
    dialog.open ||
    /INPUT|TEXTAREA|SELECT/.test(event.target.tagName) ||
    event.target.isContentEditable ||
    event.target === $("sketch")
  )
    return;
  if (event.metaKey || event.ctrlKey || event.altKey) return;
  if (event.code === "Space" && event.target.tagName !== "BUTTON") {
    event.preventDefault();
    if (!event.repeat) {
      playing = !playing;
      syncPlaying();
    }
  } else if (event.key === "Escape") {
    stopSearch();
    playing = false;
    syncPlaying();
  }
});
reducedMotion.addEventListener("change", (event) => {
  if (event.matches) {
    playing = false;
    syncPlaying();
  }
  view.setState({ reducedMotion: event.matches });
});
window.addEventListener("pagehide", persist);

syncTarget();
syncPlaying();
try {
  const result = design
    ? measureDesign(design, target)
    : fitDesignToTarget(presetDesign(preset), target);
  if (result?.design && Number.isFinite(result.error)) {
    best = result;
    candidates = [result];
    applyCandidate(result);
  }
} catch {
  /* The search will provide an initial mechanism. */
}
if (!initialShared && !initialRestored) setTimeout(launchSearch, 80);
else {
  $("search-state").textContent = initialShared
    ? "A SHARED STUDY · MAKE IT YOURS"
    : "YOUR SAVED STUDY · READY TO EXPLORE";
  $("search-detail").textContent = initialShared
    ? "Someone drew this. Where will you take it?"
    : "Right where you left it.";
}
