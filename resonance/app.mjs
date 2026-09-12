import { insideOutline, solveMembrane } from "./physics.mjs";
import { ResonanceAudio } from "./audio.mjs";
import { MembraneView } from "./render.mjs";
import {
  PRESETS,
  NAMES,
  clamp,
  clone,
  createInstrument,
  validateInstrument,
  instrumentOutline,
  frequenciesFor,
  strikeOptions,
  drawnInstrument,
} from "./instrument.mjs";

const $ = (id) => document.getElementById(id);
const storageKey = "resonance-instruments-v1";
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
const audio = new ResonanceAudio();
const cache = new Map();
let instrument = createInstrument();
let solution = null,
  solutionKey = "",
  selectedMode = null,
  modePage = 0,
  tool = "play",
  activeHandle = 0;
let rack = Array(4).fill(null),
  history = [],
  editStart = null,
  lastStrike = { x: 0.12, y: 0.08 };
let solveTimer,
  toastTimer,
  patternTimer,
  patternStep = 0,
  patternBusy = false,
  patternGeneration = 0,
  meterFrame;
let pendingId = 0,
  latestRevision = 0,
  volume = 65,
  muted = false;
let worker = null,
  workerFailed = false;
const pending = new Map();

try {
  const saved = JSON.parse(localStorage.getItem(storageKey) || "null");
  if (saved) {
    instrument = validateInstrument(saved.current) || instrument;
    rack = Array.from({ length: 4 }, (_, i) =>
      validateInstrument(saved.rack?.[i]),
    );
    volume = Number.isFinite(saved.volume) ? clamp(saved.volume, 0, 100) : 65;
  }
} catch {
  /* The playground works without storage. */
}

if (location.hash.startsWith("#v1=")) {
  try {
    const encoded = location.hash.slice(4);
    if (encoded.length > 6000) throw new Error("Oversized instrument");
    const shared = validateInstrument(
      JSON.parse(atob(encoded.replace(/-/g, "+").replace(/_/g, "/"))),
    );
    if (!shared) throw new Error("Invalid instrument");
    instrument = shared;
  } catch {
    toast("This instrument link could not be read. Here is a fresh shape.");
  }
}

function persist() {
  try {
    localStorage.setItem(
      storageKey,
      JSON.stringify({ current: instrument, rack, volume }),
    );
  } catch {
    /* Storage may be unavailable or full; creation and sharing still work. */
  }
}

function toast(message) {
  $("toast").textContent = message;
  $("toast").classList.add("visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => $("toast").classList.remove("visible"), 3300);
}

function createWorker() {
  try {
    worker = new Worker(new URL("./worker.mjs", import.meta.url), {
      type: "module",
    });
    worker.onmessage = ({ data }) => {
      const request = pending.get(data.id);
      if (!request) return;
      pending.delete(data.id);
      if (data.error) request.reject(new Error(data.error));
      else request.resolve(data);
    };
    worker.onerror = () => {
      workerFailed = true;
      worker.terminate();
      worker = null;
      for (const request of pending.values())
        request.reject(new Error("Background solver unavailable."));
      pending.clear();
    };
  } catch {
    workerFailed = true;
  }
}
createWorker();

function geometryKey(value) {
  return JSON.stringify([
    value.width,
    value.points.map((p) => [Number(p.x.toFixed(4)), Number(p.y.toFixed(4))]),
  ]);
}

async function solutionFor(value) {
  const key = geometryKey(value);
  if (cache.has(key)) return cache.get(key);
  const outline = instrumentOutline(value);
  let result;
  if (worker && !workerFailed) {
    try {
      result = await new Promise((resolve, reject) => {
        const id = ++pendingId;
        pending.set(id, { resolve, reject });
        worker.postMessage({ id, outline, resolution: 41, modeCount: 18 });
      });
    } catch (error) {
      if (!workerFailed) throw error;
    }
  }
  if (!result) {
    // A module-worker restriction should not prevent the instrument from working.
    const started = performance.now();
    result = {
      solution: solveMembrane(outline, { resolution: 41, modeCount: 18 }),
      elapsed: performance.now() - started,
    };
  }
  cache.set(key, result);
  if (cache.size > 32) cache.delete(cache.keys().next().value);
  return result;
}

const view = new MembraneView($("membrane"), {
  onActivityChange: (active) => {
    $("view-label").textContent = active
      ? selectedMode === null
        ? "LIVE VIBRATION / FULL SOUND"
        : `LIVE VIBRATION / MODE ${String(selectedMode + 1).padStart(2, "0")}`
      : selectedMode === null
        ? "VIBRATION STUDY / MODE 05"
        : `ISOLATED VIBRATION / MODE ${String(selectedMode + 1).padStart(2, "0")}`;
  },
  onStrike: (x, y) => strikeCurrent(x, y),
  onSelectHandle: (index) => {
    activeHandle = index;
    view.setState({ activeHandle });
  },
  onHandle: (index, x, y) => {
    if (!editStart) editStart = clone(instrument);
    const angle = (index * Math.PI) / 8;
    const radius = clamp(
      (x / (instrument.width / 100)) * Math.cos(angle) + y * Math.sin(angle),
      0.28,
      0.86,
    );
    instrument.points[index] = {
      x: Math.cos(angle) * radius,
      y: Math.sin(angle) * radius,
    };
    instrument.preset = "custom";
    activeHandle = index;
    syncShape();
    requestSolve(70);
  },
  onHandleEnd: () => {
    if (editStart) {
      remember(editStart);
      editStart = null;
    }
    requestSolve(0);
    persist();
  },
  onDraw: (path) => {
    const drawn = drawnInstrument(path, instrument);
    if (!drawn) {
      toast("Draw a larger closed outline across the surface.");
      return;
    }
    remember();
    instrument = drawn;
    selectedMode = null;
    modePage = 0;
    setTool("shape");
    syncControls();
    requestSolve(0);
    persist();
    toast("Your outline is ready. Pull the rim to fine-tune it.");
  },
});
view.setState({ reducedMotion: reducedMotion.matches });
reducedMotion.addEventListener("change", (event) =>
  view.setState({ reducedMotion: event.matches }),
);

function syncShape() {
  view.setState({
    outline: instrumentOutline(instrument),
    controlPoints: instrument.points.map((p) => ({
      x: (p.x * instrument.width) / 100,
      y: p.y,
    })),
    activeHandle,
  });
  document.querySelectorAll("[data-preset]").forEach((button) => {
    const selected = button.dataset.preset === instrument.preset;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-pressed", String(selected));
  });
  const serial = Math.max(0, PRESETS.indexOf(instrument.preset)) + 1;
  $("instrument-name").replaceChildren(
    document.createTextNode(NAMES[instrument.preset] + " "),
  );
  const label = document.createElement("span");
  label.textContent = `/ ${String(serial).padStart(2, "0")}`;
  $("instrument-name").append(label);
}

function syncControls() {
  for (const key of ["width", "tension", "decay", "mallet"])
    $(key).value = instrument[key];
  $("volume").value = volume;
  syncShape();
  updateReadouts();
}

function updateReadouts() {
  $("width-output").textContent = `${Math.round(instrument.width)}%`;
  $("tension-output").textContent = `${Math.round(instrument.tension)}%`;
  $("decay-output").textContent = `${instrument.decay.toFixed(1)} s`;
  $("mallet-output").textContent =
    instrument.mallet < 30
      ? "Soft felt"
      : instrument.mallet > 75
        ? "Hard wood"
        : "Balanced";
  $("volume-output").textContent = `${muted ? 0 : Math.round(volume)}%`;
  document.querySelectorAll("input[type=range]").forEach((input) => {
    input.style.setProperty(
      "--fill",
      `${((Number(input.value) - Number(input.min)) / (Number(input.max) - Number(input.min))) * 100}%`,
    );
    const output = $(`${input.id}-output`);
    if (output) input.setAttribute("aria-valuetext", output.textContent);
  });
  if (solution) {
    const frequencies = frequenciesFor(solution, instrument);
    $("frequency-output").textContent = frequencies[0].toFixed(1);
    const midi = Math.round(69 + 12 * Math.log2(frequencies[0] / 440));
    const notes = [
      "C",
      "C♯",
      "D",
      "E♭",
      "E",
      "F",
      "F♯",
      "G",
      "A♭",
      "A",
      "B♭",
      "B",
    ];
    $("note-output").textContent =
      notes[((midi % 12) + 12) % 12] + (Math.floor(midi / 12) - 1);
    $("note-output").title = "Nearest musical note";
    document.querySelectorAll(".mode-button").forEach((button) => {
      const index = Number(button.dataset.mode);
      button.querySelector(".mode-hz").textContent = Math.round(
        frequencies[index],
      );
      button.setAttribute(
        "aria-label",
        `Mode ${index + 1}, ${Math.round(frequencies[index])} hertz. Listen to this mode.`,
      );
    });
  }
}

function remember(value = instrument) {
  history.push(clone(value));
  if (history.length > 30) history.shift();
  $("undo-button").disabled = !history.length;
}

function setTool(next) {
  tool = next;
  view.setState({ tool, activeHandle });
  for (const name of ["play", "shape"]) {
    $(`${name}-tool`).classList.toggle("selected", tool === name);
    $(`${name}-tool`).setAttribute("aria-pressed", String(tool === name));
  }
  $("draw-button").classList.toggle("active", tool === "draw");
  $("draw-button").setAttribute("aria-pressed", String(tool === "draw"));
  $("stage-hint").textContent =
    tool === "play"
      ? "CLICK TO STRIKE · SPACE TO PLAY"
      : tool === "shape"
        ? "PULL A RIM POINT · ARROWS TO ADJUST"
        : "DRAW A CLOSED OUTLINE · RELEASE TO FINISH";
  $("sound-prompt").classList.toggle(
    "hidden",
    tool !== "play" || audio.state === "running",
  );
}

function requestSolve(delay = 0) {
  clearTimeout(solveTimer);
  const revision = ++latestRevision;
  $("solve-status").textContent = "Finding its voice…";
  for (const id of [
    "strike-button",
    "save-pad",
    "export-button",
    "share-button",
    "center-strike",
  ])
    $(id).disabled = true;
  if (solution) view.setState({ outline: instrumentOutline(instrument) });
  solveTimer = setTimeout(async () => {
    try {
      const snapshot = clone(instrument);
      const result = await solutionFor(snapshot);
      if (revision !== latestRevision) return;
      solution = result.solution;
      solutionKey = geometryKey(snapshot);
      if (!insideOutline(solution.outline, lastStrike.x, lastStrike.y))
        lastStrike = { x: 0.12, y: 0.08 };
      view.setState({ solution, selectedMode });
      $("loading-state").classList.add("hidden");
      $("solve-status").textContent = `${solution.modes.length} MODES · READY`;
      $("stage").dataset.solverResidual = solution.residualMax;
      $("stage").dataset.solveMs = result.elapsed.toFixed(1);
      $("stage").dataset.nodes = solution.coordinates.length;
      $("stage").dataset.geometry = solutionKey;
      for (const id of [
        "strike-button",
        "save-pad",
        "export-button",
        "share-button",
        "center-strike",
      ])
        $(id).disabled = false;
      updateReadouts();
      renderModes();
      persist();
    } catch (error) {
      if (revision !== latestRevision) return;
      $("loading-state").classList.add("hidden");
      $("solve-status").textContent = "TRY ANOTHER SHAPE";
      toast(
        "This outline could not be solved. Try a starting shape or undo your last edit.",
      );
      console.error("Membrane calculation failed:", error);
    }
  }, delay);
}

function modeThumbnail(canvas, result, index, outline = result.outline) {
  const ctx = canvas.getContext("2d");
  canvas.width = 90;
  canvas.height = 66;
  const mode = result.modes[index];
  const cell = 72 / (result.resolution - 1);
  ctx.clearRect(0, 0, 90, 66);
  for (let j = 0; j < result.coordinates.length; j++) {
    const p = result.coordinates[j],
      v = mode.values[j];
    const strength = Math.min(1, Math.abs(v));
    ctx.fillStyle =
      v > 0
        ? `rgba(184,92,54,${0.13 + strength * 0.76})`
        : `rgba(91,123,107,${0.13 + strength * 0.76})`;
    ctx.fillRect(
      45 + p.x * 27 - cell / 2,
      33 + p.y * 23 - cell / 2,
      cell + 0.7,
      cell + 0.7,
    );
  }
  ctx.beginPath();
  outline.forEach((p, i) => {
    if (!i) ctx.moveTo(45 + p.x * 27, 33 + p.y * 23);
    else ctx.lineTo(45 + p.x * 27, 33 + p.y * 23);
  });
  ctx.closePath();
  ctx.strokeStyle = "#777e6244";
  ctx.lineWidth = 0.8;
  ctx.stroke();
}

const modePageButton = document.createElement("button");
modePageButton.id = "mode-page";
modePageButton.className = "all-modes";
modePageButton.title = "Show the higher vibration modes";
modePageButton.addEventListener("click", () => {
  modePage = modePage ? 0 : 1;
  renderModes();
});
$("all-modes").before(modePageButton);

function renderModes() {
  if (!solution) return;
  const container = $("mode-strip");
  container.replaceChildren();
  const frequencies = frequenciesFor(solution, instrument);
  const start = modePage * 9,
    end = Math.min(solution.modes.length, start + 9);
  for (let index = start; index < end; index++) {
    const button = document.createElement("button");
    button.className = "mode-button";
    button.dataset.mode = index;
    button.setAttribute("aria-pressed", String(selectedMode === index));
    button.classList.toggle("selected", selectedMode === index);
    button.setAttribute(
      "aria-label",
      `Mode ${index + 1}, ${Math.round(frequencies[index])} hertz. Listen to this mode.`,
    );
    const number = document.createElement("span");
    number.className = "mode-number";
    number.textContent = String(index + 1).padStart(2, "0");
    const canvas = document.createElement("canvas");
    canvas.setAttribute("aria-hidden", "true");
    const hz = document.createElement("span");
    hz.className = "mode-hz";
    hz.textContent = Math.round(frequencies[index]);
    button.append(number, canvas, hz);
    container.append(button);
    modeThumbnail(canvas, solution, index);
    button.addEventListener("click", () => {
      selectedMode = selectedMode === index ? null : index;
      view.setState({ selectedMode });
      renderModes();
      document
        .querySelector(`[data-mode="${index}"]`)
        ?.focus({ preventScroll: true });
      if (selectedMode === null) strikeCurrent(lastStrike.x, lastStrike.y);
      else {
        // Audition a mode at its strongest point so the button always speaks.
        const values = solution.modes[index].values;
        let max = 0;
        for (let i = 1; i < values.length; i++)
          if (Math.abs(values[i]) > Math.abs(values[max])) max = i;
        const point = solution.coordinates[max];
        strikeCurrent(point.x, point.y, true);
      }
    });
  }
  $("all-modes").classList.toggle("selected", selectedMode === null);
  $("all-modes").setAttribute("aria-pressed", String(selectedMode === null));
  modePageButton.textContent = modePage ? "← 01–09" : "10–18 →";
  modePageButton.setAttribute(
    "aria-label",
    modePage ? "Show modes 1 through 9" : "Show modes 10 through 18",
  );
  $("mode-count").textContent = `${solution.modes.length} VOICES, ONE SURFACE`;
  $("mode-description").textContent =
    selectedMode === null
      ? "Tap a mode to see and hear one vibration on its own."
      : `Mode ${String(selectedMode + 1).padStart(2, "0")} · ${(frequencies[selectedMode] / frequencies[0]).toFixed(2)} × the fundamental. Click Full sound to combine them.`;
  $("view-label").textContent =
    selectedMode === null
      ? "VIBRATION STUDY / MODE 05"
      : `ISOLATED VIBRATION / MODE ${String(selectedMode + 1).padStart(2, "0")}`;
}

function startMeter() {
  if (meterFrame) return;
  let quietFrames = 0;
  const measure = () => {
    const peak = audio.readPeak?.() || 0;
    $("audio-status").dataset.signalPeak = peak.toFixed(6);
    $("audio-status").dataset.audioState = audio.state;
    if (peak > 0.00005) {
      quietFrames = 0;
      $("audio-status").dataset.maxPeak = Math.max(
        peak,
        Number($("audio-status").dataset.maxPeak || 0),
      ).toFixed(6);
    } else quietFrames++;
    $("audio-status").style.setProperty(
      "--level",
      `${Math.min(100, peak * 250)}%`,
    );
    if (quietFrames < 120 && !document.hidden)
      meterFrame = requestAnimationFrame(measure);
    else meterFrame = 0;
  };
  meterFrame = requestAnimationFrame(measure);
}

async function ensureAudio() {
  try {
    const state = await audio.unlock();
    if (state !== "running") throw new Error("Audio is paused");
    audio.setVolume(muted ? 0 : volume / 100);
    $("audio-status").textContent =
      muted || !volume
        ? "Sound is muted."
        : "Sound is on. Make a little noise.";
    $("audio-status").dataset.audioState = state;
    $("sound-prompt").classList.add("hidden");
    return true;
  } catch {
    $("audio-status").textContent = "Sound is paused. Tap Strike to try again.";
    toast("Your browser paused the audio. Tap Strike to enable sound.");
    return false;
  }
}

async function strikeCurrent(x = 0.12, y = 0.08, audition = false) {
  if (
    !solution ||
    solutionKey !== geometryKey(instrument) ||
    !insideOutline(solution.outline, x, y)
  )
    return;
  const snapshot = solution,
    voice = clone(instrument),
    mode = selectedMode;
  if (!(await ensureAudio())) return;
  const options = strikeOptions(snapshot, voice, x, y, mode);
  if (audition && mode !== null) {
    // A mode audition is heard directly; normal strikes use the contact pickup.
    options.amplitudes = options.amplitudes.map((v, i) => (i === mode ? 1 : 0));
  }
  if (Math.max(...options.amplitudes.map(Math.abs)) < 1e-10) {
    toast(
      "A quiet spot in this mode. Try striking a bright part of the surface.",
    );
    return;
  }
  audio.strike(options);
  if (snapshot === solution) {
    view.impulse(options.displacement, options.frequencies, options.decay);
    lastStrike = { x, y };
    $("view-label").textContent =
      mode === null
        ? "LIVE VIBRATION / FULL SOUND"
        : `LIVE VIBRATION / MODE ${String(mode + 1).padStart(2, "0")}`;
  }
  $("stage").dataset.strikeCount =
    Number($("stage").dataset.strikeCount || 0) + 1;
  startMeter();
}

function renderRack() {
  const container = $("rack");
  container.replaceChildren();
  rack.forEach((value, index) => {
    const slot = document.createElement("div");
    slot.className = `rack-slot${value ? "" : " empty"}`;
    slot.dataset.slot = index;
    const button = document.createElement("button");
    button.className = "pad-play";
    button.setAttribute(
      "aria-label",
      value
        ? `Play pad ${index + 1}: ${NAMES[value.preset]}`
        : `Save current instrument to pad ${index + 1}`,
    );
    const key = document.createElement("span");
    key.className = "pad-key";
    key.textContent = String(index + 1).padStart(2, "0");
    slot.append(key);
    const info = document.createElement("span");
    info.className = "pad-info";
    const title = document.createElement("strong");
    title.textContent = value ? NAMES[value.preset] : "An empty stage";
    const detail = document.createElement("span");
    if (value) {
      const canvas = document.createElement("canvas");
      canvas.setAttribute("aria-hidden", "true");
      button.append(canvas);
      detail.textContent = "FINDING ITS VOICE…";
      solutionFor(value)
        .then((result) => {
          if (!slot.isConnected) return;
          modeThumbnail(canvas, result.solution, 4, result.solution.outline);
          detail.textContent = `${Math.round(frequenciesFor(result.solution, value)[0])} HZ · ${value.decay.toFixed(1)} S`;
        })
        .catch(() => {
          detail.textContent = "TAP TO RETRY";
        });
    } else {
      const plus = document.createElement("span");
      plus.className = "pad-plus";
      plus.textContent = "+";
      button.append(plus);
      detail.className = "empty-note";
      detail.textContent = "Add your current voice";
    }
    info.append(title, detail);
    button.append(info);
    slot.append(button);
    container.append(slot);
    button.addEventListener("click", () =>
      value ? playPad(index) : savePad(index),
    );
    if (value) {
      const actions = document.createElement("div");
      actions.className = "pad-actions";
      const edit = document.createElement("button");
      edit.textContent = "↗";
      edit.title = `Edit pad ${index + 1} in the workbench`;
      edit.setAttribute("aria-label", edit.title);
      edit.addEventListener("click", () => {
        remember();
        instrument = clone(rack[index]);
        selectedMode = null;
        modePage = 0;
        syncControls();
        setTool("play");
        requestSolve();
        persist();
        $("stage").scrollIntoView({
          behavior: reducedMotion.matches ? "instant" : "smooth",
          block: "center",
        });
        toast(
          `Pad ${index + 1} is on the workbench. Changes become a new voice when saved.`,
        );
      });
      const remove = document.createElement("button");
      remove.textContent = "×";
      remove.title = `Remove pad ${index + 1}`;
      remove.setAttribute("aria-label", remove.title);
      remove.addEventListener("click", () => {
        rack[index] = null;
        if (!rack.some(Boolean)) stopPattern();
        renderRack();
        persist();
        toast(`Pad ${index + 1} cleared.`);
      });
      actions.append(edit, remove);
      slot.append(actions);
    }
  });
  $("pattern-button").disabled = !rack.some(Boolean);
  $("save-pad").textContent = rack.every(Boolean)
    ? "Rack full · clear a pad to add"
    : "＋ Add to instrument rack";
}

function savePad(index = rack.findIndex((value) => !value)) {
  if (!solution || solutionKey !== geometryKey(instrument)) {
    toast("Let the new shape finish tuning first.");
    return;
  }
  if (index < 0) {
    toast(
      "Your rack is full. Clear a pad with × to make room for another voice.",
    );
    return;
  }
  rack[index] = clone(instrument);
  renderRack();
  persist();
  toast(`Saved to pad ${index + 1}. Press ${index + 1} to play it.`);
}

async function playPad(index, fromPattern = false) {
  const generation = patternGeneration;
  const value = rack[index];
  if (!value) {
    if (!fromPattern) savePad(index);
    return;
  }
  if (!(await ensureAudio())) return;
  try {
    const result = await solutionFor(value);
    if (document.hidden || (fromPattern && generation !== patternGeneration))
      return;
    const options = strikeOptions(result.solution, value);
    audio.strike(options);
    startMeter();
    const element = document.querySelector(`[data-slot="${index}"]`);
    element?.classList.add("sounding");
    setTimeout(() => element?.classList.remove("sounding"), 250);
    if (geometryKey(value) === solutionKey)
      view.impulse(options.displacement, options.frequencies, options.decay);
  } catch {
    toast("This pad could not be played. Try saving the instrument again.");
  }
}

function stopPattern() {
  patternGeneration++;
  clearInterval(patternTimer);
  patternTimer = null;
  patternBusy = false;
  $("pattern-button").classList.remove("playing");
  $("pattern-button").innerHTML =
    '<span aria-hidden="true">▶</span> Play a pattern';
  $("pattern-button").setAttribute("aria-pressed", "false");
}

$("pattern-button").addEventListener("click", async () => {
  if (patternTimer) {
    stopPattern();
    audio.stop();
    return;
  }
  if (patternBusy) return;
  patternBusy = true;
  const generation = ++patternGeneration;
  if (!(await ensureAudio())) {
    patternBusy = false;
    return;
  }
  try {
    await Promise.all(rack.filter(Boolean).map((value) => solutionFor(value)));
  } catch {
    patternBusy = false;
    toast("One of these shapes could not be played. Try resaving it.");
    return;
  }
  if (generation !== patternGeneration || document.hidden) {
    patternBusy = false;
    return;
  }
  patternStep = 0;
  patternBusy = false;
  const beat = () => {
    const filled = rack
      .map((value, i) => (value ? i : null))
      .filter((i) => i !== null);
    if (!filled.length) {
      stopPattern();
      return;
    }
    const sequence = [0, 1, 2, 1, 0, 3, 2, 3];
    playPad(
      filled[sequence[patternStep++ % sequence.length] % filled.length],
      true,
    );
  };
  beat();
  patternTimer = setInterval(beat, 340);
  $("pattern-button").innerHTML =
    '<span aria-hidden="true">■</span> Stop the pattern';
  $("pattern-button").classList.add("playing");
  $("pattern-button").setAttribute("aria-pressed", "true");
});

document.querySelectorAll("[data-preset]").forEach((button) =>
  button.addEventListener("click", () => {
    remember();
    instrument = {
      ...createInstrument(button.dataset.preset),
      tension: instrument.tension,
      decay: instrument.decay,
      mallet: instrument.mallet,
    };
    selectedMode = null;
    modePage = 0;
    syncControls();
    setTool("play");
    requestSolve();
  }),
);
for (const key of ["width", "tension", "decay", "mallet"]) {
  $(key).addEventListener("pointerdown", () => {
    if (key === "width") editStart = clone(instrument);
  });
  $(key).addEventListener("keydown", () => {
    if (key === "width" && !editStart) editStart = clone(instrument);
  });
  $(key).addEventListener("input", () => {
    instrument[key] = Number($(key).value);
    if (key === "width") {
      syncShape();
      requestSolve(80);
    }
    updateReadouts();
    persist();
  });
  $(key).addEventListener("change", () => {
    if (key === "width") {
      if (editStart) remember(editStart);
      editStart = null;
      requestSolve();
    } else if (audio.state === "running") strikeCurrent();
  });
}
$("play-tool").addEventListener("click", () => setTool("play"));
$("shape-tool").addEventListener("click", () => setTool("shape"));
$("draw-button").addEventListener("click", () => {
  setTool(tool === "draw" ? "play" : "draw");
  if (tool === "draw") {
    toast(
      "Draw one loop around the center. We will smooth it into a membrane.",
    );
    $("membrane").focus({ preventScroll: true });
  }
});
$("strike-button").addEventListener("click", () => strikeCurrent());
$("center-strike").addEventListener("click", () => strikeCurrent(0, 0));
$("all-modes").addEventListener("click", () => {
  selectedMode = null;
  view.setState({ selectedMode });
  renderModes();
  strikeCurrent();
});
$("save-pad").addEventListener("click", () => savePad());
$("reset-button").addEventListener("click", () => {
  remember();
  const preset = PRESETS.includes(instrument.preset)
    ? instrument.preset
    : "circle";
  instrument = {
    ...createInstrument(preset),
    tension: instrument.tension,
    decay: instrument.decay,
    mallet: instrument.mallet,
  };
  selectedMode = null;
  syncControls();
  requestSolve();
});
function undo() {
  if (!history.length) return;
  instrument = history.pop();
  editStart = null;
  selectedMode = null;
  $("undo-button").disabled = !history.length;
  syncControls();
  requestSolve();
}
$("undo-button").addEventListener("click", undo);
$("mute-button").addEventListener("click", () => {
  muted = !muted;
  audio.setVolume(muted ? 0 : volume / 100);
  updateReadouts();
  $("mute-button").setAttribute("aria-pressed", String(muted));
  $("mute-button").setAttribute(
    "aria-label",
    muted ? "Unmute audio" : "Mute audio",
  );
  $("audio-status").textContent = muted
    ? "Sound is muted."
    : audio.state === "running"
      ? "Sound is on. Make a little noise."
      : "Sound starts with your first tap.";
});
$("volume").addEventListener("input", () => {
  volume = Number($("volume").value);
  muted = false;
  $("mute-button").setAttribute("aria-pressed", "false");
  $("mute-button").setAttribute("aria-label", "Mute audio");
  audio.setVolume(volume / 100);
  updateReadouts();
  persist();
  $("audio-status").textContent =
    volume === 0
      ? "Sound is muted."
      : audio.state === "running"
        ? "Sound is on. Make a little noise."
        : "Sound starts with your first tap.";
});

// Focus is a layout choice, so it works without fullscreen permissions.
let focusScroll = 0;
function setFocus(enabled) {
  if (enabled) focusScroll = window.scrollY;
  document.body.classList.toggle("instrument-focus", enabled);
  $("focus-button").setAttribute("aria-pressed", String(enabled));
  $("focus-button").innerHTML = enabled
    ? 'Exit focus <span aria-hidden="true">↙</span>'
    : 'Focus <span aria-hidden="true">⤢</span>';
  if (enabled) window.scrollTo(0, 0);
  else window.scrollTo(0, focusScroll);
}
$("focus-button").addEventListener("click", () => {
  setFocus(!document.body.classList.contains("instrument-focus"));
});

const dialog = $("science-dialog");
$("about-button").addEventListener("click", () => dialog.showModal());
dialog.addEventListener("click", (event) => {
  if (event.target !== dialog) return;
  const rect = dialog.getBoundingClientRect();
  if (
    event.clientX < rect.left ||
    event.clientX > rect.right ||
    event.clientY < rect.top ||
    event.clientY > rect.bottom
  )
    dialog.close();
});

$("export-button").addEventListener("click", async () => {
  if (!solution || solutionKey !== geometryKey(instrument)) return;
  const exportSolution = solution,
    exportInstrument = clone(instrument),
    exportStrike = { ...lastStrike },
    exportMode = selectedMode;
  $("export-button").disabled = true;
  const label = $("export-button").textContent;
  $("export-button").textContent = "Preparing your sound…";
  // Let the status paint before rendering the offline waveform.
  await new Promise((resolve) =>
    requestAnimationFrame(() => setTimeout(resolve, 0)),
  );
  try {
    const options = strikeOptions(
      exportSolution,
      exportInstrument,
      exportStrike.x,
      exportStrike.y,
      exportMode,
    );
    if (exportMode !== null)
      options.amplitudes = options.amplitudes.map((v, i) =>
        i === exportMode ? 1 : 0,
      );
    const blob = audio.exportWav(options, volume / 100);
    const url = URL.createObjectURL(blob),
      link = document.createElement("a");
    link.href = url;
    link.download = `resonance-${exportInstrument.preset}${exportMode !== null ? `-mode-${exportMode + 1}` : ""}.wav`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 15000);
    toast("Your sound is ready. A WAV file has been downloaded.");
  } catch {
    toast("The sound could not be exported. Please try again.");
  } finally {
    $("export-button").textContent = label;
    $("export-button").disabled = solutionKey !== geometryKey(instrument);
  }
});

$("share-button").addEventListener("click", async () => {
  const compact = clone(instrument);
  compact.points = compact.points.map((p) => ({
    x: Number(p.x.toFixed(4)),
    y: Number(p.y.toFixed(4)),
  }));
  const encoded = btoa(JSON.stringify(compact))
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/, "");
  const url = `${location.origin}${location.pathname}#v1=${encoded}`;
  try {
    await navigator.clipboard.writeText(url);
    toast("Instrument link copied. Your shape and tuning travel with it.");
  } catch {
    const input = document.createElement("textarea");
    input.value = url;
    input.style.position = "fixed";
    input.style.opacity = "0";
    document.body.append(input);
    input.select();
    const copied = document.execCommand("copy");
    input.remove();
    toast(
      copied
        ? "Instrument link copied."
        : "Copy the instrument URL from the address bar.",
    );
    if (!copied) window.history.replaceState(null, "", `#v1=${encoded}`);
  }
});

document.addEventListener("keydown", (event) => {
  if (
    dialog.open ||
    /INPUT|TEXTAREA|SELECT/.test(event.target.tagName) ||
    event.target.isContentEditable
  )
    return;
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "z") {
    event.preventDefault();
    undo();
    return;
  }
  if (event.metaKey || event.ctrlKey || event.altKey) return;
  if (event.code === "Space") {
    if (event.target.tagName === "BUTTON") return;
    event.preventDefault();
    if (!event.repeat) strikeCurrent();
  } else if (/^[1-4]$/.test(event.key) && !event.repeat) {
    event.preventDefault();
    playPad(Number(event.key) - 1);
  } else if (event.key === "Escape") {
    if (document.body.classList.contains("instrument-focus")) {
      setFocus(false);
      $("focus-button").focus({ preventScroll: true });
    }
    stopPattern();
    audio.stop();
    setTool("play");
  } else if (
    event.target === $("membrane") &&
    tool === "shape" &&
    event.key.startsWith("Arrow")
  ) {
    event.preventDefault();
    if (event.key === "ArrowLeft" || event.key === "ArrowRight")
      activeHandle =
        (activeHandle + (event.key === "ArrowRight" ? 1 : 15)) % 16;
    else {
      if (!event.repeat) remember();
      const point = instrument.points[activeHandle],
        angle = (activeHandle * Math.PI) / 8;
      const radius = clamp(
        Math.hypot(point.x, point.y) +
          (event.key === "ArrowUp" ? 0.035 : -0.035),
        0.28,
        0.86,
      );
      instrument.points[activeHandle] = {
        x: Math.cos(angle) * radius,
        y: Math.sin(angle) * radius,
      };
      instrument.preset = "custom";
      syncShape();
      requestSolve(70);
    }
    view.setState({ activeHandle });
  }
});
document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    stopPattern();
    audio.stop();
    if (meterFrame) cancelAnimationFrame(meterFrame);
    meterFrame = 0;
  }
});
window.addEventListener("pagehide", () => {
  persist();
  stopPattern();
  audio.stop();
});

audio.setVolume(volume / 100);
syncControls();
renderRack();
requestSolve();
