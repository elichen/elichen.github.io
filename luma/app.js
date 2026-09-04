import {
  WORLD,
  PRESETS,
  traceScene,
  vertices,
  endpoints,
  contains,
  direction,
  rotate,
  sub,
  add,
  mul,
  dot,
  wavelengthRGB,
} from "./physics.mjs";

const $ = (selector) => document.querySelector(selector);
const canvas = $("#canvas"),
  ctx = canvas.getContext("2d");
const stage = $("#stage"),
  inspector = $("#inspector-content");
const STORAGE = "luma-scene-v1",
  MAX_OBJECTS = 24;
const names = {
  source: "Light source",
  prism: "Prism",
  mirror: "Mirror",
  lens: "Lens",
};
const hints = {
  source:
    "A little light is all it takes. White light holds an entire spectrum.",
  prism:
    "Different colors take different paths through glass. Turn the prism to find them.",
  mirror:
    "The angle in equals the angle out. Sometimes a change of direction is everything.",
  lens: "Parallel rays meet at the focal point. Try a wide beam to see it happen.",
};
let objects,
  preset = "prism",
  selected = 2,
  showGrid = true;
let paths = [],
  dirty = true,
  viewport = { width: 0, height: 0, scale: 1, x: 0, y: 0, dpr: 1 };
let undoStack = [],
  redoStack = [],
  drag = null,
  sliderStart = null,
  keyStart = null;
let toastTimer,
  saveTimer,
  hovered = null;
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
const beamCanvas = document.createElement("canvas"),
  beamCtx = beamCanvas.getContext("2d");
const clone = (value) => JSON.parse(JSON.stringify(value));
const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
const normalizeAngle = (angle) => ((((angle + 180) % 360) + 360) % 360) - 180;
const sceneState = () =>
  JSON.stringify({ objects, preset, selected, showGrid });

function validScene(value) {
  if (
    !value ||
    !Array.isArray(value.objects) ||
    value.objects.length > MAX_OBJECTS ||
    !Object.hasOwn(PRESETS, value.preset)
  )
    return false;
  const ids = new Set();
  return value.objects.every((o) => {
    if (
      !o ||
      !Object.hasOwn(names, o.type) ||
      !Number.isInteger(o.id) ||
      ids.has(o.id)
    )
      return false;
    ids.add(o.id);
    if (
      ![o.x, o.y, o.angle].every(Number.isFinite) ||
      o.x < 40 ||
      o.x > 1160 ||
      o.y < 55 ||
      o.y > 705 ||
      Math.abs(o.angle) > 180
    )
      return false;
    if (o.type === "source")
      return (
        Number.isFinite(o.width) &&
        o.width >= 1 &&
        o.width <= 220 &&
        ["white", "650", "590", "530", "490", "420"].includes(o.color) &&
        typeof o.enabled === "boolean"
      );
    if (!Number.isFinite(o.size) || o.size < 50 || o.size > 350) return false;
    if (o.type === "prism")
      return (
        Number.isFinite(o.dispersion) &&
        o.dispersion >= 0.004 &&
        o.dispersion <= 0.04
      );
    if (o.type === "lens")
      return Number.isFinite(o.focal) && o.focal >= 80 && o.focal <= 500;
    return true;
  });
}
function save() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    try {
      localStorage.setItem(STORAGE, sceneState());
    } catch {
      /* Storage is optional. */
    }
  }, 180);
}
function restore(state) {
  const value = typeof state === "string" ? JSON.parse(state) : state;
  objects = clone(value.objects);
  preset = value.preset;
  selected = objects.some((o) => o.id === value.selected)
    ? value.selected
    : null;
  showGrid = value.showGrid !== false;
  updateUI();
  dirty = true;
  save();
}
function commit(before) {
  if (before === sceneState()) return;
  undoStack.push(before);
  if (undoStack.length > 60) undoStack.shift();
  redoStack = [];
  updateHistory();
  save();
}
function change(action) {
  finishKeyEdit();
  const before = sceneState();
  action();
  commit(before);
  updateUI();
  dirty = true;
}
function updateHistory() {
  $("#undo-button").disabled = !undoStack.length;
  $("#redo-button").disabled = !redoStack.length;
}
function undo() {
  finishKeyEdit();
  if (!undoStack.length) return;
  redoStack.push(sceneState());
  restore(undoStack.pop());
  updateHistory();
  toast("One step back.");
}
function redo() {
  if (!redoStack.length) return;
  undoStack.push(sceneState());
  restore(redoStack.pop());
  updateHistory();
  toast("One step forward.");
}
function toast(message) {
  $("#toast").textContent = message;
  $("#toast").classList.add("visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => $("#toast").classList.remove("visible"), 2300);
}
function control(key, title, min, max, step, value, suffix = "") {
  return `<div class="control"><label class="control-label" for="control-${key}"><span>${title}</span><output id="value-${key}">${formatValue(key, value)}${suffix}</output></label><input id="control-${key}" data-property="${key}" data-suffix="${suffix}" type="range" min="${min}" max="${max}" step="${step}" value="${value}" style="--fill:${(100 * (value - min)) / (max - min)}%"></div>`;
}
function formatValue(key, value) {
  return key === "dispersion" ? Number(value).toFixed(3) : Math.round(value);
}
function renderInspector() {
  const focused = inspector.contains(document.activeElement)
    ? document.activeElement
    : null;
  const focusSelector = focused?.id
    ? `#${focused.id}`
    : focused?.dataset.color
      ? `[data-color="${focused.dataset.color}"]`
      : null;
  const object = objects.find((o) => o.id === selected);
  $("#selected-number").textContent = object
    ? String(objects.indexOf(object) + 1).padStart(2, "0")
    : "—";
  if (!object) {
    inspector.innerHTML =
      '<p class="empty-selection">Select something.<br>See what it can do.</p>';
    if (focused) $("#object-picker").focus({ preventScroll: true });
    return;
  }
  let html = `<div class="selection-title"><svg><use href="#i-${object.type}"/></svg><span>${names[object.type]}</span><div class="selection-actions"><button class="small-icon" id="duplicate-button" aria-label="Duplicate selected object" title="Duplicate (D)"><svg><use href="#i-copy"/></svg></button><button class="small-icon" id="delete-button" aria-label="Delete selected object" title="Delete (Backspace)"><svg><use href="#i-trash"/></svg></button></div></div>`;
  html += control("angle", "Rotation", -180, 180, 1, object.angle, "°");
  if (object.type === "source") {
    html += control("width", "Beam width", 1, 220, 1, object.width);
    html +=
      '<div class="control-label"><span>Light color</span></div><div class="color-options" role="group" aria-label="Light color">';
    for (const [value, title, color] of [
      ["white", "White light", "white"],
      ["650", "Red", "#df6758"],
      ["590", "Amber", "#eac276"],
      ["530", "Green", "#83b49c"],
      ["490", "Cyan", "#83b6d0"],
      ["420", "Violet", "#ad98c4"],
    ]) {
      html += `<button class="color-chip ${value === "white" ? "white" : ""}" style="--chip:${color}" data-color="${value}" aria-label="${title}" aria-pressed="${object.color === value}"></button>`;
    }
    html += `</div><label class="source-toggle"><input id="source-enabled" type="checkbox" ${object.enabled ? "checked" : ""}> Light on</label>`;
  } else {
    html += control(
      "size",
      object.type === "prism" ? "Prism size" : "Aperture",
      object.type === "prism" ? 55 : 50,
      object.type === "prism" ? 200 : 350,
      1,
      object.size,
    );
    if (object.type === "prism")
      html += control(
        "dispersion",
        "Color spread",
        0.004,
        0.04,
        0.001,
        object.dispersion,
      );
    if (object.type === "lens")
      html += control("focal", "Focal distance", 80, 500, 1, object.focal);
  }
  html += `<p class="inspector-caption">${hints[object.type]}</p>`;
  inspector.innerHTML = html;
  $("#duplicate-button").addEventListener("click", duplicate);
  $("#delete-button").addEventListener("click", removeSelected);
  inspector.querySelectorAll("input[type=range]").forEach((input) => {
    input.addEventListener("pointerdown", () => {
      sliderStart = sceneState();
    });
    input.addEventListener("input", () => {
      if (!sliderStart) sliderStart = sceneState();
      object[input.dataset.property] = Number(input.value);
      input.style.setProperty(
        "--fill",
        `${(100 * (input.value - input.min)) / (input.max - input.min)}%`,
      );
      $(`#value-${input.dataset.property}`).textContent =
        `${formatValue(input.dataset.property, input.value)}${input.dataset.suffix}`;
      dirty = true;
      save();
    });
    input.addEventListener("change", () => {
      if (sliderStart) commit(sliderStart);
      sliderStart = null;
    });
    input.addEventListener("blur", () => {
      if (sliderStart) commit(sliderStart);
      sliderStart = null;
    });
  });
  inspector.querySelectorAll("[data-color]").forEach((button) =>
    button.addEventListener("click", () =>
      change(() => {
        object.color = button.dataset.color;
      }),
    ),
  );
  $("#source-enabled")?.addEventListener("change", (event) =>
    change(() => {
      object.enabled = event.target.checked;
    }),
  );
  if (focusSelector)
    inspector.querySelector(focusSelector)?.focus({ preventScroll: true });
}
function updateUI() {
  renderInspector();
  updateHistory();
  const index = Object.keys(PRESETS).indexOf(preset) + 1;
  $("#scene-name").innerHTML =
    `${PRESETS[preset].name}<span> / ${String(index).padStart(2, "0")}</span>`;
  document.querySelectorAll("[data-preset]").forEach((button) => {
    button.classList.toggle("active", button.dataset.preset === preset);
    button.setAttribute(
      "aria-pressed",
      String(button.dataset.preset === preset),
    );
  });
  $("#grid-button").setAttribute("aria-pressed", String(showGrid));
  $("#object-picker").innerHTML =
    '<option value="">Select an object</option>' +
    objects
      .map(
        (o, index) =>
          `<option value="${o.id}" ${o.id === selected ? "selected" : ""}>${String(index + 1).padStart(2, "0")} / ${names[o.type]}</option>`,
      )
      .join("");
}
function select(id) {
  finishKeyEdit();
  selected = id;
  updateUI();
  save();
}
function addObject(type) {
  if (objects.length >= MAX_OBJECTS) {
    toast("The workbench holds 24 objects.");
    return;
  }
  const defaults = {
    source: { width: 10, color: "white", enabled: true },
    prism: { size: 115, dispersion: 0.025 },
    mirror: { size: 180 },
    lens: { size: 230, focal: 240 },
  };
  change(() => {
    const id = Math.max(0, ...objects.map((o) => o.id)) + 1;
    let x = type === "source" ? 180 : 640,
      y = type === "source" ? 470 : 380;
    while (objects.some((o) => Math.hypot(o.x - x, o.y - y) < 65) && y < 640) {
      x += 42;
      y += 47;
    }
    objects.push({
      id,
      type,
      x,
      y,
      angle: type === "mirror" ? -30 : 0,
      ...defaults[type],
    });
    selected = id;
  });
  toast(`${names[type]} added. Drag it into place.`);
}
function duplicate() {
  const object = objects.find((o) => o.id === selected);
  if (!object) return;
  if (objects.length >= MAX_OBJECTS) {
    toast("The workbench holds 24 objects.");
    return;
  }
  change(() => {
    const copy = clone(object);
    copy.id = Math.max(...objects.map((o) => o.id)) + 1;
    copy.x = clamp(copy.x + 45, 40, 1160);
    copy.y = clamp(copy.y + 45, 55, 705);
    objects.push(copy);
    selected = copy.id;
  });
  toast("A little more possibility.");
}
function removeSelected() {
  if (!objects.some((o) => o.id === selected)) return;
  change(() => {
    objects = objects.filter((o) => o.id !== selected);
    selected = objects.at(-1)?.id ?? null;
  });
}
function loadPreset(name) {
  change(() => {
    preset = name;
    objects = clone(PRESETS[name].objects);
    selected = objects.find((o) => o.type !== "source")?.id ?? null;
  });
}

// The workbench is a fixed world fitted into any canvas; coordinates never
// change when a window is resized or a phone rotates.
function resize() {
  const { width, height } = stage.getBoundingClientRect();
  const dpr = Math.min(devicePixelRatio || 1, 2);
  const scale = Math.min(width / WORLD.width, (height - 50) / WORLD.height);
  viewport = {
    width,
    height,
    scale,
    x: (width - WORLD.width * scale) / 2,
    y: (height - WORLD.height * scale) / 2 + 10,
    dpr,
  };
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  beamCanvas.width = canvas.width;
  beamCanvas.height = canvas.height;
  dirty = true;
}
function worldTransform(context) {
  const v = viewport;
  context.setTransform(
    v.dpr * v.scale,
    0,
    0,
    v.dpr * v.scale,
    v.x * v.dpr,
    v.y * v.dpr,
  );
}
function worldPointer(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left - viewport.x) / viewport.scale,
    y: (event.clientY - rect.top - viewport.y) / viewport.scale,
  };
}
function distanceSegment(point, a, b) {
  const edge = sub(b, a),
    offset = sub(point, a);
  const fraction = clamp(dot(offset, edge) / dot(edge, edge), 0, 1);
  return Math.hypot(
    point.x - a.x - edge.x * fraction,
    point.y - a.y - edge.y * fraction,
  );
}
function handlePosition(object) {
  const radius =
    object.type === "source"
      ? 72
      : object.type === "prism"
        ? object.size + 35
        : object.size / 2 + 35;
  return add(object, mul(direction(object.angle - 90), radius));
}
function hitTest(point) {
  const hitRadius = 13 / viewport.scale;
  for (const object of [...objects].reverse()) {
    if (object.type === "source") {
      const local = rotate(sub(point, object), -object.angle);
      if (Math.abs(local.x) < 40 && Math.abs(local.y) < Math.max(25, hitRadius))
        return object;
    } else if (object.type === "prism") {
      const polygon = vertices(object);
      if (
        contains(point, polygon) ||
        polygon.some(
          (p, i) => distanceSegment(point, p, polygon[(i + 1) % 3]) < hitRadius,
        )
      )
        return object;
    } else {
      const [a, b] = endpoints(object);
      if (
        distanceSegment(point, a, b) <
        Math.max(object.type === "lens" ? 20 : 12, hitRadius)
      )
        return object;
    }
  }
  return null;
}
canvas.addEventListener("pointerdown", (event) => {
  if (event.button !== 0 || drag) return;
  canvas.focus({ preventScroll: true });
  const point = worldPointer(event),
    object = objects.find((o) => o.id === selected);
  if (
    object &&
    Math.hypot(
      point.x - handlePosition(object).x,
      point.y - handlePosition(object).y,
    ) <
      16 / viewport.scale
  ) {
    drag = {
      id: object.id,
      mode: "rotate",
      before: sceneState(),
      pointerId: event.pointerId,
    };
  } else {
    const hit = hitTest(point);
    select(hit?.id ?? null);
    if (hit)
      drag = {
        id: hit.id,
        mode: "move",
        before: sceneState(),
        offset: sub(hit, point),
        pointerId: event.pointerId,
      };
  }
  if (drag) {
    canvas.setPointerCapture(event.pointerId);
    canvas.style.cursor = "grabbing";
  }
});
canvas.addEventListener("pointermove", (event) => {
  const point = worldPointer(event);
  if (drag && drag.pointerId === event.pointerId) {
    const object = objects.find((o) => o.id === drag.id);
    if (drag.mode === "rotate") {
      object.angle = normalizeAngle(
        Math.round(
          (Math.atan2(point.y - object.y, point.x - object.x) * 180) / Math.PI +
            90,
        ),
      );
      if (event.shiftKey) object.angle = Math.round(object.angle / 15) * 15;
    } else {
      object.x = clamp(point.x + drag.offset.x, 40, 1160);
      object.y = clamp(point.y + drag.offset.y, 55, 705);
    }
    dirty = true;
    renderInspector();
  } else {
    hovered = hitTest(point)?.id ?? null;
    const object = objects.find((o) => o.id === selected);
    const onHandle =
      object &&
      Math.hypot(
        point.x - handlePosition(object).x,
        point.y - handlePosition(object).y,
      ) <
        16 / viewport.scale;
    canvas.style.cursor = hovered || onHandle ? "grab" : "default";
  }
});
function endDrag(event) {
  if (!drag || (event && event.pointerId !== drag.pointerId)) return;
  commit(drag.before);
  drag = null;
  canvas.style.cursor = "default";
}
canvas.addEventListener("pointerup", endDrag);
canvas.addEventListener("pointercancel", endDrag);
canvas.addEventListener("lostpointercapture", endDrag);
canvas.addEventListener("pointerleave", () => {
  hovered = null;
});

function finishKeyEdit() {
  if (keyStart) {
    commit(keyStart);
    keyStart = null;
  }
}
window.addEventListener("keydown", (event) => {
  if (
    $("#notes-dialog").open ||
    /INPUT|SELECT|TEXTAREA/.test(event.target.tagName) ||
    event.target.isContentEditable
  )
    return;
  if (event.metaKey || event.ctrlKey) {
    if (event.key.toLowerCase() === "z") {
      event.preventDefault();
      event.shiftKey ? redo() : undo();
    }
    return;
  }
  if (event.altKey) return;
  const key = event.key.toLowerCase(),
    object = objects.find((o) => o.id === selected);
  if (
    ["arrowup", "arrowdown", "arrowleft", "arrowright", "q", "e"].includes(
      key,
    ) &&
    object
  ) {
    event.preventDefault();
    if (!keyStart) keyStart = sceneState();
    const step = event.shiftKey ? 20 : 5;
    if (key === "arrowleft") object.x = clamp(object.x - step, 40, 1160);
    if (key === "arrowright") object.x = clamp(object.x + step, 40, 1160);
    if (key === "arrowup") object.y = clamp(object.y - step, 55, 705);
    if (key === "arrowdown") object.y = clamp(object.y + step, 55, 705);
    if (key === "q" || key === "e")
      object.angle = normalizeAngle(
        object.angle + (key === "q" ? -1 : 1) * (event.shiftKey ? 15 : 2),
      );
    renderInspector();
    dirty = true;
  } else if (key === "backspace" || key === "delete") {
    event.preventDefault();
    removeSelected();
  } else if (key === "escape") select(null);
  else if (key === "d" && !event.repeat) duplicate();
  else if (key === "g" && !event.repeat) toggleGrid();
  else if (["1", "2", "3", "4"].includes(key) && !event.repeat)
    addObject(["source", "prism", "mirror", "lens"][Number(key) - 1]);
});
window.addEventListener("keyup", (event) => {
  if (
    [
      "ArrowUp",
      "ArrowDown",
      "ArrowLeft",
      "ArrowRight",
      "q",
      "e",
      "Q",
      "E",
    ].includes(event.key)
  )
    finishKeyEdit();
});
window.addEventListener("blur", () => {
  finishKeyEdit();
  endDrag();
});
function toggleGrid() {
  showGrid = !showGrid;
  $("#grid-button").setAttribute("aria-pressed", String(showGrid));
  save();
}

// Static light paths are cached. Only the small traveling glints and selection
// affordances need to be redrawn per frame.
function buildBeams() {
  paths = traceScene(objects);
  beamCtx.setTransform(1, 0, 0, 1, 0, 0);
  beamCtx.clearRect(0, 0, beamCanvas.width, beamCanvas.height);
  worldTransform(beamCtx);
  beamCtx.globalCompositeOperation = "lighter";
  beamCtx.lineCap = "round";
  const passes = [
    { width: 11, alpha: 0.015 },
    { width: 3.2, alpha: 0.055 },
    { width: 0.9, alpha: 0.3 },
  ];
  for (const pass of passes) {
    beamCtx.lineWidth = pass.width;
    for (const path of paths) {
      const [r, g, b] = wavelengthRGB(path.wavelength);
      for (const segment of path.segments) {
        beamCtx.strokeStyle = `rgba(${r},${g},${b},${pass.alpha * segment.energy * (path.white ? 0.26 : 1.9)})`;
        beamCtx.beginPath();
        beamCtx.moveTo(segment.a.x, segment.a.y);
        beamCtx.lineTo(segment.b.x, segment.b.y);
        beamCtx.stroke();
      }
    }
  }
  beamCtx.globalCompositeOperation = "source-over";
  dirty = false;
}
function drawBackground() {
  const v = viewport;
  ctx.setTransform(v.dpr, 0, 0, v.dpr, 0, 0);
  ctx.fillStyle = "#111918";
  ctx.fillRect(0, 0, v.width, v.height);
  const gradient = ctx.createRadialGradient(
    v.width * 0.54,
    v.height * 0.47,
    20,
    v.width * 0.5,
    v.height * 0.5,
    v.width * 0.7,
  );
  gradient.addColorStop(0, "#202a2588");
  gradient.addColorStop(0.6, "#17201e44");
  gradient.addColorStop(1, "#0d151a88");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, v.width, v.height);
  if (showGrid) {
    ctx.fillStyle = "#73857429";
    const spacing = Math.max(22, 32 * v.scale);
    for (let x = (v.width / 2) % spacing; x < v.width; x += spacing)
      for (let y = (v.height / 2) % spacing; y < v.height; y += spacing)
        ctx.fillRect(x, y, 1, 1);
    ctx.strokeStyle = "#74836c21";
    ctx.lineWidth = 1;
    for (const [x, y] of [
      [v.width * 0.25, v.height * 0.25],
      [v.width * 0.75, v.height * 0.25],
      [v.width * 0.25, v.height * 0.75],
      [v.width * 0.75, v.height * 0.75],
    ]) {
      ctx.beginPath();
      ctx.moveTo(x - 4, y);
      ctx.lineTo(x + 4, y);
      ctx.moveTo(x, y - 4);
      ctx.lineTo(x, y + 4);
      ctx.stroke();
    }
  }
}
function line(a, b) {
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.stroke();
}
function drawObject(object, exporting) {
  const isSelected = object.id === selected && !exporting;
  ctx.save();
  ctx.translate(object.x, object.y);
  ctx.rotate((object.angle * Math.PI) / 180);
  const accent = isSelected
    ? "#d9dac2"
    : hovered === object.id && !exporting
      ? "#aebda0"
      : "#718873";
  if (object.type === "source") {
    if (object.width > 20) {
      ctx.strokeStyle = "#a7b89c66";
      ctx.lineWidth = 2;
      line({ x: 28, y: -object.width / 2 }, { x: 28, y: object.width / 2 });
    }
    ctx.shadowBlur = 15;
    ctx.shadowColor = "#0008";
    const body = ctx.createLinearGradient(0, -17, 0, 17);
    body.addColorStop(0, "#626653");
    body.addColorStop(0.4, "#343e33");
    body.addColorStop(1, "#252e25");
    ctx.fillStyle = body;
    ctx.strokeStyle = accent;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(-34, -17, 58, 34, 5);
    ctx.fill();
    ctx.stroke();
    ctx.shadowBlur = 0;
    ctx.fillStyle = "#222b22";
    ctx.fillRect(14, -18, 9, 36);
    ctx.strokeStyle = "#96a38288";
    for (let x = -24; x <= -4; x += 5) line({ x, y: -10 }, { x, y: 10 });
    const color =
      object.color === "white"
        ? "#f1edca"
        : `rgb(${wavelengthRGB(Number(object.color))})`;
    ctx.fillStyle = object.enabled ? color : "#55614d";
    ctx.shadowColor = color;
    ctx.shadowBlur = object.enabled ? 15 : 0;
    ctx.fillRect(24, -9, 4, 18);
    ctx.shadowBlur = 0;
    ctx.fillStyle = object.enabled ? "#becb8d" : "#58614e";
    ctx.beginPath();
    ctx.arc(-24, -12, 1.5, 0, Math.PI * 2);
    ctx.fill();
  } else if (object.type === "prism") {
    const r = object.size,
      x = (Math.sqrt(3) * r) / 2;
    ctx.beginPath();
    ctx.moveTo(0, -r);
    ctx.lineTo(x, r / 2);
    ctx.lineTo(-x, r / 2);
    ctx.closePath();
    const fill = ctx.createLinearGradient(-x, -r, x, r / 2);
    fill.addColorStop(0, "#b6ccb42a");
    fill.addColorStop(0.45, "#6b8d7712");
    fill.addColorStop(0.8, "#8cafa427");
    fill.addColorStop(1, "#ccdabe40");
    ctx.fillStyle = fill;
    ctx.fill();
    ctx.strokeStyle = accent;
    ctx.lineWidth = 1.2;
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, -r + 8);
    ctx.lineTo(x - 8, r / 2 - 5);
    ctx.lineTo(-x + 8, r / 2 - 5);
    ctx.closePath();
    ctx.strokeStyle = "#d4ebc118";
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.fillStyle = "#bacba655";
    ctx.beginPath();
    ctx.arc(0, 0, 2, 0, Math.PI * 2);
    ctx.fill();
  } else if (object.type === "mirror") {
    const half = object.size / 2;
    ctx.fillStyle = "#3e504344";
    ctx.fillRect(-7, -half, 7, object.size);
    ctx.strokeStyle = "#6b7e6488";
    ctx.lineWidth = 1;
    for (let y = -half + 6; y < half; y += 12)
      line({ x: -9, y: y + 4 }, { x: -2, y: y - 3 });
    ctx.strokeStyle = accent;
    ctx.lineWidth = 3;
    line({ x: 0, y: -half }, { x: 0, y: half });
    ctx.strokeStyle = "#d0e4cc66";
    ctx.lineWidth = 1;
    line({ x: 2, y: -half }, { x: 2, y: half });
  } else if (object.type === "lens") {
    const half = object.size / 2,
      bulge = Math.min(29, half * 0.2);
    ctx.beginPath();
    ctx.moveTo(0, -half);
    ctx.bezierCurveTo(bulge, -half * 0.45, bulge, half * 0.45, 0, half);
    ctx.bezierCurveTo(-bulge, half * 0.45, -bulge, -half * 0.45, 0, -half);
    ctx.fillStyle = "#8cbab727";
    ctx.fill();
    ctx.strokeStyle = accent;
    ctx.lineWidth = 1.2;
    ctx.stroke();
    if (isSelected) {
      ctx.strokeStyle = "#839b7944";
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 6]);
      line({ x: -object.focal - 30, y: 0 }, { x: object.focal + 30, y: 0 });
      ctx.setLineDash([]);
      ctx.strokeStyle = "#bec5a780";
      for (const x of [-object.focal, object.focal]) {
        line({ x: x - 5, y: 0 }, { x: x + 5, y: 0 });
        line({ x, y: -5 }, { x, y: 5 });
      }
      ctx.fillStyle = "#9da887";
      ctx.font = "10px monospace";
      ctx.fillText("F", object.focal - 3, 20);
    }
  }
  ctx.restore();
  if (isSelected) {
    const handle = handlePosition(object),
      d = direction(object.angle - 90);
    const baseDistance =
      object.type === "source"
        ? 27
        : object.type === "prism"
          ? object.size + 7
          : object.size / 2 + 7;
    ctx.strokeStyle = "#bdc3a76b";
    ctx.lineWidth = 1 / viewport.scale;
    ctx.setLineDash([2, 4]);
    line(add(object, mul(d, baseDistance)), handle);
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.arc(handle.x, handle.y, 5 / viewport.scale, 0, Math.PI * 2);
    ctx.fillStyle = "#19221c";
    ctx.fill();
    ctx.strokeStyle = "#d1d5b8";
    ctx.stroke();
  }
  if (!exporting) {
    const position =
      object.type === "source"
        ? add(object, { x: -31, y: 47 })
        : add(object, {
            x: -28,
            y:
              (object.type === "prism" ? object.size * 0.65 : object.size / 2) +
              35,
          });
    const index = objects.indexOf(object) + 1;
    ctx.fillStyle = isSelected ? "#d3d7bd" : "#7d8b74";
    ctx.font = `${Math.max(10, 8 / viewport.scale)}px monospace`;
    ctx.fillText(
      `${String(index).padStart(2, "0")} / ${object.type.toUpperCase()}`,
      position.x,
      position.y,
    );
  }
}
function drawGlints(time) {
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  for (let i = 0; i < paths.length; i += paths.length > 100 ? 31 : 2) {
    const path = paths[i];
    const total = path.segments.reduce(
      (sum, s) => sum + Math.hypot(s.b.x - s.a.x, s.b.y - s.a.y),
      0,
    );
    let travel = (time * 0.12 + i * 23) % (total + 500);
    for (const segment of path.segments) {
      const length = Math.hypot(
        segment.b.x - segment.a.x,
        segment.b.y - segment.a.y,
      );
      if (travel < length) {
        const t = travel / length;
        ctx.fillStyle = `rgba(${wavelengthRGB(path.wavelength)},.5)`;
        ctx.beginPath();
        ctx.arc(
          segment.a.x + (segment.b.x - segment.a.x) * t,
          segment.a.y + (segment.b.y - segment.a.y) * t,
          1,
          0,
          Math.PI * 2,
        );
        ctx.fill();
        break;
      }
      travel -= length;
    }
  }
  ctx.restore();
}
function render(time = 0, exporting = false) {
  if (dirty) buildBeams();
  drawBackground();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.drawImage(beamCanvas, 0, 0);
  worldTransform(ctx);
  if (!exporting && !reducedMotion.matches) drawGlints(time);
  for (const object of objects) drawObject(object, exporting);
}
let lastFrame = 0;
function frame(time) {
  if (!document.hidden && (time - lastFrame > 30 || dirty)) {
    render(time);
    lastFrame = time;
  }
  requestAnimationFrame(frame);
}
async function exportImage() {
  $("#export-button").disabled = true;
  try {
    render(0, true);
    const output = document.createElement("canvas");
    output.width = 2400;
    output.height = Math.round((2400 * viewport.height) / viewport.width);
    const context = output.getContext("2d");
    context.drawImage(canvas, 0, 0, output.width, output.height);
    context.fillStyle = "#c8cfb5";
    context.font = "22px Georgia";
    context.fillText("luma. / " + PRESETS[preset].name, 50, 55);
    context.fillStyle = "#92a187";
    context.font = "12px monospace";
    context.fillText("A PLAYGROUND FOR LIGHT", 50, output.height - 36);
    const blob = await new Promise((resolve) =>
      output.toBlob(resolve, "image/png"),
    );
    if (!blob) throw new Error("Image export unavailable");
    const url = URL.createObjectURL(blob),
      link = document.createElement("a");
    link.href = url;
    link.download = `luma-${preset}.png`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
    toast("A little light, saved.");
  } catch {
    toast("Could not save the image. Please try again.");
  } finally {
    $("#export-button").disabled = false;
    render();
  }
}

document
  .querySelectorAll("[data-add]")
  .forEach((button) =>
    button.addEventListener("click", () => addObject(button.dataset.add)),
  );
document
  .querySelectorAll("[data-preset]")
  .forEach((button) =>
    button.addEventListener("click", () => loadPreset(button.dataset.preset)),
  );
$("#object-picker").addEventListener("change", (event) =>
  select(event.target.value ? Number(event.target.value) : null),
);
$("#undo-button").addEventListener("click", undo);
$("#redo-button").addEventListener("click", redo);
$("#grid-button").addEventListener("click", toggleGrid);
$("#reset-button").addEventListener("click", () => {
  loadPreset(preset);
  toast("Back to the beginning. Undo to restore.");
});
$("#export-button").addEventListener("click", exportImage);
$("#about-button").addEventListener("click", () =>
  $("#notes-dialog").showModal(),
);
$("#notes-dialog").addEventListener("click", (event) => {
  if (event.target === $("#notes-dialog")) {
    const bounds = event.target.getBoundingClientRect();
    if (
      event.clientX < bounds.left ||
      event.clientX > bounds.right ||
      event.clientY < bounds.top ||
      event.clientY > bounds.bottom
    )
      event.target.close();
  }
});
window.addEventListener("pagehide", () => {
  try {
    localStorage.setItem(STORAGE, sceneState());
  } catch {
    /* Optional storage. */
  }
});

objects = clone(PRESETS.prism.objects);
try {
  const saved = JSON.parse(localStorage.getItem(STORAGE));
  if (validScene(saved)) {
    objects = saved.objects;
    preset = saved.preset;
    selected = objects.some((o) => o.id === saved.selected)
      ? saved.selected
      : null;
    showGrid = saved.showGrid !== false;
  }
} catch {
  /* A fresh workbench is also a good place to start. */
}
updateUI();
save();
new ResizeObserver(resize).observe(stage);
resize();
requestAnimationFrame(frame);
