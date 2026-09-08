import { poseAt } from "./kinematics.mjs";

const TAU = Math.PI * 2;
const PAPER = "#f3f1e9";
const INK = "#243d5b";
const BLUE = "#466eac";
const ORANGE = "#d5633d";
const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
const finitePoint = (point) =>
  point && Number.isFinite(point.x) && Number.isFinite(point.y);
const pointsOf = (value) =>
  Array.isArray(value) ? value : value?.points || [];
const distance = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const xml = (value) =>
  String(value).replace(
    /[&<>"']/g,
    (character) =>
      ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&apos;",
      })[character],
  );

function usablePose(design, theta) {
  if (!design) return null;
  try {
    const pose = poseAt(design, theta);
    if (!pose || pose.valid === false) return null;
    const result = {
      ...pose,
      O: pose.O || pose.joints?.O,
      G: pose.G || pose.joints?.G,
      A: pose.A || pose.joints?.A,
      B: pose.B || pose.joints?.B,
      P: pose.P || pose.tracer,
    };
    return ["O", "G", "A", "B", "P"].every((key) => finitePoint(result[key]))
      ? result
      : null;
  } catch {
    return null;
  }
}

function boundsOf(points) {
  let xmin = Infinity,
    ymin = Infinity,
    xmax = -Infinity,
    ymax = -Infinity;
  for (const point of points) {
    if (!finitePoint(point)) continue;
    xmin = Math.min(xmin, point.x);
    xmax = Math.max(xmax, point.x);
    ymin = Math.min(ymin, point.y);
    ymax = Math.max(ymax, point.y);
  }
  return Number.isFinite(xmin)
    ? { xmin, ymin, xmax, ymax }
    : { xmin: -1, ymin: -1, xmax: 1, ymax: 1 };
}

function cameraFor(bounds, width, height, padding) {
  const spanX = Math.max(0.2, bounds.xmax - bounds.xmin);
  const spanY = Math.max(0.2, bounds.ymax - bounds.ymin);
  const scale = Math.min(
    Math.max(1, width - padding * 2) / spanX,
    Math.max(1, height - padding * 2) / spanY,
  );
  return {
    x: (bounds.xmin + bounds.xmax) / 2,
    y: (bounds.ymin + bounds.ymax) / 2,
    scale,
  };
}

function screenPoint(point, camera, width, height) {
  return {
    x: width / 2 + (point.x - camera.x) * camera.scale,
    y: height / 2 - (point.y - camera.y) * camera.scale,
  };
}

function niceStep(value) {
  const exponent = 10 ** Math.floor(Math.log10(Math.max(0.000001, value)));
  const fraction = value / exponent;
  return (fraction < 2 ? 1 : fraction < 5 ? 2 : 5) * exponent;
}

/** Top-view drafting renderer. All callback positions and solver poses use world coordinates. */
export class MechanismView {
  constructor(canvas, callbacks = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d", { alpha: true });
    this.callbacks = callbacks;
    this.state = {
      design: null,
      target: [],
      curve: [],
      playing: true,
      speed: 0.45,
      showTarget: true,
      showTrace: true,
      showDimensions: false,
    };
    this._theta = 0;
    this.width = 900;
    this.height = 620;
    this.dpr = 1;
    this.camera = { x: 0, y: 0, scale: 200 };
    this.bounds = { xmin: -1, ymin: -1, xmax: 1, ymax: 1 };
    this.currentPose = null;
    this.fitPoints = [];
    this.guide = null;
    this.raf = 0;
    this.lastTime = null;
    this.destroyed = false;
    this.visible = true;
    this.drag = null;
    this.hover = null;
    this.needsFit = false;
    this.frames = 0;
    this.lastRenderMs = 0;
    this.motionQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
    this.reducedMotion = this.motionQuery.matches;
    this.canvas.style.touchAction = "none";
    this.canvas.style.cursor = "default";
    this.pointerListeners = {
      pointerdown: (event) => this.pointerDown(event),
      pointermove: (event) => this.pointerMove(event),
      pointerup: (event) => this.pointerUp(event),
      pointercancel: (event) => this.pointerUp(event),
      pointerleave: () => {
        if (!this.drag) {
          this.hover = null;
          this.canvas.style.cursor = "default";
          this.schedule();
        }
      },
    };
    for (const [type, listener] of Object.entries(this.pointerListeners))
      canvas.addEventListener(type, listener);
    this.onVisibility = () => {
      this.lastTime = null;
      if (document.hidden) this.cancelFrame();
      else this.schedule();
    };
    this.onMotion = (event) => {
      this.reducedMotion = event.matches;
      this.lastTime = null;
      this.schedule();
    };
    document.addEventListener("visibilitychange", this.onVisibility);
    this.motionQuery.addEventListener("change", this.onMotion);
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas);
    if (typeof IntersectionObserver !== "undefined") {
      this.intersectionObserver = new IntersectionObserver(
        (entries) => {
          this.visible = entries[0]?.isIntersecting ?? true;
          this.lastTime = null;
          if (this.visible) this.schedule();
          else this.cancelFrame();
        },
        { rootMargin: "80px" },
      );
      this.intersectionObserver.observe(canvas);
    }
    this.resize();
  }

  get theta() {
    return this._theta;
  }
  set theta(value) {
    if (Number.isFinite(value)) {
      this._theta = ((value % TAU) + TAU) % TAU;
      this.lastTime = null;
      this.schedule();
    }
  }
  get angle() {
    return this._theta;
  }

  setState(next) {
    const geometryChanged =
      "design" in next || "target" in next || "curve" in next;
    if ("theta" in next) this.theta = next.theta;
    if ("reducedMotion" in next) this.reducedMotion = !!next.reducedMotion;
    if ("playing" in next && next.playing !== this.state.playing)
      this.lastTime = null;
    Object.assign(this.state, next);
    if (geometryChanged) {
      this.prepareGeometry();
      if (this.drag) this.needsFit = true;
      else this.fitCamera();
    }
    this.schedule();
  }

  prepareGeometry() {
    const poses = [];
    this.fitPoints = [
      ...pointsOf(this.state.target),
      ...pointsOf(this.state.curve),
    ];
    for (let i = 0; i < 96; i++) {
      const pose = usablePose(this.state.design, (i / 96) * TAU);
      if (!pose) continue;
      poses.push(pose);
      this.fitPoints.push(pose.O, pose.G, pose.A, pose.B, pose.P);
    }
    this.guide = null;
    if (this.state.design?.family === "slider" && poses.length) {
      const reference = poses[0];
      const guide = reference.guide;
      let vector =
        guide?.length === 2
          ? { x: guide[1].x - guide[0].x, y: guide[1].y - guide[0].y }
          : null;
      if (!vector || Math.hypot(vector.x, vector.y) < 1e-9) {
        const farthest = poses.reduce(
          (best, pose) =>
            distance(pose.B, reference.B) > distance(best.B, reference.B)
              ? pose
              : best,
          reference,
        );
        vector = {
          x: farthest.B.x - reference.B.x,
          y: farthest.B.y - reference.B.y,
        };
      }
      const norm = Math.hypot(vector.x, vector.y) || 1;
      const direction = { x: vector.x / norm, y: vector.y / norm };
      let min = Infinity,
        max = -Infinity;
      for (const pose of poses) {
        const position =
          (pose.B.x - reference.G.x) * direction.x +
          (pose.B.y - reference.G.y) * direction.y;
        min = Math.min(min, position);
        max = Math.max(max, position);
      }
      min = Math.min(0, min);
      max = Math.max(0, max);
      const extension = Math.max((max - min) * 0.15, norm * 0.035);
      this.guide = [min - extension, max + extension].map((position) => ({
        x: reference.G.x + direction.x * position,
        y: reference.G.y + direction.y * position,
      }));
      this.fitPoints.push(...this.guide);
    }
    this.bounds = boundsOf(this.fitPoints);
  }

  fitCamera() {
    this.camera = cameraFor(
      this.bounds,
      this.width,
      this.height,
      Math.min(76, Math.max(46, this.width * 0.07)),
    );
    this.needsFit = false;
  }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    this.width = Math.max(1, rect.width || 900);
    this.height = Math.max(1, rect.height || 620);
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.canvas.width = Math.round(this.width * this.dpr);
    this.canvas.height = Math.round(this.height * this.dpr);
    this.fitCamera();
    this.schedule();
  }

  project(pointOrX, y) {
    return screenPoint(
      typeof pointOrX === "number" ? { x: pointOrX, y } : pointOrX,
      this.camera,
      this.width,
      this.height,
    );
  }

  unproject(x, y) {
    return {
      x: this.camera.x + (x - this.width / 2) / this.camera.scale,
      y: this.camera.y - (y - this.height / 2) / this.camera.scale,
    };
  }

  eventPoint(event) {
    const rect = this.canvas.getBoundingClientRect();
    return { x: event.clientX - rect.left, y: event.clientY - rect.top };
  }

  nearestHandle(point) {
    if (!this.currentPose) return null;
    let name = null,
      minimum = 22;
    for (const key of this.state.design?.family === "slider"
      ? ["O", "P"]
      : ["O", "G", "P"]) {
      const projected = this.project(this.currentPose[key]);
      const d = distance(point, projected);
      if (d < minimum) {
        minimum = d;
        name = key;
      }
    }
    return name;
  }

  pointerDown(event) {
    if (event.button !== 0 && event.pointerType !== "touch") return;
    this.canvas.focus({ preventScroll: true });
    const point = this.eventPoint(event),
      name = this.nearestHandle(point);
    if (!name) return;
    event.preventDefault();
    this.drag = { name, pointerId: event.pointerId };
    this.hover = name;
    this.canvas.style.cursor = "grabbing";
    this.canvas.setPointerCapture(event.pointerId);
    this.lastTime = null;
    this.schedule();
  }

  pointerMove(event) {
    const screen = this.eventPoint(event);
    if (this.drag) {
      if (this.drag.pointerId !== event.pointerId) return;
      event.preventDefault();
      const world = this.unproject(screen.x, screen.y);
      this.callbacks.onJointDrag?.(this.drag.name, world.x, world.y);
    } else {
      this.hover = this.nearestHandle(screen);
      this.canvas.style.cursor = this.hover ? "grab" : "default";
    }
    this.schedule();
  }

  pointerUp(event) {
    if (!this.drag || event.pointerId !== this.drag.pointerId) return;
    this.drag = null;
    if (this.canvas.hasPointerCapture(event.pointerId))
      this.canvas.releasePointerCapture(event.pointerId);
    this.canvas.style.cursor = this.hover ? "grab" : "default";
    this.lastTime = null;
    this.callbacks.onJointEnd?.();
    if (this.needsFit) this.fitCamera();
    this.schedule();
  }

  path(points, close = false) {
    const ctx = this.ctx;
    ctx.beginPath();
    points.forEach((point, index) => {
      const screen = this.project(point);
      if (index === 0) ctx.moveTo(screen.x, screen.y);
      else ctx.lineTo(screen.x, screen.y);
    });
    if (close) ctx.closePath();
  }

  drawGrid() {
    const ctx = this.ctx,
      scale = this.camera.scale;
    const step = niceStep(28 / scale),
      major = step * 5;
    const left = this.unproject(0, 0).x,
      right = this.unproject(this.width, 0).x;
    const bottom = this.unproject(0, this.height).y,
      top = this.unproject(0, 0).y;
    ctx.save();
    ctx.lineWidth = 0.6;
    for (const [spacing, color] of [
      [step, "rgba(84,112,124,0.09)"],
      [major, "rgba(84,112,124,0.14)"],
    ]) {
      ctx.strokeStyle = color;
      ctx.beginPath();
      for (
        let x = Math.ceil(left / spacing) * spacing;
        x <= right;
        x += spacing
      ) {
        const screen = this.project(x, 0);
        ctx.moveTo(screen.x, 0);
        ctx.lineTo(screen.x, this.height);
      }
      for (
        let y = Math.ceil(bottom / spacing) * spacing;
        y <= top;
        y += spacing
      ) {
        const screen = this.project(0, y);
        ctx.moveTo(0, screen.y);
        ctx.lineTo(this.width, screen.y);
      }
      ctx.stroke();
    }
    ctx.restore();
  }

  drawCurves() {
    const ctx = this.ctx,
      target = pointsOf(this.state.target),
      curve = pointsOf(this.state.curve);
    ctx.save();
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    if (this.state.showTarget && target.length > 1) {
      this.path(target, true);
      ctx.strokeStyle = "rgba(70,80,77,0.58)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([5, 5]);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    if (this.state.showTrace && curve.length > 1) {
      this.path(curve, true);
      ctx.strokeStyle = "rgba(205,91,50,0.38)";
      ctx.lineWidth = 2;
      ctx.stroke();
    }
    ctx.restore();
  }

  drawGuide() {
    if (!this.guide) return;
    const ctx = this.ctx,
      a = this.project(this.guide[0]),
      b = this.project(this.guide[1]);
    const length = distance(a, b) || 1,
      n = { x: -(b.y - a.y) / length, y: (b.x - a.x) / length };
    ctx.save();
    ctx.strokeStyle = "rgba(80,101,117,0.24)";
    ctx.lineWidth = 21;
    ctx.lineCap = "butt";
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
    ctx.strokeStyle = "#778996";
    ctx.lineWidth = 1.5;
    for (const side of [-1, 1]) {
      ctx.beginPath();
      ctx.moveTo(a.x + n.x * side * 12, a.y + n.y * side * 12);
      ctx.lineTo(b.x + n.x * side * 12, b.y + n.y * side * 12);
      ctx.stroke();
    }
    ctx.strokeStyle = "rgba(82,103,120,0.3)";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 5]);
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();
  }

  drawBar(a, b, width = 17, pale = false) {
    const ctx = this.ctx,
      p = this.project(a),
      q = this.project(b);
    ctx.save();
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(p.x, p.y + 2);
    ctx.lineTo(q.x, q.y + 2);
    ctx.strokeStyle = "rgba(29,47,68,0.12)";
    ctx.lineWidth = width + 3;
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(p.x, p.y);
    ctx.lineTo(q.x, q.y);
    ctx.strokeStyle = pale ? "#4c647c" : "#253d61";
    ctx.lineWidth = width;
    ctx.stroke();
    const face = ctx.createLinearGradient(
      p.x,
      p.y - width * 0.4,
      q.x,
      q.y + width * 0.5,
    );
    face.addColorStop(0, pale ? "#9cabb5" : "#6e90c3");
    face.addColorStop(0.4, pale ? "#8194a3" : "#496fae");
    face.addColorStop(1, pale ? "#607989" : "#365587");
    ctx.strokeStyle = face;
    ctx.lineWidth = Math.max(2, width - 3);
    ctx.stroke();
    const length = distance(p, q);
    if (length > 27) {
      const dx = (q.x - p.x) / length,
        dy = (q.y - p.y) / length;
      ctx.beginPath();
      ctx.moveTo(p.x + dx * 13, p.y + dy * 13);
      ctx.lineTo(q.x - dx * 13, q.y - dy * 13);
      ctx.strokeStyle = pale ? "rgba(36,62,80,0.25)" : "rgba(22,43,81,0.3)";
      ctx.lineWidth = Math.max(1.5, width * 0.2);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(p.x + dx * 13 - dy * 1.5, p.y + dy * 13 + dx * 1.5);
      ctx.lineTo(q.x - dx * 13 - dy * 1.5, q.y - dy * 13 + dx * 1.5);
      ctx.strokeStyle = "rgba(232,241,247,0.18)";
      ctx.lineWidth = 0.7;
      ctx.stroke();
    }
    ctx.restore();
  }

  drawCoupler(pose) {
    const ctx = this.ctx;
    ctx.save();
    this.path([pose.A, pose.B, pose.P], true);
    ctx.fillStyle = "rgba(74,105,159,0.09)";
    ctx.fill();
    ctx.strokeStyle = "rgba(58,86,127,0.43)";
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.restore();
    this.drawBar(pose.A, pose.P, 9, true);
    this.drawBar(pose.B, pose.P, 9, true);
    this.drawBar(pose.A, pose.B, 18);
  }

  drawAnchor(point) {
    const ctx = this.ctx,
      p = this.project(point);
    ctx.save();
    ctx.translate(p.x, p.y);
    ctx.fillStyle = "#d3d9d7";
    ctx.strokeStyle = "#899995";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(-17, 19);
    ctx.lineTo(-7, 3);
    ctx.lineTo(7, 3);
    ctx.lineTo(17, 19);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(-21, 21);
    ctx.lineTo(21, 21);
    ctx.stroke();
    ctx.strokeStyle = "#9aa8a4";
    ctx.lineWidth = 0.8;
    for (let x = -18; x <= 20; x += 6) {
      ctx.beginPath();
      ctx.moveTo(x, 22);
      ctx.lineTo(x - 4, 27);
      ctx.stroke();
    }
    ctx.restore();
  }

  drawHub(point, radius = 10) {
    const ctx = this.ctx,
      p = this.project(point);
    ctx.save();
    ctx.beginPath();
    ctx.arc(p.x, p.y + 1.5, radius + 1, 0, TAU);
    ctx.fillStyle = "rgba(19,40,61,0.19)";
    ctx.fill();
    const metal = ctx.createRadialGradient(
      p.x - radius * 0.3,
      p.y - radius * 0.35,
      0,
      p.x,
      p.y,
      radius,
    );
    metal.addColorStop(0, "#fcfcf6");
    metal.addColorStop(0.48, "#e0e6e5");
    metal.addColorStop(0.74, "#b9c6ce");
    metal.addColorStop(1, "#869cac");
    ctx.beginPath();
    ctx.arc(p.x, p.y, radius, 0, TAU);
    ctx.fillStyle = metal;
    ctx.fill();
    ctx.strokeStyle = "#395268";
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(p.x, p.y, radius * 0.5, 0, TAU);
    ctx.fillStyle = "#e9ede8";
    ctx.fill();
    ctx.strokeStyle = "#637d8c";
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(p.x, p.y, 1.6, 0, TAU);
    ctx.fillStyle = "#4a6578";
    ctx.fill();
    ctx.restore();
  }

  drawSlider(pose) {
    if (!this.guide) return;
    const ctx = this.ctx,
      p = this.project(pose.B),
      a = this.project(this.guide[0]),
      b = this.project(this.guide[1]);
    const angle = Math.atan2(b.y - a.y, b.x - a.x);
    ctx.save();
    ctx.translate(p.x, p.y);
    ctx.rotate(angle);
    ctx.fillStyle = "#aebfca";
    ctx.strokeStyle = "#415f77";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.roundRect(-20, -10, 40, 20, 3);
    ctx.fill();
    ctx.stroke();
    ctx.strokeStyle = "rgba(241,246,244,0.68)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(-16, -6);
    ctx.lineTo(16, -6);
    ctx.stroke();
    ctx.restore();
  }

  drawDrive(pose) {
    const ctx = this.ctx,
      p = this.project(pose.O),
      radius = 31;
    const start = -Math.PI * 0.93,
      end = -Math.PI * 1.9;
    ctx.save();
    ctx.strokeStyle = "rgba(41,66,93,0.58)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(p.x, p.y, radius, start, end, true);
    ctx.stroke();
    const point = {
      x: p.x + Math.cos(end) * radius,
      y: p.y + Math.sin(end) * radius,
    };
    const tangent = { x: Math.sin(end), y: -Math.cos(end) };
    ctx.beginPath();
    ctx.moveTo(point.x, point.y);
    ctx.lineTo(
      point.x - tangent.x * 7 - tangent.y * 3,
      point.y - tangent.y * 7 + tangent.x * 3,
    );
    ctx.lineTo(
      point.x - tangent.x * 7 + tangent.y * 3,
      point.y - tangent.y * 7 - tangent.x * 3,
    );
    ctx.closePath();
    ctx.fillStyle = "#506b83";
    ctx.fill();
    ctx.restore();
  }

  drawTail(pose) {
    if (!this.state.showTrace) return;
    const ctx = this.ctx,
      samples = [];
    for (let i = 0; i <= 28; i++) {
      const frame = usablePose(
        this.state.design,
        this._theta - (1 - i / 28) * 1.9,
      );
      if (frame) samples.push(this.project(frame.P));
    }
    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    for (let i = 1; i < samples.length; i++) {
      ctx.beginPath();
      ctx.moveTo(samples[i - 1].x, samples[i - 1].y);
      ctx.lineTo(samples[i].x, samples[i].y);
      ctx.strokeStyle = `rgba(209,94,51,${0.08 + (i / samples.length) * 0.77})`;
      ctx.lineWidth = 2.5;
      ctx.stroke();
    }
    if (samples.length > 6) {
      const a = samples[samples.length - 7],
        b = samples[samples.length - 5],
        length = distance(a, b);
      if (length > 0.4) {
        const dx = (b.x - a.x) / length,
          dy = (b.y - a.y) / length;
        ctx.beginPath();
        ctx.moveTo(b.x, b.y);
        ctx.lineTo(b.x - dx * 7 + dy * 3, b.y - dy * 7 - dx * 3);
        ctx.moveTo(b.x, b.y);
        ctx.lineTo(b.x - dx * 7 - dy * 3, b.y - dy * 7 + dx * 3);
        ctx.strokeStyle = ORANGE;
        ctx.lineWidth = 1.6;
        ctx.stroke();
      }
    }
    ctx.restore();
  }

  drawPen(point) {
    const ctx = this.ctx,
      p = this.project(point);
    ctx.save();
    ctx.beginPath();
    ctx.arc(p.x, p.y, 14, 0, TAU);
    ctx.fillStyle = "rgba(213,99,61,0.1)";
    ctx.fill();
    ctx.beginPath();
    ctx.arc(p.x, p.y, 7.5, 0, TAU);
    ctx.fillStyle = "#b84f2d";
    ctx.fill();
    ctx.beginPath();
    ctx.arc(p.x, p.y, 5, 0, TAU);
    ctx.fillStyle = "#e57e50";
    ctx.fill();
    ctx.strokeStyle = "#f2b893";
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(p.x, p.y, 1.5, 0, TAU);
    ctx.fillStyle = "#fff2de";
    ctx.fill();
    ctx.restore();
  }

  tag(text, point, dx, dy, color = INK) {
    const ctx = this.ctx,
      p = this.project(point);
    ctx.save();
    ctx.font = '10px "IBM Plex Mono", ui-monospace, monospace';
    ctx.textBaseline = "middle";
    const width = ctx.measureText(text).width;
    ctx.fillStyle = "rgba(243,241,233,0.89)";
    ctx.fillRect(p.x + dx - 3, p.y + dy - 7, width + 6, 14);
    ctx.fillStyle = color;
    ctx.fillText(text, p.x + dx, p.y + dy);
    ctx.restore();
  }

  drawDimensions(pose) {
    if (!this.state.showDimensions) return;
    const segments =
      this.state.design?.family === "slider"
        ? [
            [pose.O, pose.A],
            [pose.A, pose.B],
          ]
        : [
            [pose.O, pose.A],
            [pose.A, pose.B],
            [pose.B, pose.G],
            [pose.O, pose.G],
          ];
    const ctx = this.ctx;
    for (const [a, b] of segments) {
      const p = this.project(a),
        q = this.project(b),
        length = distance(p, q);
      if (length < 35) continue;
      const dx = (q.x - p.x) / length,
        dy = (q.y - p.y) / length;
      const normal = { x: -dy * 22, y: dx * 22 };
      ctx.save();
      ctx.strokeStyle = "rgba(49,70,84,0.46)";
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.moveTo(p.x + normal.x, p.y + normal.y);
      ctx.lineTo(q.x + normal.x, q.y + normal.y);
      for (const point of [p, q]) {
        ctx.moveTo(point.x + normal.x * 0.67, point.y + normal.y * 0.67);
        ctx.lineTo(point.x + normal.x * 1.15, point.y + normal.y * 1.15);
      }
      ctx.stroke();
      const text = distance(a, b).toFixed(2),
        middle = {
          x: (p.x + q.x) / 2 + normal.x,
          y: (p.y + q.y) / 2 + normal.y,
        };
      ctx.font = '9px "IBM Plex Mono", ui-monospace, monospace';
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      const textWidth = ctx.measureText(text).width;
      ctx.fillStyle = PAPER;
      ctx.fillRect(
        middle.x - textWidth / 2 - 4,
        middle.y - 6,
        textWidth + 8,
        12,
      );
      ctx.fillStyle = "#62717a";
      ctx.fillText(text, middle.x, middle.y);
      ctx.restore();
    }
  }

  drawScale() {
    if (!this.state.design) return;
    const ctx = this.ctx,
      unit = niceStep(80 / this.camera.scale),
      pixels = unit * this.camera.scale;
    ctx.save();
    ctx.strokeStyle = "rgba(67,88,96,0.48)";
    ctx.lineWidth = 1;
    const x = 27,
      y = this.height - 25;
    ctx.beginPath();
    ctx.moveTo(x, y - 4);
    ctx.lineTo(x, y);
    ctx.lineTo(x + pixels, y);
    ctx.lineTo(x + pixels, y - 4);
    ctx.stroke();
    ctx.font = '9px "IBM Plex Mono", ui-monospace, monospace';
    ctx.fillStyle = "#7d8784";
    ctx.fillText(`${Number(unit.toPrecision(3))} units`, x, y - 9);
    ctx.restore();
  }

  draw(timestamp) {
    this.raf = 0;
    if (this.destroyed || document.hidden || !this.visible) return;
    const started = performance.now();
    if (
      this.lastTime !== null &&
      this.state.playing &&
      !this.reducedMotion &&
      !this.drag
    ) {
      const dt = Math.min(
        0.06,
        Math.max(0, (timestamp - this.lastTime) / 1000),
      );
      this._theta =
        (this._theta +
          dt * TAU * clamp(Number(this.state.speed) || 0.45, 0.01, 8)) %
        TAU;
    }
    this.lastTime = timestamp;
    const ctx = this.ctx;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, this.width, this.height);
    this.drawGrid();
    this.drawCurves();
    this.drawGuide();
    const pose = usablePose(this.state.design, this._theta);
    this.currentPose = pose;
    if (pose) {
      if (this.state.design?.family !== "slider") {
        ctx.save();
        ctx.setLineDash([3, 5]);
        this.path([pose.O, pose.G]);
        ctx.strokeStyle = "rgba(62,83,93,0.24)";
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.restore();
      }
      this.drawAnchor(pose.O);
      this.drawAnchor(pose.G);
      this.drawDrive(pose);
      this.drawBar(pose.O, pose.A, 19);
      if (this.state.design?.family === "slider") this.drawSlider(pose);
      else this.drawBar(pose.G, pose.B, 17, true);
      this.drawCoupler(pose);
      this.drawTail(pose);
      this.drawHub(pose.O, 11);
      this.drawHub(pose.G, 9);
      this.drawHub(pose.A, 10);
      this.drawHub(pose.B, 10);
      this.drawPen(pose.P);
      this.drawDimensions(pose);
      this.tag("O · DRIVE", pose.O, -21, 41, "#677675");
      this.tag(
        this.state.design?.family === "slider" ? "G · GUIDE" : "G · FIXED",
        pose.G,
        -21,
        41,
        "#677675",
      );
      this.tag("A", pose.A, 13, -13, "#657887");
      this.tag("B", pose.B, 13, -13, "#657887");
      this.tag("P", pose.P, 13, -12, "#b25432");
      if (this.hover && pose[this.hover]) {
        const point = this.project(pose[this.hover]);
        ctx.save();
        ctx.beginPath();
        ctx.arc(point.x, point.y, this.hover === "P" ? 19 : 18, 0, TAU);
        ctx.strokeStyle = this.drag ? "#d16b42" : "rgba(49,84,124,0.5)";
        ctx.lineWidth = 1;
        ctx.setLineDash(this.drag ? [] : [2, 3]);
        ctx.stroke();
        ctx.restore();
      }
    }
    this.drawScale();
    this.frames++;
    this.lastRenderMs = performance.now() - started;
    if (
      this.state.playing &&
      !this.reducedMotion &&
      this.state.design &&
      !this.drag
    )
      this.schedule();
  }

  schedule() {
    if (!this.raf && !this.destroyed && !document.hidden && this.visible)
      this.raf = requestAnimationFrame((time) => this.draw(time));
  }

  cancelFrame() {
    if (this.raf) cancelAnimationFrame(this.raf);
    this.raf = 0;
  }

  getRenderInfo() {
    return {
      width: this.width,
      height: this.height,
      dpr: this.dpr,
      theta: this._theta,
      scale: this.camera.scale,
      frames: this.frames,
      lastRenderMs: Number(this.lastRenderMs.toFixed(2)),
      playing: this.state.playing && !this.reducedMotion,
      visible: this.visible && !document.hidden,
      handles: this.currentPose
        ? Object.fromEntries(
            (this.state.design?.family === "slider"
              ? ["O", "P"]
              : ["O", "G", "P"]
            ).map((key) => [key, this.project(this.currentPose[key])]),
          )
        : {},
    };
  }

  /** Standalone SVG: native, discrete sample animation preserves exact link closure in every frame. */
  snapshotSVG({
    title = "Motion Foundry — mechanism study",
    animated = true,
  } = {}) {
    if (!this.state.design) throw new Error("There is no mechanism to export.");
    const frameCount = animated ? 120 : 1;
    const poses = Array.from({ length: frameCount }, (_, index) =>
      usablePose(
        this.state.design,
        animated ? (index / frameCount) * TAU : this._theta,
      ),
    );
    if (poses.some((pose) => !pose))
      throw new Error("The mechanism does not close through its full cycle.");
    if (animated) poses.push(poses[0]);
    const width = 1200,
      height = 850;
    const camera = cameraFor(this.bounds, width, height - 70, 86);
    const project = (point) => {
      const p = screenPoint(point, camera, width, height - 70);
      return { x: Number(p.x.toFixed(3)), y: Number((p.y + 70).toFixed(3)) };
    };
    const screenPoses = poses.map((pose) =>
      Object.fromEntries(
        ["O", "G", "A", "B", "P"].map((key) => [key, project(pose[key])]),
      ),
    );
    const duration = (
      1 / clamp(Number(this.state.speed) || 0.45, 0.01, 8)
    ).toFixed(3);
    const animation = (attribute, values) =>
      animated
        ? `<animate attributeName="${attribute}" values="${values.join(";")}" dur="${duration}s" repeatCount="indefinite" calcMode="discrete"/>`
        : "";
    const list = (keys) =>
      screenPoses.map((pose) =>
        keys.map((key) => `${pose[key].x},${pose[key].y}`).join(" "),
      );
    const line = (keys, stroke, lineWidth, extra = "") => {
      const values = list(keys);
      return `<polyline points="${values[0]}" fill="none" stroke="${stroke}" stroke-width="${lineWidth}" stroke-linecap="round" stroke-linejoin="round" ${extra}>${animation("points", values)}</polyline>`;
    };
    const circle = (key, radius, fill, stroke = "#405b70", lineWidth = 1) =>
      `<circle cx="${screenPoses[0][key].x}" cy="${screenPoses[0][key].y}" r="${radius}" fill="${fill}" stroke="${stroke}" stroke-width="${lineWidth}">${animation(
        "cx",
        screenPoses.map((pose) => pose[key].x),
      )}${animation(
        "cy",
        screenPoses.map((pose) => pose[key].y),
      )}</circle>`;
    const tracePath = (points) =>
      points
        .map((point, index) => {
          const p = project(point);
          return `${index ? "L" : "M"}${p.x},${p.y}`;
        })
        .join(" ") + " Z";
    const target = pointsOf(this.state.target),
      curve = pointsOf(this.state.curve);
    const parts = [
      `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="study-title study-description">`,
      `<title id="study-title">${xml(title)}</title><desc id="study-description">A ${this.state.design.family === "slider" ? "crank-slider" : "four-bar"} mechanism and its traced path.${animated ? " Animation uses 120 exact kinematic poses per revolution, with discrete frame changes." : " Static construction view."} Links form rigid bodies; units are arbitrary. Kinematic study, without force or physical clearance analysis.</desc>`,
      '<defs><pattern id="small-grid" width="25" height="25" patternUnits="userSpaceOnUse"><path d="M25 0H0V25" fill="none" stroke="#8097a1" stroke-opacity="0.15" stroke-width="0.7"/></pattern><pattern id="grid" width="125" height="125" patternUnits="userSpaceOnUse"><rect width="125" height="125" fill="url(#small-grid)"/><path d="M125 0H0V125" fill="none" stroke="#738c99" stroke-opacity="0.17" stroke-width="1"/></pattern><radialGradient id="steel" cx="35%" cy="30%"><stop stop-color="#ffffff"/><stop offset=".55" stop-color="#e2e8e8"/><stop offset="1" stop-color="#90a4b2"/></radialGradient></defs>',
      `<rect width="${width}" height="${height}" fill="${PAPER}"/><rect y="70" width="${width}" height="${height - 70}" fill="url(#grid)"/>`,
      `<text x="38" y="35" fill="${INK}" font-family="Georgia,serif" font-size="24">${xml(title)}</text><text x="38" y="55" fill="#7f8781" font-family="monospace" font-size="9" letter-spacing="1.5">KINEMATIC CONSTRUCTION / ${animated ? "ONE CONTINUOUS CRANK REVOLUTION" : "STATIC STUDY"}</text>`,
    ];
    if (this.state.showTarget && target.length > 1)
      parts.push(
        `<path d="${tracePath(target)}" fill="none" stroke="#58625c" stroke-opacity=".65" stroke-width="1.6" stroke-dasharray="6 6"/>`,
      );
    if (this.state.showTrace && curve.length > 1)
      parts.push(
        `<path d="${tracePath(curve)}" fill="none" stroke="${ORANGE}" stroke-opacity=".7" stroke-width="2"/>`,
      );
    if (this.guide) {
      const [a, b] = this.guide.map(project),
        length = distance(a, b) || 1,
        nx = (-(b.y - a.y) / length) * 13,
        ny = ((b.x - a.x) / length) * 13;
      parts.push(
        `<path d="M${a.x},${a.y}L${b.x},${b.y}" fill="none" stroke="#b9c6ca" stroke-opacity=".65" stroke-width="21"/>`,
      );
      for (const sign of [-1, 1])
        parts.push(
          `<path d="M${a.x + nx * sign},${a.y + ny * sign}L${b.x + nx * sign},${b.y + ny * sign}" stroke="#758b98" stroke-width="1.5"/>`,
        );
    } else parts.push(line(["O", "G"], "#849492", 1, 'stroke-dasharray="4 5"'));
    for (const key of ["O", "G"]) {
      const point = screenPoses[0][key];
      parts.push(
        `<path d="M${point.x - 18},${point.y + 20}L${point.x - 7},${point.y + 3}H${point.x + 7}L${point.x + 18},${point.y + 20}Z" fill="#d5ddda" stroke="#8a9a95"/>`,
      );
      parts.push(
        `<path d="M${point.x - 23},${point.y + 24}H${point.x + 23}" stroke="#869990"/>`,
      );
      for (let x = -20; x <= 22; x += 7)
        parts.push(
          `<path d="M${point.x + x},${point.y + 24}l-5,6" stroke="#9aaba0" stroke-width=".8"/>`,
        );
    }
    const triangle = list(["A", "B", "P"]);
    parts.push(
      `<polygon points="${triangle[0]}" fill="#4b70a1" fill-opacity=".1" stroke="#577597" stroke-opacity=".4" stroke-width="1">${animation("points", triangle)}</polygon>`,
    );
    const bars =
      this.state.design.family === "slider"
        ? [
            ["O", "A"],
            ["A", "P"],
            ["B", "P"],
            ["A", "B"],
          ]
        : [
            ["O", "A"],
            ["G", "B"],
            ["A", "P"],
            ["B", "P"],
            ["A", "B"],
          ];
    for (const keys of bars) {
      const narrow = keys.includes("P");
      parts.push(line(keys, "#2a4364", narrow ? 11 : 20));
      parts.push(line(keys, narrow ? "#8199aa" : BLUE, narrow ? 8 : 16));
      parts.push(line(keys, narrow ? "#aec0ca" : "#789ac8", 1.2));
    }
    if (this.guide) {
      // The carriage is a rigid rectangle translated along the actual guide.
      const a = project(this.guide[0]),
        b = project(this.guide[1]),
        angle = Math.atan2(b.y - a.y, b.x - a.x);
      const corners = screenPoses.map((pose) =>
        [
          [-19, -10],
          [19, -10],
          [19, 10],
          [-19, 10],
        ]
          .map(
            ([x, y]) =>
              `${(pose.B.x + x * Math.cos(angle) - y * Math.sin(angle)).toFixed(3)},${(pose.B.y + x * Math.sin(angle) + y * Math.cos(angle)).toFixed(3)}`,
          )
          .join(" "),
      );
      parts.push(
        `<polygon points="${corners[0]}" fill="#b9c8d1" stroke="#4d6c82" stroke-width="1.4">${animation("points", corners)}</polygon>`,
      );
    }
    for (const key of ["O", "G", "A", "B"]) {
      parts.push(circle(key, key === "O" ? 12 : 10.5, "url(#steel)"));
      parts.push(circle(key, 4.8, "#e5ece7", "#6e8796"));
      parts.push(circle(key, 1.5, "#3e5b71", "none"));
    }
    parts.push(circle("P", 9, "#ce6038", "#ab462a", 1.2));
    parts.push(circle("P", 4, "#f3ad74", "#f5d1a7"));
    parts.push(circle("P", 1.5, "#fff3dd", "none"));
    for (const key of ["O", "G", "A", "B", "P"]) {
      const pose = screenPoses[0],
        dx = key === "O" || key === "G" ? -17 : 15,
        dy = key === "O" || key === "G" ? 43 : -13;
      const valuesX = screenPoses.map((frame) => frame[key].x + dx),
        valuesY = screenPoses.map((frame) => frame[key].y + dy);
      parts.push(
        `<text x="${pose[key].x + dx}" y="${pose[key].y + dy}" fill="${key === "P" ? ORANGE : INK}" font-family="monospace" font-size="11" paint-order="stroke" stroke="${PAPER}" stroke-width="4" stroke-linejoin="round">${key}${animation("x", valuesX)}${animation("y", valuesY)}</text>`,
      );
    }
    if (this.state.showDimensions) {
      const segments =
        this.state.design.family === "slider"
          ? [
              ["O", "A"],
              ["A", "B"],
            ]
          : [
              ["O", "A"],
              ["A", "B"],
              ["B", "G"],
              ["O", "G"],
            ];
      for (const [first, second] of segments) {
        const measurements = screenPoses.map((frame) => {
          const a = frame[first],
            b = frame[second],
            length = distance(a, b) || 1;
          const nx = (-(b.y - a.y) / length) * 26,
            ny = ((b.x - a.x) / length) * 26;
          const number = (value) => Number(value.toFixed(3));
          return {
            path: `M${number(a.x + nx)},${number(a.y + ny)}L${number(b.x + nx)},${number(b.y + ny)}M${number(a.x + nx * 0.67)},${number(a.y + ny * 0.67)}L${number(a.x + nx * 1.15)},${number(a.y + ny * 1.15)}M${number(b.x + nx * 0.67)},${number(b.y + ny * 0.67)}L${number(b.x + nx * 1.15)},${number(b.y + ny * 1.15)}`,
            x: number((a.x + b.x) / 2 + nx),
            y: number((a.y + b.y) / 2 + ny),
          };
        });
        const text = distance(poses[0][first], poses[0][second]).toFixed(2);
        parts.push(
          `<path d="${measurements[0].path}" fill="none" stroke="#7b8d99" stroke-width=".8">${animation(
            "d",
            measurements.map((value) => value.path),
          )}</path>`,
        );
        parts.push(
          `<text x="${measurements[0].x}" y="${measurements[0].y}" text-anchor="middle" dominant-baseline="middle" font-family="monospace" font-size="10" fill="#6b7f8e" paint-order="stroke" stroke="${PAPER}" stroke-width="5" stroke-linejoin="round">${text}${animation(
            "x",
            measurements.map((value) => value.x),
          )}${animation(
            "y",
            measurements.map((value) => value.y),
          )}</text>`,
        );
      }
    }
    parts.push(
      `<text x="${width - 38}" y="${height - 23}" text-anchor="end" fill="#7b8980" font-family="monospace" font-size="10">MOTION FOUNDRY · ${animated ? "120 EXACT POSES / CYCLE" : "CONSTRUCTION STUDY"}</text></svg>`,
    );
    return parts.join("");
  }

  destroy() {
    this.destroyed = true;
    this.cancelFrame();
    this.resizeObserver.disconnect();
    this.intersectionObserver?.disconnect();
    document.removeEventListener("visibilitychange", this.onVisibility);
    this.motionQuery.removeEventListener("change", this.onMotion);
    for (const [type, listener] of Object.entries(this.pointerListeners))
      this.canvas.removeEventListener(type, listener);
  }
}
