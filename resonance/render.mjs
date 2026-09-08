const TAU = Math.PI * 2;
const clamp = (value, low, high) => Math.max(low, Math.min(high, value));

function insidePolygon(x, y, outline) {
  let inside = false;
  for (let i = 0, j = outline.length - 1; i < outline.length; j = i++) {
    const a = outline[i],
      b = outline[j];
    if (
      a.y > y !== b.y > y &&
      x < ((b.x - a.x) * (y - a.y)) / (b.y - a.y) + a.x
    )
      inside = !inside;
  }
  return inside;
}

function barycentric(x, y, a, b, c) {
  const denominator =
    (b.sy - c.sy) * (a.sx - c.sx) + (c.sx - b.sx) * (a.sy - c.sy);
  if (Math.abs(denominator) < 0.01) return null;
  const u =
    ((b.sy - c.sy) * (x - c.sx) + (c.sx - b.sx) * (y - c.sy)) / denominator;
  const v =
    ((c.sy - a.sy) * (x - c.sx) + (a.sx - c.sx) * (y - c.sy)) / denominator;
  const w = 1 - u - v;
  return u >= -0.002 && v >= -0.002 && w >= -0.002 ? { u, v, w } : null;
}

/** A self-contained, perspective canvas instrument surface. Coordinates are in [-1, 1]. */
export class MembraneView {
  constructor(canvas, callbacks = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d", { alpha: true });
    this.callbacks = callbacks;
    this.state = {
      solution: null,
      outline: [],
      controlPoints: [],
      selectedMode: null,
      tool: "play",
      activeHandle: 0,
      reducedMotion: false,
    };
    this.width = 900;
    this.height = 620;
    this.scale = 300;
    this.origin = { x: 450, y: 330 };
    this.dpr = 1;
    this.raf = 0;
    this.destroyed = false;
    this.inViewport = true;
    this.pointer = null;
    this.drag = null;
    this.drawPoints = [];
    this.voices = [];
    this.active = false;
    this.ripples = [];
    this.hitTriangles = [];
    this.gridVertices = [];
    this.cells = [];
    this.field = new Float32Array(0);
    this.frameCount = 0;
    this.lastRenderMs = 0;
    this.canvas.style.touchAction = "none";
    this.canvas.style.cursor = "crosshair";

    this.listeners = {
      pointerdown: (event) => this.pointerDown(event),
      pointermove: (event) => this.pointerMove(event),
      pointerup: (event) => this.pointerUp(event),
      pointercancel: (event) => this.pointerUp(event, true),
      pointerleave: () => {
        if (!this.drag) {
          this.pointer = null;
          this.schedule();
        }
      },
    };
    for (const [type, listener] of Object.entries(this.listeners))
      canvas.addEventListener(type, listener);
    this.onVisibility = () => {
      if (document.hidden) this.cancelFrame();
      else this.schedule();
    };
    document.addEventListener("visibilitychange", this.onVisibility);
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas);
    if (typeof IntersectionObserver !== "undefined") {
      this.intersectionObserver = new IntersectionObserver(
        (entries) => {
          this.inViewport = entries[0]?.isIntersecting ?? true;
          if (this.inViewport) this.schedule();
          else this.cancelFrame();
        },
        { rootMargin: "80px" },
      );
      this.intersectionObserver.observe(canvas);
    }
    this.resize();
  }

  setState(next) {
    const changedSolution =
      "solution" in next && next.solution !== this.state.solution;
    const changedTool = "tool" in next && next.tool !== this.state.tool;
    Object.assign(this.state, next);
    if (changedSolution) {
      this.prepareGrid();
      this.voices = [];
      if (this.active) {
        this.active = false;
        this.callbacks.onActivityChange?.(false);
      }
    }
    if (changedTool) {
      this.drag = null;
      this.drawPoints = [];
      this.pointer = null;
      this.canvas.style.cursor =
        next.tool === "shape" ? "default" : "crosshair";
    }
    this.schedule();
  }

  impulse(amplitudes, frequencies, decay = 2.5) {
    const coefficients = Array.from(amplitudes, (value) =>
      Number.isFinite(value) ? value : 0,
    );
    const total = coefficients.reduce((sum, value) => sum + Math.abs(value), 0);
    if (total < 1e-14) return;
    // Physical displacement is tiny. One common gain exaggerates motion while
    // preserving signs and modal ratios; the L1 bound also limits visual peaks.
    const visualGain = 1.6 / total;
    this.voices.push({
      amplitudes: coefficients.map((value) => value * visualGain),
      frequencies: Array.from(frequencies),
      decay: Math.max(0.1, decay),
      startedAt: performance.now() / 1000,
    });
    if (this.voices.length > 8) this.voices.shift();
    const point = this.pointer?.world;
    if (point && insidePolygon(point.x, point.y, this.outline())) {
      this.ripples.push({
        x: point.x,
        y: point.y,
        startedAt: performance.now() / 1000,
      });
      if (this.ripples.length > 6) this.ripples.shift();
    }
    this.schedule();
  }

  outline() {
    return this.state.outline.length
      ? this.state.outline
      : this.state.solution?.outline || [];
  }

  resize() {
    const bounds = this.canvas.getBoundingClientRect();
    this.width = Math.max(1, bounds.width || 900);
    this.height = Math.max(1, bounds.height || 620);
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.canvas.width = Math.round(this.width * this.dpr);
    this.canvas.height = Math.round(this.height * this.dpr);
    this.scale = Math.min(this.width / 2.65, this.height / 1.83);
    this.origin = { x: this.width * 0.5, y: this.height * 0.55 };
    this.schedule();
  }

  prepareGrid() {
    const solution = this.state.solution;
    this.gridVertices = [];
    this.cells = [];
    this.hitTriangles = [];
    this.field = new Float32Array(solution?.coordinates.length || 0);
    if (!solution) return;
    const size = solution.resolution;
    const spacing = solution.spacing || 2 / (size - 1);
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const index = y * size + x;
        this.gridVertices.push({
          x: -1 + x * spacing,
          y: -1 + y * spacing,
          dataIndex: solution.indices[index],
          z: 0,
          sx: 0,
          sy: 0,
        });
      }
    }
    for (let y = 0; y < size - 1; y++) {
      for (let x = 0; x < size - 1; x++) {
        const a = y * size + x,
          b = a + 1,
          d = a + size,
          c = d + 1;
        const vertices = [a, b, c, d].map((index) => this.gridVertices[index]);
        if (vertices.every((vertex) => vertex.dataIndex < 0)) continue;
        this.cells.push({ vertices, depth: y - x * 0.143 });
      }
    }
    this.cells.sort((a, b) => a.depth - b.depth);
  }

  project(x, y, z = 0) {
    return {
      x: this.origin.x + (x + 0.15 * y) * this.scale,
      y: this.origin.y + (0.56 * y - 0.08 * x - z) * this.scale,
    };
  }

  unproject(x, y) {
    const u = (x - this.origin.x) / this.scale;
    const v = (y - this.origin.y) / this.scale;
    return { x: (0.56 * u - 0.15 * v) / 0.572, y: (0.08 * u + v) / 0.572 };
  }

  pointFromEvent(event) {
    const rect = this.canvas.getBoundingClientRect();
    return { x: event.clientX - rect.left, y: event.clientY - rect.top };
  }

  surfacePoint(screen) {
    // Pick the visible triangles, including raised lobes, rather than only the rest plane.
    for (let i = this.hitTriangles.length - 1; i >= 0; i--) {
      const triangle = this.hitTriangles[i];
      const weights = barycentric(screen.x, screen.y, ...triangle);
      if (!weights) continue;
      const [a, b, c] = triangle,
        { u, v, w } = weights;
      const point = {
        x: a.x * u + b.x * v + c.x * w,
        y: a.y * u + b.y * v + c.y * w,
      };
      if (insidePolygon(point.x, point.y, this.outline())) return point;
    }
    const point = this.unproject(screen.x, screen.y);
    return insidePolygon(point.x, point.y, this.outline()) ? point : null;
  }

  closestHandle(screen) {
    let closest = -1,
      distance = 23;
    this.state.controlPoints.forEach((point, index) => {
      const position = this.project(point.x, point.y);
      const d = Math.hypot(screen.x - position.x, screen.y - position.y);
      if (d < distance) {
        distance = d;
        closest = index;
      }
    });
    return closest;
  }

  pointerDown(event) {
    if (event.button !== 0 && event.pointerType !== "touch") return;
    event.preventDefault();
    this.canvas.focus({ preventScroll: true });
    const screen = this.pointFromEvent(event);
    const world =
      this.state.tool === "play"
        ? this.surfacePoint(screen)
        : this.unproject(screen.x, screen.y);
    this.pointer = { screen, world };
    if (this.state.tool === "play") {
      if (world) this.callbacks.onStrike?.(world.x, world.y);
    } else if (this.state.tool === "shape") {
      const index = this.closestHandle(screen);
      if (index < 0) return;
      this.drag = { type: "handle", index, pointerId: event.pointerId };
      this.callbacks.onSelectHandle?.(index);
      this.canvas.style.cursor = "grabbing";
      this.canvas.setPointerCapture(event.pointerId);
    } else if (this.state.tool === "draw") {
      this.drag = { type: "draw", pointerId: event.pointerId };
      this.drawPoints = [world];
      this.canvas.setPointerCapture(event.pointerId);
    }
    this.schedule();
  }

  pointerMove(event) {
    const screen = this.pointFromEvent(event);
    const world =
      this.state.tool === "play"
        ? this.surfacePoint(screen)
        : this.unproject(screen.x, screen.y);
    this.pointer = { screen, world };
    if (this.drag && event.pointerId !== this.drag.pointerId) return;
    if (this.drag?.type === "handle") {
      this.callbacks.onHandle?.(this.drag.index, world.x, world.y);
    } else if (this.drag?.type === "draw") {
      const previous = this.drawPoints[this.drawPoints.length - 1];
      if (
        Math.hypot(world.x - previous.x, world.y - previous.y) * this.scale >
        3
      )
        this.drawPoints.push(world);
    } else if (this.state.tool === "shape") {
      this.canvas.style.cursor =
        this.closestHandle(screen) >= 0 ? "grab" : "default";
    } else if (this.state.tool === "play") {
      this.canvas.style.cursor = world ? "crosshair" : "default";
    }
    this.schedule();
  }

  pointerUp(event, cancelled = false) {
    if (!this.drag || this.drag.pointerId !== event.pointerId) return;
    const drag = this.drag;
    this.drag = null;
    if (this.canvas.hasPointerCapture(event.pointerId))
      this.canvas.releasePointerCapture(event.pointerId);
    if (drag.type === "handle") {
      this.callbacks.onHandleEnd?.();
      this.canvas.style.cursor = "grab";
    } else if (!cancelled && this.drawPoints.length >= 8) {
      this.callbacks.onDraw?.(this.drawPoints.map((point) => ({ ...point })));
    }
    this.drawPoints = [];
    this.schedule();
  }

  calculateField(now) {
    const solution = this.state.solution;
    if (!solution) return;
    const modes = solution.modes;
    this.field.fill(0);
    this.voices = this.voices.filter(
      (voice) => now - voice.startedAt < Math.max(1.2, voice.decay * 3),
    );
    const active = this.voices.length > 0;
    if (active !== this.active) {
      this.active = active;
      this.callbacks.onActivityChange?.(active);
    }
    const selected = this.state.selectedMode;
    const reduced = this.state.reducedMotion;
    if (active && !reduced) {
      for (const voice of this.voices) {
        const age = now - voice.startedAt;
        const envelope = Math.exp((-age * 2.3) / voice.decay);
        const base = voice.frequencies[0] || 1;
        for (
          let modeIndex = 0;
          modeIndex < Math.min(modes.length, voice.amplitudes.length);
          modeIndex++
        ) {
          if (selected !== null && modeIndex !== selected) continue;
          const coefficient =
            voice.amplitudes[modeIndex] *
            envelope *
            Math.sin(age * (voice.frequencies[modeIndex] / base) * 5.4);
          if (Math.abs(coefficient) < 0.0001) continue;
          const values = modes[modeIndex].values;
          for (let index = 0; index < this.field.length; index++)
            this.field[index] += values[index] * coefficient;
        }
      }
      for (let i = 0; i < this.field.length; i++)
        this.field[i] = Math.tanh(this.field[i] * 1.15) * 0.26;
    } else {
      const index =
        selected !== null
          ? clamp(selected, 0, modes.length - 1)
          : Math.min(4, modes.length - 1);
      const values = modes[index]?.values;
      const wave = reduced
        ? 0.85
        : selected !== null
          ? Math.cos(now * 1.4)
          : 0.82 + 0.18 * Math.cos(now * 0.72);
      const amplitude = (selected !== null ? 0.215 : 0.16) * wave;
      if (values)
        for (let i = 0; i < this.field.length; i++)
          this.field[i] = values[i] * amplitude;
    }
    for (const vertex of this.gridVertices) {
      vertex.z = vertex.dataIndex < 0 ? 0 : this.field[vertex.dataIndex];
      const screen = this.project(vertex.x, vertex.y, vertex.z);
      vertex.sx = screen.x;
      vertex.sy = screen.y;
    }
  }

  outlinePath(offset = 0) {
    const path = new Path2D();
    const outline = this.outline();
    outline.forEach((point, index) => {
      const screen = this.project(point.x, point.y, offset);
      if (index === 0) path.moveTo(screen.x, screen.y);
      else path.lineTo(screen.x, screen.y);
    });
    path.closePath();
    return path;
  }

  drawGround() {
    const ctx = this.ctx;
    const center = this.project(0, 0, -0.06);
    // The elliptical field is subtle enough to let the enclosing surface define the backdrop.
    ctx.save();
    ctx.translate(center.x, center.y + this.scale * 0.09);
    ctx.scale(1, 0.48);
    const pool = ctx.createRadialGradient(0, 0, 0, 0, 0, this.scale * 1.17);
    pool.addColorStop(0, "rgba(0,0,0,0.43)");
    pool.addColorStop(0.66, "rgba(0,0,0,0.2)");
    pool.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = pool;
    ctx.fillRect(
      -this.scale * 1.2,
      -this.scale * 1.2,
      this.scale * 2.4,
      this.scale * 2.4,
    );
    ctx.restore();

    ctx.save();
    ctx.strokeStyle = "rgba(199,178,150,0.055)";
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 8]);
    const radius = 1.06;
    ctx.beginPath();
    for (let i = 0; i <= 120; i++) {
      const angle = (i / 120) * TAU;
      const point = this.project(
        Math.cos(angle) * radius,
        Math.sin(angle) * radius,
        -0.045,
      );
      if (!i) ctx.moveTo(point.x, point.y);
      else ctx.lineTo(point.x, point.y);
    }
    ctx.stroke();
    ctx.setLineDash([]);
    for (let i = 0; i < 4; i++) {
      const angle = (i * Math.PI) / 2;
      const a = this.project(
        Math.cos(angle) * 1.05,
        Math.sin(angle) * 1.05,
        -0.045,
      );
      const b = this.project(
        Math.cos(angle) * 1.09,
        Math.sin(angle) * 1.09,
        -0.045,
      );
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    }
    ctx.restore();
  }

  drawMembrane() {
    const ctx = this.ctx;
    const path = this.outlinePath();
    ctx.save();
    ctx.fillStyle = "#1c201f";
    ctx.shadowColor = "rgba(0,0,0,0.6)";
    ctx.shadowBlur = 24;
    ctx.shadowOffsetY = 13;
    ctx.fill(this.outlinePath(-0.024));
    ctx.restore();

    ctx.save();
    const skin = ctx.createLinearGradient(
      this.origin.x - this.scale,
      this.origin.y - this.scale,
      this.origin.x + this.scale,
      this.origin.y + this.scale * 0.4,
    );
    skin.addColorStop(0, "#35322d");
    skin.addColorStop(0.5, "#292827");
    skin.addColorStop(1, "#1b2525");
    ctx.fillStyle = skin;
    ctx.fill(path);
    ctx.clip(path);
    this.hitTriangles = [];
    for (const cell of this.cells) {
      const [a, b, c, d] = cell.vertices;
      const z = (a.z + b.z + c.z + d.z) * 0.25;
      const slope =
        (a.z + d.z - (b.z + c.z)) * 1.6 + (c.z + d.z - (a.z + b.z)) * 1.4;
      const amount = clamp(Math.abs(z) / 0.22, 0, 1);
      const light = clamp(0.94 + slope * 1.8, 0.42, 1.2);
      const low = [39, 40, 37];
      const high = z >= 0 ? [240, 132, 82] : [66, 155, 153];
      const color = low.map((value, i) =>
        Math.min(255, Math.round((value + (high[i] - value) * amount) * light)),
      );
      ctx.fillStyle = `rgb(${color.join(",")})`;
      ctx.beginPath();
      ctx.moveTo(a.sx, a.sy);
      ctx.lineTo(b.sx, b.sy);
      ctx.lineTo(c.sx, c.sy);
      ctx.lineTo(d.sx, d.sy);
      ctx.closePath();
      ctx.fill();
      ctx.strokeStyle =
        z >= 0
          ? `rgba(249,172,120,${0.1 + amount * 0.35})`
          : `rgba(113,183,174,${0.11 + amount * 0.28})`;
      ctx.lineWidth = 0.65;
      ctx.stroke();
      this.hitTriangles.push([a, b, c], [a, c, d]);
    }
    // Fine crossing threads make the neutral areas read as a woven physical surface.
    ctx.globalAlpha = 0.13;
    ctx.strokeStyle = "#d0b898";
    ctx.lineWidth = 0.4;
    const size = this.state.solution?.resolution || 0;
    for (let y = 1; y < size - 1; y++) {
      for (let x = 1; x < size - 1; x++) {
        const vertex = this.gridVertices[y * size + x];
        if (vertex.dataIndex < 0 || (x + y) % 3 !== 0) continue;
        ctx.beginPath();
        ctx.moveTo(vertex.sx - 1, vertex.sy + 1.5);
        ctx.lineTo(vertex.sx + 1, vertex.sy - 1.5);
        ctx.stroke();
      }
    }
    ctx.restore();

    // A heavy lower rim and a narrow highlight make a solid copper frame.
    ctx.save();
    ctx.strokeStyle = "#201b17";
    ctx.lineWidth = 8;
    ctx.stroke(this.outlinePath(-0.01));
    const copper = ctx.createLinearGradient(
      this.origin.x,
      this.origin.y - this.scale * 0.55,
      this.origin.x + this.scale * 0.3,
      this.origin.y + this.scale * 0.55,
    );
    copper.addColorStop(0, "#8c6751");
    copper.addColorStop(0.43, "#c98e68");
    copper.addColorStop(0.62, "#775140");
    copper.addColorStop(1, "#d3a383");
    ctx.strokeStyle = copper;
    ctx.lineWidth = 3.8;
    ctx.stroke(path);
    ctx.strokeStyle = "rgba(255,219,178,0.32)";
    ctx.lineWidth = 0.7;
    ctx.stroke(this.outlinePath(0.003));
    ctx.restore();
  }

  sampleHeight(x, y) {
    const solution = this.state.solution;
    if (!solution) return 0;
    const size = solution.resolution;
    const gx = clamp(
      (x + 1) / (solution.spacing || 2 / (size - 1)),
      0,
      size - 1.00001,
    );
    const gy = clamp(
      (y + 1) / (solution.spacing || 2 / (size - 1)),
      0,
      size - 1.00001,
    );
    const ix = Math.floor(gx),
      iy = Math.floor(gy),
      fx = gx - ix,
      fy = gy - iy;
    const get = (i, j) => {
      const index = solution.indices[j * size + i];
      return index < 0 ? 0 : this.field[index] || 0;
    };
    return (
      get(ix, iy) * (1 - fx) * (1 - fy) +
      get(ix + 1, iy) * fx * (1 - fy) +
      get(ix, iy + 1) * (1 - fx) * fy +
      get(ix + 1, iy + 1) * fx * fy
    );
  }

  drawInteraction(now) {
    const ctx = this.ctx;
    if (this.state.tool === "shape") {
      ctx.save();
      this.state.controlPoints.forEach((point, index) => {
        const screen = this.project(point.x, point.y);
        const active =
          index === this.state.activeHandle || index === this.drag?.index;
        ctx.beginPath();
        ctx.arc(screen.x, screen.y, active ? 8.5 : 5, 0, TAU);
        ctx.fillStyle = active ? "#e8b58c" : "#252622";
        ctx.fill();
        ctx.strokeStyle = active ? "#f6d5af" : "#d0a781";
        ctx.lineWidth = active ? 1.5 : 1.1;
        ctx.stroke();
        if (active) {
          ctx.beginPath();
          ctx.arc(screen.x, screen.y, 15, 0, TAU);
          ctx.strokeStyle = "rgba(229,180,136,0.2)";
          ctx.lineWidth = 1;
          ctx.stroke();
          ctx.beginPath();
          ctx.arc(screen.x, screen.y, 2, 0, TAU);
          ctx.fillStyle = "#50402f";
          ctx.fill();
        }
      });
      ctx.restore();
    }
    if (this.state.tool === "draw") {
      ctx.save();
      ctx.globalAlpha = this.drawPoints.length ? 1 : 0.5;
      ctx.strokeStyle = "#eec9a4";
      ctx.lineWidth = 2;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.beginPath();
      this.drawPoints.forEach((point, index) => {
        const screen = this.project(point.x, point.y);
        if (index === 0) ctx.moveTo(screen.x, screen.y);
        else ctx.lineTo(screen.x, screen.y);
      });
      ctx.stroke();
      if (this.drawPoints.length > 1) {
        const start = this.project(this.drawPoints[0].x, this.drawPoints[0].y);
        const end = this.project(
          this.drawPoints.at(-1).x,
          this.drawPoints.at(-1).y,
        );
        ctx.setLineDash([4, 6]);
        ctx.lineWidth = 1;
        ctx.strokeStyle = "rgba(238,201,164,0.45)";
        ctx.beginPath();
        ctx.moveTo(end.x, end.y);
        ctx.lineTo(start.x, start.y);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.arc(start.x, start.y, 5, 0, TAU);
        ctx.fillStyle = "#edc39b";
        ctx.fill();
      }
      ctx.restore();
    }
    if (this.pointer?.world && this.state.tool === "play") {
      const point = this.pointer.world;
      const screen = this.project(
        point.x,
        point.y,
        this.sampleHeight(point.x, point.y),
      );
      ctx.save();
      ctx.strokeStyle = "rgba(255,225,188,0.9)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.ellipse(screen.x, screen.y, 11, 7, -0.06, 0, TAU);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(screen.x, screen.y, 2, 0, TAU);
      ctx.fillStyle = "#f6d4b2";
      ctx.fill();
      ctx.restore();
    }
    this.ripples = this.ripples.filter(
      (ripple) => now - ripple.startedAt < 0.8,
    );
    if (!this.state.reducedMotion)
      for (const ripple of this.ripples) {
        const age = now - ripple.startedAt;
        const screen = this.project(
          ripple.x,
          ripple.y,
          this.sampleHeight(ripple.x, ripple.y),
        );
        ctx.save();
        ctx.strokeStyle = `rgba(255,215,171,${(1 - age / 0.8) * 0.6})`;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.ellipse(
          screen.x,
          screen.y,
          5 + age * 45,
          3 + age * 25,
          -0.07,
          0,
          TAU,
        );
        ctx.stroke();
        ctx.restore();
      }
  }

  draw(timestamp) {
    this.raf = 0;
    if (this.destroyed || document.hidden || !this.inViewport) return;
    const start = performance.now();
    const now = timestamp / 1000;
    const ctx = this.ctx;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, this.width, this.height);
    this.calculateField(now);
    this.drawGround();
    if (this.outline().length > 2) this.drawMembrane();
    this.drawInteraction(now);
    this.frameCount++;
    this.lastRenderMs = performance.now() - start;
    if (!this.state.reducedMotion && this.state.solution) this.schedule();
  }

  schedule() {
    if (!this.raf && !this.destroyed && !document.hidden && this.inViewport)
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
      frames: this.frameCount,
      lastRenderMs: Number(this.lastRenderMs.toFixed(2)),
      vertices: this.field.length,
      cells: this.cells.length,
      activeVoices: this.voices.length,
      visible: this.inViewport && !document.hidden,
    };
  }

  destroy() {
    this.destroyed = true;
    this.cancelFrame();
    this.resizeObserver.disconnect();
    this.intersectionObserver?.disconnect();
    document.removeEventListener("visibilitychange", this.onVisibility);
    for (const [type, listener] of Object.entries(this.listeners))
      this.canvas.removeEventListener(type, listener);
    this.voices = [];
    this.hitTriangles = [];
    if (this.active) this.callbacks.onActivityChange?.(false);
    this.active = false;
  }
}
