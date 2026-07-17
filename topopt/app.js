"use strict";

/* Topology Optimizer — drafting-sheet UI over the SIMP solver. */

const INK = "#20262d";
const PAPER_RGB = [253, 253, 250];
const INK_RGB = [32, 38, 45];
const BLUE = "#2b5aa6";
const RED = "#c23d26";
const FAINT = "rgba(32, 38, 45, 0.5)";
const CONSTRUCTION = "rgba(43, 90, 166, 0.4)";

const ARROW_LEN = 34;
const VOID_BRUSH = 0.035; // radius as a fraction of domain width

const PRESETS = {
  cantilever: {
    walls: ["left"],
    pins: [],
    loads: [{ x: 1, y: 0.55, dx: 0, dy: 1 }],
    voids: [],
  },
  bridge: {
    walls: [],
    pins: [{ x: 0.015, y: 1 }, { x: 0.985, y: 1 }],
    loads: [{ x: 0.5, y: 1, dx: 0, dy: 1 }],
    voids: [],
  },
  canopy: {
    walls: ["top"],
    pins: [],
    loads: [{ x: 0.62, y: 0.82, dx: 0, dy: 1 }],
    voids: [],
  },
  table: {
    walls: [],
    pins: [{ x: 0.06, y: 1 }, { x: 0.94, y: 1 }],
    loads: [
      { x: 0.2, y: 0, dx: 0, dy: 1 },
      { x: 0.35, y: 0, dx: 0, dy: 1 },
      { x: 0.5, y: 0, dx: 0, dy: 1 },
      { x: 0.65, y: 0, dx: 0, dy: 1 },
      { x: 0.8, y: 0, dx: 0, dy: 1 },
    ],
    voids: [],
  },
};

const TOOL_HINTS = {
  load: "Load — press on the sheet, drag to aim the force, release to place it.",
  pin: "Pin — click to anchor the structure. Two pins stop rotation.",
  void: "Void — drag to carve keep-out holes the material must route around.",
  erase: "Erase — click a load, pin, or void to remove it.",
};

const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);

function fmtC(c) {
  if (!isFinite(c)) return "—";
  if (c >= 1000) return c.toFixed(0);
  if (c >= 10) return c.toFixed(1);
  return c.toFixed(2);
}

class App {
  constructor() {
    this.canvas = document.getElementById("field");
    this.ctx = this.canvas.getContext("2d");
    this.spark = document.getElementById("spark");
    this.sparkCtx = this.spark.getContext("2d");

    this.els = {};
    for (const id of ["run", "reset", "clear", "volfrac", "rmin", "spec-mesh",
      "volfrac-out", "rmin-out", "tb-iter", "tb-c", "tb-vol", "tb-chg",
      "status-state", "status-hint", "spark-readout"]) {
      this.els[id] = document.getElementById(id);
    }

    this.state = structuredClone(PRESETS.cantilever);
    this.presetName = "cantilever";
    this.tool = "load";
    this.nely = 70;
    this.volfrac = 0.4;
    this.rminBase = 2.25;
    this.history = [];
    this.running = false;
    this.drag = null;
    this.layout = { ox: 0, oy: 0, w: 1, h: 1 };
    this.reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

    this.buildSolver();
    this.bindPanel();
    this.bindPointer();
    this.bindSpark();
    this.setTool("load");

    new ResizeObserver(() => this.resize()).observe(this.canvas);
    this.resize();
    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(() => this.draw());
    }
    if (!this.reducedMotion) setTimeout(() => this.start(), 700);
  }

  /* ---------- solver plumbing ---------- */

  buildSolver() {
    const nely = this.nely;
    const nelx = nely * 2;
    const rmin = this.rminBase * (nelx / 96);
    this.solver = new TopOpt({ nelx, nely, volfrac: this.volfrac, rmin });
    this.els["spec-mesh"].textContent = `${nelx} × ${nely}`;
    this.rebuildBCs();
    this.solver.resetDesign();
    this.history = [];
    this.updateReadouts(null);
  }

  rebuildBCs() {
    const s = this.solver;
    const { nelx, nely } = s;
    s.clearBCs();
    for (const wall of this.state.walls) {
      if (wall === "left") for (let iy = 0; iy <= nely; iy++) s.fixNode(0, iy);
      if (wall === "right") for (let iy = 0; iy <= nely; iy++) s.fixNode(nelx, iy);
      if (wall === "top") for (let ix = 0; ix <= nelx; ix++) s.fixNode(ix, 0);
      if (wall === "bottom") for (let ix = 0; ix <= nelx; ix++) s.fixNode(ix, nely);
    }
    for (const pin of this.state.pins) {
      const ix = Math.round(pin.x * nelx);
      const iy = Math.round(pin.y * nely);
      for (const [ox, oy] of [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const jx = ix + ox, jy = iy + oy;
        if (jx >= 0 && jx <= nelx && jy >= 0 && jy <= nely) s.fixNode(jx, jy);
      }
    }
    for (const load of this.state.loads) {
      const ix = Math.round(load.x * nelx);
      const iy = Math.round(load.y * nely);
      const mag = Math.hypot(load.dx, load.dy) || 1;
      s.addLoad(ix, iy, load.dx / mag, load.dy / mag);
    }
    const passive = new Uint8Array(s.nel);
    for (const v of this.state.voids) {
      const rEl = v.r * nelx;
      const cx = v.x * nelx, cy = v.y * nely;
      const x0 = Math.max(0, Math.floor(cx - rEl)), x1 = Math.min(nelx - 1, Math.ceil(cx + rEl));
      const y0 = Math.max(0, Math.floor(cy - rEl)), y1 = Math.min(nely - 1, Math.ceil(cy + rEl));
      for (let ex = x0; ex <= x1; ex++) {
        for (let ey = y0; ey <= y1; ey++) {
          const dx = ex + 0.5 - cx, dy = ey + 0.5 - cy;
          if (dx * dx + dy * dy <= rEl * rEl) passive[ex * nely + ey] = 1;
        }
      }
    }
    s.setPassive(passive);
    s.applyPassive(s.xPhys);
  }

  /* ---------- run loop ---------- */

  start() {
    if (this.running) return;
    if (!this.solver.hasValidBCs()) {
      this.setStatus("CHECK SETUP", "Needs at least one pin or wall, and one load.", true);
      return;
    }
    this.running = true;
    this.els.run.textContent = "Pause";
    this.els.run.classList.add("running");
    this.setStatus("ITERATING", TOOL_HINTS[this.tool]);
    setTimeout(() => this.tick(), 0);
  }

  pause(label = "PAUSED") {
    this.running = false;
    this.els.run.textContent = "Run optimization";
    this.els.run.classList.remove("running");
    this.setStatus(label, TOOL_HINTS[this.tool]);
  }

  tick() {
    if (!this.running) return;
    if (!this.solver.hasValidBCs()) {
      this.pause("CHECK SETUP");
      this.setStatus("CHECK SETUP", "Needs at least one pin or wall, and one load.", true);
      return;
    }
    const res = this.solver.step();
    this.history.push(res.compliance);
    this.updateReadouts(res);
    this.draw();
    this.drawSpark();
    if (this.isConverged(res)) {
      this.pause("CONVERGED");
      this.setStatus("CONVERGED", `Design settled after ${res.iter} iterations.`);
    } else if (res.iter >= 600) {
      this.pause("STOPPED");
    } else {
      // setTimeout instead of rAF: keep iterating when the window is occluded
      setTimeout(() => this.tick(), 0);
    }
  }

  isConverged(res) {
    if (res.change < 0.01 && res.iter >= 15) return true;
    // OC designs often keep a few boundary elements flickering; treat a flat
    // compliance window as settled too
    if (res.iter >= 40 && this.history.length >= 12) {
      const win = this.history.slice(-12);
      const hi = Math.max(...win), lo = Math.min(...win);
      if (hi > 0 && (hi - lo) / hi < 0.001) return true;
    }
    return false;
  }

  /* ---------- panel wiring ---------- */

  bindPanel() {
    document.querySelectorAll(".preset").forEach((btn) => {
      btn.addEventListener("click", () => this.setPreset(btn.dataset.preset));
    });
    document.querySelectorAll(".tool").forEach((btn) => {
      btn.addEventListener("click", () => this.setTool(btn.dataset.tool));
    });
    document.querySelectorAll(".seg button").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".seg button").forEach((b) =>
          b.setAttribute("aria-pressed", b === btn ? "true" : "false"));
        this.nely = parseInt(btn.dataset.mesh, 10);
        const wasRunning = this.running;
        this.pause("READY");
        this.buildSolver();
        this.draw();
        this.drawSpark();
        if (wasRunning) this.start();
      });
    });
    this.els.volfrac.addEventListener("input", () => {
      this.volfrac = parseFloat(this.els.volfrac.value);
      this.els["volfrac-out"].textContent = this.volfrac.toFixed(2);
      this.solver.volfrac = this.volfrac;
    });
    this.els.rmin.addEventListener("change", () => {
      this.rminBase = parseFloat(this.els.rmin.value);
      const wasRunning = this.running;
      this.pause("READY");
      this.buildSolver();
      this.draw();
      this.drawSpark();
      if (wasRunning) this.start();
    });
    this.els.rmin.addEventListener("input", () => {
      this.els["rmin-out"].textContent = parseFloat(this.els.rmin.value).toFixed(2);
    });
    this.els.run.addEventListener("click", () => {
      if (this.running) this.pause();
      else this.start();
    });
    this.els.reset.addEventListener("click", () => {
      this.solver.resetDesign();
      this.history = [];
      this.updateReadouts(null);
      this.draw();
      this.drawSpark();
      if (!this.running) this.setStatus("READY", TOOL_HINTS[this.tool]);
    });
    this.els.clear.addEventListener("click", () => {
      this.state = { walls: [], pins: [], loads: [], voids: [] };
      this.markCustom();
      this.pause("READY");
      this.rebuildBCs();
      this.solver.resetDesign();
      this.history = [];
      this.updateReadouts(null);
      this.draw();
      this.drawSpark();
      this.setStatus("BLANK SHEET", "Draw a load and a pin (or pick a preset), then run.");
    });
  }

  setPreset(name) {
    this.state = structuredClone(PRESETS[name]);
    this.presetName = name;
    document.querySelectorAll(".preset").forEach((b) =>
      b.setAttribute("aria-pressed", b.dataset.preset === name ? "true" : "false"));
    this.pause("READY");
    this.rebuildBCs();
    this.solver.resetDesign();
    this.history = [];
    this.updateReadouts(null);
    this.draw();
    this.drawSpark();
    this.start();
  }

  markCustom() {
    this.presetName = null;
    document.querySelectorAll(".preset").forEach((b) => b.setAttribute("aria-pressed", "false"));
  }

  setTool(tool) {
    this.tool = tool;
    document.querySelectorAll(".tool").forEach((b) =>
      b.setAttribute("aria-pressed", b.dataset.tool === tool ? "true" : "false"));
    if (!this.running) this.setStatus(this.els["status-state"].textContent, TOOL_HINTS[tool]);
    else this.els["status-hint"].textContent = TOOL_HINTS[tool];
  }

  setStatus(state, hint, warn = false) {
    this.els["status-state"].textContent = state;
    this.els["status-state"].classList.toggle("warn", warn);
    if (hint !== undefined) this.els["status-hint"].textContent = hint;
  }

  updateReadouts(res) {
    this.els["tb-iter"].textContent = String(res ? res.iter : 0).padStart(3, "0");
    this.els["tb-c"].textContent = res ? fmtC(res.compliance) : "—";
    this.els["tb-vol"].textContent = res ? (res.volume * 100).toFixed(1) + "%" : "—";
    this.els["tb-chg"].textContent = res ? res.change.toFixed(3) : "—";
  }

  /* ---------- pointer tools ---------- */

  bindPointer() {
    const c = this.canvas;
    c.addEventListener("pointerdown", (ev) => {
      c.setPointerCapture(ev.pointerId);
      const p = this.toNorm(ev);
      if (this.tool === "load") {
        this.drag = { kind: "load", x: clamp(p.x, 0, 1), y: clamp(p.y, 0, 1), px: p.px, py: p.py, cx: p.px, cy: p.py };
      } else if (this.tool === "pin") {
        this.state.pins.push({ x: clamp(p.x, 0, 1), y: clamp(p.y, 0, 1) });
        this.afterEdit();
      } else if (this.tool === "void") {
        this.drag = { kind: "void", lastX: null, lastY: null };
        this.paintVoid(p);
      } else if (this.tool === "erase") {
        this.drag = { kind: "erase" };
        this.eraseAt(p);
      }
      this.draw();
    });
    c.addEventListener("pointermove", (ev) => {
      if (!this.drag) return;
      const p = this.toNorm(ev);
      if (this.drag.kind === "load") {
        this.drag.cx = p.px;
        this.drag.cy = p.py;
        this.draw();
      } else if (this.drag.kind === "void") {
        this.paintVoid(p);
        this.draw();
      } else if (this.drag.kind === "erase") {
        this.eraseAt(p, true);
        this.draw();
      }
    });
    const finish = (ev) => {
      if (!this.drag) return;
      if (this.drag.kind === "load") {
        const d = this.drag;
        let dx = d.cx - d.px, dy = d.cy - d.py;
        if (Math.hypot(dx, dy) < 8) { dx = 0; dy = 1; }
        const mag = Math.hypot(dx, dy);
        this.state.loads.push({ x: d.x, y: d.y, dx: dx / mag, dy: dy / mag });
        this.afterEdit();
      }
      this.drag = null;
      this.draw();
    };
    c.addEventListener("pointerup", finish);
    c.addEventListener("pointercancel", () => { this.drag = null; this.draw(); });
  }

  toNorm(ev) {
    const rect = this.canvas.getBoundingClientRect();
    const px = ev.clientX - rect.left;
    const py = ev.clientY - rect.top;
    const { ox, oy, w, h } = this.layout;
    return { px, py, x: (px - ox) / w, y: (py - oy) / h };
  }

  paintVoid(p) {
    const x = clamp(p.x, 0, 1), y = clamp(p.y, 0, 1);
    const d = this.drag;
    if (d.lastX !== null && Math.hypot(x - d.lastX, (y - d.lastY) * 0.5) < VOID_BRUSH * 0.5) return;
    d.lastX = x; d.lastY = y;
    this.state.voids.push({ x, y, r: VOID_BRUSH });
    this.afterEdit();
  }

  eraseAt(p, dragging = false) {
    const { w, h } = this.layout;
    const hitR = 14;
    if (!dragging) {
      for (let i = this.state.loads.length - 1; i >= 0; i--) {
        const L = this.state.loads[i];
        const lx = L.x * w, ly = L.y * h;
        const tx = lx + L.dx * ARROW_LEN, ty = ly + L.dy * ARROW_LEN;
        const px = p.x * w, py = p.y * h;
        if (distToSegment(px, py, lx, ly, tx, ty) < hitR) {
          this.state.loads.splice(i, 1);
          this.afterEdit();
          return;
        }
      }
      for (let i = this.state.pins.length - 1; i >= 0; i--) {
        const P = this.state.pins[i];
        if (Math.hypot((P.x - p.x) * w, (P.y - p.y) * h) < hitR) {
          this.state.pins.splice(i, 1);
          this.afterEdit();
          return;
        }
      }
    }
    for (let i = this.state.voids.length - 1; i >= 0; i--) {
      const V = this.state.voids[i];
      if (Math.hypot((V.x - p.x) * w, (V.y - p.y) * h) < V.r * w + (dragging ? 4 : 0)) {
        this.state.voids.splice(i, 1);
        this.afterEdit();
        if (!dragging) return;
      }
    }
  }

  afterEdit() {
    this.markCustom();
    this.rebuildBCs();
    if (!this.running) {
      this.solver.filterForward(this.solver.x, this.solver.xPhys);
      this.solver.applyPassive(this.solver.xPhys);
      this.setStatus("READY", TOOL_HINTS[this.tool]);
    }
  }

  /* ---------- rendering ---------- */

  resize() {
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    if (rect.width < 10 || rect.height < 10) return;
    this.canvas.width = Math.round(rect.width * dpr);
    this.canvas.height = Math.round(rect.height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const pad = { t: 38, r: 58, b: 40, l: 34 };
    let w = rect.width - pad.l - pad.r;
    let h = w / 2;
    const maxH = rect.height - pad.t - pad.b;
    if (h > maxH) { h = maxH; w = h * 2; }
    this.layout = {
      ox: pad.l + (rect.width - pad.l - pad.r - w) / 2,
      oy: pad.t + (rect.height - pad.t - pad.b - h) / 2,
      w, h,
      cssW: rect.width, cssH: rect.height,
    };

    const sRect = this.spark.getBoundingClientRect();
    this.spark.width = Math.round(sRect.width * dpr);
    this.spark.height = Math.round(44 * dpr);
    this.sparkCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

    this.draw();
    this.drawSpark();
  }

  draw() {
    const ctx = this.ctx;
    const { ox, oy, w, h, cssW, cssH } = this.layout;
    const s = this.solver;
    ctx.clearRect(0, 0, cssW, cssH);

    // sheet under the design domain
    ctx.fillStyle = "#fdfdfa";
    ctx.fillRect(ox, oy, w, h);

    // density field
    if (!this.off || this.off.width !== s.nelx || this.off.height !== s.nely) {
      this.off = document.createElement("canvas");
      this.off.width = s.nelx;
      this.off.height = s.nely;
      this.offCtx = this.off.getContext("2d");
      this.img = this.offCtx.createImageData(s.nelx, s.nely);
    }
    const data = this.img.data;
    const xp = s.xPhys;
    for (let ey = 0; ey < s.nely; ey++) {
      for (let ex = 0; ex < s.nelx; ex++) {
        const t = Math.pow(xp[ex * s.nely + ey], 1.25);
        const i = (ey * s.nelx + ex) * 4;
        data[i] = PAPER_RGB[0] + (INK_RGB[0] - PAPER_RGB[0]) * t;
        data[i + 1] = PAPER_RGB[1] + (INK_RGB[1] - PAPER_RGB[1]) * t;
        data[i + 2] = PAPER_RGB[2] + (INK_RGB[2] - PAPER_RGB[2]) * t;
        data[i + 3] = 255;
      }
    }
    this.offCtx.putImageData(this.img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(this.off, ox, oy, w, h);

    // construction outline of the design envelope
    ctx.strokeStyle = "rgba(32, 38, 45, 0.3)";
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 4]);
    ctx.strokeRect(ox + 0.5, oy + 0.5, w, h);
    ctx.setLineDash([]);

    this.drawVoids(ctx);
    this.drawWalls(ctx);
    this.drawPins(ctx);
    this.drawLoads(ctx);
    this.drawDims(ctx);

    // load-drag preview
    if (this.drag && this.drag.kind === "load") {
      const d = this.drag;
      const lx = ox + d.x * w, ly = oy + d.y * h;
      let dx = d.cx - d.px, dy = d.cy - d.py;
      if (Math.hypot(dx, dy) < 8) { dx = 0; dy = 1; }
      const mag = Math.hypot(dx, dy);
      ctx.globalAlpha = 0.55;
      this.drawArrow(ctx, lx, ly, dx / mag, dy / mag, true);
      ctx.globalAlpha = 1;
    }
  }

  drawVoids(ctx) {
    const { ox, oy, w, h } = this.layout;
    for (const v of this.state.voids) {
      const cx = ox + v.x * w, cy = oy + v.y * h, r = v.r * w;
      ctx.save();
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.clip();
      ctx.strokeStyle = CONSTRUCTION;
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let d = -r * 2; d <= r * 2; d += 6) {
        ctx.moveTo(cx + d - r, cy + r);
        ctx.lineTo(cx + d + r, cy - r);
      }
      ctx.stroke();
      ctx.restore();
      ctx.strokeStyle = CONSTRUCTION;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  drawWalls(ctx) {
    const { ox, oy, w, h } = this.layout;
    ctx.strokeStyle = INK;
    for (const wall of this.state.walls) {
      let x0, y0, x1, y1, nx, ny; // edge segment + outward normal
      if (wall === "left") { x0 = ox; y0 = oy; x1 = ox; y1 = oy + h; nx = -1; ny = 0; }
      else if (wall === "right") { x0 = ox + w; y0 = oy; x1 = ox + w; y1 = oy + h; nx = 1; ny = 0; }
      else if (wall === "top") { x0 = ox; y0 = oy; x1 = ox + w; y1 = oy; nx = 0; ny = -1; }
      else { x0 = ox; y0 = oy + h; x1 = ox + w; y1 = oy + h; nx = 0; ny = 1; }
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      ctx.stroke();
      // ground hatching on the outside
      const len = Math.hypot(x1 - x0, y1 - y0);
      const tx = (x1 - x0) / len, ty = (y1 - y0) / len;
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let d = 6; d < len; d += 11) {
        const bx = x0 + tx * d, by = y0 + ty * d;
        ctx.moveTo(bx, by);
        ctx.lineTo(bx + (nx - tx) * 8, by + (ny - ty) * 8);
      }
      ctx.stroke();
    }
  }

  drawPins(ctx) {
    const { ox, oy, w, h } = this.layout;
    const size = 13;
    for (const p of this.state.pins) {
      const px = ox + p.x * w, py = oy + p.y * h;
      ctx.fillStyle = "#fdfdfa";
      ctx.strokeStyle = INK;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.lineTo(px - size * 0.62, py + size);
      ctx.lineTo(px + size * 0.62, py + size);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      // ground line + hatch
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(px - size, py + size + 2.5);
      ctx.lineTo(px + size, py + size + 2.5);
      ctx.stroke();
      ctx.beginPath();
      for (let i = -1; i <= 1; i++) {
        ctx.moveTo(px + i * 8 + 2, py + size + 2.5);
        ctx.lineTo(px + i * 8 - 3, py + size + 8);
      }
      ctx.stroke();
      ctx.fillStyle = INK;
      ctx.beginPath();
      ctx.arc(px, py, 2.4, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  drawLoads(ctx) {
    const { ox, oy, w, h } = this.layout;
    for (const L of this.state.loads) {
      this.drawArrow(ctx, ox + L.x * w, oy + L.y * h, L.dx, L.dy, false);
    }
  }

  drawArrow(ctx, x, y, dx, dy, dashed) {
    const tx = x + dx * ARROW_LEN, ty = y + dy * ARROW_LEN;
    ctx.strokeStyle = RED;
    ctx.fillStyle = RED;
    ctx.lineWidth = 2;
    if (dashed) ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(tx - dx * 8, ty - dy * 8);
    ctx.stroke();
    ctx.setLineDash([]);
    // head
    const px = -dy, py = dx;
    ctx.beginPath();
    ctx.moveTo(tx, ty);
    ctx.lineTo(tx - dx * 9 + px * 3.6, ty - dy * 9 + py * 3.6);
    ctx.lineTo(tx - dx * 9 - px * 3.6, ty - dy * 9 - py * 3.6);
    ctx.closePath();
    ctx.fill();
    // application point
    ctx.fillStyle = INK;
    ctx.beginPath();
    ctx.arc(x, y, 2.4, 0, Math.PI * 2);
    ctx.fill();
  }

  drawDims(ctx) {
    const { ox, oy, w, h } = this.layout;
    const s = this.solver;
    ctx.strokeStyle = FAINT;
    ctx.fillStyle = FAINT;
    ctx.lineWidth = 1;
    ctx.font = "9px 'IBM Plex Mono', monospace";

    // top dimension
    const dy = oy - 15;
    ctx.beginPath();
    ctx.moveTo(ox + 0.5, oy - 3); ctx.lineTo(ox + 0.5, dy - 4);
    ctx.moveTo(ox + w - 0.5, oy - 3); ctx.lineTo(ox + w - 0.5, dy - 4);
    ctx.moveTo(ox, dy); ctx.lineTo(ox + w, dy);
    // architectural tick slashes
    ctx.moveTo(ox - 3, dy + 3); ctx.lineTo(ox + 3, dy - 3);
    ctx.moveTo(ox + w - 3, dy + 3); ctx.lineTo(ox + w + 3, dy - 3);
    ctx.stroke();
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.fillText(`${s.nelx} ELEMENTS`, ox + w / 2, dy - 3);

    // right dimension
    const dx = ox + w + 15;
    ctx.beginPath();
    ctx.moveTo(ox + w + 3, oy + 0.5); ctx.lineTo(dx + 4, oy + 0.5);
    ctx.moveTo(ox + w + 3, oy + h - 0.5); ctx.lineTo(dx + 4, oy + h - 0.5);
    ctx.moveTo(dx, oy); ctx.lineTo(dx, oy + h);
    ctx.moveTo(dx - 3, oy + 3); ctx.lineTo(dx + 3, oy - 3);
    ctx.moveTo(dx - 3, oy + h + 3); ctx.lineTo(dx + 3, oy + h - 3);
    ctx.stroke();
    ctx.save();
    ctx.translate(dx + 6, oy + h / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textBaseline = "top";
    ctx.fillText(`${s.nely}`, 0, 0);
    ctx.restore();
  }

  /* ---------- sparkline ---------- */

  bindSpark() {
    this.sparkHover = null;
    this.spark.addEventListener("pointermove", (ev) => {
      if (this.history.length < 2) return;
      const rect = this.spark.getBoundingClientRect();
      const f = clamp((ev.clientX - rect.left) / rect.width, 0, 1);
      this.sparkHover = Math.round(f * (this.history.length - 1));
      this.drawSpark();
    });
    this.spark.addEventListener("pointerleave", () => {
      this.sparkHover = null;
      this.drawSpark();
    });
  }

  drawSpark() {
    const ctx = this.sparkCtx;
    const rect = this.spark.getBoundingClientRect();
    const W = rect.width, H = 44;
    ctx.clearRect(0, 0, W, H);
    const hist = this.history;

    ctx.strokeStyle = "rgba(32, 38, 45, 0.18)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, H - 0.5);
    ctx.lineTo(W, H - 0.5);
    ctx.stroke();

    if (hist.length < 2) {
      this.els["spark-readout"].textContent = "";
      return;
    }
    let lo = Infinity, hi = -Infinity;
    for (const c of hist) {
      const v = Math.log(Math.max(c, 1e-12));
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    if (hi - lo < 1e-9) hi = lo + 1e-9;
    const px = (i) => (i / (hist.length - 1)) * (W - 6) + 2;
    const py = (c) => 4 + (1 - (Math.log(Math.max(c, 1e-12)) - lo) / (hi - lo)) * (H - 12);

    ctx.strokeStyle = BLUE;
    ctx.lineWidth = 1.5;
    ctx.lineJoin = "round";
    ctx.beginPath();
    for (let i = 0; i < hist.length; i++) {
      if (i === 0) ctx.moveTo(px(i), py(hist[i]));
      else ctx.lineTo(px(i), py(hist[i]));
    }
    ctx.stroke();

    const hover = this.sparkHover;
    const iShow = hover === null ? hist.length - 1 : Math.min(hover, hist.length - 1);
    if (hover !== null) {
      ctx.strokeStyle = "rgba(43, 90, 166, 0.35)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(px(iShow) + 0.5, 2);
      ctx.lineTo(px(iShow) + 0.5, H - 2);
      ctx.stroke();
    }
    ctx.fillStyle = BLUE;
    ctx.beginPath();
    ctx.arc(px(iShow), py(hist[iShow]), 2.4, 0, Math.PI * 2);
    ctx.fill();

    this.els["spark-readout"].textContent =
      hover === null ? "" : ` — it ${iShow + 1} · C ${fmtC(hist[iShow])}`;
  }
}

function distToSegment(px, py, x0, y0, x1, y1) {
  const dx = x1 - x0, dy = y1 - y0;
  const len2 = dx * dx + dy * dy;
  const t = len2 === 0 ? 0 : clamp(((px - x0) * dx + (py - y0) * dy) / len2, 0, 1);
  return Math.hypot(px - (x0 + t * dx), py - (y0 + t * dy));
}

window.topoptApp = new App();
