/** Pointer and keyboard contour drawing, with a consistent normalized square. */
export class PathSketch {
  constructor(canvas, onChange, onDraft = () => {}) {
    this.canvas = canvas;
    this.context = canvas.getContext("2d");
    this.onChange = onChange;
    this.onDraft = onDraft;
    this.path = [];
    this.draft = [];
    this.pointer = null;
    this.cursor = { x: 0, y: 0 };
    this.drag = null;
    this.width = 200;
    this.height = 190;
    this.observer = new ResizeObserver(() => this.resize());
    this.observer.observe(canvas);
    canvas.addEventListener("pointerdown", (e) => {
      if (e.button !== 0 && e.pointerType !== "touch") return;
      e.preventDefault();
      canvas.focus({ preventScroll: true });
      const point = this.point(e);
      this.drag = {
        id: e.pointerId,
        start: point,
        moved: false,
        prior: this.draft.slice(),
      };
      this.draft.push(point);
      this.cursor = point;
      canvas.setPointerCapture(e.pointerId);
      this.draw();
      this.onDraft(this.draft.length);
    });
    canvas.addEventListener("pointermove", (e) => {
      this.pointer = this.point(e);
      if (this.drag && e.pointerId === this.drag.id) {
        const last = this.draft.at(-1);
        if (
          Math.hypot(last.x - this.pointer.x, last.y - this.pointer.y) > 0.025
        ) {
          if (
            !this.drag.moved &&
            Math.hypot(
              this.pointer.x - this.drag.start.x,
              this.pointer.y - this.drag.start.y,
            ) > 0.07
          ) {
            this.draft = [this.drag.start];
            this.drag.moved = true;
          }
          this.draft.push(this.pointer);
          this.cursor = this.pointer;
          this.onDraft(this.draft.length);
        }
      }
      this.draw();
    });
    canvas.addEventListener("pointerup", (e) => {
      if (!this.drag || this.drag.id !== e.pointerId) return;
      if (canvas.hasPointerCapture(e.pointerId))
        canvas.releasePointerCapture(e.pointerId);
      const moved = this.drag.moved;
      this.drag = null;
      if (moved && this.draft.length >= 6) this.finish();
      else this.draw();
    });
    canvas.addEventListener("pointercancel", (e) => {
      if (!this.drag) return;
      this.draft = this.drag.prior;
      this.drag = null;
      if (canvas.hasPointerCapture(e.pointerId))
        canvas.releasePointerCapture(e.pointerId);
      this.draw();
      this.onDraft(this.draft.length);
    });
    canvas.addEventListener("pointerleave", () => {
      if (!this.drag) this.pointer = null;
      this.draw();
    });
    canvas.addEventListener("keydown", (e) => {
      if (e.key.startsWith("Arrow")) {
        e.preventDefault();
        const step = e.shiftKey ? 0.15 : 0.06;
        if (e.key === "ArrowLeft") this.cursor.x -= step;
        if (e.key === "ArrowRight") this.cursor.x += step;
        if (e.key === "ArrowUp") this.cursor.y += step;
        if (e.key === "ArrowDown") this.cursor.y -= step;
        this.cursor.x = Math.max(-0.98, Math.min(0.98, this.cursor.x));
        this.cursor.y = Math.max(-0.98, Math.min(0.98, this.cursor.y));
        this.pointer = { ...this.cursor };
        this.draw();
      } else if (e.code === "Space") {
        e.preventDefault();
        e.stopPropagation();
        this.draft.push({ ...this.cursor });
        this.draw();
        this.onDraft(this.draft.length);
      } else if (e.key === "Enter") {
        e.preventDefault();
        e.stopPropagation();
        this.finish();
      } else if (e.key === "Backspace") {
        e.preventDefault();
        this.draft.pop();
        this.draw();
        this.onDraft(this.draft.length);
      } else if (e.key === "Escape") {
        this.draft = [];
        this.draw();
        this.onDraft(0);
      }
    });
    this.resize();
  }
  resize() {
    const bounds = this.canvas.getBoundingClientRect();
    this.width = bounds.width;
    this.height = bounds.height;
    const dpr = Math.min(devicePixelRatio || 1, 2);
    this.canvas.width = Math.round(this.width * dpr);
    this.canvas.height = Math.round(this.height * dpr);
    this.context.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.scale = Math.min(this.width, this.height) * 0.4;
    this.draw();
  }
  point(event) {
    const b = this.canvas.getBoundingClientRect();
    return {
      x: Math.max(
        -1,
        Math.min(1, (event.clientX - b.left - this.width / 2) / this.scale),
      ),
      y: Math.max(
        -1,
        Math.min(1, (this.height / 2 - event.clientY + b.top) / this.scale),
      ),
    };
  }
  setPath(points) {
    this.path = points.map((p) => ({ ...p }));
    this.draft = [];
    this.draw();
    this.onDraft(0);
  }
  clear() {
    this.path = [];
    this.draft = [];
    this.draw();
    this.onDraft(0);
  }
  finish() {
    if (this.draft.length < 3) {
      this.onDraft(
        this.draft.length,
        "Add at least three points to close a loop.",
      );
      return false;
    }
    const path = this.draft.slice();
    let length = 0;
    for (let i = 1; i < path.length; i++)
      length += Math.hypot(
        path[i].x - path[i - 1].x,
        path[i].y - path[i - 1].y,
      );
    if (length < 0.25) {
      this.onDraft(
        path.length,
        "Make a larger gesture before closing the loop.",
      );
      return false;
    }
    this.path = path;
    this.draft = [];
    this.draw();
    this.onChange(path);
    this.onDraft(0);
    return true;
  }
  draw() {
    const ctx = this.context,
      w = this.width,
      h = this.height,
      s = this.scale;
    if (!s) return;
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = "#8aa08b35";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(w / 2, 17);
    ctx.lineTo(w / 2, h - 17);
    ctx.moveTo(17, h / 2);
    ctx.lineTo(w - 17, h / 2);
    ctx.stroke();
    const trace = (points, close) => {
      ctx.beginPath();
      points.forEach((p, i) => {
        const x = w / 2 + p.x * s,
          y = h / 2 - p.y * s;
        if (!i) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      if (close) ctx.closePath();
    };
    if (this.path.length) {
      trace(this.path, true);
      ctx.fillStyle = this.draft.length ? "#345c9310" : "#345c9317";
      ctx.fill();
      ctx.strokeStyle = this.draft.length ? "#567a8970" : "#355b7c";
      ctx.lineWidth = this.draft.length ? 1 : 2;
      ctx.lineJoin = "round";
      ctx.stroke();
      if (!this.draft.length) {
        const start = this.path[0];
        ctx.beginPath();
        ctx.arc(w / 2 + start.x * s, h / 2 - start.y * s, 3, 0, Math.PI * 2);
        ctx.fillStyle = "#cc4d29";
        ctx.fill();
      }
    }
    if (this.draft.length) {
      trace(this.draft, false);
      ctx.strokeStyle = "#cc4d29";
      ctx.lineWidth = 2;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.stroke();
      const first = this.draft[0],
        last = this.draft.at(-1);
      ctx.setLineDash([3, 4]);
      ctx.strokeStyle = "#cc4d2955";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(w / 2 + last.x * s, h / 2 - last.y * s);
      ctx.lineTo(w / 2 + first.x * s, h / 2 - first.y * s);
      ctx.stroke();
      ctx.setLineDash([]);
      if (!this.drag?.moved)
        for (const p of this.draft) {
          ctx.beginPath();
          ctx.arc(w / 2 + p.x * s, h / 2 - p.y * s, 2.3, 0, Math.PI * 2);
          ctx.fillStyle = "#cc4d29";
          ctx.fill();
        }
    }
    if (this.pointer) {
      const x = w / 2 + this.pointer.x * s,
        y = h / 2 - this.pointer.y * s;
      ctx.strokeStyle = "#cc4d2988";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x - 5, y);
      ctx.lineTo(x + 5, y);
      ctx.moveTo(x, y - 5);
      ctx.lineTo(x, y + 5);
      ctx.stroke();
    }
    if (!this.path.length && !this.draft.length) {
      ctx.font = "11px Manrope, sans-serif";
      ctx.textAlign = "center";
      ctx.fillStyle = "#829576";
      ctx.fillText("A loop begins here.", w / 2, h / 2 - 14);
    }
  }
}
