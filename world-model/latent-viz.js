// Interactive 2D visualization of the language-free Trajectory JEPA behavior manifold
// and the single orthogonal Procrustes rotation R.
// Plots 500+ held-out robot trajectory clips colored by behavior family with subtle cluster
// labels, and animates the active English sentence rotating via R from its unrotated identity
// position (amber) straight into the matching robot behavior cluster (cyan), tethered to its
// nearest held-out trajectory clip!

const SKILL_HUES = [
  354, // throw (coral-red)
  270, // rotate_base (violet)
  175, // place (cyan-teal)
  32,  // push (amber-orange)
  142, // lift_hold (emerald)
  52,  // lift_drop (gold)
  205, // wave (sky blue)
  232, // reach_up (indigo)
  15,  // bow (terracotta)
  315, // circle (magenta)
  190, // sweep (aqua)
  95,  // clap_gripper (lime)
  335, // poke_cube (rose)
  250, // point (periwinkle)
  160, // nod (mint)
  290, // dance (orchid)
  215, // home (slate-blue)
];

export class LatentSpaceViz {
  constructor(canvas, arenaPol) {
    this.canvas = canvas;
    this.pol = arenaPol;
    this.state = null;
    this.animStart = 0;
    this.animDur = 540;
    this._raf = null;

    // Robust 2D bounds across held-out cloud
    const M = arenaPol.cloudCount;
    const xs = [], ys = [];
    for (let i = 0; i < M; i++) {
      xs.push(arenaPol.cloud2D[i * 2]);
      ys.push(arenaPol.cloud2D[i * 2 + 1]);
    }
    xs.sort((a, b) => a - b);
    ys.sort((a, b) => a - b);
    const xLo = xs[Math.floor(M * 0.01)], xHi = xs[Math.floor(M * 0.99)];
    const yLo = ys[Math.floor(M * 0.01)], yHi = ys[Math.floor(M * 0.99)];
    this.rangeX = Math.max(Math.abs(xLo), Math.abs(xHi), 0.12) * 1.24;
    this.rangeY = Math.max(Math.abs(yLo), Math.abs(yHi), 0.12) * 1.24;

    this.skillColors = SKILL_HUES.map(h => `hsla(${h}, 76%, 62%, 0.62)`);
    this.skillColorsSolid = SKILL_HUES.map(h => `hsla(${h}, 84%, 72%, 0.95)`);

    this._onResize = () => this.draw(1);
    if (typeof ResizeObserver !== 'undefined') {
      new ResizeObserver(this._onResize).observe(canvas);
    }
    this.draw(1);
  }

  setRotation(rotResult) {
    this.state = rotResult;
    this.animStart = performance.now();
    if (this._raf) cancelAnimationFrame(this._raf);
    const tick = (now) => {
      const t = Math.min(1, (now - this.animStart) / this.animDur);
      const ease = 1 - Math.pow(1 - t, 3);
      this.draw(ease);
      if (t < 1) this._raf = requestAnimationFrame(tick);
      else this._raf = null;
    };
    this._raf = requestAnimationFrame(tick);
  }

  draw(t = 1) {
    const canvas = this.canvas;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const cssW = canvas.clientWidth || 360;
    const cssH = canvas.clientHeight || 240;
    const W = Math.round(cssW * dpr), H = Math.round(cssH * dpr);
    if (canvas.width !== W || canvas.height !== H) {
      canvas.width = W;
      canvas.height = H;
    }

    const ctx = canvas.getContext('2d');
    ctx.save();
    ctx.scale(dpr, dpr);

    ctx.fillStyle = '#141923';
    ctx.fillRect(0, 0, cssW, cssH);

    const pad = 16;
    const pw = cssW - 2 * pad, ph = cssH - 2 * pad;
    const cx0 = cssW / 2, cy0 = cssH / 2;
    const sx = (pw / 2) / this.rangeX;
    const sy = (ph / 2) / this.rangeY;
    const clamp = (v, r) => Math.max(-r * 0.96, Math.min(r * 0.96, v));
    const toPx = (u, v) => [cx0 + clamp(u, this.rangeX) * sx, cy0 - clamp(v, this.rangeY) * sy];

    // Subtle axes & rings
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad, cy0); ctx.lineTo(cssW - pad, cy0);
    ctx.moveTo(cx0, pad); ctx.lineTo(cx0, cssH - pad);
    ctx.stroke();

    // Draw held-out trajectory clips colored by skill family
    const M = this.pol.cloudCount;
    const c2d = this.pol.cloud2D;
    const csk = this.pol.cloudSkill;
    const activeSkill = this.state?.topSkill;

    for (let i = 0; i < M; i++) {
      const [px, py] = toPx(c2d[i * 2], c2d[i * 2 + 1]);
      const sid = csk[i];
      const isActive = activeSkill && this.pol.skills[sid] === activeSkill;
      ctx.fillStyle = isActive ? this.skillColorsSolid[sid % SKILL_HUES.length] : this.skillColors[sid % SKILL_HUES.length];
      ctx.beginPath();
      ctx.arc(px, py, isActive ? 2.6 : 1.9, 0, Math.PI * 2);
      ctx.fill();
    }

    // Subtle cluster labels (prioritizing well-spaced centroids + always highlighting the active cluster!)
    ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.textAlign = 'center';
    const placedBoxes = [];
    const centroids = [...(this.pol.centroids2D || [])];
    // Sort so active cluster is drawn first & never hidden
    centroids.sort((a, b) => (this.pol.skills[b.id] === activeSkill ? 1 : 0) - (this.pol.skills[a.id] === activeSkill ? 1 : 0));
    for (const c of centroids) {
      const [px, py] = toPx(c.x, c.y);
      const isAct = this.pol.skills[c.id] === activeSkill;
      const bw = c.name.length * 6.2 + 8, bh = 13;
      const overlaps = !isAct && placedBoxes.some(b => Math.abs(b[0] - px) < (b[2] + bw) * 0.46 && Math.abs(b[1] - py) < 12);
      if (overlaps) continue;
      placedBoxes.push([px, py, bw]);
      ctx.fillStyle = isAct ? 'rgba(20, 25, 35, 0.90)' : 'rgba(20, 25, 35, 0.62)';
      ctx.fillRect(px - bw / 2, py - 9, bw, bh);
      ctx.fillStyle = isAct ? '#ffffff' : 'rgba(236, 239, 244, 0.56)';
      ctx.fillText(c.name, px, py + 1);
    }

    if (this.state) {
      const s = this.state;
      const [x0, y0] = toPx(s.pre2D[0], s.pre2D[1]);
      const [x1, y1] = toPx(s.post2D[0], s.post2D[1]);
      const [xn, yn] = toPx(s.nearest2D[0], s.nearest2D[1]);

      const mx = (x0 + x1) / 2, my = (y0 + y1) / 2;
      const dx = x1 - x0, dy = y1 - y0;
      const dist = Math.hypot(dx, dy);
      const bulge = Math.min(34, dist * 0.25);
      const qx = mx - (dy / (dist || 1)) * bulge;
      const qy = my + (dx / (dist || 1)) * bulge;

      const curX = (1 - t) * (1 - t) * x0 + 2 * (1 - t) * t * qx + t * t * x1;
      const curY = (1 - t) * (1 - t) * y0 + 2 * (1 - t) * t * qy + t * t * y1;

      const isGood = s.nKey !== 'shuffled' && s.nKey !== 'untrained';
      const postColor = isGood ? '#56d4c8' : '#f59e0b';

      // Curved rotation arc
      ctx.strokeStyle = isGood ? 'rgba(86, 212, 200, 0.55)' : 'rgba(245, 158, 11, 0.55)';
      ctx.lineWidth = 1.7;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.quadraticCurveTo(qx, qy, curX, curY);
      ctx.stroke();
      ctx.setLineDash([]);

      // Nearest clip tether
      if (t > 0.72) {
        const a = (t - 0.72) / 0.28;
        ctx.strokeStyle = `rgba(245, 66, 84, ${(0.65 * a).toFixed(2)})`;
        ctx.lineWidth = 1.3;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(xn, yn);
        ctx.stroke();
        ctx.fillStyle = `rgba(245, 66, 84, ${(0.95 * a).toFixed(2)})`;
        ctx.beginPath();
        ctx.arc(xn, yn, 3.8, 0, Math.PI * 2);
        ctx.fill();
      }

      // Pre-rotation dot (amber hollow ring)
      ctx.fillStyle = '#141923';
      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x0, y0, 4.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      // Rotated target dot
      ctx.fillStyle = isGood ? 'rgba(86, 212, 200, 0.25)' : 'rgba(245, 158, 11, 0.25)';
      ctx.beginPath();
      ctx.arc(curX, curY, 10.5, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = postColor;
      ctx.strokeStyle = '#141923';
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      ctx.arc(curX, curY, 5.4, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      if (dist > 28) {
        const lx = 0.25 * x0 + 0.5 * qx + 0.25 * x1;
        const ly = 0.25 * y0 + 0.5 * qy + 0.25 * y1;
        ctx.fillStyle = 'rgba(20, 25, 35, 0.88)';
        ctx.fillRect(lx - 9, ly - 8, 18, 15);
        ctx.fillStyle = postColor;
        ctx.font = 'bold 11px ui-monospace, SFMono-Regular, Menlo, monospace';
        ctx.textAlign = 'center';
        ctx.fillText('R', lx, ly + 3);
      }
    }

    ctx.restore();
  }
}
