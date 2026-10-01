// Turn a visitor's pen strokes into the network's input, the same way the
// training lines were prepared: level the line, find its baseline and
// x-height, measure everything in x-heights, and resample each stroke at a
// fixed step along its length.
(function (root) {
  "use strict";

  function quantile(sorted, q) {
    const i = (sorted.length - 1) * q, lo = Math.floor(i), hi = Math.ceil(i);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo);
  }

  // points spaced `step` apart along each stroke (strokes: arrays of [x, y])
  function resampleStroke(stroke, step, keepEnds) {
    if (stroke.length < 2) return stroke.slice();
    const cum = [0];
    for (let i = 1; i < stroke.length; i++) {
      cum.push(cum[i - 1] + Math.hypot(stroke[i][0] - stroke[i - 1][0], stroke[i][1] - stroke[i - 1][1]));
    }
    const total = cum[cum.length - 1];
    if (total < 1e-9) return [stroke[0]];
    let n;
    if (keepEnds) {
      if (total < step * 0.5) return [stroke[0], stroke[stroke.length - 1]];
      n = Math.max(2, Math.round(total / step) + 1);
    } else {
      n = Math.floor(total / step) + 1;
    }
    const out = [];
    let j = 0;
    for (let k = 0; k < n; k++) {
      const u = keepEnds ? (total * k) / (n - 1) : step * k;
      while (j < cum.length - 2 && cum[j + 1] < u) j++;
      const seg = cum[j + 1] - cum[j], t = seg > 0 ? (u - cum[j]) / seg : 0;
      out.push([stroke[j][0] + (stroke[j + 1][0] - stroke[j][0]) * t, stroke[j][1] + (stroke[j + 1][1] - stroke[j][1]) * t]);
    }
    return out;
  }

  // light smoothing for mouse jitter; endpoints stay put
  function smooth(stroke) {
    if (stroke.length < 3) return stroke;
    return stroke.map((p, i) => {
      if (i === 0 || i === stroke.length - 1) return p;
      const a = stroke[i - 1], b = stroke[i + 1];
      return [(a[0] + 2 * p[0] + b[0]) / 4, (a[1] + 2 * p[1] + b[1]) / 4];
    });
  }

  // Estimate tilt, baseline and x-height of a written prompt line.
  function measure(strokes, calib) {
    let ymin = Infinity, ymax = -Infinity;
    for (const s of strokes) for (const p of s) { ymin = Math.min(ymin, p[1]); ymax = Math.max(ymax, p[1]); }
    const h = Math.max(ymax - ymin, 1e-6);
    const dense = [];
    for (const s of strokes) for (const p of resampleStroke(s, h / 150, false)) dense.push(p);
    let mx = 0, my = 0;
    for (const p of dense) { mx += p[0]; my += p[1]; }
    mx /= dense.length; my /= dense.length;
    let sxy = 0, sxx = 0;
    for (const p of dense) { sxy += (p[0] - mx) * (p[1] - my); sxx += (p[0] - mx) ** 2; }
    const slope = sxx > 0 ? sxy / sxx : 0;
    const level = dense.map((p) => p[1] - slope * (p[0] - mx)).sort((a, b) => a - b);
    const q = calib.q.map((qq) => quantile(level, qq));
    const med = q[calib.q.indexOf(0.5)];
    let xh = 0, base = med;
    q.forEach((v, i) => { xh += (v - med) * calib.w[i]; base += (v - med) * calib.v[i]; });
    return { tilt: slope - calib.slope, xh, base, cx: mx };
  }

  // strokes (canvas px) -> { pts: [[x, y]...], lift: [0/1...] } in x-heights, baseline at y = 0
  function normalize(strokes, calib, spacing) {
    const kept = strokes.filter((s) => s.length > 0);
    if (!kept.length) return null;
    const m = measure(kept, calib);
    if (!(m.xh > 1)) return null;
    // The prompt itself leans a little in the training data (its letters, not the writers),
    // so remove only the visitor's extra tilt, as a shear about the line's centre.
    let xmin = Infinity;
    const level = kept.map((s) => s.map(([x, y]) => {
      xmin = Math.min(xmin, x);
      return [x, y - m.tilt * (x - m.cx)];
    }));
    const pts = [], lift = [];
    for (const s of level) {
      const unit = smooth(s).map(([x, y]) => [(x - xmin) / m.xh, (y - m.base) / m.xh]);
      const r = resampleStroke(unit, spacing, true);
      r.forEach((p, i) => { pts.push(p); lift.push(i === r.length - 1 ? 1 : 0); });
    }
    return { pts, lift, xh: m.xh, tilt: m.tilt };
  }

  // normalized points -> network input moves [dx, dy, pen] (pen: lifted before this point)
  function toMoves(line, spacing) {
    const x = new Float32Array(line.pts.length * 3);
    for (let i = 0; i < line.pts.length; i++) {
      const p = line.pts[i], q = i ? line.pts[i - 1] : p;
      x[3 * i] = (p[0] - q[0]) / spacing;
      x[3 * i + 1] = (p[1] - q[1]) / spacing;
      x[3 * i + 2] = i === 0 ? 1 : line.lift[i - 1];
    }
    return x;
  }

  const api = { normalize, toMoves, measure, resampleStroke };
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.Ink = api;
})(typeof self !== "undefined" ? self : this);
