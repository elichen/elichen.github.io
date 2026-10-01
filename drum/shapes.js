// Drum outlines: example shapes, and the clean-up a freehand drawing gets
// before it becomes a drum (the same as the training doodles in
// drum/train/shapes.py: resample, smooth, resample to 96 points).

const TAU = Math.PI * 2;
export const N_OUT = 96;

export function resample(p, n) {
  const m = p.length / 2;
  const cum = new Float64Array(m + 1);
  for (let i = 0; i < m; i++) {
    const j = (i + 1) % m;
    cum[i + 1] = cum[i] + Math.hypot(p[2 * j] - p[2 * i], p[2 * j + 1] - p[2 * i + 1]);
  }
  const out = new Float64Array(2 * n);
  let k = 0;
  for (let i = 0; i < n; i++) {
    const s = (cum[m] * i) / n;
    while (k < m - 1 && cum[k + 1] <= s) k++;
    const w = cum[k + 1] > cum[k] ? (s - cum[k]) / (cum[k + 1] - cum[k]) : 0;
    const j = (k + 1) % m;
    out[2 * i] = p[2 * k] + (p[2 * j] - p[2 * k]) * w;
    out[2 * i + 1] = p[2 * k + 1] + (p[2 * j + 1] - p[2 * k + 1]) * w;
  }
  return out;
}

function smooth(p, passes) {
  let q = Float64Array.from(p);
  const n = q.length / 2;
  for (let r = 0; r < passes; r++) {
    const s = new Float64Array(q.length);
    for (let i = 0; i < n; i++) {
      const a = (i - 1 + n) % n, b = (i + 1) % n;
      s[2 * i] = (q[2 * a] + 2 * q[2 * i] + q[2 * b]) / 4;
      s[2 * i + 1] = (q[2 * a + 1] + 2 * q[2 * i + 1] + q[2 * b + 1]) / 4;
    }
    q = s;
  }
  return q;
}

// A freehand loop, cleaned up like a training doodle.
export function fromStroke(pts) {
  return resample(smooth(resample(pts, 192), 3), N_OUT);
}

function area(p) {
  let a = 0;
  for (let i = 0, n = p.length / 2; i < n; i++) {
    const j = (i + 1) % n;
    a += p[2 * i] * p[2 * j + 1] - p[2 * j] * p[2 * i + 1];
  }
  return a / 2;
}

function perimeter(p) {
  let s = 0;
  for (let i = 0, n = p.length / 2; i < n; i++) {
    const j = (i + 1) % n;
    s += Math.hypot(p[2 * j] - p[2 * i], p[2 * j + 1] - p[2 * i + 1]);
  }
  return s;
}

function crosses(p) {
  const n = p.length / 2;
  for (let i = 0; i < n; i++) {
    const ax = p[2 * i], ay = p[2 * i + 1], bx = p[2 * ((i + 1) % n)], by = p[2 * ((i + 1) % n) + 1];
    for (let j = i + 2; j < n; j++) {
      if (i === 0 && j === n - 1) continue;
      const cx = p[2 * j], cy = p[2 * j + 1], dx = p[2 * ((j + 1) % n)], dy = p[2 * ((j + 1) % n) + 1];
      const den = (bx - ax) * (dy - cy) - (by - ay) * (dx - cx);
      if (den === 0) continue;
      const t = ((cx - ax) * (dy - cy) - (cy - ay) * (dx - cx)) / den;
      const u = ((cx - ax) * (by - ay) - (cy - ay) * (bx - ax)) / den;
      if (t > 1e-9 && t < 1 - 1e-9 && u > 1e-9 && u < 1 - 1e-9) return true;
    }
  }
  return false;
}

// Why an outline can't be a drum, or null if it can.
export function problem(p) {
  if (crosses(p)) return 'The outline crosses itself. Draw a loop that doesn’t cross over.';
  const iso = (4 * Math.PI * Math.abs(area(p))) / perimeter(p) ** 2;
  if (!(iso > 0.18)) return 'That shape is too thin to ring. Try something rounder.';
  return null;
}

function param(f, n = 240) {
  const out = [];
  for (let i = 0; i < n; i++) out.push(...f((TAU * i) / n));
  return out;
}

function polygon(corners) {
  return resample(Float64Array.from(corners.flat()), N_OUT);
}

export const EXAMPLES = {
  circle: { label: 'Circle', points: resample(Float64Array.from(param((t) => [Math.cos(t), Math.sin(t)])), N_OUT) },
  square: { label: 'Square', points: polygon([[-1, -1], [1, -1], [1, 1], [-1, 1]]) },
  triangle: { label: 'Triangle', points: polygon([[0, 1.2], [-1.04, -0.6], [1.04, -0.6]]) },
  star: {
    label: 'Star',
    points: polygon(Array.from({ length: 10 }, (_, k) => {
      const r = k % 2 ? 0.45 : 1, a = Math.PI / 2 + (k * Math.PI) / 5;
      return [r * Math.cos(a), r * Math.sin(a)];
    })),
  },
  heart: {
    label: 'Heart',
    points: resample(smooth(resample(Float64Array.from(param((t) => [16 * Math.sin(t) ** 3,
      13 * Math.cos(t) - 5 * Math.cos(2 * t) - 2 * Math.cos(3 * t) - Math.cos(4 * t)])), 192), 1), N_OUT),
  },
  ell: { label: 'L shape', points: polygon([[-1, -1], [1, -1], [1, -0.2], [-0.2, -0.2], [-0.2, 1], [-1, 1]]) },
  fish: {
    label: 'Fish',
    points: resample(smooth(resample(Float64Array.from(param((t) => [Math.cos(t) - Math.sin(t) ** 2 / Math.SQRT2,
      Math.cos(t) * Math.sin(t)])), 192), 2), N_OUT),
  },
};

// The Gordon–Webb–Wolpert drums (Driscoll 1997, Moler 2012): seven
// half-squares each, glued two different ways.
export const GWW = [
  Float64Array.from([0, 0, 0, 1, 2, 3, 2, 2, 3, 2, 2, 1, 1, 1, 1, 0]),
  Float64Array.from([1, 0, 0, 1, 0, 2, 2, 2, 2, 3, 3, 2, 2, 1, 1, 1]),
];

export function iconPath(p) {
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (let i = 0; i < p.length; i += 2) {
    x0 = Math.min(x0, p[i]); x1 = Math.max(x1, p[i]); y0 = Math.min(y0, p[i + 1]); y1 = Math.max(y1, p[i + 1]);
  }
  const s = 18 / Math.max(x1 - x0, y1 - y0), cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  let d = '';
  for (let i = 0; i < p.length; i += 2) d += `${i ? 'L' : 'M'}${(12 + (p[i] - cx) * s).toFixed(2)} ${(12 - (p[i + 1] - cy) * s).toFixed(2)}`;
  return d + 'Z';
}
