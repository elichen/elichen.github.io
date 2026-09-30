// Example drawings, as flat [x0, y0, x1, y1, ...] arrays centred on the
// origin with a radius of about 3 plate units.

const TAU = Math.PI * 2;

function param(f, n = 240) {
  const out = [];
  for (let i = 0; i < n; i++) {
    const [x, y] = f((TAU * i) / n);
    out.push(x, y);
  }
  return out;
}

function fit(pts, radius = 3) {
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (let i = 0; i < pts.length; i += 2) {
    x0 = Math.min(x0, pts[i]); x1 = Math.max(x1, pts[i]);
    y0 = Math.min(y0, pts[i + 1]); y1 = Math.max(y1, pts[i + 1]);
  }
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2, s = (2 * radius) / Math.max(x1 - x0, y1 - y0);
  return pts.map((v, i) => (i % 2 ? (v - cy) * s : (v - cx) * s));
}

function polygon(corners, n = 240, round = 3) {
  const seg = corners.map((p, i) => Math.hypot(corners[(i + 1) % corners.length][0] - p[0], corners[(i + 1) % corners.length][1] - p[1]));
  const total = seg.reduce((a, b) => a + b, 0);
  let pts = [];
  for (let k = 0; k < n; k++) {
    let s = (total * k) / n, i = 0;
    while (s > seg[i]) { s -= seg[i]; i++; }
    const a = corners[i], b = corners[(i + 1) % corners.length], w = s / seg[i];
    pts.push([a[0] + (b[0] - a[0]) * w, a[1] + (b[1] - a[1]) * w]);
  }
  for (let r = 0; r < round; r++) {
    pts = pts.map((p, i) => {
      const q = pts[(i - 1 + n) % n], s = pts[(i + 1) % n];
      return [(q[0] + 2 * p[0] + s[0]) / 4, (q[1] + 2 * p[1] + s[1]) / 4];
    });
  }
  return pts.flat();
}

export const SHAPES = {
  heart: {
    label: 'Heart',
    points: fit(param((t) => [16 * Math.sin(t) ** 3, 13 * Math.cos(t) - 5 * Math.cos(2 * t) - 2 * Math.cos(3 * t) - Math.cos(4 * t)])),
  },
  star: {
    label: 'Star',
    points: fit(polygon(Array.from({ length: 10 }, (_, k) => {
      const r = k % 2 ? 0.42 : 1, a = Math.PI / 2 + (k * Math.PI) / 5;
      return [r * Math.cos(a), r * Math.sin(a)];
    }))),
  },
  fish: {
    label: 'Fish',
    points: fit(param((t) => [Math.cos(t) - Math.sin(t) ** 2 / Math.SQRT2, Math.cos(t) * Math.sin(t)])),
  },
  moon: {
    label: 'Crescent moon',
    points: fit(polygon([
      ...Array.from({ length: 40 }, (_, i) => { const a = -2.3 + (4.6 * i) / 39; return [Math.cos(a), Math.sin(a)]; }),
      ...Array.from({ length: 40 }, (_, i) => { const a = 2.0 - (4.0 * i) / 39; return [0.5 + 0.78 * Math.cos(a), 0.92 * Math.sin(a)]; }),
    ], 240, 2)),
  },
  cloud: {
    label: 'Cloud',
    points: fit(param((t) => {
      const r = 1 + 0.13 * Math.abs(Math.sin(2.5 * t));
      return [1.5 * r * Math.cos(t), 0.85 * r * Math.sin(t)];
    })),
  },
  bolt: {
    label: 'Lightning bolt (open stroke)',
    open: true,
    points: fit([0.35, 1, -0.25, 0.05, 0.2, 0.05, -0.35, -1]),
  },
};

// SVG path for a button icon, fitted into a 24 x 24 box.
export function iconPath(pts, open = false) {
  const p = fit(pts, 9);
  let d = '';
  for (let i = 0; i < p.length; i += 2) d += `${i ? 'L' : 'M'}${(12 + p[i]).toFixed(2)} ${(12 - p[i + 1]).toFixed(2)}`;
  return open ? d : d + 'Z';
}

// Densify an open polyline so the out-and-back path has even samples.
export function densify(pts, n = 160) {
  const seg = [];
  for (let i = 0; i + 3 < pts.length; i += 2) seg.push(Math.hypot(pts[i + 2] - pts[i], pts[i + 3] - pts[i + 1]));
  const total = seg.reduce((a, b) => a + b, 0), out = [];
  for (let k = 0; k < n; k++) {
    let s = (total * k) / (n - 1), i = 0;
    while (i < seg.length - 1 && s > seg[i]) { s -= seg[i]; i++; }
    const w = seg[i] ? Math.min(1, s / seg[i]) : 0;
    out.push(pts[2 * i] + (pts[2 * i + 2] - pts[2 * i]) * w, pts[2 * i + 1] + (pts[2 * i + 3] - pts[2 * i + 1]) * w);
  }
  return out;
}

// An open stroke becomes the closed path that runs along it and back.
export function outAndBack(pts) {
  const out = pts.slice();
  for (let i = pts.length - 4; i >= 2; i -= 2) out.push(pts[i], pts[i + 1]);
  return out;
}
