// Vibration modes of a drum: -Δu = λu inside the outline, u = 0 on the rim.
// A port of drum/train/fem.py (checked by drum/tools/golden_test.mjs):
// linear finite elements on a jittered-lattice Delaunay mesh of the drum
// scaled to unit area, solved by shift-invert Lanczos on a skyline Cholesky
// factorisation of the stiffness matrix.

import Delaunator from './vendor/delaunator.js';

export const H = 0.025;
export const K_MODES = 40;

export function polygonArea(p) {
  let a = 0;
  for (let i = 0, n = p.length / 2; i < n; i++) {
    const j = (i + 1) % n;
    a += p[2 * i] * p[2 * j + 1] - p[2 * j] * p[2 * i + 1];
  }
  return a / 2;
}

// Counter-clockwise, centroid at the origin, unit area. p: flat [x0, y0, ...].
export function normalise(p) {
  let q = Float64Array.from(p);
  const n = q.length / 2;
  let a = polygonArea(q);
  if (a < 0) {
    const r = new Float64Array(q.length);
    for (let i = 0; i < n; i++) { r[2 * i] = q[2 * (n - 1 - i)]; r[2 * i + 1] = q[2 * (n - 1 - i) + 1]; }
    q = r;
    a = -a;
  }
  let cx = 0, cy = 0;
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    const cr = q[2 * i] * q[2 * j + 1] - q[2 * j] * q[2 * i + 1];
    cx += (q[2 * i] + q[2 * j]) * cr;
    cy += (q[2 * i + 1] + q[2 * j + 1]) * cr;
  }
  cx /= 6 * a; cy /= 6 * a;
  const s = 1 / Math.sqrt(a);
  for (let i = 0; i < n; i++) { q[2 * i] = (q[2 * i] - cx) * s; q[2 * i + 1] = (q[2 * i + 1] - cy) * s; }
  return q;
}

function resampleRim(p, h) {
  const out = [];
  const n = p.length / 2;
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    const ax = p[2 * i], ay = p[2 * i + 1], bx = p[2 * j], by = p[2 * j + 1];
    const L = Math.hypot(bx - ax, by - ay);
    const k = Math.max(1, Math.ceil(L / h - 1e-9));
    for (let s = 0; s < k; s++) out.push(ax + (s / k) * (bx - ax), ay + (s / k) * (by - ay));
  }
  return out;
}

export function inside(p, x, y) {
  let c = 0;
  const n = p.length / 2;
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    const ax = p[2 * i], ay = p[2 * i + 1], bx = p[2 * j], by = p[2 * j + 1];
    if ((ay > y) !== (by > y)) {
      const xint = ((bx - ax) * (y - ay)) / (by - ay) + ax;
      if (x < xint) c++;
    }
  }
  return c % 2 === 1;
}

function distToRim(p, x, y) {
  let best = Infinity;
  const n = p.length / 2;
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    const ax = p[2 * i], ay = p[2 * i + 1], abx = p[2 * j] - ax, aby = p[2 * j + 1] - ay;
    let t = ((x - ax) * abx + (y - ay) * aby) / Math.max(abx * abx + aby * aby, 1e-18);
    t = Math.min(1, Math.max(0, t));
    const dx = x - (ax + t * abx), dy = y - (ay + t * aby);
    best = Math.min(best, dx * dx + dy * dy);
  }
  return Math.sqrt(best);
}

function lattice(p, h, jit = 0.12) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i < p.length; i += 2) {
    x0 = Math.min(x0, p[i]); x1 = Math.max(x1, p[i]);
    y0 = Math.min(y0, p[i + 1]); y1 = Math.max(y1, p[i + 1]);
  }
  const dy = (h * Math.sqrt(3)) / 2;
  const j0 = Math.floor(y0 / dy) - 1, j1 = Math.ceil(y1 / dy) + 1;
  const i0 = Math.floor(x0 / h) - 1, i1 = Math.ceil(x1 / h) + 1;
  const out = [];
  for (let I = i0; I <= i1; I++) {
    for (let J = j0; J <= j1; J++) {
      let s = (Math.imul(I, 73856093) ^ Math.imul(J, 19349663)) >>> 0;
      s = Math.imul(s ^ (s >>> 13), 1274126177) >>> 0;
      const u = ((s & 0xffff) / 65536) * 2 - 1, v = (((s >>> 16) & 0xffff) / 65536) * 2 - 1;
      const x = I * h + (J & 1) * h / 2 + jit * h * u;
      const y = J * dy + jit * h * v;
      if (inside(p, x, y) && distToRim(p, x, y) > 0.55 * h) out.push(x, y);
    }
  }
  return out;
}

// Mesh of a normalised outline: nodes (rim first), triangles, rim count.
export function mesh(p, h = H) {
  const rim = resampleRim(p, h);
  const nodes = Float64Array.from([...rim, ...lattice(p, h)]);
  const d = new Delaunator(nodes);
  const tri = [];
  for (let t = 0; t < d.triangles.length; t += 3) {
    const a = d.triangles[t], b = d.triangles[t + 1], c = d.triangles[t + 2];
    const cx = (nodes[2 * a] + nodes[2 * b] + nodes[2 * c]) / 3;
    const cy = (nodes[2 * a + 1] + nodes[2 * b + 1] + nodes[2 * c + 1]) / 3;
    if (!inside(p, cx, cy)) continue;
    const cr = (nodes[2 * b] - nodes[2 * a]) * (nodes[2 * c + 1] - nodes[2 * a + 1]) -
      (nodes[2 * b + 1] - nodes[2 * a + 1]) * (nodes[2 * c] - nodes[2 * a]);
    tri.push(...(cr < 0 ? [a, c, b] : [a, b, c]));
  }
  return { nodes, tri: Uint32Array.from(tri), nRim: rim.length / 2 };
}

// Structured mesh for polygons on the unit grid and its diagonals (the
// isospectral drums): every grid square of side 1/n split by both diagonals.
export function crisscrossMesh(p, n) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i < p.length; i += 2) {
    x0 = Math.min(x0, p[i]); x1 = Math.max(x1, p[i]);
    y0 = Math.min(y0, p[i + 1]); y1 = Math.max(y1, p[i + 1]);
  }
  x0 = Math.floor(x0); y0 = Math.floor(y0); x1 = Math.ceil(x1); y1 = Math.ceil(y1);
  const key = new Map(), xy = [], tri = [];
  const node = (x, y) => {
    const k = `${Math.round(x * 2 * n)},${Math.round(y * 2 * n)}`;
    if (!key.has(k)) { key.set(k, xy.length / 2); xy.push(x, y); }
    return key.get(k);
  };
  for (let i = x0 * n; i < x1 * n; i++) {
    for (let j = y0 * n; j < y1 * n; j++) {
      const xa = i / n, xb = (i + 1) / n, ya = j / n, yb = (j + 1) / n, xm = (xa + xb) / 2, ym = (ya + yb) / 2;
      const c = [[xa, ya], [xb, ya], [xb, yb], [xa, yb]];
      for (let k = 0; k < 4; k++) {
        const [ax, ay] = c[k], [bx, by] = c[(k + 1) % 4];
        if (inside(p, (ax + bx + xm) / 3, (ay + by + ym) / 3)) tri.push(node(ax, ay), node(bx, by), node(xm, ym));
      }
    }
  }
  const N = xy.length / 2, onRim = new Uint8Array(N);
  for (let i = 0; i < N; i++) onRim[i] = distToRim(p, xy[2 * i], xy[2 * i + 1]) < 1e-9 ? 1 : 0;
  const order = [];
  for (let i = 0; i < N; i++) if (onRim[i]) order.push(i);
  const nRim = order.length;
  for (let i = 0; i < N; i++) if (!onRim[i]) order.push(i);
  const remap = new Uint32Array(N);
  order.forEach((o, k) => { remap[o] = k; });
  const nodes = new Float64Array(2 * N);
  order.forEach((o, k) => { nodes[2 * k] = xy[2 * o]; nodes[2 * k + 1] = xy[2 * o + 1]; });
  return { nodes, tri: Uint32Array.from(tri, (t) => remap[t]), nRim };
}

// --- assembly --------------------------------------------------------------

// Stiffness and consistent mass matrices on the interior nodes, as maps from
// row to {col: value} (small and sparse).
function assemble(nodes, tri, nRim) {
  const n = nodes.length / 2 - nRim;
  const K = Array.from({ length: n }, () => new Map());
  const M = Array.from({ length: n }, () => new Map());
  const add = (A, i, j, v) => A[i].set(j, (A[i].get(j) || 0) + v);
  for (let t = 0; t < tri.length; t += 3) {
    const v = [tri[t], tri[t + 1], tri[t + 2]];
    const x = v.map((k) => nodes[2 * k]), y = v.map((k) => nodes[2 * k + 1]);
    const b = [y[1] - y[2], y[2] - y[0], y[0] - y[1]];
    const c = [x[2] - x[1], x[0] - x[2], x[1] - x[0]];
    const A = 0.5 * (b[0] * c[1] - b[1] * c[0]);
    for (let r = 0; r < 3; r++) {
      const i = v[r] - nRim;
      if (i < 0) continue;
      for (let s = 0; s < 3; s++) {
        const j = v[s] - nRim;
        if (j < 0) continue;
        add(K, i, j, (b[r] * b[s] + c[r] * c[s]) / (4 * A));
        add(M, i, j, (A / 12) * (r === s ? 2 : 1));
      }
    }
  }
  return { n, K, M };
}

// Reverse Cuthill–McKee ordering, to keep the Cholesky factor's profile small.
function rcm(K, n) {
  const deg = K.map((r) => r.size);
  const perm = [], seen = new Uint8Array(n);
  while (perm.length < n) {
    let start = -1;
    for (let i = 0; i < n; i++) if (!seen[i] && (start < 0 || deg[i] < deg[start])) start = i;
    seen[start] = 1;
    const q = [start];
    for (let h = 0; h < q.length; h++) {
      const u = q[h];
      perm.push(u);
      const nb = [...K[u].keys()].filter((v) => !seen[v]).sort((a, b) => deg[a] - deg[b]);
      for (const v of nb) { seen[v] = 1; q.push(v); }
    }
  }
  return perm.reverse();
}

// Skyline (envelope) Cholesky of a symmetric positive-definite matrix given in
// a permuted order. Row i stores columns first[i]..i.
class Skyline {
  constructor(K, perm) {
    const n = perm.length;
    const inv = new Int32Array(n);
    perm.forEach((p, k) => { inv[p] = k; });
    this.n = n;
    this.first = new Int32Array(n);
    this.start = new Int32Array(n + 1);
    for (let i = 0; i < n; i++) {
      let f = i;
      for (const j of K[perm[i]].keys()) f = Math.min(f, inv[j]);
      this.first[i] = f;
      this.start[i + 1] = this.start[i] + (i - f + 1);
    }
    const L = (this.L = new Float64Array(this.start[n]));
    for (let i = 0; i < n; i++) {
      for (const [j, v] of K[perm[i]]) {
        const jj = inv[j];
        if (jj <= i) L[this.start[i] + jj - this.first[i]] = v;
      }
    }
    // in-place factorisation
    for (let i = 0; i < n; i++) {
      const fi = this.first[i], si = this.start[i];
      for (let j = fi; j < i; j++) {
        const fj = this.first[j], sj = this.start[j];
        let s = L[si + j - fi];
        for (let k = Math.max(fi, fj); k < j; k++) s -= L[si + k - fi] * L[sj + k - fj];
        L[si + j - fi] = s / L[sj + j - fj];
      }
      let d = L[si + i - fi];
      for (let k = fi; k < i; k++) d -= L[si + k - fi] ** 2;
      if (!(d > 0)) throw new Error('stiffness matrix is not positive definite');
      L[si + i - fi] = Math.sqrt(d);
    }
    this.perm = perm;
    this.inv = inv;
  }

  // Solve K x = b (b in original order); returns x in original order.
  solve(b) {
    const { n, L, first, start, perm } = this;
    const y = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let s = b[perm[i]];
      const fi = first[i], si = start[i];
      for (let k = fi; k < i; k++) s -= L[si + k - fi] * y[k];
      y[i] = s / L[si + i - fi];
    }
    for (let i = n - 1; i >= 0; i--) {
      const fi = first[i], si = start[i];
      y[i] /= L[si + i - fi];
      const yi = y[i];
      for (let k = fi; k < i; k++) y[k] -= L[si + k - fi] * yi;
    }
    const x = new Float64Array(n);
    for (let i = 0; i < n; i++) x[perm[i]] = y[i];
    return x;
  }
}

function matvec(A, x) {
  const y = new Float64Array(x.length);
  for (let i = 0; i < A.length; i++) {
    let s = 0;
    for (const [j, v] of A[i]) s += v * x[j];
    y[i] = s;
  }
  return y;
}

// Eigen-decomposition of a symmetric tridiagonal matrix (implicit QL with
// shifts). d: diagonal, e: off-diagonal (e[0] unused). Returns values and
// column eigenvectors Z (m x m, row-major).
function tridiagEig(dIn, eIn) {
  const m = dIn.length, d = Float64Array.from(dIn), e = new Float64Array(m);
  for (let i = 1; i < m; i++) e[i - 1] = eIn[i];
  const Z = new Float64Array(m * m);
  for (let i = 0; i < m; i++) Z[i * m + i] = 1;
  for (let l = 0; l < m; l++) {
    let iter = 0, mm;
    do {
      for (mm = l; mm < m - 1; mm++) {
        const dd = Math.abs(d[mm]) + Math.abs(d[mm + 1]);
        if (Math.abs(e[mm]) <= 1e-15 * dd) break;
      }
      if (mm !== l) {
        if (iter++ > 60) break;
        let g = (d[l + 1] - d[l]) / (2 * e[l]);
        let r = Math.hypot(g, 1);
        g = d[mm] - d[l] + e[l] / (g + (g >= 0 ? Math.abs(r) : -Math.abs(r)));
        let s = 1, c = 1, p = 0, i;
        for (i = mm - 1; i >= l; i--) {
          let f = s * e[i];
          const b = c * e[i];
          r = Math.hypot(f, g);
          e[i + 1] = r;
          if (r === 0) { d[i + 1] -= p; e[mm] = 0; break; }
          s = f / r; c = g / r;
          g = d[i + 1] - p;
          r = (d[i] - g) * s + 2 * c * b;
          p = s * r;
          d[i + 1] = g + p;
          g = c * r - b;
          for (let k = 0; k < m; k++) {
            f = Z[k * m + i + 1];
            Z[k * m + i + 1] = s * Z[k * m + i] + c * f;
            Z[k * m + i] = c * Z[k * m + i] - s * f;
          }
        }
        if (r === 0 && i >= l) continue;
        d[l] -= p; e[l] = g; e[mm] = 0;
      }
    } while (mm !== l);
  }
  return { values: d, Z };
}

// Lowest k eigenpairs of K x = λ M x by Lanczos on K⁻¹M in the M inner
// product, with full reorthogonalisation.
function lanczos(K, M, n, k, steps) {
  const chol = new Skyline(K, rcm(K, n));
  const m = Math.min(n, steps);
  const V = [], alpha = new Float64Array(m), beta = new Float64Array(m + 1);
  let v = new Float64Array(n);
  let seed = 12345;
  for (let i = 0; i < n; i++) { seed = (Math.imul(seed, 1103515245) + 12345) >>> 0; v[i] = seed / 4294967296 - 0.5; }
  let Mv = matvec(M, v);
  let nrm = Math.sqrt(v.reduce((s, x, i) => s + x * Mv[i], 0));
  v = v.map((x) => x / nrm);
  Mv = Mv.map((x) => x / nrm);
  const MV = [];
  let used = m;
  for (let j = 0; j < m; j++) {
    V.push(v); MV.push(Mv);
    let w = chol.solve(Mv);                       // w = K⁻¹ M v
    let a = 0;
    for (let i = 0; i < n; i++) a += w[i] * Mv[i];
    alpha[j] = a;
    // full reorthogonalisation against all previous vectors (twice)
    for (let pass = 0; pass < 2; pass++) {
      for (let q = 0; q < V.length; q++) {
        let c = 0;
        const mq = MV[q];
        for (let i = 0; i < n; i++) c += w[i] * mq[i];
        const vq = V[q];
        for (let i = 0; i < n; i++) w[i] -= c * vq[i];
      }
    }
    const Mw = matvec(M, w);
    const b = Math.sqrt(Math.max(0, w.reduce((s, x, i) => s + x * Mw[i], 0)));
    beta[j + 1] = b;
    if (b < 1e-12 || j === m - 1) { used = j + 1; break; }
    v = w.map((x) => x / b);
    Mv = Mw.map((x) => x / b);
  }
  const { values, Z } = tridiagEig(alpha.subarray(0, used), beta.subarray(0, used));
  // θ = 1/λ: the largest θ are the lowest modes
  const order = Array.from({ length: used }, (_, i) => i).sort((a, b) => values[b] - values[a]).slice(0, k);
  const lam = new Float64Array(order.length);
  const vecs = [];
  order.forEach((col, r) => {
    lam[r] = 1 / values[col];
    const x = new Float64Array(n);
    for (let q = 0; q < used; q++) {
      const z = Z[q * used + col];
      const vq = V[q];
      for (let i = 0; i < n; i++) x[i] += z * vq[i];
    }
    vecs.push(x);
  });
  return { lam, vecs };
}

// Modes of a meshed drum: eigenvalues and mode shapes on every node (zero on
// the rim), each mode normalised to unit M-norm.
export function modes(meshed, k = K_MODES, steps = 3 * K_MODES) {
  const { nodes, tri, nRim } = meshed;
  const { n, K, M } = assemble(nodes, tri, nRim);
  const { lam, vecs } = lanczos(K, M, n, k, Math.max(steps, k + 20));
  const N = nodes.length / 2;
  const shapes = vecs.map((v) => {
    const full = new Float64Array(N);
    full.set(v, nRim);
    return full;
  });
  return { lam, shapes };
}

// Everything the page needs for an outline: the normalised outline, its
// mesh and its lowest modes.
export function drum(outline, k = K_MODES) {
  const p = normalise(outline);
  const meshed = mesh(p);
  return { outline: p, ...meshed, ...modes(meshed, k) };
}
