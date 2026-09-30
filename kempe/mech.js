// Planar linkages built from dyads, their tracer curves, similarity-invariant
// curve alignment, and Levenberg–Marquardt refinement of a machine towards a
// drawing. A JavaScript port of kempe/train/linkage.py and refine.py; the two
// must agree (see kempe/train/golden.py).
//
// A machine is a list of joints in construction order. kind 0 is a fixed
// pivot; kind 1 is a crank tip turning about pivot a at `gear` turns per turn
// of the motor (the first crank is the motor itself, gear 1); kind 2 is a dyad,
// a joint tied by two bars to earlier joints a and b, placed on the side of
// line a→b it started on. The pen is the last joint.

export const T_SIM = 200;
export const N_CURVE = 64;
const TAU = Math.PI * 2;
const MIN_SIN_FLOOR = 0.25;
const EXTENT_MAX = 3.0;
const BAR_MIN = 0.25;

export class Machine {
  // spec: { kind, a, b, gear: Int arrays; pos: Float64Array(2n) initial pose }
  constructor(spec) {
    const n = spec.kind.length;
    this.n = n;
    this.kind = Int8Array.from(spec.kind);
    this.a = Int8Array.from(spec.a);
    this.b = Int8Array.from(spec.b);
    this.gear = Int8Array.from(spec.gear);
    this.grounds = [];
    this.cranks = [];
    this.dyads = [];
    for (let j = 0; j < n; j++) {
      (this.kind[j] === 0 ? this.grounds : this.kind[j] === 1 ? this.cranks : this.dyads).push(j);
    }
    const pos = spec.pos;
    this.p0x = pos[0];
    this.p0y = pos[1];
    const m = this.cranks[0];
    const dx = pos[2 * m] - pos[2 * this.a[m]];
    const dy = pos[2 * m + 1] - pos[2 * this.a[m] + 1];
    this.r = Math.hypot(dx, dy);
    this.phi = Math.atan2(dy, dx);
    this.side = new Float64Array(n);
    for (const j of this.dyads) {
      const a = this.a[j], b = this.b[j];
      const ax = pos[2 * a], ay = pos[2 * a + 1];
      const cr = (pos[2 * b] - ax) * (pos[2 * j + 1] - ay) - (pos[2 * b + 1] - ay) * (pos[2 * j] - ax);
      this.side[j] = Math.sign(cr) || 1;
    }
    this.tracer = n - 1;
    this.D = 2 * (this.grounds.length - 1) + 2 * (this.cranks.length - 1) + 2 * this.dyads.length;
    this.initial = this.paramsFromPos(pos);
    // bars: [joint, joint] pairs, cranks first (pivot→tip), then dyad bars
    this.bars = [];
    for (const c of this.cranks) this.bars.push([this.a[c], c]);
    for (const j of this.dyads) this.bars.push([this.a[j], j], [this.b[j], j]);
  }

  paramsFromPos(pos) {
    const P = new Float64Array(this.D);
    let i = 0;
    for (const g of this.grounds.slice(1)) { P[i++] = pos[2 * g]; P[i++] = pos[2 * g + 1]; }
    for (const c of this.cranks.slice(1)) {
      const dx = pos[2 * c] - pos[2 * this.a[c]], dy = pos[2 * c + 1] - pos[2 * this.a[c] + 1];
      P[i++] = Math.hypot(dx, dy);
      P[i++] = Math.atan2(dy, dx);
    }
    for (const j of this.dyads) {
      const a = this.a[j], b = this.b[j];
      P[i++] = Math.hypot(pos[2 * j] - pos[2 * a], pos[2 * j + 1] - pos[2 * a + 1]);
      P[i++] = Math.hypot(pos[2 * j] - pos[2 * b], pos[2 * j + 1] - pos[2 * b + 1]);
    }
    return P;
  }

  // Joint positions for each crank angle in `thetas`, written to out
  // (n * T * 2, joint-major). Returns the smallest transmission-angle sine,
  // or -1 if some dyad cannot close (the machine would lock).
  simulate(P, thetas, out) {
    const n = this.n, T = thetas.length;
    const X = (j, t) => 2 * (j * T + t);
    let i = 0;
    for (let t = 0; t < T; t++) { out[X(0, t)] = this.p0x; out[X(0, t) + 1] = this.p0y; }
    for (const g of this.grounds.slice(1)) {
      const x = P[i++], y = P[i++];
      for (let t = 0; t < T; t++) { out[X(g, t)] = x; out[X(g, t) + 1] = y; }
    }
    let first = true;
    for (const c of this.cranks) {
      let r, phi;
      if (first) { r = this.r; phi = this.phi; first = false; }
      else { r = P[i++]; phi = P[i++]; }
      const piv = this.a[c], g = this.gear[c];
      for (let t = 0; t < T; t++) {
        const ang = g * thetas[t] + phi;
        out[X(c, t)] = out[X(piv, t)] + r * Math.cos(ang);
        out[X(c, t) + 1] = out[X(piv, t) + 1] + r * Math.sin(ang);
      }
    }
    let minSin = 1;
    for (const j of this.dyads) {
      const la = P[i++], lb = P[i++];
      const a = this.a[j], b = this.b[j], side = this.side[j];
      for (let t = 0; t < T; t++) {
        const ax = out[X(a, t)], ay = out[X(a, t) + 1];
        const dx = out[X(b, t)] - ax, dy = out[X(b, t) + 1] - ay;
        const d = Math.hypot(dx, dy);
        const al = (la * la - lb * lb + d * d) / (2 * d);
        const h2 = la * la - al * al;
        if (!(h2 > 0) || !(d > 0)) return -1;
        const h = Math.sqrt(h2);
        const ux = dx / d, uy = dy / d;
        out[X(j, t)] = ax + al * ux - side * h * uy;
        out[X(j, t) + 1] = ay + al * uy + side * h * ux;
        const s = (d * h) / (la * lb);
        if (s < minSin) minSin = s;
      }
    }
    return minSin;
  }

  // Crank radii and bar lengths for design P (for the stubby-bar penalty).
  barLengths(P) {
    const out = [this.r];
    let i = 2 * (this.grounds.length - 1);
    for (let c = 1; c < this.cranks.length; c++) { out.push(Math.abs(P[i])); i += 2; }
    for (let d = 0; d < this.dyads.length; d++) { out.push(P[i], P[i + 1]); i += 2; }
    return out;
  }
}

export const thetaGrid = (T) => Float64Array.from({ length: T }, (_, t) => (TAU * t) / T);
const THETAS = thetaGrid(T_SIM);

// Arc-length resampling of a closed curve (xy interleaved, T points) at
// s_i = (i + delta) / n of the perimeter, starting at the first point.
export function resampleClosed(xy, T, n, delta, re, im) {
  const cum = new Float64Array(T + 1);
  for (let t = 0; t < T; t++) {
    const u = (t + 1) % T;
    cum[t + 1] = cum[t] + Math.hypot(xy[2 * u] - xy[2 * t], xy[2 * u + 1] - xy[2 * t + 1]);
  }
  const total = cum[T] || 1e-12;
  let k = 0;
  for (let i = 0; i < n; i++) {
    let u = ((i + delta) / n) % 1;
    if (u < 0) u += 1;
    const s = u * total;
    // cum is increasing; u grows with i except at the wrap
    if (k > 0 && cum[k] > s) k = 0;
    while (k < T - 1 && cum[k + 1] <= s) k++;
    const c0 = cum[k], c1 = cum[k + 1];
    const w = c1 > c0 ? (s - c0) / (c1 - c0) : 0;
    const k1 = (k + 1) % T;
    re[i] = xy[2 * k] * (1 - w) + xy[2 * k1] * w;
    im[i] = xy[2 * k + 1] * (1 - w) + xy[2 * k1 + 1] * w;
  }
}

// Centre and scale to unit RMS in place; returns [mean re, mean im, rms].
export function normalise(re, im) {
  const n = re.length;
  let mx = 0, my = 0;
  for (let i = 0; i < n; i++) { mx += re[i]; my += im[i]; }
  mx /= n; my /= n;
  let s = 0;
  for (let i = 0; i < n; i++) { re[i] -= mx; im[i] -= my; s += re[i] * re[i] + im[i] * im[i]; }
  s = Math.sqrt(s / n) || 1e-12;
  for (let i = 0; i < n; i++) { re[i] /= s; im[i] /= s; }
  return [mx, my, s];
}

// --- small radix-2 FFT for the correlation over start shifts -------------
const fftCache = new Map();
function fftPlan(n) {
  if (fftCache.has(n)) return fftCache.get(n);
  const bits = Math.log2(n);
  const rev = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    let r = 0;
    for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
    rev[i] = r;
  }
  const cos = new Float64Array(n / 2), sin = new Float64Array(n / 2);
  for (let i = 0; i < n / 2; i++) { cos[i] = Math.cos((TAU * i) / n); sin[i] = -Math.sin((TAU * i) / n); }
  const plan = { rev, cos, sin };
  fftCache.set(n, plan);
  return plan;
}
// In-place FFT (sign -1) or inverse without 1/n (sign +1).
function fft(re, im, inverse = false) {
  const n = re.length, { rev, cos, sin } = fftPlan(n);
  for (let i = 0; i < n; i++) {
    const j = rev[i];
    if (j > i) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  const sg = inverse ? -1 : 1;
  for (let size = 2; size <= n; size <<= 1) {
    const half = size >> 1, step = n / size;
    for (let st = 0; st < n; st += size) {
      for (let k = 0; k < half; k++) {
        const wr = cos[k * step], wi = sg * sin[k * step];
        const i1 = st + k, i2 = i1 + half;
        const tr = re[i2] * wr - im[i2] * wi, ti = re[i2] * wi + im[i2] * wr;
        re[i2] = re[i1] - tr; im[i2] = im[i1] - ti;
        re[i1] += tr; im[i1] += ti;
      }
    }
  }
}

// The four traversals a machine can realise of the same path: as is, crank
// reversed, mirrored, mirrored and reversed. rev[k] = z[-k mod n].
export function variant(re, im, v, outRe, outIm) {
  const n = re.length;
  for (let k = 0; k < n; k++) {
    const src = v & 1 ? (n - k) % n : k;
    outRe[k] = re[src];
    outIm[k] = v & 2 ? -im[src] : im[src];
  }
}

// Precomputed spectrum of a normalised target, for fast alignment.
export class Target {
  constructor(re, im) {
    this.n = re.length;
    this.re = Float64Array.from(re);
    this.im = Float64Array.from(im);
    this.Fre = Float64Array.from(re);
    this.Fim = Float64Array.from(im);
    fft(this.Fre, this.Fim);
    const n = this.n;
    this._a = new Float64Array(n); this._b = new Float64Array(n);
    this._c = new Float64Array(n); this._d = new Float64Array(n);
  }

  // Best (1 - |corr|^2) over start shift and the four variants of candidate c.
  align(cre, cim) {
    const n = this.n, vr = this._a, vi = this._b, pr = this._c, pi = this._d;
    let best = -1, bv = 0, bs = 0;
    for (let v = 0; v < 4; v++) {
      variant(cre, cim, v, vr, vi);
      fft(vr, vi);
      for (let k = 0; k < n; k++) {   // Ft * conj(Fc)
        pr[k] = this.Fre[k] * vr[k] + this.Fim[k] * vi[k];
        pi[k] = this.Fim[k] * vr[k] - this.Fre[k] * vi[k];
      }
      fft(pr, pi, true);
      for (let s = 0; s < n; s++) {
        const m = Math.hypot(pr[s], pi[s]) / (n * n);
        if (m > best) { best = m; bv = v; bs = s; }
      }
    }
    return { dist: 1 - best * best, variant: bv, shift: bs };
  }

  // Target re-expressed to line up index-for-index with the candidate's own
  // samples for a given variant and shift.
  frame(v, shift) {
    const n = this.n, fr = new Float64Array(n), fi = new Float64Array(n);
    variant(this.re, this.im, v, fr, fi);
    const s = v === 0 || v === 2 ? shift : -shift;
    const tr = new Float64Array(n), ti = new Float64Array(n);
    for (let k = 0; k < n; k++) {
      const src = (((k + s) % n) + n) % n;
      tr[k] = fr[src]; ti[k] = fi[src];
    }
    return { re: tr, im: ti };
  }
}

// Prepare a drawing (xy interleaved, closed) as a normalised target.
export function targetFromPoints(xy, n = N_CURVE) {
  const T = xy.length / 2;
  const re = new Float64Array(n), im = new Float64Array(n);
  resampleClosed(xy, T, n, 0, re, im);
  const [mx, my, s] = normalise(re, im);
  const t = new Target(re, im);
  t.mean = [mx, my];
  t.scale = s;
  return t;
}

// Workspace for evaluating one machine's residuals.
export class Fitter {
  constructor(machine, target, T = T_SIM) {
    this.m = machine;
    this.target = target;
    this.T = T;
    this.thetas = T === T_SIM ? THETAS : thetaGrid(T);
    this.buf = new Float64Array(machine.n * T * 2);
    this.curve = new Float64Array(T * 2);
    const n = target.n;
    this.cre = new Float64Array(n);
    this.cim = new Float64Array(n);
    this.R = 2 * n + 3;
  }

  // Tracer curve of design P, resampled from offset delta. Returns
  // { mean, scale, minSin } or null if the machine locks.
  curveOf(P, delta) {
    const minSin = this.m.simulate(P, this.thetas, this.buf);
    if (minSin < 0) return null;
    const T = this.T, base = 2 * this.m.tracer * T;
    for (let t = 0; t < 2 * T; t++) this.curve[t] = this.buf[base + t];
    resampleClosed(this.curve, T, this.target.n, delta, this.cre, this.cim);
    const [mx, my, s] = normalise(this.cre, this.cim);
    return { mean: [mx, my], scale: s, minSin };
  }

  // Residual vector (2n + 3) into out; returns the squared norm, or Infinity.
  residuals(P, delta, tt, out) {
    const c = this.curveOf(P, delta);
    if (!c) return Infinity;
    const n = this.target.n, cre = this.cre, cim = this.cim;
    let ar = 0, ai = 0;                       // alpha = <c, tt> / n
    for (let k = 0; k < n; k++) {
      ar += cre[k] * tt.re[k] + cim[k] * tt.im[k];
      ai += cre[k] * tt.im[k] - cim[k] * tt.re[k];
    }
    ar /= n; ai /= n;
    const sq = Math.sqrt(n);
    let cost = 0;
    for (let k = 0; k < n; k++) {
      const rr = (tt.re[k] - (ar * cre[k] - ai * cim[k])) / sq;
      const ri = (tt.im[k] - (ar * cim[k] + ai * cre[k])) / sq;
      out[k] = rr; out[n + k] = ri;
      cost += rr * rr + ri * ri;
    }
    // drivable: transmission angles stay open
    const pSin = 2.0 * Math.max(0, MIN_SIN_FLOOR - c.minSin);
    // compact: every joint stays within EXTENT_MAX curve radii of the curve
    const buf = this.buf, [mx, my] = c.mean;
    let ext = 0;
    for (let i = 0; i < buf.length; i += 2) {
      const e = (buf[i] - mx) ** 2 + (buf[i + 1] - my) ** 2;
      if (e > ext) ext = e;
    }
    ext = Math.sqrt(ext) / c.scale;
    const pExt = 0.3 * Math.max(0, ext - EXTENT_MAX);
    let bs = 0;
    for (const l of this.m.barLengths(P)) {
      const d = Math.max(0, BAR_MIN - l / c.scale);
      bs += d * d;
    }
    const pBar = 0.5 * Math.sqrt(bs);
    out[2 * n] = pSin; out[2 * n + 1] = pExt; out[2 * n + 2] = pBar;
    cost += pSin * pSin + pExt * pExt + pBar * pBar;
    this.alpha = [ar, ai];
    this.last = c;
    return Number.isFinite(cost) ? cost : Infinity;
  }
}

// Solve (A + lam * diag(A)) x = -g for small dense symmetric A.
function solveDamped(A, g, lam, D) {
  const M = new Float64Array(D * D), x = new Float64Array(D);
  for (let i = 0; i < D; i++) {
    for (let j = 0; j < D; j++) M[i * D + j] = A[i * D + j];
    M[i * D + i] += lam * A[i * D + i] + 1e-9;
    x[i] = -g[i];
  }
  // Gaussian elimination with partial pivoting
  for (let c = 0; c < D; c++) {
    let p = c;
    for (let r = c + 1; r < D; r++) if (Math.abs(M[r * D + c]) > Math.abs(M[p * D + c])) p = r;
    if (p !== c) {
      for (let k = 0; k < D; k++) [M[c * D + k], M[p * D + k]] = [M[p * D + k], M[c * D + k]];
      [x[c], x[p]] = [x[p], x[c]];
    }
    const piv = M[c * D + c];
    if (Math.abs(piv) < 1e-300) return null;
    for (let r = c + 1; r < D; r++) {
      const f = M[r * D + c] / piv;
      if (f === 0) continue;
      for (let k = c; k < D; k++) M[r * D + k] -= f * M[c * D + k];
      x[r] -= f * x[c];
    }
  }
  for (let c = D - 1; c >= 0; c--) {
    let s = x[c];
    for (let k = c + 1; k < D; k++) s -= M[c * D + k] * x[k];
    x[c] = s / M[c * D + c];
  }
  return x;
}

// Levenberg–Marquardt over the design and the start offset. Call step()
// repeatedly; state is kept so the UI can show every improvement.
export class Refiner {
  constructor(machine, target, P0 = machine.initial) {
    this.m = machine;
    this.target = target;
    this.fit = new Fitter(machine, target);
    this.x = new Float64Array(machine.D + 1);
    this.x.set(P0);
    this.lam = 1e-3;
    this.iter = 0;
    this.R = new Float64Array(this.fit.R);
    this.Rn = new Float64Array(this.fit.R);
    this.stalled = false;
    const c = this.fit.curveOf(P0, 0);
    if (!c) { this.cost = Infinity; this.tt = null; return; }
    const al = target.align(this.fit.cre, this.fit.cim);
    this.variant = al.variant;
    this.tt = target.frame(al.variant, al.shift);
    this.cost = this.fit.residuals(this.P, 0, this.tt, this.R);
  }

  get P() { return this.x.subarray(0, this.m.D); }
  get delta() { return this.x[this.m.D]; }

  realign() {
    const c = this.fit.curveOf(this.P, this.delta);
    if (!c) return;
    const al = this.target.align(this.fit.cre, this.fit.cim);
    const tt = this.target.frame(al.variant, al.shift);
    const cn = this.fit.residuals(this.P, this.delta, tt, this.Rn);
    if (cn < this.cost) { this.tt = tt; this.cost = cn; this.variant = al.variant; this.R.set(this.Rn); }
  }

  step() {
    if (!this.tt || this.stalled) return false;
    if (this.iter % 10 === 9) this.realign();   // re-pick the discrete alignment
    this.iter++;
    const D = this.x.length, Rl = this.R.length, x = this.x, fit = this.fit;
    const J = new Float64Array(Rl * D);
    const R0 = new Float64Array(Rl);
    const c0 = fit.residuals(x.subarray(0, D - 1), x[D - 1], this.tt, R0);
    if (!Number.isFinite(c0)) { this.stalled = true; return false; }
    const xh = Float64Array.from(x), Rh = new Float64Array(Rl);
    for (let p = 0; p < D; p++) {
      const h = 1e-5 * Math.max(1, Math.abs(x[p]));
      xh[p] = x[p] + h;
      const ch = fit.residuals(xh.subarray(0, D - 1), xh[D - 1], this.tt, Rh);
      xh[p] = x[p];
      for (let r = 0; r < Rl; r++) J[r * D + p] = Number.isFinite(ch) ? (Rh[r] - R0[r]) / h : 0;
    }
    const A = new Float64Array(D * D), g = new Float64Array(D);
    for (let i = 0; i < D; i++) {
      let gi = 0;
      for (let r = 0; r < Rl; r++) gi += J[r * D + i] * R0[r];
      g[i] = gi;
      for (let j = i; j < D; j++) {
        let s = 0;
        for (let r = 0; r < Rl; r++) s += J[r * D + i] * J[r * D + j];
        A[i * D + j] = s; A[j * D + i] = s;
      }
    }
    const xn = new Float64Array(D);
    for (let tries = 0; tries < 8; tries++) {
      const dx = solveDamped(A, g, this.lam, D);
      if (dx) {
        for (let i = 0; i < D; i++) xn[i] = x[i] + dx[i];
        const cn = fit.residuals(xn.subarray(0, D - 1), xn[D - 1], this.tt, this.Rn);
        if (cn < c0) {
          x.set(xn);
          this.cost = cn;
          this.R.set(this.Rn);
          this.lam = Math.max(this.lam / 3, 1e-7);
          return true;
        }
      }
      this.lam *= 4;
    }
    if (this.lam > 1e6) this.stalled = true;
    return false;
  }

  // Shape error alone: RMS distance between the aligned curves, relative to
  // the drawing's RMS radius.
  shapeError() {
    const n = this.target.n;
    const c = this.fit.residuals(this.P, this.delta, this.tt, this.Rn);
    if (!Number.isFinite(c)) return Infinity;
    let s = 0;
    for (let k = 0; k < 2 * n; k++) s += this.Rn[k] ** 2;
    return Math.sqrt(s);
  }

  // How to place the machine over the drawing: world = mean_t + scale_t *
  // M(alpha * (p - mean_c) / scale_c), M = mirror for variants 2 and 3.
  placement() {
    this.fit.residuals(this.P, this.delta, this.tt, this.Rn);
    const c = this.fit.last;
    return {
      alpha: this.fit.alpha.slice(),
      mean: c.mean.slice(),
      scale: c.scale,
      mirror: this.variant >= 2,
    };
  }
}

// Map a machine-frame point to world coordinates with a placement.
export function toWorld(pl, target, x, y) {
  const u = (x - pl.mean[0]) / pl.scale, v = (y - pl.mean[1]) / pl.scale;
  let wx = pl.alpha[0] * u - pl.alpha[1] * v;
  let wy = pl.alpha[0] * v + pl.alpha[1] * u;
  if (pl.mirror) wy = -wy;
  return [target.mean[0] + target.scale * wx, target.mean[1] + target.scale * wy];
}
