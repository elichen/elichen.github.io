// Live statistics of the beads' motion, fed with bead records (every few steps).
//
// Mean squared displacement uses a multiple-tau scheme: level k keeps the last
// M positions sampled every 2^k records, and each new sample is compared with
// those, so lags from one record to hours of simulated time cost O(M log) per
// sample.

const M = 16;
const LEVELS = 22;

class BeadHistory {
  constructor() {
    this.buf = Array.from({ length: LEVELS }, () => new Float64Array(2 * M));
    this.len = new Int32Array(LEVELS);
    this.head = new Int32Array(LEVELS);
    this.count = 0;
  }
  clear() { this.len.fill(0); this.head.fill(0); this.count = 0; }
}

export class MotionStats {
  constructor(nb, recordTime) {
    this.nb = nb;
    this.recordTime = recordTime;
    this.beads = Array.from({ length: nb }, () => new BeadHistory());
    this.sum = new Float64Array(LEVELS * (M + 1));
    this.n = new Float64Array(LEVELS * (M + 1));
    this.ke = 0;
    this.keCount = 0;
  }

  reset() {
    for (const b of this.beads) b.clear();
    this.sum.fill(0);
    this.n.fill(0);
    this.ke = 0;
    this.keCount = 0;
  }

  // Forget one bead's recent positions (e.g. after it was dragged), keeping the sums.
  forget(b) { this.beads[b].clear(); }

  // x, y: unwrapped position; v: velocity; m: mass
  add(b, x, y, vx, vy, m) {
    const h = this.beads[b];
    const c = h.count++;
    for (let k = 0; k < LEVELS; k++) {
      if (c % (1 << k) !== 0) break;
      const buf = h.buf[k], len = h.len[k];
      const jmin = k === 0 ? 1 : M / 2 + 1;
      for (let j = jmin; j <= Math.min(len, M); j++) {
        const p = ((h.head[k] - j) % M + M) % M;
        const dx = x - buf[2 * p], dy = y - buf[2 * p + 1];
        this.sum[k * (M + 1) + j] += dx * dx + dy * dy;
        this.n[k * (M + 1) + j]++;
      }
      buf[2 * h.head[k]] = x;
      buf[2 * h.head[k] + 1] = y;
      h.head[k] = (h.head[k] + 1) % M;
      h.len[k] = Math.min(len + 1, M);
    }
    this.ke += 0.5 * m * (vx * vx + vy * vy);
    this.keCount++;
  }

  // Bead temperature from equipartition: in 2D, <½Mv²> = kT.
  get beadT() { return this.keCount ? this.ke / this.keCount : NaN; }

  // [{t, msd, n}] at lags with at least minCount samples, shortest first.
  msd(minCount = 20) {
    const out = [];
    for (let k = 0; k < LEVELS; k++) {
      for (let j = k === 0 ? 1 : M / 2 + 1; j <= M; j++) {
        const i = k * (M + 1) + j;
        if (this.n[i] >= minCount) out.push({ t: j * (1 << k) * this.recordTime, msd: this.sum[i] / this.n[i], n: this.n[i] });
      }
    }
    return out;
  }

  // Diffusion coefficient from the long-lag points (MSD = 4 D t in 2D), weighting
  // each lag by its number of independent samples (about samples·record / lag).
  diffusion(minLag) {
    let num = 0, den = 0;
    for (const p of this.msd(40)) {
      if (p.t < minLag) continue;
      const w = p.n * this.recordTime / p.t;
      num += w * p.msd / (4 * p.t);
      den += w;
    }
    return den > 0 ? num / den : NaN;
  }
}
