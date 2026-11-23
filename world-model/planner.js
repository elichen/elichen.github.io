// Receding-horizon visual MPC planner over the 9.95M latent world model.
// Uses ONLY the 224x224 front camera image (zero privileged simulator state).
//
// Modes:
//   'hyb-cem' : Hybrid H10/H5 R5 + goal-directed action bank + near-only CEM (default)
//   'hyb'     : Hybrid H10/H5 R5 + goal-directed action bank (no CEM)
//   'near'    : Hybrid H10/H5 R5 + pure visual nearest-neighbor proposals (no goal filter)

import { halfToFloat } from './model.js';

function makeRng(seed = 1000) {
  let s = seed >>> 0;
  const uniform = () => {
    s = (s + 0x6D2B79F5) >>> 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const normal = () => {
    const u1 = Math.max(1e-7, 1 - uniform());
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * uniform());
  };
  return { uniform, normal };
}

// Partial top-K indices (smallest or largest) for a Float32Array of scores
function argTopK(scores, k, largest = false) {
  const n = scores.length;
  k = Math.min(k, n);
  const idx = new Int32Array(n);
  for (let i = 0; i < n; i++) idx[i] = i;
  idx.sort(largest ? (a, b) => scores[b] - scores[a] : (a, b) => scores[a] - scores[b]);
  return idx.subarray(0, k);
}

export class ActionBank {
  constructor(meta, buf, actMean, actStd) {
    this.N = meta.N;
    this.D = meta.D;       // 192
    this.H = meta.H;       // 10
    const N = this.N, D = this.D, H = this.H;

    // Decode int8 embeddings -> Float32Array [N, D] + precompute ||z||^2
    const embQ = new Int8Array(buf, meta.embOffset, N * D);
    const embScH = new Uint16Array(buf, meta.embScaleOffset, N);
    this.emb = new Float32Array(N * D);
    this.embNorm2 = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      const sc = halfToFloat(embScH[i]);
      const base = i * D;
      let n2 = 0;
      for (let j = 0; j < D; j++) {
        const v = embQ[base + j] * sc;
        this.emb[base + j] = v;
        n2 += v * v;
      }
      this.embNorm2[i] = n2;
    }

    // Decode int8 raw actions [-1, 1] -> normalized Float32Array [N, H, 5]
    const actQ = new Int8Array(buf, meta.actOffset, N * H * 5);
    this.actNorm = new Float32Array(N * H * 5);
    const inv127 = 1 / 127;
    for (let i = 0; i < N * H; i++) {
      const base = i * 5;
      for (let d = 0; d < 5; d++) {
        const raw = actQ[base + d] * inv127;
        this.actNorm[base + d] = (raw - actMean[d]) / actStd[d];
      }
    }

    // Decode fp16 dCu_xy [N, 2] and appr [N]
    const dCuH = new Uint16Array(buf, meta.dCuOffset, N * 2);
    this.dCu = new Float32Array(N * 2);
    for (let i = 0; i < N * 2; i++) this.dCu[i] = halfToFloat(dCuH[i]);

    const apprH = new Uint16Array(buf, meta.apprOffset, N);
    this.appr = new Float32Array(N);
    for (let i = 0; i < N; i++) this.appr[i] = halfToFloat(apprH[i]);
  }

  // Retrieve RS candidate normalized action sequences of length H into Float32Array(RS * T * 5)
  retrieve(z0, cRead, goal, { RS = 128, RETK = 8, NEAR_KEEP = 40, mode = 'hyb-cem', T = 10 }) {
    const N = this.N, D = this.D, H = this.H;
    const emb = this.emb, n2 = this.embNorm2;
    let z0n2 = 0;
    for (let j = 0; j < D; j++) z0n2 += z0[j] * z0[j];

    // Squared Euclidean distance in projected CLS space
    const dist2 = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      let dot = 0;
      const base = i * D;
      for (let j = 0; j < D; j++) dot += z0[j] * emb[base + j];
      dist2[i] = z0n2 + n2[i] - 2 * dot;
    }

    let chosen;
    if (mode === 'near') {
      chosen = argTopK(dist2, RS, false);
    } else {
      const kTop = Math.min(N, RS * RETK);
      const nb = argTopK(dist2, kTop, false);
      const k0 = Math.min(NEAR_KEEP, Math.floor(RS / 3));
      const rx = goal[0] - cRead[0], ry = goal[1] - cRead[1];
      const rNorm = Math.max(1e-6, Math.hypot(rx, ry));
      const invR = 1 / rNorm;
      const rest = kTop - k0;
      const sc = new Float32Array(rest);
      const dCu = this.dCu, appr = this.appr;
      for (let i = 0; i < rest; i++) {
        const idx = nb[k0 + i];
        const prog = (dCu[2 * idx] * rx + dCu[2 * idx + 1] * ry) * invR;
        sc[i] = prog + 0.5 * appr[idx];
      }
      const bestRest = argTopK(sc, RS - k0, true);
      chosen = new Int32Array(RS);
      for (let i = 0; i < k0; i++) chosen[i] = nb[i];
      for (let i = 0; i < RS - k0; i++) chosen[k0 + i] = nb[k0 + bestRest[i]];
    }

    const out = new Float32Array(RS * T * 5);
    const src = this.actNorm;
    const stride = H * 5, outStride = T * 5;
    for (let i = 0; i < RS; i++) {
      const sOff = chosen[i] * stride;
      out.set(src.subarray(sOff, sOff + outStride), i * outStride);
    }
    return out;
  }
}

export async function loadActionBank(actMean, actStd, baseUrl = 'model') {
  const [meta, buf] = await Promise.all([
    fetch(`${baseUrl}/abank.json`).then(r => r.json()),
    fetch(`${baseUrl}/abank.bin`).then(r => r.arrayBuffer()),
  ]);
  return new ActionBank(meta, buf, actMean, actStd);
}

export class Planner {
  constructor(wm, bank, opts = {}) {
    this.wm = wm;
    this.bank = bank;
    this.RS = opts.RS ?? 128;
    this.RETK = opts.RETK ?? 8;
    this.NEAR_KEEP = opts.NEAR_KEEP ?? 40;
    this.BETA = opts.BETA ?? 1.5;
    this.NEAR_CEM = opts.NEAR_CEM ?? 3;
    this.REPLAN = opts.REPLAN ?? 5;
    this.mode = opts.mode ?? 'hyb-cem';
    this.rng = makeRng(opts.seed ?? 1000);
    this.resetHistory();
  }

  resetHistory(seed = null) {
    this.sPrev = null;
    this.aPrevNorm = [];   // normalized actions executed since last replan
    if (seed !== null) this.rng = makeRng(seed);
  }

  recordExecutedAction(rawClippedAction) {
    const m = this.wm.actMean, sd = this.wm.actStd;
    for (let d = 0; d < 5; d++) {
      this.aPrevNorm.push((rawClippedAction[d] - m[d]) / sd[d]);
    }
  }

  // Evaluate candidate sequences with LatentSkip1V2 and return final costs + full trajectories
  async _scoreCandidates(s0, zLat, candArr, B, T, goal) {
    const tf = this.wm.tf;
    const { s, ls, ih } = this.wm.rolloutTensors(s0, zLat, candArr, B, T);
    const [sData, lsData, ihData] = await Promise.all([s.data(), ls.data(), ih.data()]);
    s.dispose();
    ls.dispose();
    ih.dispose();

    const costs = new Float32Array(B);
    const sigFinal = new Float32Array(B);
    const gx = goal[0], gy = goal[1], gz = goal[2];
    const beta = this.BETA;
    for (let b = 0; b < B; b++) {
      const sOff = (b * T + (T - 1)) * 7;
      const cx = sData[sOff], cy = sData[sOff + 1], cz = sData[sOff + 2];
      const hx = sData[sOff + 3], hy = sData[sOff + 4], hz = sData[sOff + 5];
      const lOff = (b * T + (T - 1)) * 3;
      const sx = Math.exp(lsData[lOff]), sy = Math.exp(lsData[lOff + 1]), sz = Math.exp(lsData[lOff + 2]);
      const sig = Math.hypot(sx, sy, sz);
      sigFinal[b] = sig;
      costs[b] = Math.hypot(cx - gx, cy - gy, cz - gz) + Math.hypot(hx - cx, hy - cy, hz - cz) + beta * sig;
    }
    return { costs, sigFinal, sData, lsData, ihData };
  }

  // Run one full visual replan from a 224x224 pixel buffer (RGB or RGBA uint8)
  async plan(pixels, goal, channels = 3) {
    const t0 = performance.now();
    const { z0, s0, rawReadout } = await this.wm.encodeFrame(pixels, channels);
    const tEnc = performance.now();

    // 5-step Kalman warm-start using previous visual readout + executed actions
    const prevAct = this.sPrev && this.aPrevNorm.length >= 5
      ? Float32Array.from(this.aPrevNorm.slice(-5 * 5)) // last replan window (usually 5 steps = 25 floats)
      : null;
    const zLat = this.wm.warmstartZTensor(s0, prevAct ? this.sPrev : null, prevAct);
    // Reset action history buffer for the upcoming replan window
    this.sPrev = Float32Array.from(s0);
    this.aPrevNorm = [];

    const dx = s0[3] - s0[0], dy = s0[4] - s0[1], dz = s0[5] - s0[2];
    const handCubeDist = Math.hypot(dx, dy, dz);
    const isNear = (s0[2] > 0.04) || (handCubeDist < 0.05);
    const T = isNear ? 5 : 10;
    const RS = this.RS;

    let cand = this.bank.retrieve(z0, s0, goal, {
      RS, RETK: this.RETK, NEAR_KEEP: this.NEAR_KEEP, mode: this.mode, T,
    });
    let res = await this._scoreCandidates(s0, zLat, cand, RS, T, goal);

    let bestIdx = 0;
    for (let i = 1; i < RS; i++) if (res.costs[i] < res.costs[bestIdx]) bestIdx = i;
    let bestCost = res.costs[bestIdx];
    let bestAct = cand.slice(bestIdx * T * 5, (bestIdx + 1) * T * 5);

    // Near-only CEM refinement when in hyb-cem mode
    const nCem = (this.mode === 'hyb-cem' && isNear) ? this.NEAR_CEM : 0;
    if (nCem > 0) {
      const topK = Math.min(24, Math.floor(RS / 4));
      const mA = this.wm.actMean, sdA = this.wm.actStd;
      const mu = new Float32Array(T * 5);
      const sd = new Float32Array(T * 5);

      const fitElites = (arr, costs, minSd) => {
        const el = argTopK(costs, topK, false);
        mu.fill(0);
        for (let i = 0; i < topK; i++) {
          const off = el[i] * T * 5;
          for (let j = 0; j < T * 5; j++) mu[j] += arr[off + j];
        }
        for (let j = 0; j < T * 5; j++) mu[j] /= topK;
        sd.fill(0);
        for (let i = 0; i < topK; i++) {
          const off = el[i] * T * 5;
          for (let j = 0; j < T * 5; j++) {
            const diff = arr[off + j] - mu[j];
            sd[j] += diff * diff;
          }
        }
        const denom = Math.max(1, topK - 1);
        for (let j = 0; j < T * 5; j++) {
          sd[j] = Math.min(1.5, Math.max(minSd, Math.sqrt(sd[j] / denom)));
        }
      };

      fitElites(cand, res.costs, 0.05);
      const samp = new Float32Array(RS * T * 5);
      for (let iter = 0; iter < nCem; iter++) {
        // Candidate 0 = current best, Candidate 1 = elite mean
        samp.set(bestAct, 0);
        samp.set(mu, T * 5);
        for (let b = 2; b < RS; b++) {
          const off = b * T * 5;
          for (let t = 0; t < T; t++) {
            for (let d = 0; d < 5; d++) {
              const j = t * 5 + d;
              const nVal = mu[j] + sd[j] * this.rng.normal();
              const raw = Math.min(1, Math.max(-1, nVal * sdA[d] + mA[d]));
              samp[off + j] = (raw - mA[d]) / sdA[d];
            }
          }
        }
        const r2 = await this._scoreCandidates(s0, zLat, samp, RS, T, goal);
        let j0 = 0;
        for (let i = 1; i < RS; i++) if (r2.costs[i] < r2.costs[j0]) j0 = i;
        if (r2.costs[j0] < bestCost) {
          bestCost = r2.costs[j0];
          bestAct = samp.slice(j0 * T * 5, (j0 + 1) * T * 5);
        }
        cand = samp;
        res = r2;
        bestIdx = j0;
        fitElites(samp, r2.costs, 0.03);
      }
      // Make sure bestIdx points to candidate 0 if bestAct won earlier
      let minI = 0;
      for (let i = 1; i < RS; i++) if (res.costs[i] < res.costs[minI]) minI = i;
      bestIdx = minI;
    }

    zLat.dispose();
    const tEnd = performance.now();

    // Unnormalize chosen action sequence (first REPLAN steps)
    const mA = this.wm.actMean, sdA = this.wm.actStd;
    const actions = [];
    for (let t = 0; t < Math.min(this.REPLAN, T); t++) {
      const a = new Float32Array(5);
      for (let d = 0; d < 5; d++) {
        a[d] = Math.min(1, Math.max(-1, bestAct[t * 5 + d] * sdA[d] + mA[d]));
      }
      actions.push(a);
    }

    // Extract best + top-12 imagined trajectories for 3D visualization
    const topVis = argTopK(res.costs, Math.min(12, RS), false);
    const extractTraj = (b) => {
      const cube = [], hand = [], sig = [];
      for (let t = 0; t < T; t++) {
        const sOff = (b * T + t) * 7;
        cube.push([res.sData[sOff], res.sData[sOff + 1], res.sData[sOff + 2]]);
        hand.push([res.sData[sOff + 3], res.sData[sOff + 4], res.sData[sOff + 5]]);
        const lOff = (b * T + t) * 3;
        sig.push(Math.hypot(Math.exp(res.lsData[lOff]), Math.exp(res.lsData[lOff + 1]), Math.exp(res.lsData[lOff + 2])));
      }
      const pih = 1 / (1 + Math.exp(-res.ihData[b * T + (T - 1)]));
      return { cube, hand, sig, pih, cost: res.costs[b] };
    };

    const bestTraj = extractTraj(bestIdx);
    const topTrajs = [];
    for (let i = 0; i < topVis.length; i++) topTrajs.push(extractTraj(topVis[i]));

    return {
      actions,
      s0,
      rawReadout,
      isNear,
      horizon: T,
      bestCost,
      bestSig: res.sigFinal[bestIdx],
      bestTraj,
      topTrajs,
      encodeMs: tEnc - t0,
      rolloutMs: tEnd - tEnc,
      totalMs: tEnd - t0,
    };
  }
}
