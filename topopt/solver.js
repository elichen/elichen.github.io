"use strict";

/**
 * SIMP topology optimization core (top88-style).
 *
 * Minimizes compliance c(x) = U^T K(x) U on a regular grid of bilinear quad
 * elements (plane stress, unit thickness), subject to a volume fraction
 * constraint. Density filter + optimality criteria update. The linear solve
 * is a matrix-free Jacobi-preconditioned conjugate gradient, warm-started
 * from the previous iteration's displacement field.
 *
 * Grid convention: element (ex, ey) with ex in [0, nelx), ey in [0, nely),
 * ey increasing downward. Node (ix, iy) has id ix*(nely+1)+iy and dofs
 * (2*id, 2*id+1) for (x, y).
 */

const E0 = 1.0;
const EMIN = 1e-9;
const NU = 0.3;

function buildKE(nu) {
  const A11 = [12, 3, -6, -3, 3, 12, 3, 0, -6, 3, 12, -3, -3, 0, -3, 12];
  const A12 = [-6, -3, 0, 3, -3, -6, -3, -6, 0, -3, -6, 3, 3, -6, 3, -6];
  const B11 = [-4, 3, -2, 9, 3, -4, -9, 4, -2, -9, -4, -3, 9, 4, -3, -4];
  const B12 = [2, -3, 4, -9, -3, 2, 9, -2, 4, 9, 2, 3, -9, -2, 3, 2];
  const KE = new Float64Array(64);
  const at = (m, r, c) => m[r * 4 + c];
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const rBlock = r >> 2, cBlock = c >> 2, ri = r & 3, ci = c & 3;
      let a, b;
      if (rBlock === cBlock) {
        a = at(A11, ri, ci);
        b = at(B11, ri, ci);
      } else if (rBlock < cBlock) {
        a = at(A12, ri, ci);
        b = at(B12, ri, ci);
      } else {
        a = at(A12, ci, ri); // transpose blocks below the diagonal
        b = at(B12, ci, ri);
      }
      KE[r * 8 + c] = (a + nu * b) / ((1 - nu * nu) * 24);
    }
  }
  return KE;
}

class TopOpt {
  constructor({ nelx, nely, volfrac = 0.4, penal = 3, rmin = 2.4 }) {
    this.nelx = nelx;
    this.nely = nely;
    this.nel = nelx * nely;
    this.ndof = 2 * (nelx + 1) * (nely + 1);
    this.volfrac = volfrac;
    this.penal = penal;
    this.rmin = rmin;

    this.KE = buildKE(NU);
    this.edof = this.buildEdof();
    this.buildFilter();

    this.x = new Float64Array(this.nel).fill(volfrac);
    this.xPhys = new Float64Array(this.nel);
    this.F = new Float64Array(this.ndof);
    this.U = new Float64Array(this.ndof);
    this.fixed = new Uint8Array(this.ndof);
    // passive: 0 = free design, 1 = forced void, 2 = forced solid
    this.passive = new Uint8Array(this.nel);

    this.dc = new Float64Array(this.nel);
    this.dv = new Float64Array(this.nel);
    this.xNew = new Float64Array(this.nel);
    this.xTilde = new Float64Array(this.nel);

    // CG scratch
    this.cgR = new Float64Array(this.ndof);
    this.cgZ = new Float64Array(this.ndof);
    this.cgP = new Float64Array(this.ndof);
    this.cgQ = new Float64Array(this.ndof);
    this.cgMinv = new Float64Array(this.ndof);

    this.iter = 0;
    this.compliance = NaN;
    this.change = 1;
    this.cgIters = 0;
    this.resetDesign();
  }

  nodeId(ix, iy) {
    return ix * (this.nely + 1) + iy;
  }

  buildEdof() {
    const { nelx, nely } = this;
    const edof = new Int32Array(this.nel * 8);
    for (let ex = 0; ex < nelx; ex++) {
      for (let ey = 0; ey < nely; ey++) {
        const e = ex * nely + ey;
        const tl = this.nodeId(ex, ey);
        const bl = tl + 1;
        const tr = this.nodeId(ex + 1, ey);
        const br = tr + 1;
        const o = e * 8;
        // top88 local node order: BL, BR, TR, TL
        edof[o] = 2 * bl; edof[o + 1] = 2 * bl + 1;
        edof[o + 2] = 2 * br; edof[o + 3] = 2 * br + 1;
        edof[o + 4] = 2 * tr; edof[o + 5] = 2 * tr + 1;
        edof[o + 6] = 2 * tl; edof[o + 7] = 2 * tl + 1;
      }
    }
    return edof;
  }

  buildFilter() {
    const { nelx, nely, rmin } = this;
    const reach = Math.ceil(rmin) - 1;
    const offsets = [];
    const counts = new Int32Array(this.nel);
    const starts = new Int32Array(this.nel + 1);
    // first pass: count neighbors per element
    const span = 2 * reach + 1;
    const kernel = [];
    for (let dx = -reach; dx <= reach; dx++) {
      for (let dy = -reach; dy <= reach; dy++) {
        const w = rmin - Math.hypot(dx, dy);
        if (w > 0) kernel.push([dx, dy, w]);
      }
    }
    for (let ex = 0; ex < nelx; ex++) {
      for (let ey = 0; ey < nely; ey++) {
        const e = ex * nely + ey;
        let n = 0;
        for (const [dx, dy] of kernel) {
          const jx = ex + dx, jy = ey + dy;
          if (jx >= 0 && jx < nelx && jy >= 0 && jy < nely) n++;
        }
        counts[e] = n;
      }
    }
    for (let e = 0; e < this.nel; e++) starts[e + 1] = starts[e] + counts[e];
    const total = starts[this.nel];
    const nIdx = new Int32Array(total);
    const nW = new Float64Array(total);
    const Hs = new Float64Array(this.nel);
    const fill = new Int32Array(this.nel);
    for (let ex = 0; ex < nelx; ex++) {
      for (let ey = 0; ey < nely; ey++) {
        const e = ex * nely + ey;
        for (const [dx, dy, w] of kernel) {
          const jx = ex + dx, jy = ey + dy;
          if (jx < 0 || jx >= nelx || jy < 0 || jy >= nely) continue;
          const j = jx * nely + jy;
          const at = starts[e] + fill[e]++;
          nIdx[at] = j;
          nW[at] = w;
          Hs[e] += w;
        }
      }
    }
    this.filterStarts = starts;
    this.filterIdx = nIdx;
    this.filterW = nW;
    this.filterHs = Hs;
  }

  /** out_i = sum_j H_ij * v_j / Hs_i  (density filter forward) */
  filterForward(v, out) {
    const { filterStarts: s, filterIdx: idx, filterW: w, filterHs: Hs } = this;
    for (let e = 0; e < this.nel; e++) {
      let acc = 0;
      for (let k = s[e]; k < s[e + 1]; k++) acc += w[k] * v[idx[k]];
      out[e] = acc / Hs[e];
    }
  }

  /** out_j = sum_i H_ij * v_i / Hs_i  (chain rule back through the filter) */
  filterBackward(v, out) {
    const { filterStarts: s, filterIdx: idx, filterW: w, filterHs: Hs } = this;
    out.fill(0);
    for (let e = 0; e < this.nel; e++) {
      const scaled = v[e] / Hs[e];
      for (let k = s[e]; k < s[e + 1]; k++) out[idx[k]] += w[k] * scaled;
    }
  }

  clearBCs() {
    this.F.fill(0);
    this.fixed.fill(0);
  }

  addLoad(ix, iy, fx, fy) {
    const n = this.nodeId(ix, iy);
    this.F[2 * n] += fx;
    this.F[2 * n + 1] += fy;
  }

  fixNode(ix, iy, fixX = true, fixY = true) {
    const n = this.nodeId(ix, iy);
    if (fixX) this.fixed[2 * n] = 1;
    if (fixY) this.fixed[2 * n + 1] = 1;
  }

  setPassive(passive) {
    this.passive.set(passive);
  }

  resetDesign() {
    this.x.fill(this.volfrac);
    this.U.fill(0);
    this.iter = 0;
    this.change = 1;
    this.compliance = NaN;
    this.applyPassive(this.x);
    this.filterForward(this.x, this.xPhys);
    this.applyPassive(this.xPhys);
  }

  applyPassive(arr) {
    const p = this.passive;
    for (let e = 0; e < this.nel; e++) {
      if (p[e] === 1) arr[e] = 0;
      else if (p[e] === 2) arr[e] = 1;
    }
  }

  hasValidBCs() {
    let nFixed = 0;
    let hasLoad = false;
    for (let d = 0; d < this.ndof; d++) {
      if (this.fixed[d]) nFixed++;
      else if (this.F[d] !== 0) hasLoad = true;
    }
    return nFixed >= 3 && hasLoad;
  }

  elementStiffness(e) {
    const xp = this.xPhys[e];
    return EMIN + Math.pow(xp, this.penal) * (E0 - EMIN);
  }

  /** out = K(xPhys) * v, with fixed dofs projected out. */
  applyK(v, out) {
    out.fill(0);
    const { KE, edof, nel } = this;
    for (let e = 0; e < nel; e++) {
      const Ee = this.elementStiffness(e);
      const o = e * 8;
      const d0 = edof[o], d1 = edof[o + 1], d2 = edof[o + 2], d3 = edof[o + 3];
      const d4 = edof[o + 4], d5 = edof[o + 5], d6 = edof[o + 6], d7 = edof[o + 7];
      const u0 = v[d0], u1 = v[d1], u2 = v[d2], u3 = v[d3];
      const u4 = v[d4], u5 = v[d5], u6 = v[d6], u7 = v[d7];
      for (let r = 0; r < 8; r++) {
        const kr = r * 8;
        out[edof[o + r]] += Ee * (
          KE[kr] * u0 + KE[kr + 1] * u1 + KE[kr + 2] * u2 + KE[kr + 3] * u3 +
          KE[kr + 4] * u4 + KE[kr + 5] * u5 + KE[kr + 6] * u6 + KE[kr + 7] * u7
        );
      }
    }
    const fixed = this.fixed;
    for (let d = 0; d < this.ndof; d++) if (fixed[d]) out[d] = 0;
  }

  solve(tol = 1e-4, maxIter = 4000) {
    const { F, U, fixed, ndof, cgR: r, cgZ: z, cgP: p, cgQ: q, cgMinv: Minv } = this;
    // Jacobi preconditioner from the current stiffness diagonal
    Minv.fill(0);
    const { KE, edof, nel } = this;
    for (let e = 0; e < nel; e++) {
      const Ee = this.elementStiffness(e);
      const o = e * 8;
      for (let i = 0; i < 8; i++) Minv[edof[o + i]] += Ee * KE[i * 8 + i];
    }
    for (let d = 0; d < ndof; d++) Minv[d] = fixed[d] || Minv[d] === 0 ? 0 : 1 / Minv[d];

    for (let d = 0; d < ndof; d++) if (fixed[d]) U[d] = 0;
    this.applyK(U, q);
    let fNorm2 = 0;
    for (let d = 0; d < ndof; d++) {
      r[d] = fixed[d] ? 0 : F[d] - q[d];
      if (!fixed[d]) fNorm2 += F[d] * F[d];
    }
    if (fNorm2 === 0) { U.fill(0); this.cgIters = 0; return; }
    const tol2 = tol * tol * fNorm2;

    let rz = 0;
    for (let d = 0; d < ndof; d++) {
      z[d] = Minv[d] * r[d];
      p[d] = z[d];
      rz += r[d] * z[d];
    }
    let it = 0;
    for (; it < maxIter; it++) {
      let r2 = 0;
      for (let d = 0; d < ndof; d++) r2 += r[d] * r[d];
      if (r2 <= tol2) break;
      this.applyK(p, q);
      let pq = 0;
      for (let d = 0; d < ndof; d++) pq += p[d] * q[d];
      if (pq <= 0) break; // lost positive definiteness (under-constrained)
      const alpha = rz / pq;
      for (let d = 0; d < ndof; d++) {
        U[d] += alpha * p[d];
        r[d] -= alpha * q[d];
      }
      let rzNew = 0;
      for (let d = 0; d < ndof; d++) {
        z[d] = Minv[d] * r[d];
        rzNew += r[d] * z[d];
      }
      const beta = rzNew / rz;
      rz = rzNew;
      for (let d = 0; d < ndof; d++) p[d] = z[d] + beta * p[d];
    }
    this.cgIters = it;
  }

  /** One optimization iteration: FEA solve, sensitivities, filter, OC update. */
  step() {
    const { KE, edof, nel, xPhys, penal } = this;
    this.solve(this.iter < 5 ? 1e-4 : 2e-4);

    // compliance and sensitivities
    let c = 0;
    const dc = this.dc, dv = this.dv, U = this.U;
    for (let e = 0; e < nel; e++) {
      const o = e * 8;
      let ueKue = 0;
      const u0 = U[edof[o]], u1 = U[edof[o + 1]], u2 = U[edof[o + 2]], u3 = U[edof[o + 3]];
      const u4 = U[edof[o + 4]], u5 = U[edof[o + 5]], u6 = U[edof[o + 6]], u7 = U[edof[o + 7]];
      const ue = [u0, u1, u2, u3, u4, u5, u6, u7];
      for (let r = 0; r < 8; r++) {
        const kr = r * 8;
        let ku = 0;
        for (let cc = 0; cc < 8; cc++) ku += KE[kr + cc] * ue[cc];
        ueKue += ue[r] * ku;
      }
      const xp = xPhys[e];
      c += (EMIN + Math.pow(xp, penal) * (E0 - EMIN)) * ueKue;
      dc[e] = -penal * Math.pow(xp, penal - 1) * (E0 - EMIN) * ueKue;
      dv[e] = 1;
    }
    // chain rule back through the density filter
    this.filterBackward(dc, this.xTilde);
    dc.set(this.xTilde);
    this.filterBackward(dv, this.xTilde);
    dv.set(this.xTilde);

    // optimality criteria with bisection on the Lagrange multiplier
    const x = this.x, xNew = this.xNew, xPhysNew = this.xTilde;
    const move = 0.2;
    let l1 = 0, l2 = 1e9;
    while ((l2 - l1) / (l1 + l2 + 1e-30) > 1e-4) {
      const lmid = 0.5 * (l1 + l2);
      for (let e = 0; e < nel; e++) {
        const be = Math.max(0, -dc[e] / (dv[e] * lmid + 1e-30));
        let xn = x[e] * Math.sqrt(be);
        const lo = Math.max(0, x[e] - move);
        const hi = Math.min(1, x[e] + move);
        xNew[e] = xn < lo ? lo : xn > hi ? hi : xn;
      }
      this.filterForward(xNew, xPhysNew);
      this.applyPassive(xPhysNew);
      let vol = 0;
      for (let e = 0; e < nel; e++) vol += xPhysNew[e];
      if (vol > this.volfrac * nel) l1 = lmid;
      else l2 = lmid;
    }
    let change = 0;
    for (let e = 0; e < nel; e++) {
      const d = Math.abs(xNew[e] - x[e]);
      if (d > change) change = d;
    }
    x.set(xNew);
    this.filterForward(x, this.xPhys);
    this.applyPassive(this.xPhys);

    this.iter++;
    this.compliance = c;
    this.change = change;
    let vol = 0;
    for (let e = 0; e < nel; e++) vol += this.xPhys[e];
    return {
      iter: this.iter,
      compliance: c,
      volume: vol / nel,
      change,
      cgIters: this.cgIters,
    };
  }
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { TopOpt, buildKE };
}
if (typeof window !== "undefined") {
  window.TopOpt = TopOpt;
}
