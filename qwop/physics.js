// 2D ragdoll-runner physics. Line-for-line mirror of train/sim.py — the neural
// nets were trained against that implementation, so keep the two in sync
// (train/parity.py verifies they agree).
const QwopPhysics = (() => {
  const NB = 9;
  const TORSO = 0, THIGH_R = 1, THIGH_L = 2, CALF_R = 3, CALF_L = 4,
    FOOT_R = 5, FOOT_L = 6, ARM_R = 7, ARM_L = 8;
  const MASS = [40.0, 7.0, 7.0, 4.0, 4.0, 2.0, 2.0, 3.5, 3.5];
  const INERTIA = [2.2, 0.13, 0.13, 0.072, 0.072, 0.03, 0.03, 0.09, 0.09];
  const INV_M = MASS.map((m) => 1.0 / m);
  const INV_I = INERTIA.map((i) => 1.0 / i);

  const NJ = 8;
  const HIP_R = 0, HIP_L = 1, KNEE_R = 2, KNEE_L = 3, ANKLE_R = 4, ANKLE_L = 5,
    SHO_R = 6, SHO_L = 7;
  const J_A = [TORSO, TORSO, THIGH_R, THIGH_L, CALF_R, CALF_L, TORSO, TORSO];
  const J_B = [THIGH_R, THIGH_L, CALF_R, CALF_L, FOOT_R, FOOT_L, ARM_R, ARM_L];
  const J_AX = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
  const J_AY = [-0.30, -0.30, -0.225, -0.225, -0.225, -0.225, 0.27, 0.27];
  const J_BX = [0.0, 0.0, 0.0, 0.0, -0.06, -0.06, 0.0, 0.0];
  const J_BY = [0.225, 0.225, 0.225, 0.225, 0.03, 0.03, 0.275, 0.275];
  const J_LO = [-0.8, -0.8, -2.4, -2.4, -0.6, -0.6, -1.6, -1.6];
  const J_HI = [1.5, 1.5, 0.0, 0.0, 0.6, 0.6, 1.6, 1.6];
  const J_TORQUE = [450.0, 450.0, 350.0, 350.0, 40.0, 40.0, 60.0, 60.0];
  const HIP_SPEED = 4.5;
  const KNEE_SPEED = 5.5;
  const ARM_SPEED = 3.0;
  const ANKLE_GAIN = 8.0;
  const ANKLE_SPEED = 4.0;

  const NC = 11;
  const C_BODY = [FOOT_R, FOOT_R, FOOT_L, FOOT_L, CALF_R, CALF_L,
    TORSO, TORSO, TORSO, ARM_R, ARM_L];
  const C_X = [-0.13, 0.13, -0.13, 0.13, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
  const C_Y = [0.0, 0.0, 0.0, 0.0, 0.225, 0.225, -0.30, 0.30, 0.45, -0.275, -0.275];
  const C_R = [0.04, 0.04, 0.04, 0.04, 0.055, 0.055, 0.10, 0.10, 0.12, 0.045, 0.045];
  const C_FATAL0 = 6;

  const GRAVITY = 9.81;
  const FRICTION = 0.9;
  const NSUB = 4;
  const H = 1.0 / 120.0;
  const DT = NSUB * H;
  const ITERS = 4;
  const BETA = 0.2;
  const SLOP = 0.005;
  const MAX_CORR = 3.0;
  const MAX_V = 50.0;
  const MAX_W = 40.0;
  const FATAL_Y = 0.01;

  const NACC = NJ * 5 + NC * 2;
  const NACT = 9;
  const NOBS = 6 + 7 * (NB - 1) + 6 + NACT;
  const GOAL_X = 100.0;

  const POSE0 = [0.4, -0.26, -0.19, -0.01, -0.18, 0.29, 0.0, 0.0];
  const POSE0_TORSO = -0.03;

  // s is a flat Float64Array [NB*6]: x, y, angle, vx, vy, w per body.
  function setPose(s, x, torsoA, q) {
    s.fill(0.0);
    s[TORSO * 6 + 0] = x;
    s[TORSO * 6 + 2] = torsoA;
    for (let j = 0; j < NJ; j++) {
      const a = J_A[j] * 6, b = J_B[j] * 6;
      const ca = Math.cos(s[a + 2]), sa = Math.sin(s[a + 2]);
      const px = s[a] + ca * J_AX[j] - sa * J_AY[j];
      const py = s[a + 1] + sa * J_AX[j] + ca * J_AY[j];
      const ang = s[a + 2] + q[j];
      const cb = Math.cos(ang), sb = Math.sin(ang);
      s[b + 2] = ang;
      s[b] = px - (cb * J_BX[j] - sb * J_BY[j]);
      s[b + 1] = py - (sb * J_BX[j] + cb * J_BY[j]);
    }
    let low = 1e9;
    for (let c = 0; c < NC; c++) {
      const i = C_BODY[c] * 6;
      const y = s[i + 1] + Math.sin(s[i + 2]) * C_X[c] + Math.cos(s[i + 2]) * C_Y[c] - C_R[c];
      if (y < low) low = y;
    }
    for (let i = 0; i < NB; i++) s[i * 6 + 1] -= low;
  }

  const tgt = new Float64Array(NJ);
  const cs = new Float64Array(NB), sn = new Float64Array(NB);
  const jrax = new Float64Array(NJ), jray = new Float64Array(NJ);
  const jrbx = new Float64Array(NJ), jrby = new Float64Array(NJ);
  const jk11 = new Float64Array(NJ), jk12 = new Float64Array(NJ), jk22 = new Float64Array(NJ);
  const jbx = new Float64Array(NJ), jby = new Float64Array(NJ), jka = new Float64Array(NJ);
  const jlo = new Float64Array(NJ), jhi = new Float64Array(NJ);
  const crx = new Float64Array(NC), cry = new Float64Array(NC);
  const ckn = new Float64Array(NC), ckt = new Float64Array(NC), cb = new Float64Array(NC);

  function setTargets(action) {
    const hip = Math.floor(action / 3), knee = action % 3;
    tgt.fill(0.0);
    if (hip === 1) {
      tgt[HIP_R] = HIP_SPEED; tgt[HIP_L] = -HIP_SPEED;
      tgt[SHO_R] = -ARM_SPEED; tgt[SHO_L] = ARM_SPEED;
    } else if (hip === 2) {
      tgt[HIP_R] = -HIP_SPEED; tgt[HIP_L] = HIP_SPEED;
      tgt[SHO_R] = ARM_SPEED; tgt[SHO_L] = -ARM_SPEED;
    }
    if (knee === 1) {
      tgt[KNEE_L] = -KNEE_SPEED; tgt[KNEE_R] = KNEE_SPEED;
    } else if (knee === 2) {
      tgt[KNEE_L] = KNEE_SPEED; tgt[KNEE_R] = -KNEE_SPEED;
    }
  }

  function substep(s, acc) {
    for (let i = 0; i < NB; i++) {
      s[i * 6 + 4] -= GRAVITY * H;
      cs[i] = Math.cos(s[i * 6 + 2]);
      sn[i] = Math.sin(s[i * 6 + 2]);
    }

    for (let j = 0; j < NJ; j++) {
      const ia = J_A[j], ib = J_B[j], a = ia * 6, b = ib * 6;
      const rax = cs[ia] * J_AX[j] - sn[ia] * J_AY[j];
      const ray = sn[ia] * J_AX[j] + cs[ia] * J_AY[j];
      const rbx = cs[ib] * J_BX[j] - sn[ib] * J_BY[j];
      const rby = sn[ib] * J_BX[j] + cs[ib] * J_BY[j];
      jrax[j] = rax; jray[j] = ray; jrbx[j] = rbx; jrby[j] = rby;
      const ms = INV_M[ia] + INV_M[ib];
      const k11 = ms + INV_I[ia] * ray * ray + INV_I[ib] * rby * rby;
      const k12 = -INV_I[ia] * rax * ray - INV_I[ib] * rbx * rby;
      const k22 = ms + INV_I[ia] * rax * rax + INV_I[ib] * rbx * rbx;
      const det = k11 * k22 - k12 * k12;
      jk11[j] = k22 / det;
      jk12[j] = -k12 / det;
      jk22[j] = k11 / det;
      jbx[j] = (BETA / H) * (s[b] + rbx - s[a] - rax);
      jby[j] = (BETA / H) * (s[b + 1] + rby - s[a + 1] - ray);
      jka[j] = 1.0 / (INV_I[ia] + INV_I[ib]);
      const ang = s[b + 2] - s[a + 2];
      const cLo = ang - J_LO[j];
      const cHi = J_HI[j] - ang;
      jlo[j] = cLo > 0.0 ? cLo / H : (BETA / H) * cLo;
      jhi[j] = cHi > 0.0 ? cHi / H : (BETA / H) * cHi;
      if (j === ANKLE_R || j === ANKLE_L) {
        let t = -ANKLE_GAIN * ang;
        if (t > ANKLE_SPEED) t = ANKLE_SPEED;
        else if (t < -ANKLE_SPEED) t = -ANKLE_SPEED;
        tgt[j] = t;
      }
    }

    for (let c = 0; c < NC; c++) {
      const ib = C_BODY[c], i = ib * 6;
      const rx = cs[ib] * C_X[c] - sn[ib] * C_Y[c];
      const ry = sn[ib] * C_X[c] + cs[ib] * C_Y[c] - C_R[c];
      crx[c] = rx; cry[c] = ry;
      ckn[c] = 1.0 / (INV_M[ib] + INV_I[ib] * rx * rx);
      ckt[c] = 1.0 / (INV_M[ib] + INV_I[ib] * ry * ry);
      const sep = s[i + 1] + ry;
      if (sep > 0.0) {
        cb[c] = sep / H;
      } else {
        let bias = (BETA / H) * (sep + SLOP);
        if (bias > 0.0) bias = 0.0;
        if (bias < -MAX_CORR) bias = -MAX_CORR;
        cb[c] = bias;
      }
    }

    // warm start
    for (let j = 0; j < NJ; j++) {
      const ia = J_A[j], ib = J_B[j], a = ia * 6, b = ib * 6, o = j * 5;
      const px = acc[o], py = acc[o + 1];
      const aimp = acc[o + 2] + acc[o + 3] - acc[o + 4];
      s[a + 3] -= INV_M[ia] * px;
      s[a + 4] -= INV_M[ia] * py;
      s[a + 5] -= INV_I[ia] * (jrax[j] * py - jray[j] * px + aimp);
      s[b + 3] += INV_M[ib] * px;
      s[b + 4] += INV_M[ib] * py;
      s[b + 5] += INV_I[ib] * (jrbx[j] * py - jrby[j] * px + aimp);
    }
    for (let c = 0; c < NC; c++) {
      const ib = C_BODY[c], i = ib * 6, o = NJ * 5 + c * 2;
      const pn = acc[o], pt = acc[o + 1];
      s[i + 3] += INV_M[ib] * pt;
      s[i + 4] += INV_M[ib] * pn;
      s[i + 5] += INV_I[ib] * (crx[c] * pn - cry[c] * pt);
    }

    for (let it = 0; it < ITERS; it++) {
      for (let j = 0; j < NJ; j++) {
        const ia = J_A[j], ib = J_B[j], a = ia * 6, b = ib * 6, o = j * 5;
        let lam, mx, old, nw;
        // motor
        lam = (tgt[j] - (s[b + 5] - s[a + 5])) * jka[j];
        mx = J_TORQUE[j] * H;
        old = acc[o + 2];
        nw = old + lam;
        if (nw > mx) nw = mx;
        else if (nw < -mx) nw = -mx;
        acc[o + 2] = nw;
        lam = nw - old;
        s[a + 5] -= INV_I[ia] * lam;
        s[b + 5] += INV_I[ib] * lam;
        // lower limit
        lam = -((s[b + 5] - s[a + 5]) + jlo[j]) * jka[j];
        old = acc[o + 3];
        nw = old + lam;
        if (nw < 0.0) nw = 0.0;
        acc[o + 3] = nw;
        lam = nw - old;
        s[a + 5] -= INV_I[ia] * lam;
        s[b + 5] += INV_I[ib] * lam;
        // upper limit
        lam = -((s[a + 5] - s[b + 5]) + jhi[j]) * jka[j];
        old = acc[o + 4];
        nw = old + lam;
        if (nw < 0.0) nw = 0.0;
        acc[o + 4] = nw;
        lam = nw - old;
        s[a + 5] += INV_I[ia] * lam;
        s[b + 5] -= INV_I[ib] * lam;
        // point
        const cdx = s[b + 3] - s[b + 5] * jrby[j] - s[a + 3] + s[a + 5] * jray[j] + jbx[j];
        const cdy = s[b + 4] + s[b + 5] * jrbx[j] - s[a + 4] - s[a + 5] * jrax[j] + jby[j];
        const px = -(jk11[j] * cdx + jk12[j] * cdy);
        const py = -(jk12[j] * cdx + jk22[j] * cdy);
        acc[o] += px;
        acc[o + 1] += py;
        s[a + 3] -= INV_M[ia] * px;
        s[a + 4] -= INV_M[ia] * py;
        s[a + 5] -= INV_I[ia] * (jrax[j] * py - jray[j] * px);
        s[b + 3] += INV_M[ib] * px;
        s[b + 4] += INV_M[ib] * py;
        s[b + 5] += INV_I[ib] * (jrbx[j] * py - jrby[j] * px);
      }
      for (let c = 0; c < NC; c++) {
        const ib = C_BODY[c], i = ib * 6, o = NJ * 5 + c * 2;
        let lam, mx, old, nw;
        // friction
        const vt = s[i + 3] - s[i + 5] * cry[c];
        lam = -vt * ckt[c];
        mx = FRICTION * acc[o];
        old = acc[o + 1];
        nw = old + lam;
        if (nw > mx) nw = mx;
        else if (nw < -mx) nw = -mx;
        acc[o + 1] = nw;
        lam = nw - old;
        s[i + 3] += INV_M[ib] * lam;
        s[i + 5] -= INV_I[ib] * cry[c] * lam;
        // normal
        const vn = s[i + 4] + s[i + 5] * crx[c];
        lam = -(vn + cb[c]) * ckn[c];
        old = acc[o];
        nw = old + lam;
        if (nw < 0.0) nw = 0.0;
        acc[o] = nw;
        lam = nw - old;
        s[i + 4] += INV_M[ib] * lam;
        s[i + 5] += INV_I[ib] * crx[c] * lam;
      }
    }

    for (let i = 0; i < NB; i++) {
      const o = i * 6;
      for (let k = 3; k < 5; k++) {
        if (s[o + k] > MAX_V) s[o + k] = MAX_V;
        else if (s[o + k] < -MAX_V) s[o + k] = -MAX_V;
      }
      if (s[o + 5] > MAX_W) s[o + 5] = MAX_W;
      else if (s[o + 5] < -MAX_W) s[o + 5] = -MAX_W;
      s[o] += H * s[o + 3];
      s[o + 1] += H * s[o + 4];
      s[o + 2] += H * s[o + 5];
    }
  }

  function fatalTouch(s) {
    for (let c = C_FATAL0; c < NC; c++) {
      const i = C_BODY[c] * 6;
      const y = s[i + 1] + Math.sin(s[i + 2]) * C_X[c] + Math.cos(s[i + 2]) * C_Y[c] - C_R[c];
      if (y < FATAL_Y) return true;
    }
    return false;
  }

  // One control step (NSUB substeps). Returns true if the runner fell.
  function step(s, acc, action) {
    setTargets(action);
    for (let sub = 0; sub < NSUB; sub++) substep(s, acc);
    return fatalTouch(s);
  }

  function observe(s, acc, prevAction, out) {
    const tx = s[0], ty = s[1], ta = s[2], tvx = s[3], tvy = s[4], tw = s[5];
    out[0] = (ty - 1.2) * 2.0;
    out[1] = Math.sin(ta);
    out[2] = Math.cos(ta);
    out[3] = tvx * 0.2;
    out[4] = tvy * 0.2;
    out[5] = tw * 0.2;
    let k = 6;
    for (let i = 1; i < NB; i++) {
      const o = i * 6;
      out[k] = s[o] - tx;
      out[k + 1] = s[o + 1] - ty;
      out[k + 2] = Math.sin(s[o + 2]);
      out[k + 3] = Math.cos(s[o + 2]);
      out[k + 4] = (s[o + 3] - tvx) * 0.2;
      out[k + 5] = (s[o + 4] - tvy) * 0.2;
      out[k + 6] = s[o + 5] * 0.1;
      k += 7;
    }
    for (let c = 0; c < 6; c++) out[k++] = acc[NJ * 5 + c * 2] > 0.0 ? 1.0 : 0.0;
    for (let a = 0; a < NACT; a++) out[k++] = prevAction === a ? 1.0 : 0.0;
  }

  function jointAngle(s, j) {
    return s[J_B[j] * 6 + 2] - s[J_A[j] * 6 + 2];
  }

  function create() {
    const s = new Float64Array(NB * 6);
    const acc = new Float64Array(NACC);
    setPose(s, 0.0, POSE0_TORSO, POSE0);
    return { s, acc };
  }

  return {
    NB, NJ, NC, NACC, NACT, NOBS, NSUB, H, DT, GOAL_X,
    J_A, J_B, J_AX, J_AY, J_BX, J_BY, C_BODY, C_X, C_Y, C_R, POSE0, POSE0_TORSO,
    HIP_R, HIP_L, KNEE_R, KNEE_L,
    setPose, setTargets, substep, fatalTouch, step, observe, jointAngle, create,
  };
})();

if (typeof module !== 'undefined') module.exports = QwopPhysics;
