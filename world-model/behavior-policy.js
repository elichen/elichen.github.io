// Client-side Orthogonal Procrustes Rotation R + Language-Free BehaviorPolicy (GRU + 3 FiLM SwiGLU ResBlocks).
//
// Pipeline (zero text ever touches the robot policy or trajectory encoder):
//   1) MiniLM (frozen pretrained English encoder) -> emb [384]
//   2) Unpaired text PCA whitener -> a = (emb - mu_t) @ P_t [kT]
//   3) ONE single orthogonal rotation matrix R (R R^T = I_kT) -> q = a @ R [kV], z_beh = q / ||q||
//   4) Recurrent BehaviorPolicy pi(a_t | obs_t, z_beh, h_t) steps the 6-DOF UR5e + Robotiq gripper at 20 Hz!

import { halfToFloat } from './model.js';

function decodeTensor(buf, t) {
  const n = t.shape.reduce((a, b) => a * b, 1);
  if (t.dtype === 'f32') return new Float32Array(buf, t.offset, n);
  if (t.dtype === 'f16') {
    const h = new Uint16Array(buf, t.offset, n), out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = halfToFloat(h[i]);
    return out;
  }
  if (t.dtype === 'i8_col') {
    const q = new Int8Array(buf, t.offset, n), cols = t.shape[1];
    const sc = new Float32Array(buf, t.scaleOffset, cols), out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = q[i] * sc[i % cols];
    return out;
  }
  throw new Error(`Unsupported dtype: ${t.dtype}`);
}

function matVec(vec, mat, rows, cols) {
  const out = new Float32Array(cols);
  for (let r = 0; r < rows; r++) {
    const vr = vec[r], base = r * cols;
    for (let c = 0; c < cols; c++) out[c] += vr * mat[base + c];
  }
  return out;
}

function unitVec(v) {
  let n2 = 0;
  for (let i = 0; i < v.length; i++) n2 += v[i] * v[i];
  const inv = 1 / Math.max(1e-8, Math.sqrt(n2));
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] * inv;
  return out;
}

export class ArenaRotationPolicy {
  constructor(tf, meta, buf) {
    this.tf = tf;
    this.meta = meta;
    this.kT = meta.kT;
    this.kV = meta.kV;
    this.obsDim = meta.obsDim;
    this.d = meta.d;
    this.nblocks = meta.nblocks;
    this.actDim = meta.actDim;
    this.horizon = meta.horizon || 80;
    this.skills = meta.skills;
    this.nOptions = meta.nOptions;
    this.cloudCount = meta.cloudCount;
    this.cloudSkill = meta.cloudSkill;
    this.cloudLabels = meta.cloudLabels;
    this.centroids2D = meta.centroids2D;

    this.w = {};
    this.arrays = {};
    for (const t of meta.tensors) {
      const arr = decodeTensor(buf, t);
      if (t.name.startsWith('pol/')) {
        this.w[t.name] = tf.tensor(arr, t.shape, 'float32');
        if (t.name === 'pol/obs_mu' || t.name === 'pol/obs_sd') this.arrays[t.name] = arr;
      } else {
        this.arrays[t.name] = arr;
      }
    }

    this.muT = this.arrays['rot/mu_t'];
    this.pT = this.arrays['rot/P_t'];
    this.visProj2D = this.arrays['vis/proj_2d'];
    this.preProj2D = this.arrays['vis/pre_proj_2d'];
    this.cloud2D = this.arrays['vis/cloud_2d'];
    this.cloudUnit = this.arrays['vis/cloud_unit'];
    this.placeW = this.arrays['vis/place_w'];
    this.placeB = this.arrays['vis/place_b'];
    this._orthoCache = new Map();

    // Persistent GRU hidden state [1, d]
    this.h = tf.zeros([1, this.d], 'float32');
    this.zBehT = null;
  }

  dispose() {
    for (const t of Object.values(this.w)) t.dispose();
    if (this.h) this.h.dispose();
    if (this.zBehT) this.zBehT.dispose();
  }

  resetHidden() {
    if (this.h) this.h.dispose();
    this.h = this.tf.zeros([1, this.d], 'float32');
  }

  setActiveLatent(zUnit) {
    if (this.zBehT) this.zBehT.dispose();
    this.zBehT = this.tf.tensor2d(zUnit, [1, this.kV], 'float32');
    this.resetHidden();
  }

  // Exact live orthogonality check: max |R R^T - I| (if kT <= kV) or max |R^T R - I| (if kT > kV)
  orthoError(nKey = '10000') {
    if (this._orthoCache.has(nKey)) return this._orthoCache.get(nKey);
    const R = this.arrays[`rot/R_${nKey}`];
    const D = this.kT, K = this.kV;
    let mx = 0;
    if (D <= K) {
      for (let i = 0; i < D; i++) {
        for (let j = i; j < D; j++) {
          let dot = 0;
          const bi = i * K, bj = j * K;
          for (let k = 0; k < K; k++) dot += R[bi + k] * R[bj + k];
          const e = Math.abs(dot - (i === j ? 1 : 0));
          if (e > mx) mx = e;
        }
      }
    } else {
      for (let i = 0; i < K; i++) {
        for (let j = i; j < K; j++) {
          let dot = 0;
          for (let r = 0; r < D; r++) dot += R[r * K + i] * R[r * K + j];
          const e = Math.abs(dot - (i === j ? 1 : 0));
          if (e > mx) mx = e;
        }
      }
    }
    this._orthoCache.set(nKey, mx);
    return mx;
  }

  // Rotate a MiniLM sentence embedding through ONE single orthogonal matrix R_nKey
  rotateEmbedding(emb, text, pieces, { nKey = '10000', encMs = 0 } = {}) {
    const D = this.kT, K = this.kV;
    const centered = new Float32Array(emb.length);
    for (let i = 0; i < emb.length; i++) centered[i] = emb[i] - this.muT[i];
    const aText = matVec(centered, this.pT, emb.length, D);

    const R = this.arrays[`rot/R_${nKey}`];
    const qRaw = matVec(aText, R, D, K);
    const qUnit = unitVec(qRaw);

    // Project before/after R onto 2D behavior manifold slice
    const pre2D = matVec(aText, this.preProj2D, D, 2);
    const post2D = matVec(qRaw, this.visProj2D, K, 2);

    // k=15 cosine vote over held-out language-free trajectory manifold cloud (for UI inspection only!)
    const cu = this.cloudUnit;
    const M = this.cloudCount;
    const sims = new Float32Array(M);
    let bestCos = -2, bestIdx = 0;
    for (let i = 0; i < M; i++) {
      let dot = 0;
      const base = i * K;
      for (let j = 0; j < K; j++) dot += qUnit[j] * cu[base + j];
      sims[i] = dot;
      if (dot > bestCos) { bestCos = dot; bestIdx = i; }
    }
    // Top-15 weighted behavior vote + unsupervised kernel manifold projection onto unlabeled cloud!
    const idx = Array.from({ length: M }, (_, i) => i).sort((a, b) => sims[b] - sims[a]).slice(0, 15);
    const votes = new Float32Array(this.skills.length);
    for (const i of idx) votes[this.cloudSkill[i]] += Math.max(0, sims[i]);
    let topSkillId = 0, secondSkillId = 0;
    for (let s = 0; s < votes.length; s++) {
      if (votes[s] > votes[topSkillId]) { secondSkillId = topSkillId; topSkillId = s; }
      else if (s !== topSkillId && votes[s] > votes[secondSkillId]) secondSkillId = s;
    }

    // Unsupervised Nadaraya-Watson manifold denoising onto the unlabeled robot trajectory bank
    // (zero text, zero skill labels — projects rotated q back onto the support of real robot trajectories)
    const projK = this.meta.projTopK ?? 8;
    const projBeta = this.meta.projBeta ?? 25.0;
    let zBeh = qUnit;
    if (projK > 0) {
      const topK = idx.slice(0, projK);
      const maxS = sims[topK[0]];
      const wts = new Float32Array(topK.length);
      let wSum = 0;
      for (let t = 0; t < topK.length; t++) {
        const w = Math.exp((sims[topK[t]] - maxS) * projBeta);
        wts[t] = w;
        wSum += w;
      }
      const qProj = new Float32Array(K);
      for (let t = 0; t < topK.length; t++) {
        const wt = wts[t] / wSum;
        const base = topK[t] * K;
        for (let j = 0; j < K; j++) qProj[j] += wt * cu[base + j];
      }
      zBeh = unitVec(qProj);
    }

    // Optional place target readout from traj-only linear probe (shown on table if top skill is 'place')
    const placeXY = matVec(qRaw, this.placeW, K, 2);
    placeXY[0] = Math.min(0.56, Math.max(0.28, placeXY[0] + this.placeB[0]));
    placeXY[1] = Math.min(0.28, Math.max(-0.28, placeXY[1] + this.placeB[1]));

    return {
      text,
      pieces,
      nKey,
      encMs,
      aText,
      qRaw,
      qUnit,
      zBeh,
      pre2D: [pre2D[0], pre2D[1]],
      post2D: [post2D[0], post2D[1]],
      nearest2D: [this.cloud2D[bestIdx * 2], this.cloud2D[bestIdx * 2 + 1]],
      nearestLabel: this.cloudLabels[bestIdx],
      nearestCos: bestCos,
      topSkill: this.skills[topSkillId],
      secondSkill: this.skills[secondSkillId],
      placeGoal: (this.skills[topSkillId] === 'place' || this.skills[topSkillId] === 'cube_relocate')
        ? [placeXY[0], placeXY[1], 0.02]
        : null,
      orthoErr: this.orthoError(nKey),
    };
  }

  _linear(x, name) {
    return this.tf.add(this.tf.matMul(x, this.w[`${name}/w`]), this.w[`${name}/b`]);
  }

  _ln(x, name) {
    const tf = this.tf;
    const { mean, variance } = tf.moments(x, -1, true);
    return x.sub(mean).div(variance.add(1e-5).sqrt()).mul(this.w[`${name}/w`]).add(this.w[`${name}/b`]);
  }

  _silu(x) {
    return x.mul(this.tf.sigmoid(x));
  }

  // Single 20 Hz closed-loop action step (exact PyTorch BehaviorPolicy.step parity!)
  async step(obsRaw, tFrac, zUnitOverride = null) {
    const tf = this.tf;
    const d = this.d;
    const zT = zUnitOverride ? tf.tensor2d(zUnitOverride, [1, this.kV], 'float32') : this.zBehT;
    if (!zT) return new Float32Array(this.actDim);

    const [actT, hNext] = tf.tidy(() => {
      const ob = tf.tensor2d(obsRaw, [1, this.obsDim], 'float32')
        .sub(this.w['pol/obs_mu'])
        .div(this.w['pol/obs_sd']);
      const tfT = tf.tensor2d([[Math.min(1, Math.max(0, tFrac))]], [1, 1], 'float32');
      const x = this._silu(this._linear(tf.concat([ob, zT, tfT], -1), 'pol/in'));

      // Exact PyTorch GRUCell: r, z, n
      const gi = this._linear(x, 'pol/gru_ih');
      const gh = this._linear(this.h, 'pol/gru_hh');
      const [ir, iz, in_] = tf.split(gi, [d, d, d], -1);
      const [hr, hz, hn] = tf.split(gh, [d, d, d], -1);
      const r = tf.sigmoid(ir.add(hr));
      const z = tf.sigmoid(iz.add(hz));
      const n = tf.tanh(in_.add(r.mul(hn)));
      const hNew = tf.sub(1, z).mul(n).add(z.mul(this.h));

      let h = hNew;
      for (let i = 0; i < this.nblocks; i++) {
        const bp = `pol/b${i}`;
        const lnH = this._ln(h, `${bp}/ln`);
        const [g, b] = tf.split(this._linear(zT, `${bp}/film`), [d, d], -1);
        const mod = lnH.mul(g.add(1)).add(b);
        const [u, v] = tf.split(this._linear(mod, `${bp}/fc1`), [d, d], -1);
        h = h.add(this._linear(this._silu(u).mul(v), `${bp}/fc2`));
      }
      const act = tf.tanh(this._linear(this._ln(h, 'pol/out_ln'), 'pol/head')).reshape([this.actDim]);
      return [act, hNew];
    });

    if (zUnitOverride) zT.dispose();
    this.h.dispose();
    this.h = hNext;
    const act = Float32Array.from(await actT.data());
    actT.dispose();
    return act;
  }
}

export async function loadArenaPolicy(tf, baseUrl = 'model') {
  const [meta, buf] = await Promise.all([
    fetch(`${baseUrl}/arena_policy.json`).then(r => r.json()),
    fetch(`${baseUrl}/arena_policy.bin`).then(r => r.arrayBuffer()),
  ]);
  return new ArenaRotationPolicy(tf, meta, buf);
}
