// Latent visual world model (9.95M params) on TensorFlow.js (WebGPU / WebGL / CPU):
//   1) Frozen ViT-tiny encoder (5.50M) + JEPA projector (0.79M)
//   2) Frozen Readout2 transformer (1.65M) over 8x8 pooled patch tokens
//   3) LatentSkip1V2 1-step autoregressive latent world model (2.80M) with 5-step Kalman warm-start
//
// Weights come from export_browser.py: dense kernels int8 (per output column), conv/small f16, biases/LN f32.

const INV_SQRT2 = Math.SQRT1_2;

export function halfToFloat(h) {
  const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

function decodeTensor(buf, t) {
  const n = t.shape.reduce((a, b) => a * b, 1);
  if (t.dtype === 'f32') return new Float32Array(buf, t.offset, n);
  if (t.dtype === 'f16') {
    const h = new Uint16Array(buf, t.offset, n), out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = halfToFloat(h[i]);
    return out;
  }
  const q = new Int8Array(buf, t.offset, n), cols = t.shape[1];
  const scale = new Float32Array(buf, t.scaleOffset, cols), out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = q[i] * scale[i % cols];
  return out;
}

export class WorldModel {
  constructor(tf, meta, buf) {
    this.tf = tf;
    this.meta = meta;
    this.w = {};
    for (const t of meta.tensors) {
      this.w[t.name] = tf.tensor(decodeTensor(buf, t), t.shape, 'float32');
    }
    this.imgMean = tf.tensor(meta.imgMean, [1, 1, 1, 3], 'float32');
    this.imgStd = tf.tensor(meta.imgStd, [1, 1, 1, 3], 'float32');
    this.actMean = Float32Array.from(meta.actMean);
    this.actStd = Float32Array.from(meta.actStd);
    this.actMeanT = tf.tensor(meta.actMean, [1, 5], 'float32');
    this.actStdT = tf.tensor(meta.actStd, [1, 5], 'float32');
    this.vitEps = meta.vit.eps;
    this.roEps = meta.ro.eps;
    this.skEps = meta.sk.eps;
    this.skDz = meta.sk.dz;
  }

  dispose() {
    for (const t of Object.values(this.w)) t.dispose();
    this.imgMean.dispose();
    this.imgStd.dispose();
    this.actMeanT.dispose();
    this.actStdT.dispose();
  }

  linear(x, name) {
    return this.tf.add(this.tf.matMul(x, this.w[`${name}/w`]), this.w[`${name}/b`]);
  }

  layerNorm(x, name, eps) {
    const tf = this.tf;
    const { mean, variance } = tf.moments(x, -1, true);
    const norm = x.sub(mean).div(variance.add(eps).sqrt());
    return norm.mul(this.w[`${name}/w`]).add(this.w[`${name}/b`]);
  }

  gelu(x) {
    const tf = this.tf;
    return x.mul(0.5).mul(tf.erf(x.mul(INV_SQRT2)).add(1));
  }

  silu(x) {
    return x.mul(this.tf.sigmoid(x));
  }

  // Pre-norm transformer block for rank-2 token matrix [T, D] (batch=1)
  transformerBlock2D(x, prefix, heads, eps) {
    const tf = this.tf;
    const [T, D] = x.shape;
    const hd = D / heads;
    const h1 = this.layerNorm(x, `${prefix}/ln1`, eps);
    const qkv = this.linear(h1, `${prefix}/qkv`).reshape([T, 3, heads, hd]);
    const [q, k, v] = tf.unstack(qkv, 1); // each [T, heads, hd]
    const qt = q.transpose([1, 0, 2]);    // [heads, T, hd]
    const kt = k.transpose([1, 2, 0]);    // [heads, hd, T]
    const vt = v.transpose([1, 0, 2]);    // [heads, T, hd]
    const attn = tf.softmax(tf.matMul(qt, kt).mul(1 / Math.sqrt(hd)), -1);
    const ctx = tf.matMul(attn, vt).transpose([1, 0, 2]).reshape([T, D]);
    const x1 = x.add(this.linear(ctx, `${prefix}/o`));
    const h2 = this.layerNorm(x1, `${prefix}/ln2`, eps);
    const mlp = this.linear(this.gelu(this.linear(h2, `${prefix}/fc1`)), `${prefix}/fc2`);
    return x1.add(mlp);
  }

  // Encode a single 224x224 RGB uint8 (or RGBA uint8) image into:
  //   z0: Float32Array(192) — projected CLS embedding for action-bank retrieval
  //   s0: Float32Array(7)   — [cube_xyz(3), hand_xyz(3), contact(1)] from Readout2
  //   rawReadout: Float32Array(10) — [cube(3), hand_direct(3), rel(3), contact(1)]
  async encodeFrame(pixels, channels = 3) {
    const tf = this.tf;
    const [z0T, rawT] = tf.tidy(() => {
      let im = tf.tensor(pixels, [1, 224, 224, channels], 'float32');
      if (channels === 4) im = im.slice([0, 0, 0, 0], [1, 224, 224, 3]);
      im = im.div(255).sub(this.imgMean).div(this.imgStd);

      // ViT-tiny patch embedding: [1, 224, 224, 3] -> [1, 16, 16, 192] -> [256, 192]
      const patch = tf.conv2d(im, this.w['enc/patch/w'], 14, 'valid')
        .add(this.w['enc/patch/b'])
        .reshape([256, 192]);
      let x = tf.concat([this.w['enc/cls'].reshape([1, 192]), patch], 0)
        .add(this.w['enc/pos'].reshape([257, 192]));

      for (let l = 0; l < 12; l++) {
        x = this.transformerBlock2D(x, `enc/l${l}`, 3, this.vitEps);
      }
      x = this.layerNorm(x, 'enc/ln', this.vitEps); // [257, 192]

      // Projector on CLS token -> z0 [192]
      const cls = x.slice([0, 0], [1, 192]);
      const p1 = this.linear(cls, 'proj/fc1')
        .mul(this.w['proj/bn/scale'])
        .add(this.w['proj/bn/bias']);
      const z0 = this.linear(this.gelu(p1), 'proj/fc2').reshape([192]);

      // 2x2 average pool on 16x16 patch tokens -> 8x8 = 64 standardized tokens
      const patches = x.slice([1, 0], [256, 192]).reshape([1, 16, 16, 192]);
      const pooled = tf.avgPool(patches, 2, 2, 'valid').reshape([64, 192]);
      const tok = pooled.sub(this.w['tok/mu']).div(this.w['tok/sd']);

      // Readout2 (3-layer pre-norm transformer, d=256, heads=4)
      let rx = tf.concat([
        this.w['ro/cls'].reshape([1, 256]),
        this.linear(tok, 'ro/inp').add(this.w['ro/pos'].reshape([64, 256])),
      ], 0);
      for (let l = 0; l < 3; l++) {
        rx = this.transformerBlock2D(rx, `ro/l${l}`, 4, this.roEps);
      }
      const rcls = rx.slice([0, 0], [1, 256]);
      const raw = this.linear(this.layerNorm(rcls, 'ro/out_ln', this.roEps), 'ro/out_fc')
        .reshape([10])
        .mul(this.w['ro/ys'])
        .add(this.w['ro/ym']);
      return [z0, raw];
    });

    const [z0, rawReadout] = await Promise.all([z0T.data(), rawT.data()]);
    z0T.dispose();
    rawT.dispose();
    const s0 = new Float32Array(7);
    s0[0] = rawReadout[0]; s0[1] = rawReadout[1]; s0[2] = rawReadout[2];
    s0[3] = rawReadout[0] + rawReadout[6];
    s0[4] = rawReadout[1] + rawReadout[7];
    s0[5] = rawReadout[2] + rawReadout[8];
    s0[6] = Math.min(1, Math.max(0, rawReadout[9]));
    return { z0: Float32Array.from(z0), s0, rawReadout: Float32Array.from(rawReadout) };
  }

  // One transition step of LatentSkip1V2: (s [B,7], z [B,128], a [B,5], c0 [B,3]) -> { s, z, h }
  _skip1Step(s, z, a, c0) {
    const tf = this.tf;
    const cube = s.slice([0, 0], [-1, 3]);
    const hand = s.slice([0, 3], [-1, 3]);
    const con = s.slice([0, 6], [-1, 1]);
    const rel = hand.sub(cube);
    const dist = rel.square().sum(-1, true).add(1e-12).sqrt();
    const dc0 = cube.sub(c0);
    let h = this.linear(tf.concat([s, rel, dist, dc0, a, z], -1), 'sk/inp');

    for (let b = 0; b < 3; b++) {
      const f1 = this.linear(this.layerNorm(h, `sk/b${b}/ln`, this.skEps), `sk/b${b}/f1`);
      const [u, v] = tf.split(f1, 2, -1);
      h = h.add(this.linear(this.silu(u).mul(v), `sk/b${b}/f2`));
    }
    h = this.layerNorm(h, 'sk/ln', this.skEps);
    const ds = this.linear(h, 'sk/ds');
    const s6 = s.slice([0, 0], [-1, 6]).add(ds.slice([0, 0], [-1, 6]));
    const sc = con.add(ds.slice([0, 6], [-1, 1])).clipByValue(-0.2, 1.2);
    const sNext = tf.concat([s6, sc], -1);

    // GRUCell(512, 128) matching PyTorch gate order [r, z, n]
    const gi = tf.add(tf.matMul(h, this.w['sk/gru/w_ih']), this.w['sk/gru/b_ih']);
    const gh = tf.add(tf.matMul(z, this.w['sk/gru/w_hh']), this.w['sk/gru/b_hh']);
    const [ir, iz, in_] = tf.split(gi, 3, -1);
    const [hr, hz, hn] = tf.split(gh, 3, -1);
    const rGate = tf.sigmoid(ir.add(hr));
    const zGate = tf.sigmoid(iz.add(hz));
    const nGate = tf.tanh(in_.add(rGate.mul(hn)));
    const zNext = tf.sub(1, zGate).mul(nGate).add(zGate.mul(z));
    return { s: sNext, z: zNext, h };
  }

  // 5-step Kalman warm-start (returns a [1, 128] tf.Tensor — caller disposes)
  warmstartZTensor(sNowArr, sPrevArr = null, aPrevNormArr = null) {
    const tf = this.tf;
    return tf.tidy(() => {
      const sNow = tf.tensor2d(sNowArr, [1, 7], 'float32');
      const rel0 = sNow.slice([0, 3], [1, 3]).sub(sNow.slice([0, 0], [1, 3]));
      const zInit = (s, rel) => this.linear(this.silu(this.linear(tf.concat([s, rel], -1), 'sk/zi0')), 'sk/zi1');
      if (!sPrevArr || !aPrevNormArr || aPrevNormArr.length === 0) {
        return zInit(sNow, rel0);
      }
      const nSteps = aPrevNormArr.length / 5;
      let s = tf.tensor2d(sPrevArr, [1, 7], 'float32');
      const c0p = s.slice([0, 0], [1, 3]);
      const relp = s.slice([0, 3], [1, 3]).sub(c0p);
      let z = zInit(s, relp);
      for (let k = 0; k < nSteps; k++) {
        const ak = tf.tensor2d(aPrevNormArr.subarray(k * 5, k * 5 + 5), [1, 5], 'float32');
        ({ s, z } = this._skip1Step(s, z, ak, c0p));
      }
      const innov = sNow.sub(s);
      const fuseIn = tf.concat([z, sNow, rel0, innov], -1);
      return z.add(this.linear(this.silu(this.linear(fuseIn, 'sk/zf0')), 'sk/zf1'));
    });
  }

  // Roll out B candidate normalized action sequences of length T.
  // actsNorm: Float32Array(B * T * 5) or tf.Tensor3D [B, T, 5]
  // Returns { s: [B, T, 7], ls: [B, T, 3], ih: [B, T] } as tf.Tensors (caller disposes).
  rolloutTensors(s0Arr, z0Tensor, actsNorm, B, T) {
    const tf = this.tf;
    return tf.tidy(() => {
      const s0 = tf.tensor2d(s0Arr, [1, 7], 'float32').tile([B, 1]);
      const c0 = s0.slice([0, 0], [B, 3]);
      let s = s0;
      let z = z0Tensor.tile([B, 1]);
      const acts = actsNorm instanceof tf.Tensor ? actsNorm : tf.tensor3d(actsNorm, [B, T, 5], 'float32');
      const ss = [], lss = [], ihs = [];
      for (let k = 0; k < T; k++) {
        const ak = acts.slice([0, k, 0], [B, 1, 5]).reshape([B, 5]);
        let h;
        ({ s, z, h } = this._skip1Step(s, z, ak, c0));
        ss.push(s);
        lss.push(this.linear(h, 'sk/ls').clipByValue(-5.5, -1.0));
        ihs.push(this.linear(h, 'sk/ih').reshape([B]));
      }
      return {
        s: tf.stack(ss, 1),     // [B, T, 7]
        ls: tf.stack(lss, 1),   // [B, T, 3]
        ih: tf.stack(ihs, 1),   // [B, T]
      };
    });
  }
}

export async function loadWorldModel(tf, baseUrl = 'model') {
  const [meta, buf] = await Promise.all([
    fetch(`${baseUrl}/weights.json`).then(r => r.json()),
    fetch(`${baseUrl}/weights.bin`).then(r => r.arrayBuffer()),
  ]);
  return new WorldModel(tf, meta, buf);
}
