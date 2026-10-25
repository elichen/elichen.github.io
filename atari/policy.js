// BBF's network (Impala CNN x4 width, 2048 hidden, dueling distributional head) on TensorFlow.js.
// Weights come from tools/export.py: conv kernels float16, dense kernels int8 with a scale per output column.

function halfToFloat(h) {
  const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

function decode(meta, buf, t) {
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

export class Policy {
  // meta: parsed policy.json; buf: ArrayBuffer of policy.bin
  constructor(tf, meta, buf) {
    this.tf = tf;
    this.numActions = meta.numActions;
    this.trainingSteps = meta.trainingSteps;
    this.w = {};
    for (const t of meta.tensors) this.w[t.name] = tf.tensor(decode(meta, buf, t), t.shape);
    this.support = tf.linspace(-meta.vmax, meta.vmax, meta.numAtoms);
  }

  dispose() {
    for (const t of Object.values(this.w)) t.dispose();
    this.support.dispose();
  }

  conv(x, name) {
    return this.tf.add(this.tf.conv2d(x, this.w[`${name}/kernel`], 1, 'same'), this.w[`${name}/bias`]);
  }

  dense(x, name) {
    return this.tf.add(this.tf.matMul(x, this.w[`${name}/kernel`]), this.w[`${name}/bias`]);
  }

  // states: uint8 [N, 84, 84, 4] tensor or TypedArray. Returns Q [N, A] tensor.
  q(states) {
    const tf = this.tf;
    return tf.tidy(() => {
      let x = (states instanceof tf.Tensor ? states : tf.tensor(states, [states.length / 28224, 84, 84, 4], 'float32'))
        .toFloat().div(255);
      for (let s = 0; s < 3; s++) {
        x = tf.maxPool(this.conv(x, `stage${s}/conv0`), 3, 2, 'same');
        for (let b = 0; b < 2; b++) {
          let y = this.conv(tf.relu(x), `stage${s}/conv${1 + 2 * b}`);
          y = this.conv(tf.relu(y), `stage${s}/conv${2 + 2 * b}`);
          x = tf.add(x, y);
        }
      }
      x = tf.relu(x);
      const n = x.shape[0];
      x = x.reshape([n, -1]);
      // renormalize: each sample's latent scaled to [0, 1]
      const lo = x.min(1, true), hi = x.max(1, true);
      x = x.sub(lo).div(hi.sub(lo).add(1e-5));
      x = tf.relu(this.dense(x, 'projection'));
      const A = this.numActions, atoms = this.support.shape[0];
      const adv = this.dense(x, 'advantage').reshape([n, A, atoms]);
      const value = this.dense(x, 'value').reshape([n, 1, atoms]);
      const logits = value.add(adv.sub(adv.mean(1, true)));
      return tf.softmax(logits).mul(this.support).sum(2);
    });
  }
}

export async function loadPolicy(tf, url) {
  const [meta, buf] = await Promise.all([
    fetch(`${url}/policy.json`).then(r => r.json()),
    fetch(`${url}/policy.bin`).then(r => r.arrayBuffer()),
  ]);
  return new Policy(tf, meta, buf);
}
