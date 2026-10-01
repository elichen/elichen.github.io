// The listener (drum/train/train_hear.py) in plain JavaScript: from the
// ratios of a drum's notes to its lowest, M candidate outlines and how sure
// the network is of each. Weights: model/listener.json + listener.bin
// (float16).

export const KMAX = 40;
const N = 64;
const NF = 32;

const gelu = (x) => 0.5 * x * (1 + Math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));

function halfToFloat(h) {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
  if (e === 0) return s * 2 ** -14 * (f / 1024);
  if (e === 31) return f ? NaN : s * Infinity;
  return s * 2 ** (e - 15) * (1 + f / 1024);
}

function layerNorm(x, g, b) {
  let m = 0, v = 0;
  for (const t of x) m += t;
  m /= x.length;
  for (const t of x) v += (t - m) ** 2;
  const inv = 1 / Math.sqrt(v / x.length + 1e-5);
  return x.map((t, i) => (t - m) * inv * g[i] + b[i]);
}

function linear(x, W, B) {
  const out = B.length, inp = x.length, y = new Float32Array(out);
  for (let o = 0; o < out; o++) {
    let s = B[o];
    const r = o * inp;
    for (let i = 0; i < inp; i++) s += W[r + i] * x[i];
    y[o] = s;
  }
  return y;
}

export class Listener {
  static async load(url) {
    const meta = await (await fetch(url)).json();
    const bin = await (await fetch(new URL(meta.file, new URL(url, location.href)))).arrayBuffer();
    return new Listener(meta, bin);
  }

  constructor(meta, bin) {
    const half = new Uint16Array(bin);
    this.t = {};
    for (const [name, [off, len]] of Object.entries(meta.tensors)) {
      const a = new Float32Array(len);
      for (let i = 0; i < len; i++) a[i] = halfToFloat(half[off + i]);
      this.t[name] = a;
    }
    this.depth = meta.depth;
    this.M = meta.M;
  }

  // lam: sorted eigenvalues (at least K of them); K: notes heard.
  // Returns { outlines: M arrays of 2*N (x, y), confidence: M softmax weights }.
  hear(lam, K) {
    const t = this.t;
    const x = new Float32Array(2 * (KMAX - 1));
    for (let k = 1; k < KMAX; k++) {
      if (k < K) {
        x[k - 1] = Math.log(lam[k] / lam[0]);
        x[KMAX - 1 + k - 1] = 1;
      }
    }
    let h = linear(x, t['inp.weight'], t['inp.bias']);
    for (let d = 0; d < this.depth; d++) {
      const p = `blocks.${d}.`;
      let u = layerNorm(h, t[p + '0.weight'], t[p + '0.bias']);
      u = linear(u, t[p + '1.weight'], t[p + '1.bias']).map(gelu);
      u = linear(u, t[p + '3.weight'], t[p + '3.bias']);
      for (let i = 0; i < h.length; i++) h[i] += u[i];
    }
    const o = linear(layerNorm(h, t['norm.weight'], t['norm.bias']), t['out.weight'], t['out.bias']);
    const M = this.M, outlines = [];
    for (let m = 0; m < M; m++) {
      const re = new Float64Array(N), im = new Float64Array(N);
      for (let f = 0; f < NF; f++) {
        const k = f < NF / 2 ? f : f - NF;
        let cr = o[(m * NF + f) * 2], ci = o[(m * NF + f) * 2 + 1];
        if (k === 1) cr += 1;   // the network starts from a circle
        for (let n = 0; n < N; n++) {
          const a = (2 * Math.PI * k * n) / N;
          re[n] += cr * Math.cos(a) - ci * Math.sin(a);
          im[n] += cr * Math.sin(a) + ci * Math.cos(a);
        }
      }
      outlines.push(unitOutline(re, im));
    }
    const logits = Array.from(o.subarray(M * 2 * NF, M * 2 * NF + M));
    const mx = Math.max(...logits), ex = logits.map((v) => Math.exp(v - mx)), s = ex.reduce((a, b) => a + b, 0);
    return { outlines, confidence: ex.map((v) => v / s) };
  }
}

// Centre and scale to unit RMS radius, as flat [x, y, ...].
function unitOutline(re, im) {
  const n = re.length;
  let mx = 0, my = 0;
  for (let i = 0; i < n; i++) { mx += re[i]; my += im[i]; }
  mx /= n; my /= n;
  let s = 0;
  for (let i = 0; i < n; i++) s += (re[i] - mx) ** 2 + (im[i] - my) ** 2;
  s = Math.sqrt(s / n) || 1;
  const out = new Float64Array(2 * n);
  for (let i = 0; i < n; i++) { out[2 * i] = (re[i] - mx) / s; out[2 * i + 1] = (im[i] - my) / s; }
  return out;
}
