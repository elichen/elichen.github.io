// The curve encoder (kempe/train/encoder.py) in plain JavaScript: a closed
// curve -> a unit vector whose dot products rank machine curves by how well
// they align with it. Weights come from model/encoder.json + encoder.bin.

const LAGS = [1, 2, 4, 8, 16, 32];
const TLAGS = [1, 2, 4];
const C_IN = 1 + 2 * LAGS.length + 2 * TLAGS.length + 1;

const gelu = (x) => 0.5 * x * (1 + Math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));

function features(re, im) {
  const n = re.length, F = new Float32Array(C_IN * n);
  let c = 0;
  for (let i = 0; i < n; i++) F[i] = Math.hypot(re[i], im[i]);
  c = 1;
  for (const k of LAGS) {
    for (let i = 0; i < n; i++) {
      const j = (i + k) % n;   // z[i+k] * conj(z[i])
      F[c * n + i] = re[j] * re[i] + im[j] * im[i];
      F[(c + 1) * n + i] = im[j] * re[i] - re[j] * im[i];
    }
    c += 2;
  }
  const tr = new Float64Array(n), ti = new Float64Array(n);
  let tn = 0, perim = 0;
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    tr[i] = re[j] - re[i]; ti[i] = im[j] - im[i];
    const m2 = tr[i] * tr[i] + ti[i] * ti[i];
    tn += m2; perim += Math.sqrt(m2);
  }
  tn = tn / n + 1e-9;
  for (const k of TLAGS) {
    for (let i = 0; i < n; i++) {
      const j = (i + k) % n;
      F[c * n + i] = (tr[j] * tr[i] + ti[j] * ti[i]) / tn;
      F[(c + 1) * n + i] = (ti[j] * tr[i] - tr[j] * ti[i]) / tn;
    }
    c += 2;
  }
  for (let i = 0; i < n; i++) F[c * n + i] = perim / 10;
  return F;
}

// Circular conv1d, kernel 3, dilation d: out[o][i] = b[o] + sum w[o][c][k] x[c][i + (k-1)d]
function conv(x, cin, n, W, B, cout, d) {
  const y = new Float32Array(cout * n);
  for (let o = 0; o < cout; o++) {
    const yo = o * n;
    for (let i = 0; i < n; i++) y[yo + i] = B[o];
    for (let c = 0; c < cin; c++) {
      const w0 = W[(o * cin + c) * 3], w1 = W[(o * cin + c) * 3 + 1], w2 = W[(o * cin + c) * 3 + 2];
      const xc = c * n;
      for (let i = 0; i < n; i++) {
        const im = (i - d + n) % n, ip = (i + d) % n;
        y[yo + i] += w0 * x[xc + im] + w1 * x[xc + i] + w2 * x[xc + ip];
      }
    }
  }
  return y;
}

// LayerNorm over channels at each position.
function lnChannels(x, C, n, g, b) {
  const y = new Float32Array(C * n);
  for (let i = 0; i < n; i++) {
    let m = 0, v = 0;
    for (let c = 0; c < C; c++) m += x[c * n + i];
    m /= C;
    for (let c = 0; c < C; c++) { const t = x[c * n + i] - m; v += t * t; }
    const inv = 1 / Math.sqrt(v / C + 1e-5);
    for (let c = 0; c < C; c++) y[c * n + i] = (x[c * n + i] - m) * inv * g[c] + b[c];
  }
  return y;
}

function linear(x, W, B, out, inp) {
  const y = new Float32Array(out);
  for (let o = 0; o < out; o++) {
    let s = B[o];
    for (let i = 0; i < inp; i++) s += W[o * inp + i] * x[i];
    y[o] = s;
  }
  return y;
}

export class CurveEncoder {
  static async load(url) {
    const meta = await (await fetch(url)).json();
    const bin = await (await fetch(new URL(meta.file, new URL(url, location.href)))).arrayBuffer();
    return new CurveEncoder(meta, bin);
  }

  constructor(meta, bin) {
    this.width = meta.width;
    this.dim = meta.dim;
    this.dils = meta.dils;
    this.t = {};
    for (const [name, [off, len]] of Object.entries(meta.tensors)) {
      this.t[name] = new Float32Array(bin, off, len);
    }
  }

  branch(re, im) {
    const n = re.length, Wd = this.width, t = this.t;
    let x = conv(features(re, im), C_IN, n, t['inp.weight'], t['inp.bias'], Wd, 1);
    this.dils.forEach((d, k) => {
      const p = `blocks.${k}.`;
      let h = lnChannels(x, Wd, n, t[p + 'norm.weight'], t[p + 'norm.bias']);
      h = conv(h, Wd, n, t[p + 'c1.weight'], t[p + 'c1.bias'], Wd, d);
      for (let i = 0; i < h.length; i++) h[i] = gelu(h[i]);
      h = conv(h, Wd, n, t[p + 'c2.weight'], t[p + 'c2.bias'], Wd, d);
      for (let i = 0; i < x.length; i++) x[i] += h[i];
    });
    const pooled = new Float32Array(2 * Wd);
    for (let c = 0; c < Wd; c++) {
      let s = 0, m = -Infinity;
      for (let i = 0; i < n; i++) { const v = x[c * n + i]; s += v; if (v > m) m = v; }
      pooled[c] = s / n; pooled[Wd + c] = m;
    }
    let h = lnChannels(pooled, 2 * Wd, 1, t['norm.weight'], t['norm.bias']);
    h = linear(h, t['fc1.weight'], t['fc1.bias'], 2 * Wd, 2 * Wd);
    for (let i = 0; i < h.length; i++) h[i] = gelu(h[i]);
    return linear(h, t['fc2.weight'], t['fc2.bias'], this.dim, 2 * Wd);
  }

  // re, im: normalised curve samples (64). Returns a unit Float32Array(dim).
  embed(re, im) {
    const n = re.length, e = new Float32Array(this.dim);
    const vr = new Float64Array(n), vi = new Float64Array(n);
    for (let v = 0; v < 4; v++) {
      for (let k = 0; k < n; k++) {
        const src = v & 1 ? (n - k) % n : k;
        vr[k] = re[src]; vi[k] = v & 2 ? -im[src] : im[src];
      }
      const b = this.branch(vr, vi);
      for (let i = 0; i < this.dim; i++) e[i] += b[i];
    }
    return unit(e);
  }
}

function unit(e) {
  let s = 0;
  for (const v of e) s += v * v;
  s = Math.sqrt(s) || 1;
  for (let i = 0; i < e.length; i++) e[i] /= s;
  return e;
}

// Non-learned fallback: symmetrised Fourier magnitudes (encoder.py
// fourier_descriptor), used only when no trained encoder is shipped.
export function fourierDescriptor(re, im, K = 10) {
  const n = re.length, d = new Float32Array(2 * K);
  const coef = (k) => {
    let a = 0, b = 0;
    for (let j = 0; j < n; j++) {
      const ang = (-2 * Math.PI * k * j) / n;
      a += re[j] * Math.cos(ang) - im[j] * Math.sin(ang);
      b += re[j] * Math.sin(ang) + im[j] * Math.cos(ang);
    }
    return Math.hypot(a, b) / n;
  };
  for (let k = 1; k <= K; k++) {
    const p = coef(k), q = coef(n - k);
    d[k - 1] = p + q;
    d[K + k - 1] = Math.abs(p - q);
  }
  return unit(d);
}
