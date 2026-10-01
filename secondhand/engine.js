// Handwriting synthesis network (Graves 2013), inference only.
// Runs in a Worker or Node. Weights: fp16 blob + JSON layout from export.py.
(function (root) {
  "use strict";

  function halfToFloat(u16) {
    const out = new Float32Array(u16.length);
    for (let i = 0; i < u16.length; i++) {
      const h = u16[i], s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
      out[i] = e === 0 ? s * f * 5.960464477539063e-8 : e === 31 ? (f ? NaN : s * Infinity) : s * Math.pow(2, e - 15) * (1 + f / 1024);
    }
    return out;
  }

  // y = W x + b, W row-major [rows][cols]
  function matvec(W, b, x, y, rows, cols) {
    for (let r = 0; r < rows; r++) {
      let s0 = 0, s1 = 0, s2 = 0, s3 = 0, o = r * cols, c = 0;
      for (; c + 3 < cols; c += 4) {
        s0 += W[o + c] * x[c]; s1 += W[o + c + 1] * x[c + 1];
        s2 += W[o + c + 2] * x[c + 2]; s3 += W[o + c + 3] * x[c + 3];
      }
      for (; c < cols; c++) s0 += W[o + c] * x[c];
      y[r] = b[r] + s0 + s1 + s2 + s3;
    }
  }

  const sig = (v) => 1 / (1 + Math.exp(-v));
  const JUMP = 2.5;

  function lstm(g, c, h, H) {
    for (let j = 0; j < H; j++) {
      const i = sig(g[j]), f = sig(g[H + j]), gg = Math.tanh(g[2 * H + j]), o = sig(g[3 * H + j]);
      c[j] = f * c[j] + i * gg;
      h[j] = o * Math.tanh(c[j]);
    }
  }

  class HandModel {
    constructor(meta, blob) {
      this.meta = meta;
      const { H, M, K, V } = meta;
      Object.assign(this, { H, M, K, V });
      const u16 = new Uint16Array(blob);
      this.w = {};
      for (const [name, { offset, shape }] of Object.entries(meta.layout)) {
        const n = shape.reduce((a, b) => a * b, 1);
        this.w[name] = halfToFloat(u16.subarray(offset, offset + n));
      }
      this.vocab = new Map([...meta.vocab].map((ch, i) => [ch, i + 1]));
      this.spacing = meta.spacing;
      this.in1 = 3 + V + H; this.in2 = 3 + V + 2 * H;
      this.buf1 = new Float32Array(this.in1); this.buf2 = new Float32Array(this.in2);
      this.gates = new Float32Array(4 * H); this.win = new Float32Array(3 * K);
      this.hcat = new Float32Array(3 * H); this.y = new Float32Array(1 + 6 * M);
    }

    encode(text) {
      return Int32Array.from([...text], (ch) => this.vocab.get(ch) || 0);
    }

    newState(text) {
      const { H, K, V } = this;
      return {
        codes: this.encode(text),
        h1: new Float32Array(H), c1: new Float32Array(H),
        h2: new Float32Array(H), c2: new Float32Array(H),
        h3: new Float32Array(H), c3: new Float32Array(H),
        kappa: new Float32Array(K), w: new Float32Array(V),
        phi: new Float32Array(text.length), phiEnd: 0, y: new Float32Array(1 + 6 * this.M)
      };
    }

    cloneState(s) {
      const o = {};
      for (const [k, v] of Object.entries(s)) o[k] = ArrayBuffer.isView(v) ? v.slice() : v;
      return o;
    }

    // Feed one pen move [dx, dy, pen]; leaves the prediction for the next move in s.y.
    step(s, dx, dy, pen) {
      const { H, K, V, w } = this;
      const b1 = this.buf1;
      b1[0] = dx; b1[1] = dy; b1[2] = pen;
      b1.set(s.w, 3); b1.set(s.h1, 3 + V);
      matvec(w.W1, w.b1, b1, this.gates, 4 * H, this.in1);
      lstm(this.gates, s.c1, s.h1, H);

      // attention window over the characters
      matvec(w.Wwin, w.bwin, s.h1, this.win, 3 * K, H);
      const U = s.codes.length;
      s.w.fill(0);
      let phiEnd = 0;
      for (let k = 0; k < K; k++) s.kappa[k] += Math.exp(this.win[2 * K + k]);
      for (let u = 0; u <= U; u++) {
        let p = 0;
        for (let k = 0; k < K; k++) {
          const d = s.kappa[k] - u;
          p += Math.exp(this.win[k]) * Math.exp(-Math.exp(this.win[K + k]) * d * d);
        }
        if (u < U) { s.phi[u] = p; if (s.codes[u]) s.w[s.codes[u]] += p; }
        else phiEnd = p;
      }
      s.phiEnd = phiEnd;

      const b2 = this.buf2;
      b2[0] = dx; b2[1] = dy; b2[2] = pen;
      b2.set(s.w, 3); b2.set(s.h1, 3 + V); b2.set(s.h2, 3 + V + H);
      matvec(w.W2, w.b2, b2, this.gates, 4 * H, this.in2);
      lstm(this.gates, s.c2, s.h2, H);
      b2.set(s.h2, 3 + V); b2.set(s.h3, 3 + V + H);
      matvec(w.W3, w.b3, b2, this.gates, 4 * H, this.in2);
      lstm(this.gates, s.c3, s.h3, H);

      this.hcat.set(s.h1, 0); this.hcat.set(s.h2, H); this.hcat.set(s.h3, 2 * H);
      matvec(w.Wout, w.bout, this.hcat, s.y, 1 + 6 * this.M, 3 * H);
      return s;
    }

    // Draw the next move from s.y. bias > 0 sharpens toward the most likely stroke.
    sample(s, bias, rand) {
      const { M } = this, y = s.y;
      let mx = -Infinity;
      const pis = new Float32Array(M);
      for (let m = 0; m < M; m++) { pis[m] = y[1 + m] * (1 + bias); if (pis[m] > mx) mx = pis[m]; }
      let tot = 0;
      for (let m = 0; m < M; m++) { pis[m] = Math.exp(pis[m] - mx); tot += pis[m]; }
      let r = rand() * tot, m = 0;
      for (; m < M - 1; m++) { r -= pis[m]; if (r <= 0) break; }
      const mux = y[1 + M + m], muy = y[1 + 2 * M + m];
      const clampS = (v) => Math.exp(Math.max(-6, Math.min(6, v - bias)));
      const sx = clampS(y[1 + 3 * M + m]), sy = clampS(y[1 + 4 * M + m]);
      const rho = Math.tanh(y[1 + 5 * M + m]) * 0.999;
      const u1 = Math.max(rand(), 1e-12), u2 = rand();
      const z1 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      const z2 = Math.sqrt(-2 * Math.log(u1)) * Math.sin(2 * Math.PI * u2);
      const dx = mux + sx * z1, dy = muy + sy * (rho * z1 + Math.sqrt(1 - rho * rho) * z2);
      // Pen moves are resampled at one unit per step, so anything much longer
      // is a jump between strokes, whatever the separate lift draw said.
      const pen = rand() < sig(y[0]) || Math.hypot(dx, dy) > JUMP ? 1 : 0;
      return [dx, dy, pen];
    }

    // The k likeliest components of the next-move mixture, after the same
    // sharpening as sample(): [weight, mux, muy, sx, sy, rho] each, plus P(lift).
    mixture(s, bias, k, out) {
      const { M } = this, y = s.y;
      const logits = new Float32Array(M);
      let mx = -Infinity, tot = 0;
      for (let m = 0; m < M; m++) { logits[m] = y[1 + m] * (1 + bias); if (logits[m] > mx) mx = logits[m]; }
      for (let m = 0; m < M; m++) { logits[m] = Math.exp(logits[m] - mx); tot += logits[m]; }
      const order = Array.from({ length: M }, (_, m) => m).sort((a, b) => logits[b] - logits[a]);
      const clampS = (v) => Math.exp(Math.max(-6, Math.min(6, v - bias)));
      for (let j = 0; j < k; j++) {
        const m = order[j];
        out[6 * j] = logits[m] / tot;
        out[6 * j + 1] = y[1 + M + m]; out[6 * j + 2] = y[1 + 2 * M + m];
        out[6 * j + 3] = clampS(y[1 + 3 * M + m]); out[6 * j + 4] = clampS(y[1 + 4 * M + m]);
        out[6 * j + 5] = Math.tanh(y[1 + 5 * M + m]) * 0.999;
      }
      out[6 * k] = sig(y[0]);
      return out;
    }

    // Index of the character the window is centred on (for the attention strip).
    focus(s) {
      let best = 0;
      for (let u = 1; u < s.phi.length; u++) if (s.phi[u] > s.phi[best]) best = u;
      return best;
    }

    finished(s) {
      let mx = 0;
      for (let u = 0; u < s.phi.length; u++) if (s.phi[u] > mx) mx = s.phi[u];
      return s.phiEnd > mx;
    }
  }

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  const api = { HandModel, mulberry32 };
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.HandEngine = api;
})(typeof self !== "undefined" ? self : this);
