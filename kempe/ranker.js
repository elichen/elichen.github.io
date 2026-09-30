// The tuning predictor: given a drawing and a machine from the shortlist, a
// small network predicts how close the machine will get once its bars are
// tuned. The page tunes the machines it rates best, not merely the ones whose
// untuned path looks most like the drawing. Trained by kempe/train/train_ranker.py
// on how 3 million (doodle, machine) pairs actually tuned.

export const N_STRUCT = 15;

// Structure of a machine, from its spec and one simulated cycle (a Fitter
// that has just run curveOf on the machine's initial design).
export function machineFeatures(m, fitter, curve) {
  const cranks = m.cranks.length;
  const g2 = cranks > 1 ? m.gear[m.cranks[1]] : 0;
  const g3 = cranks > 2 ? m.gear[m.cranks[2]] : 0;
  const buf = fitter.buf, [mx, my] = curve.mean;
  let ext = 0;
  for (let i = 0; i < buf.length; i += 2) ext = Math.max(ext, (buf[i] - mx) ** 2 + (buf[i + 1] - my) ** 2);
  ext = Math.sqrt(ext) / curve.scale;
  return Float32Array.of(
    m.n / 16, m.bars.length / 20,
    cranks === 1 ? 1 : 0, cranks === 2 ? 1 : 0, cranks === 3 ? 1 : 0,
    g2 / 4, Math.abs(g2) / 4, g3 / 4, Math.abs(g3) / 4,
    m.grounds.length / 4, m.dyads.length / 10,
    ext / 5, curve.minSin, m.r / curve.scale, m.D / 30,
  );
}

// Pair features: raw alignment error, both Fourier fingerprints and their
// difference, and the machine's structure.
export function pairFeatures(rawErr, fq, fm, struct) {
  const K = fq.length, out = new Float32Array(2 + 3 * K + struct.length);
  out[0] = rawErr;
  out[1] = Math.log(rawErr + 1e-3);
  for (let k = 0; k < K; k++) {
    out[2 + k] = fq[k];
    out[2 + K + k] = fm[k];
    out[2 + 2 * K + k] = Math.abs(fq[k] - fm[k]);
  }
  out.set(struct, 2 + 3 * K);
  return out;
}

export class Ranker {
  static async load(url) {
    const meta = await (await fetch(url)).json();
    return new Ranker(meta);
  }

  // meta: { mean, std, layers: [{ w: [out][in], b: [out] }, ...] } with GELU between layers
  constructor(meta) {
    this.mean = Float32Array.from(meta.mean);
    this.std = Float32Array.from(meta.std);
    this.layers = meta.layers.map((l) => ({
      out: l.b.length, inp: l.w[0].length, w: Float32Array.from(l.w.flat()), b: Float32Array.from(l.b),
    }));
  }

  // Predicted log tuned error; lower is better.
  score(x) {
    let h = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) h[i] = (x[i] - this.mean[i]) / this.std[i];
    this.layers.forEach((l, li) => {
      const y = new Float32Array(l.out);
      for (let o = 0; o < l.out; o++) {
        let s = l.b[o];
        const row = o * l.inp;
        for (let i = 0; i < l.inp; i++) s += l.w[row + i] * h[i];
        y[o] = li < this.layers.length - 1 ? 0.5 * s * (1 + Math.tanh(0.7978845608028654 * (s + 0.044715 * s * s * s))) : s;
      }
      h = y;
    });
    return h[0];
  }
}
