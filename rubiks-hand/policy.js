// Pure-JS neural network policy evaluator (zero dependencies).
// Reads model/{reorient,twist}/policy.{json,bin} exported from training/export.py.
// Supports observation normalization (mean/std, optional clip) and ELU / Swish / Tanh MLP layers.

export class Policy {
    constructor(meta, weights) {
        this.mean = Float64Array.from(meta.mean);
        this.std = Float64Array.from(meta.std);
        this.obsClip = meta.obsClip ?? 10.0;
        this.act = meta.act;
        this.activation = meta.activation || 'elu';
        this.tanhOut = meta.tanhOut ?? true;
        let k = 0;
        this.layers = meta.layers.map(({ in: n, out }) => {
            const w = weights.subarray(k, k += n * out);
            const b = weights.subarray(k, k += out);
            return { n, out, w, b, y: new Float64Array(out) };
        });
        this.x = new Float64Array(meta.obs);
    }

    action(obs) {
        let x = this.x;
        const clip = this.obsClip;
        for (let i = 0; i < x.length; i++) {
            let v = (obs[i] - this.mean[i]) / this.std[i];
            if (v > clip) v = clip;
            else if (v < -clip) v = -clip;
            x[i] = v;
        }
        const actFn = this.activation;
        const lastIdx = this.layers.length - 1;
        for (let l = 0; l <= lastIdx; l++) {
            const { n, out, w, b, y } = this.layers[l];
            const last = l === lastIdx;
            const outDim = last ? this.act : out;
            for (let o = 0; o < outDim; o++) {
                let s = b[o];
                for (let i = 0, r = o * n; i < n; i++) s += w[r + i] * x[i];
                if (last) {
                    y[o] = this.tanhOut ? Math.tanh(s) : Math.max(-1, Math.min(1, s));
                } else if (actFn === 'swish') {
                    y[o] = s / (1 + Math.exp(-s));
                } else if (actFn === 'tanh') {
                    y[o] = Math.tanh(s);
                } else {
                    // ELU
                    y[o] = s >= 0 ? s : Math.exp(s) - 1;
                }
            }
            x = y;
        }
        return x.subarray(0, this.act);
    }
}
