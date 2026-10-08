// The trained policy: Brax's PPO actor (an MLP with swish activations) run in plain
// JS. Observations are normalized with the statistics gathered during training, and
// the action is tanh of the Gaussian's mean (the noise is for exploration only).
export class Policy {
    // meta: model/policy.json; weights: model/policy.bin as a Float32Array
    constructor(meta, weights) {
        this.mean = Float64Array.from(meta.mean);
        this.std = Float64Array.from(meta.std);
        this.act = meta.act;
        let k = 0;
        this.layers = meta.layers.map(({ in: n, out }) => {
            const w = weights.subarray(k, k += n * out), b = weights.subarray(k, k += out);
            return { n, out, w, b, y: new Float64Array(out) };
        });
        this.x = new Float64Array(meta.obs);
    }

    // obs: 66 numbers -> 8 actions in [-1, 1]
    action(obs) {
        let x = this.x;
        for (let i = 0; i < x.length; i++) x[i] = (obs[i] - this.mean[i]) / this.std[i];
        this.layers.forEach(({ n, out, w, b, y }, l) => {
            const last = l === this.layers.length - 1;
            for (let o = 0; o < (last ? this.act : out); o++) {
                let s = b[o];
                for (let i = 0, r = o * n; i < n; i++) s += w[r + i] * x[i];
                y[o] = last ? Math.tanh(s) : s / (1 + Math.exp(-s));  // swish(s) = s · sigmoid(s)
            }
            x = y;
        });
        return x.subarray(0, this.act);
    }
}
