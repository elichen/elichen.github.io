// The trained policy: Brax's PPO actor, an MLP (204 -> 512 -> 256 -> 128 -> 4 with
// SiLU activations), run in plain JS. Observations are normalized with the statistics
// gathered during training; the action is the Gaussian's mean, clipped to [-1, 1]
// (the noise is for exploration only).
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
        this.output = new Float64Array(meta.act);
    }

    // obs: 204 numbers -> [collective, roll rate, pitch rate, yaw rate] in [-1, 1]
    action(obs) {
        let x = this.x;
        for (let i = 0; i < x.length; i++) x[i] = (obs[i] - this.mean[i]) / this.std[i];
        const last = this.layers.length - 1;
        this.layers.forEach(({ n, out, w, b, y }, l) => {
            for (let o = 0; o < out; o++) {
                let s = b[o];
                for (let i = 0, r = o * n; i < n; i++) s += w[r + i] * x[i];
                y[o] = l === last ? s : s / (1 + Math.exp(-s));  // silu(s) = s · sigmoid(s)
            }
            x = y;
        });
        for (let i = 0; i < this.act; i++) this.output[i] = Math.min(1, Math.max(-1, x[i]));
        return this.output;
    }
}

export async function loadPolicy(base = 'model/') {
    const [meta, bin] = await Promise.all([
        fetch(base + 'policy.json').then(r => r.json()),
        fetch(base + 'policy.bin').then(r => r.arrayBuffer()),
    ]);
    return new Policy(meta, new Float32Array(bin));
}
