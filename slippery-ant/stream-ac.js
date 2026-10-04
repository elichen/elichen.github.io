// Stream-AC from "Streaming Deep Reinforcement Learning Finally Works"
// (Elsayed, Vasan, Mahmood; arXiv 2410.14606, 2026 revision), ported from the
// authors' code (github.com/mohmdelsayed/streaming-drl, branch 2026) to plain
// typed-array math. At batch size 1 this is ~30x faster than TensorFlow.js.
//
// Each network: in -> 128 -> LayerNorm -> LeakyReLU -> 128 -> LayerNorm -> LeakyReLU -> heads.
// The actor has a mean head and a softplus std head; the critic has one value head.

const SLOPE = 0.01, LN_EPS = 1e-5;

export class Trunk {
    constructor(inDim, hidden, headDims) {
        this.inDim = inDim;
        this.h = hidden;
        this.headDims = headDims;
        // Flat parameter layout: W1 (h x in), b1, W2 (h x h), b2, then per head W (k x h), b
        const sizes = [hidden * inDim, hidden, hidden * hidden, hidden];
        for (const k of headDims) sizes.push(k * hidden, k);
        this.offsets = [];
        let o = 0;
        for (const s of sizes) { this.offsets.push(o); o += s; }
        this.n = o;
        this.w = new Float64Array(o);
        this.g = new Float64Array(o);
        // [weight offset index, fan_out, fan_in] per dense layer
        this.layers = [[0, hidden, inDim], [2, hidden, hidden]];
        headDims.forEach((k, i) => this.layers.push([4 + 2 * i, k, hidden]));
        // Activations cached by forward() for backward()
        this.x = new Float64Array(inDim);
        this.z1 = new Float64Array(hidden); this.n1 = new Float64Array(hidden); this.a1 = new Float64Array(hidden); this.s1 = 0;
        this.z2 = new Float64Array(hidden); this.n2 = new Float64Array(hidden); this.a2 = new Float64Array(hidden); this.s2 = 0;
        this.heads = headDims.map(k => new Float64Array(k));
        this.d1 = new Float64Array(hidden);
        this.d2 = new Float64Array(hidden);
        // Scratch for forward passes that shouldn't overwrite the cache
        this.tmp = [1, 2, 3, 4].map(() => new Float64Array(hidden));
        this.tmpHeads = headDims.map(k => new Float64Array(k));
    }

    // Uniform ±1/sqrt(fan_in) with ceil(s·fan_in) inputs of every row zeroed; biases zero
    sparseInit(sparsities) {
        this.w.fill(0);
        this.layers.forEach(([li, out, inn], k) => {
            const W = this.offsets[li], bound = 1 / Math.sqrt(inn);
            const zeros = Math.ceil(sparsities[k] * inn);
            for (let j = 0; j < out; j++) {
                const perm = Array.from({ length: inn }, (_, i) => i);
                for (let i = inn - 1; i > 0; i--) {
                    const r = Math.floor(Math.random() * (i + 1));
                    [perm[i], perm[r]] = [perm[r], perm[i]];
                }
                const zero = new Set(perm.slice(0, zeros));
                for (let i = 0; i < inn; i++) this.w[W + j * inn + i] = zero.has(i) ? 0 : (Math.random() * 2 - 1) * bound;
            }
        });
    }

    static dense(w, W, b, x, inn, out, y) {
        for (let j = 0; j < out; j++) {
            let s = w[b + j];
            const row = W + j * inn;
            for (let i = 0; i < inn; i++) s += w[row + i] * x[i];
            y[j] = s;
        }
    }

    // LayerNorm (no affine) then LeakyReLU; returns 1/std for the backward pass
    static lnAct(z, n, a) {
        const H = z.length;
        let m = 0;
        for (let i = 0; i < H; i++) m += z[i];
        m /= H;
        let v = 0;
        for (let i = 0; i < H; i++) { const d = z[i] - m; v += d * d; }
        v /= H;
        const s = 1 / Math.sqrt(v + LN_EPS);
        for (let i = 0; i < H; i++) {
            n[i] = (z[i] - m) * s;
            a[i] = n[i] > 0 ? n[i] : SLOPE * n[i];
        }
        return s;
    }

    // cache = false evaluates without touching the activations backward() needs
    forward(x, cache = true) {
        const { w, offsets: o, h, inDim } = this;
        const [z1, n1, a1, z2, n2, a2] = cache
            ? [this.z1, this.n1, this.a1, this.z2, this.n2, this.a2]
            : [this.tmp[0], this.tmp[1], this.tmp[2], this.tmp[3], this.tmp[1], this.tmp[2]];
        if (cache) this.x.set(x);
        Trunk.dense(w, o[0], o[1], x, inDim, h, z1);
        const s1 = Trunk.lnAct(z1, n1, a1);
        Trunk.dense(w, o[2], o[3], a1, h, h, z2);
        const s2 = Trunk.lnAct(z2, n2, a2);
        if (cache) { this.s1 = s1; this.s2 = s2; }
        const outs = cache ? this.heads : this.tmpHeads;
        this.headDims.forEach((k, i) => Trunk.dense(w, o[4 + 2 * i], o[5 + 2 * i], a2, h, k, outs[i]));
        return outs;
    }

    // Given d(loss)/d(head outputs), write d(loss)/d(parameters) into this.g
    backward(dHeads) {
        const { w, g, offsets: o, h, inDim } = this;
        const d2 = this.d2.fill(0), d1 = this.d1.fill(0);
        this.headDims.forEach((k, hi) => {
            const W = o[4 + 2 * hi], B = o[5 + 2 * hi], dy = dHeads[hi];
            for (let j = 0; j < k; j++) {
                const dj = dy[j];
                g[B + j] = dj;
                const row = W + j * h;
                for (let i = 0; i < h; i++) {
                    g[row + i] = dj * this.a2[i];
                    d2[i] += w[row + i] * dj;
                }
            }
        });
        Trunk.lnActBack(this.n2, this.s2, d2);
        for (let j = 0; j < h; j++) {
            const dj = d2[j];
            g[o[3] + j] = dj;
            const row = o[2] + j * h;
            for (let i = 0; i < h; i++) {
                g[row + i] = dj * this.a1[i];
                d1[i] += w[row + i] * dj;
            }
        }
        Trunk.lnActBack(this.n1, this.s1, d1);
        for (let j = 0; j < h; j++) {
            const dj = d1[j];
            g[o[1] + j] = dj;
            const row = o[0] + j * inDim;
            for (let i = 0; i < inDim; i++) g[row + i] = dj * this.x[i];
        }
    }

    // In place: gradient w.r.t. the activation -> gradient w.r.t. the pre-LayerNorm input
    static lnActBack(n, s, d) {
        const H = n.length;
        let m1 = 0, m2 = 0;
        for (let i = 0; i < H; i++) {
            d[i] *= n[i] > 0 ? 1 : SLOPE;
            m1 += d[i];
            m2 += d[i] * n[i];
        }
        m1 /= H;
        m2 /= H;
        for (let i = 0; i < H; i++) d[i] = s * (d[i] - m1 - n[i] * m2);
    }
}

// ERK sparse init (sparse_init.py): one global nonzero budget shared by every
// layer, density ∝ (fan_in + fan_out) / params; layers that would exceed 1 go dense
export function erkInit(nets, targetSparsity = 0.95) {
    const infos = nets.flatMap(net => net.layers.map((l, k) => ({ net, k, fanOut: l[1], fanIn: l[2], count: l[1] * l[2] })));
    const target = (1 - targetSparsity) * infos.reduce((s, i) => s + i.count, 0);
    const dense = new Set();
    let eps = 0;
    for (;;) {
        const rem = infos.filter(i => !dense.has(i));
        const nzDense = [...dense].reduce((s, i) => s + i.count, 0);
        eps = (target - nzDense) / Math.max(rem.reduce((s, i) => s + i.fanIn + i.fanOut, 0), 1e-12);
        const fresh = rem.filter(i => eps * (i.fanIn + i.fanOut) / i.count >= 1);
        if (!fresh.length) break;
        fresh.forEach(i => dense.add(i));
    }
    for (const net of nets) {
        net.sparseInit(net.layers.map((l, k) => {
            const info = infos.find(i => i.net === net && i.k === k);
            return dense.has(info) ? 0 : 1 - Math.min(1, eps * (info.fanIn + info.fanOut) / info.count);
        }));
    }
}

// StreamingOptimizer (optimizer.py, 2026): e = γλe + g; v = max(βv, |δe|);
// w += lr·δ·e/(v + eps). No weight ever moves more than lr in one step.
export class StreamingOpt {
    constructor(net, lr, gamma = 0.99, lambda = 0.8, beta = 0.99995, eps = 1e-8) {
        this.net = net;
        this.lr = lr;
        this.gl = gamma * lambda;
        this.beta = beta;
        this.eps = eps;
        this.e = new Float64Array(net.n);
        this.v = new Float64Array(net.n);
        this.pushed = 0;  // mean |δe|/v this step: how hard the update pushed, as a fraction of the cap
    }

    step(delta, reset) {
        const { e, v, gl, beta, eps } = this, g = this.net.g, w = this.net.w;
        const ad = Math.abs(delta), k = this.lr * delta;
        let pushed = 0;
        for (let i = 0; i < e.length; i++) {
            const ei = gl * e[i] + g[i];
            const de = Math.abs(ei) * ad;
            const vi = Math.max(beta * v[i], de);
            v[i] = vi;
            w[i] += k * ei / (vi + eps);
            pushed += de / (vi + eps);
            e[i] = reset ? 0 : ei;
        }
        this.pushed = pushed / e.length;
    }

    resetTraces() {
        this.e.fill(0);
    }
}

// SampleMeanStd from obs_reward_transforms.py (sample variance; var = 1 until 2 samples)
export class SampleMeanStd {
    constructor(n) {
        this.mean = new Float64Array(n);
        this.var = new Float64Array(n).fill(1);
        this.p = new Float64Array(n);
        this.count = 0;
    }

    update(x) {
        const n = ++this.count;
        for (let i = 0; i < this.mean.length; i++) {
            if (n === 1) { this.mean[i] = x[i]; this.p[i] = 0; this.var[i] = 1; continue; }
            const m = this.mean[i] + (x[i] - this.mean[i]) / n;
            this.p[i] += (x[i] - this.mean[i]) * (x[i] - m);
            this.mean[i] = m;
            this.var[i] = this.p[i] / (n - 1);
        }
    }
}

const softplus = x => x > 20 ? x : Math.log1p(Math.exp(x));
const sigmoid = x => 1 / (1 + Math.exp(-x));

function gauss() {
    let u = 0;
    while (u === 0) u = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random());
}

// The whole streaming learner: running observation normalization, reward scaling
// by the return's running variance, the actor and critic, and their optimizers.
// It sees each transition once and keeps nothing: no replay buffer, no batches.
export class StreamAC {
    constructor(obsDim, actDim, { hidden = 128, gamma = 0.99, lambda = 0.8, beta = 0.99995,
                                  lrPolicy = 1e-4, lrValue = 2e-4, entropy = 0.01, sparsity = 0.95 } = {}) {
        this.obsDim = obsDim;
        this.actDim = actDim;
        this.hp = { hidden, gamma, lambda, beta, lrPolicy, lrValue, entropy, sparsity };
        this.actor = new Trunk(obsDim, hidden, [actDim, actDim]);
        this.critic = new Trunk(obsDim, hidden, [1]);
        erkInit([this.actor, this.critic], sparsity);
        this.optPi = new StreamingOpt(this.actor, lrPolicy, gamma, lambda, beta);
        this.optV = new StreamingOpt(this.critic, lrValue, gamma, lambda, beta);
        this.obsStats = new SampleMeanStd(obsDim);
        this.retStats = new SampleMeanStd(1);
        this.retTrace = 0;
        this.a = new Float64Array(actDim);
        this.mu = new Float64Array(actDim);
        this.std = new Float64Array(actDim);
        this.dMu = new Float64Array(actDim);
        this.dPre = new Float64Array(actDim);
        this.one = new Float64Array([1]);
        this.retBuf = new Float64Array(1);
        // Diagnostics for the page
        this.lastDelta = 0;
        this.lastValue = 0;
    }

    // NormalizeObservation: update the running stats, then standardize
    normalize(obs, update = true) {
        if (update) this.obsStats.update(obs);
        const o = new Float64Array(this.obsDim), { mean, var: v } = this.obsStats;
        for (let i = 0; i < this.obsDim; i++) o[i] = (obs[i] - mean[i]) / Math.sqrt(v[i] + 1e-8);
        return o;
    }

    // ScaleReward: divide by the running std of the discounted return
    scaleReward(r, done) {
        this.retTrace = this.retTrace * this.hp.gamma + r;
        this.retBuf[0] = this.retTrace;
        this.retStats.update(this.retBuf);
        const scaled = r / Math.sqrt(this.retStats.var[0] + 1e-8);
        if (done) this.retTrace = 0;
        return scaled;
    }

    // Sample a ~ N(μ(s), σ(s)); caches the actor's activations for learn()
    act(s) {
        const [mu, pre] = this.actor.forward(s);
        for (let i = 0; i < this.actDim; i++) {
            this.mu[i] = mu[i];
            this.std[i] = softplus(pre[i]);
            this.a[i] = mu[i] + this.std[i] * gauss();
        }
        return this.a;
    }

    // One streaming update from (s, a, r, s'), right after act(s) and env.step().
    // r is the already-scaled reward. Returns the TD error.
    learn(s, r, s2, terminated, truncated) {
        // A time-limit cutoff still bootstraps; only a fall is terminal
        const vNext = terminated ? 0 : this.critic.forward(s2, false)[0][0];
        const v = this.critic.forward(s)[0][0];
        const delta = r + this.hp.gamma * vNext - v;
        this.lastDelta = delta;
        this.lastValue = v;

        // Critic follows ∇V(s); actor follows ∇[log π(a|s) + c·sign(δ)·H(π(·|s))]
        this.critic.backward([this.one]);
        const sgn = Math.sign(delta), pre = this.actor.heads[1];
        for (let i = 0; i < this.actDim; i++) {
            const sd = this.std[i], u = this.a[i] - this.mu[i];
            this.dMu[i] = u / (sd * sd);
            const dStd = (u * u) / (sd * sd * sd) - 1 / sd + this.hp.entropy * sgn / sd;
            this.dPre[i] = dStd * sigmoid(pre[i]);
        }
        this.actor.backward([this.dMu, this.dPre]);

        const done = terminated || truncated;
        this.optPi.step(delta, done);
        this.optV.step(delta, done);
        return delta;
    }

    resetTraces() {
        this.optPi.resetTraces();
        this.optV.resetTraces();
    }

    meanStd() {
        let s = 0;
        for (let i = 0; i < this.actDim; i++) s += this.std[i];
        return s / this.actDim;
    }

    // Everything needed to resume learning exactly: weights, the optimizers' step
    // scales (traces restart at an episode boundary), and the normalizers.
    save() {
        const parts = {
            actor: this.actor.w, critic: this.critic.w, actorScale: this.optPi.v, criticScale: this.optV.v,
            obsMean: this.obsStats.mean, obsVar: this.obsStats.var, obsP: this.obsStats.p,
            retMean: this.retStats.mean, retVar: this.retStats.var, retP: this.retStats.p
        };
        const layout = {};
        let n = 0;
        for (const [k, arr] of Object.entries(parts)) { layout[k] = [n, arr.length]; n += arr.length; }
        const data = new Float32Array(n);
        for (const [k, arr] of Object.entries(parts)) data.set(arr, layout[k][0]);
        const meta = {
            obsDim: this.obsDim, actDim: this.actDim, hp: this.hp, layout,
            obsCount: this.obsStats.count, retCount: this.retStats.count
        };
        return { meta, data };
    }

    static load(meta, data) {
        const agent = new StreamAC(meta.obsDim, meta.actDim, meta.hp);
        const get = k => data.subarray(meta.layout[k][0], meta.layout[k][0] + meta.layout[k][1]);
        agent.actor.w.set(get('actor'));
        agent.critic.w.set(get('critic'));
        agent.optPi.v.set(get('actorScale'));
        agent.optV.v.set(get('criticScale'));
        agent.obsStats.mean.set(get('obsMean'));
        agent.obsStats.var.set(get('obsVar'));
        agent.obsStats.p.set(get('obsP'));
        agent.obsStats.count = meta.obsCount;
        agent.retStats.mean.set(get('retMean'));
        agent.retStats.var.set(get('retVar'));
        agent.retStats.p.set(get('retP'));
        agent.retStats.count = meta.retCount;
        return agent;
    }
}
