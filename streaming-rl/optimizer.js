// StreamingOptimizer from the paper's 2026 revision (github.com/mohmdelsayed/streaming-drl,
// branch 2026). Each weight's step is divided by a slowly decaying max of |δ·e|, so no
// weight moves more than lr per step. It starts from our ObGD-trained weights, so it
// first spends warmupSteps measuring that max without moving any weights.
class StreamingOptimizer {
    constructor(params, learningRate = 3e-4, gamma = 0.99, lambda = 0.8, beta = 0.99995, warmupSteps = 1000, eps = 1e-8) {
        this.lr = learningRate;
        this.gamma = gamma;
        this.lambda = lambda;
        this.beta = beta;
        this.warmupSteps = warmupSteps;
        this.eps = eps;
        this.params = params;
        this.traces = new Map();
        this.maxV = new Map();
        this.t = 0;
        this.lastDelta = 0;

        params.forEach(param => {
            this.traces.set(param.name, tf.variable(tf.zeros(param.shape)));
            this.maxV.set(param.name, tf.variable(tf.zeros(param.shape)));
        });
    }

    async step(delta, grads, reset) {
        const gammaLambda = this.gamma * this.lambda;
        const lr = ++this.t > this.warmupSteps ? this.lr : 0;
        this.lastDelta = delta;

        grads.forEach((grad, index) => {
            if (!grad) return;
            const param = this.params[index];
            if (!param || !param.name) return;

            const e = this.traces.get(param.name);
            const v = this.maxV.get(param.name);
            if (!e) return;

            tf.tidy(() => {
                // e = γλe + grad; v = max(βv, |δe|)
                e.assign(e.mul(gammaLambda).add(grad));
                v.assign(tf.maximum(v.mul(this.beta), e.abs().mul(Math.abs(delta))));

                // Grads are of -q, so w = w - lr·δ·e / v
                param.write(param.read().sub(e.div(v.add(this.eps)).mul(lr * delta)));

                if (reset) {
                    e.assign(tf.zerosLike(e));
                }
            });
        });
    }

    resetTraces() {
        for (const e of this.traces.values()) {
            tf.tidy(() => e.assign(tf.zerosLike(e)));
        }
    }

    // Forget everything, including the step scale, for a fresh start from new weights
    reset() {
        this.resetTraces();
        for (const v of this.maxV.values()) {
            tf.tidy(() => v.assign(tf.zerosLike(v)));
        }
        this.t = 0;
    }

    getLastStats() {
        const scales = this.params.map(param => {
            const mean = tf.tidy(() => this.maxV.get(param.name).mean().dataSync()[0]);
            return `  ${param.name}: ${mean.toFixed(6)}`;
        }).join('\n');
        const warmup = this.t < this.warmupSteps ? ` (warming up, ${this.warmupSteps - this.t} steps left)` : '';

        return `StreamingOptimizer${warmup}
Learning rate: ${this.lr}   β: ${this.beta}
Delta: ${this.lastDelta.toFixed(6)}

Mean step scale max|δ·e| per layer:
${scales}`;
    }
}
