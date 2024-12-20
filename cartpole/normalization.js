class SampleMeanStd {
    constructor(shape) {
        this.mean = tf.variable(tf.zeros(shape));
        this.var = tf.variable(tf.ones(shape));
        this.count = 0;
    }

    update(x) {
        return tf.tidy(() => {
            this.count += 1;
            const delta = tf.sub(x, this.mean);
            const newMean = this.mean.add(delta.div(tf.scalar(this.count)));
            const m_a = this.var.mul(tf.scalar(this.count - 1));
            const m_b = delta.mul(tf.sub(x, newMean));
            const newVar = tf.add(m_a, m_b).div(tf.scalar(this.count));
            
            this.mean.assign(newMean);
            this.var.assign(newVar);
        });
    }

    normalize(x) {
        return tf.tidy(() => {
            const std = tf.sqrt(this.var.add(tf.scalar(1e-8)));
            return tf.div(tf.sub(x, this.mean), std);
        });
    }

    dispose() {
        this.mean.dispose();
        this.var.dispose();
    }
}

class NormalizeObservation {
    constructor(env) {
        this.env = env;
        const shape = [env.getState().length];
        this.normalizer = new SampleMeanStd(shape);
    }

    reset() {
        const state = this.env.reset();
        return tf.tidy(() => {
            const normalized = this.normalizer.normalize(tf.tensor1d(state));
            return normalized.dataSync();
        });
    }

    step(action) {
        const { state, reward, done } = this.env.step(action);
        return tf.tidy(() => {
            const stateTensor = tf.tensor1d(state);
            this.normalizer.update(stateTensor);
            const normalizedState = this.normalizer.normalize(stateTensor);
            return {
                state: normalizedState.dataSync(),
                reward,
                done
            };
        });
    }

    getState() {
        return this.env.getState();
    }

    render() {
        this.env.render();
    }

    dispose() {
        this.normalizer.dispose();
    }
}

class ScaleReward {
    constructor(env, scale = 1.0) {
        this.env = env;
        this.scale = scale;
    }

    reset() {
        return this.env.reset();
    }

    step(action) {
        const { state, reward, done } = this.env.step(action);
        return {
            state,
            reward: reward * this.scale,
            done
        };
    }

    render() {
        this.env.render();
    }
} 