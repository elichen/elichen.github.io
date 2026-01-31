class SampleMeanStd {
    constructor(shape) {
        this.mean = tf.variable(tf.zeros(shape));
        this.var = tf.variable(tf.ones(shape));
        this.p = tf.variable(tf.zeros(shape));
        this.count = 0;
        this.frozen = false; // When true, don't update stats
    }

    update(x) {
        if (this.frozen) return;
        return tf.tidy(() => {
            if (this.count === 0) {
                this.mean.assign(x);
                this.p.assign(tf.zerosLike(x));
                this.count = 1;
                return;
            }

            const newCount = this.count + 1;
            const delta = tf.sub(x, this.mean);
            const newMean = tf.add(this.mean, tf.div(delta, tf.scalar(newCount)));
            
            // Update p and var like in Python version
            const deltaPrime = tf.sub(x, newMean);
            const newP = tf.add(this.p, tf.mul(delta, deltaPrime));
            const newVar = newCount < 2 ? 
                tf.onesLike(x) : 
                tf.maximum(
                    tf.div(newP, tf.scalar(newCount - 1)),
                    tf.scalar(1e-8)  // Match Python's epsilon
                );
            
            this.count = newCount;
            this.mean.assign(newMean);
            this.var.assign(newVar);
            this.p.assign(newP);
        });
    }

    normalize(x) {
        return tf.tidy(() => {
            const std = tf.sqrt(tf.add(this.var, tf.scalar(1e-8)));
            return tf.div(tf.sub(x, this.mean), std);
        });
    }

    loadStats(stats) {
        this.mean.assign(tf.tensor(stats.mean));
        this.var.assign(tf.tensor(stats.var));
        this.count = stats.count;
    }

    dispose() {
        this.mean.dispose();
        this.var.dispose();
        this.p.dispose();
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
        const result = this.env.step(action);
        return tf.tidy(() => {
            const stateTensor = tf.tensor1d(result.state);
            this.normalizer.update(stateTensor);
            const normalizedState = this.normalizer.normalize(stateTensor);
            return {
                state: Array.from(normalizedState.dataSync()),
                reward: result.reward,
                done: result.done,
                info: result.info
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
    constructor(env, gamma = 0.99, epsilon = 1e-8) {
        this.env = env;
        this.gamma = gamma;
        this.epsilon = epsilon;
        this.rewardStats = new SampleMeanStd([1]);  // Shape for scalar reward
        this.rewardTrace = 0;  // Single env case
    }

    reset() {
        this.rewardTrace = 0;
        return this.env.reset();
    }

    step(action) {
        return tf.tidy(() => {
            const { state, reward, done, info } = this.env.step(action);
            
            // First update reward trace with original reward
            const oldTrace = this.rewardTrace;
            this.rewardTrace = oldTrace * this.gamma * (1 - (done ? 1 : 0)) + reward;
            
            // Then update stats with reward trace (not the reward)
            this.rewardStats.update(tf.tensor1d([this.rewardTrace]));

            // Finally normalize the reward using the trace's statistics
            // Important: We normalize the original reward, not the trace
            const var_sqrt = Math.sqrt(this.rewardStats.var.dataSync()[0] + this.epsilon);
            const normalizedReward = reward / var_sqrt;

            return {
                state,
                reward: normalizedReward,
                done,
                info
            };
        });
    }

    render() {
        this.env.render();
    }

    dispose() {
        this.rewardStats.dispose();
    }

    getState() {
        return this.env.getState();
    }
}

class AddTimeInfo {
    constructor(env) {
        this.env = env;
        this.epiTime = -0.5;
        this.timeLimit = 10000;  // Same as Python's max_episode_steps
    }

    reset() {
        this.epiTime = -0.5;
        const state = this.env.reset();
        return [...state, this.epiTime];
    }

    step(action) {
        const result = this.env.step(action);
        this.epiTime += 1.0 / this.timeLimit;
        
        return {
            state: [...result.state, this.epiTime],
            reward: result.reward,
            done: result.done,
            info: result.info
        };
    }

    render() {
        this.env.render();
    }

    getState() {
        const baseState = this.env.getState();
        return [...baseState, this.epiTime];
    }
} 