class SampleMeanStd {
    constructor(shape) {
        this.mean = tf.variable(tf.zeros(shape));
        this.var = tf.variable(tf.ones(shape));
        this.p = tf.variable(tf.zeros(shape));  // Add p variable for tracking
        this.count = 0;
    }

    update(x) {
        return tf.tidy(() => {
            if (this.count === 0) {
                this.mean.assign(x);
                this.p.assign(tf.zerosLike(x));
                this.count = 1;
                return;
            }

            this.count += 1;
            const delta = tf.sub(x, this.mean);
            const newMean = this.mean.add(delta.div(tf.scalar(this.count)));
            
            // Update p and var like in Python version
            const deltaPrime = tf.sub(x, newMean);
            const newP = this.p.add(tf.mul(delta, deltaPrime));
            const newVar = this.count < 2 ? 
                tf.onesLike(x) : 
                newP.div(tf.scalar(this.count - 1)).maximum(tf.scalar(1e-2));  // Add minimum clip like Python
            
            this.mean.assign(newMean);
            this.var.assign(newVar);
            this.p.assign(newP);
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
        const { state, reward, done } = this.env.step(action);
        
        // Update reward trace
        this.rewardTrace = this.rewardTrace * this.gamma * (1 - (done ? 1 : 0)) + reward;
        
        // Update stats with reward trace and normalize reward
        this.rewardStats.update(tf.tensor1d([this.rewardTrace]));
        const normalizedReward = tf.tidy(() => {
            const std = tf.sqrt(this.rewardStats.var.add(tf.scalar(this.epsilon)));
            return reward / std.dataSync()[0];  // Use raw reward but scale by trace's std
        });

        return {
            state,
            reward: normalizedReward,
            done
        };
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
        const { state, reward, done } = this.env.step(action);
        this.epiTime += 1.0 / this.timeLimit;
        
        return {
            state: [...state, this.epiTime],
            reward,
            done
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