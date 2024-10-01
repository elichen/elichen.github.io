class PrioritizedReplayBuffer {
    constructor(capacity, alpha = 0.6, beta = 0.4, betaIncrement = 0.001, epsilon = 1e-6) {
        this.capacity = capacity;
        this.alpha = alpha;
        this.beta = beta;
        this.betaIncrement = betaIncrement;
        this.epsilon = epsilon;
        this.buffer = [];
        this.priorities = [];
        this.maxPriority = 1.0;
    }

    add(state, action, reward, nextState, done) {
        const experience = [state, action, reward, nextState, done];
        if (this.buffer.length >= this.capacity) {
            this.buffer.shift();
            this.priorities.shift();
        }
        this.buffer.push(experience);
        this.priorities.push(this.maxPriority);
    }

    sample(batchSize) {
        const totalPriority = this.priorities.reduce((a, b) => a + Math.pow(b, this.alpha), 0);
        const probabilities = this.priorities.map(p => Math.pow(p, this.alpha) / totalPriority);
        const indices = this._sampleProportional(probabilities, batchSize);
        
        const samples = indices.map(i => this.buffer[i]);
        const weights = indices.map(i => Math.pow(this.buffer.length * probabilities[i], -this.beta));
        const maxWeight = Math.max(...weights);
        const normalizedWeights = weights.map(w => w / maxWeight);

        this.beta = Math.min(1.0, this.beta + this.betaIncrement);

        return { samples, indices, weights: normalizedWeights };
    }

    update(indices, errors) {
        for (let i = 0; i < indices.length; i++) {
            const priority = Math.pow(Math.abs(errors[i]) + this.epsilon, this.alpha);
            this.priorities[indices[i]] = priority;
            this.maxPriority = Math.max(this.maxPriority, priority);
        }
    }

    _sampleProportional(probabilities, size) {
        const indices = [];
        const cumSum = probabilities.reduce((acc, p, i) => {
            acc.push((acc[i-1] || 0) + p);
            return acc;
        }, []);
        
        for (let i = 0; i < size; i++) {
            const r = Math.random();
            const index = cumSum.findIndex(cs => cs > r);
            indices.push(index);
        }
        return indices;
    }
}