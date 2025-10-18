/**
 * Experience Replay Buffer for DQN-style learning
 * Stores transitions and allows sampling for off-policy learning
 */
class ReplayBuffer {
    constructor(capacity = 100000) {
        this.capacity = capacity;
        this.buffer = [];
        this.position = 0;
    }

    /**
     * Add a transition to the buffer
     * @param {Array} state - Current state
     * @param {number} action - Action taken
     * @param {number} reward - Reward received
     * @param {Array} nextState - Next state
     * @param {boolean} done - Whether episode ended
     */
    push(state, action, reward, nextState, done) {
        const transition = {
            state: state.slice(), // Copy arrays to avoid reference issues
            action: action,
            reward: reward,
            nextState: nextState ? nextState.slice() : null,
            done: done
        };

        if (this.buffer.length < this.capacity) {
            this.buffer.push(transition);
        } else {
            this.buffer[this.position] = transition;
        }
        this.position = (this.position + 1) % this.capacity;
    }

    /**
     * Sample a random batch from the buffer
     * @param {number} batchSize - Number of samples to return
     * @returns {Array} Array of sampled transitions
     */
    sample(batchSize) {
        if (this.buffer.length < batchSize) {
            return null; // Not enough samples yet
        }

        const batch = [];
        const indices = new Set();

        // Sample without replacement
        while (indices.size < batchSize) {
            const idx = Math.floor(Math.random() * this.buffer.length);
            if (!indices.has(idx)) {
                indices.add(idx);
                batch.push(this.buffer[idx]);
            }
        }

        return batch;
    }

    /**
     * Get current size of the buffer
     */
    size() {
        return this.buffer.length;
    }

    /**
     * Clear the buffer
     */
    clear() {
        this.buffer = [];
        this.position = 0;
    }
}

/**
 * Prioritized Experience Replay Buffer
 * Samples transitions based on their TD error (priority)
 */
class PrioritizedReplayBuffer extends ReplayBuffer {
    constructor(capacity = 100000, alpha = 0.6, beta = 0.4, betaSchedule = 100000) {
        super(capacity);
        this.alpha = alpha; // Priority exponent (0 = uniform, 1 = full prioritization)
        this.beta = beta; // Importance sampling exponent
        this.betaSchedule = betaSchedule; // Steps to anneal beta to 1
        this.priorities = [];
        this.maxPriority = 1.0;
        this.steps = 0;
    }

    push(state, action, reward, nextState, done, tdError = null) {
        // Use TD error as priority if available, otherwise use max priority
        const priority = tdError ? Math.abs(tdError) + 1e-6 : this.maxPriority;

        super.push(state, action, reward, nextState, done);

        if (this.priorities.length < this.capacity) {
            this.priorities.push(priority);
        } else {
            this.priorities[this.position - 1] = priority;
        }

        this.maxPriority = Math.max(this.maxPriority, priority);
    }

    /**
     * Sample a batch based on priorities
     * @param {number} batchSize - Number of samples to return
     * @returns {Object} Batch of transitions and importance weights
     */
    sample(batchSize) {
        if (this.buffer.length < batchSize) {
            return null;
        }

        // Calculate probabilities
        const priorities = this.priorities.slice(0, this.buffer.length);
        const probs = priorities.map(p => Math.pow(p, this.alpha));
        const probSum = probs.reduce((a, b) => a + b, 0);
        const normalizedProbs = probs.map(p => p / probSum);

        // Sample based on priorities
        const batch = [];
        const indices = [];
        const weights = [];

        // Calculate current beta
        const currentBeta = Math.min(1.0, this.beta + (1.0 - this.beta) * this.steps / this.betaSchedule);

        for (let i = 0; i < batchSize; i++) {
            const idx = this.sampleIndex(normalizedProbs);
            indices.push(idx);
            batch.push(this.buffer[idx]);

            // Calculate importance sampling weight
            const weight = Math.pow(this.buffer.length * normalizedProbs[idx], -currentBeta);
            weights.push(weight);
        }

        // Normalize weights
        const maxWeight = Math.max(...weights);
        const normalizedWeights = weights.map(w => w / maxWeight);

        this.steps++;

        return {
            batch: batch,
            indices: indices,
            weights: normalizedWeights
        };
    }

    /**
     * Sample an index based on probabilities
     */
    sampleIndex(probs) {
        const rand = Math.random();
        let cumSum = 0;
        for (let i = 0; i < probs.length; i++) {
            cumSum += probs[i];
            if (rand < cumSum) {
                return i;
            }
        }
        return probs.length - 1;
    }

    /**
     * Update priorities for sampled transitions
     * @param {Array} indices - Indices of transitions to update
     * @param {Array} tdErrors - New TD errors for these transitions
     */
    updatePriorities(indices, tdErrors) {
        for (let i = 0; i < indices.length; i++) {
            const priority = Math.abs(tdErrors[i]) + 1e-6;
            this.priorities[indices[i]] = priority;
            this.maxPriority = Math.max(this.maxPriority, priority);
        }
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { ReplayBuffer, PrioritizedReplayBuffer };
}