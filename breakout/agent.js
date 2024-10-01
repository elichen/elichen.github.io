class DQNAgent {
    constructor(stateShape, numActions, batchSize = 1000, memorySize = 100000, gamma = 0.99, epsilonDecay = 0.995) {
        this.stateShape = stateShape; // [42, 42]
        this.numActions = numActions;
        this.batchSize = batchSize;
        this.gamma = gamma;
        this.epsilon = 1.0;
        this.epsilonMin = 0.01;
        this.epsilonDecay = epsilonDecay;
        this.model = new DQNModel([...stateShape, 1], numActions);
        this.memory = new PrioritizedReplayBuffer(memorySize);
    }

    act(state, training = true) {
        if (training && Math.random() < this.epsilon) {
            return Math.floor(Math.random() * this.numActions);
        } else {
            return tf.tidy(() => {
                const stateTensor = tf.tensor2d(state.flat(), [1, state.flat().length]);
                const prediction = this.model.predict(stateTensor);
                return prediction.argMax(1).dataSync()[0];
            });
        }
    }

    remember(state, action, reward, nextState, done) {
        this.memory.add(state, action, reward, nextState, done);
    }

    async trainOnEpisode(episodeMemory) {
        // Add episode memory to the agent's memory
        for (const experience of episodeMemory) {
            this.remember(...experience);
        }

        // Perform multiple replay steps
        const replaySteps = Math.min(10, Math.floor(episodeMemory.length / this.batchSize));
        for (let i = 0; i < replaySteps; i++) {
            await this.replay();
        }

        // Decay epsilon
        if (this.epsilon > this.epsilonMin) {
            this.epsilon *= this.epsilonDecay;
        }
    }

    async replay() {
        if (this.memory.buffer.length < this.batchSize) return;

        try {
            const { samples, indices, weights } = this.memory.sample(this.batchSize);

            const states = tf.tensor2d(samples.map(x => x[0].flat()), [this.batchSize, this.stateShape[0] * this.stateShape[1]]);
            const nextStates = tf.tensor2d(samples.map(x => x[3].flat()), [this.batchSize, this.stateShape[0] * this.stateShape[1]]);

            const currentQs = this.model.predict(states);
            const nextQs = this.model.predict(nextStates);

            const updatedQs = currentQs.arraySync();
            const errors = [];

            for (let i = 0; i < this.batchSize; i++) {
                const [, action, reward, , done] = samples[i];
                const oldQ = updatedQs[i][action];
                if (done) {
                    updatedQs[i][action] = reward;
                } else {
                    updatedQs[i][action] = reward + this.gamma * Math.max(...nextQs.arraySync()[i]);
                }
                errors.push(Math.abs(oldQ - updatedQs[i][action]));
            }

            const targets = tf.tensor2d(updatedQs);

            // Apply importance sampling weights manually
            const weightedTargets = tf.tidy(() => {
                const weightsTensor = tf.tensor1d(weights);
                return targets.mul(weightsTensor.expandDims(1));
            });

            await this.model.train(states, weightedTargets);

            this.memory.update(indices, errors);

            states.dispose();
            nextStates.dispose();
            currentQs.dispose();
            nextQs.dispose();
            targets.dispose();
            weightedTargets.dispose();
        } catch (error) {
            console.error('Error in replay method:', error);
        }
    }

    // ... (remove getBatch and flattenState methods as they're no longer needed)
}