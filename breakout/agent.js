class DQNAgent {
    constructor(stateShape, numActions, batchSize = 32, memorySize = 10000, gamma = 0.99, epsilonDecay = 0.995) {
        this.stateShape = stateShape; // [42, 42]
        this.numActions = numActions;
        this.batchSize = batchSize;
        this.memory = [];
        this.memorySize = memorySize;
        this.gamma = gamma;
        this.epsilon = 1.0;
        this.epsilonMin = 0.01;
        this.epsilonDecay = epsilonDecay;
        this.model = new DQNModel([...stateShape, 1], numActions);
    }

    act(state, training = true) {
        if (training && Math.random() < this.epsilon) {
            return Math.floor(Math.random() * this.numActions);
        } else {
            return tf.tidy(() => {
                const flatState = this.flattenState(state);
                const stateTensor = tf.tensor4d(flatState, [1, ...this.stateShape, 1]);
                const prediction = this.model.predict(stateTensor);
                return prediction.argMax(1).dataSync()[0];
            });
        }
    }

    remember(state, action, reward, nextState, done) {
        this.memory.push([state, action, reward, nextState, done]);
        if (this.memory.length > this.memorySize) {
            this.memory.shift();
        }
    }

    async trainOnEpisode(episodeMemory) {
        // Add episode memory to the agent's memory
        this.memory.push(...episodeMemory);
        if (this.memory.length > this.memorySize) {
            this.memory = this.memory.slice(-this.memorySize);
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
        if (this.memory.length < this.batchSize) return;

        try {
            const batch = this.getBatch();

            const flatStates = batch.flatMap(x => this.flattenState(x[0]));
            const flatNextStates = batch.flatMap(x => this.flattenState(x[3]));

            const states = tf.tensor4d(flatStates, [this.batchSize, ...this.stateShape, 1]);
            const nextStates = tf.tensor4d(flatNextStates, [this.batchSize, ...this.stateShape, 1]);

            const currentQs = this.model.predict(states);
            const nextQs = this.model.predict(nextStates);

            const updatedQs = currentQs.arraySync();

            for (let i = 0; i < this.batchSize; i++) {
                const [, action, reward, , done] = batch[i];
                if (done) {
                    updatedQs[i][action] = reward;
                } else {
                    updatedQs[i][action] = reward + this.gamma * Math.max(...nextQs.arraySync()[i]);
                }
            }

            const targets = tf.tensor2d(updatedQs.flat(), [this.batchSize, this.numActions]);

            await this.model.train(states, targets);

            states.dispose();
            nextStates.dispose();
            currentQs.dispose();
            nextQs.dispose();
            targets.dispose();
        } catch (error) {
            console.error('Error in replay method:', error);
        }
    }

    getBatch() {
        const batchIndices = [];
        while (batchIndices.length < this.batchSize) {
            const randomIndex = Math.floor(Math.random() * this.memory.length);
            if (!batchIndices.includes(randomIndex)) {
                batchIndices.push(randomIndex);
            }
        }
        return batchIndices.map(index => this.memory[index]);
    }

    flattenState(state) {
        if (!Array.isArray(state) || !Array.isArray(state[0])) {
            console.error('Invalid state format:', state);
            return [];
        }
        return state.flat();
    }
}