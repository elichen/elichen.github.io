class DQNAgent {
    constructor(inputSize, numActions, batchSize = 1000, memorySize = 100000, gamma = 0.99, epsilonStart = 1.0, epsilonEnd = 0.1, fixedEpsilonEpisodes = 250, decayEpsilonEpisodes = 250) {
        this.inputSize = inputSize;
        this.numActions = numActions;
        this.batchSize = batchSize;
        this.memorySize = memorySize;
        this.gamma = gamma;
        this.epsilonStart = epsilonStart;
        this.epsilonEnd = epsilonEnd;
        this.fixedEpsilonEpisodes = fixedEpsilonEpisodes;
        this.decayEpsilonEpisodes = decayEpsilonEpisodes;
        this.epsilon = epsilonStart;
        this.episodeCount = 0;

        this.model = new DQNModel(inputSize, numActions);
        this.targetModel = new DQNModel(inputSize, numActions);
        this.updateTargetModel();

        this.memory = [];
        this.updateFrequency = 1000;
        this.stepsSinceUpdate = 0;
        this.losses = [];
    }

    act(state, training = true) {
        if (training && Math.random() < this.epsilon) {
            return Math.floor(Math.random() * this.numActions);
        } else {
            const prediction = this.model.predict(state);
            const actionTensor = prediction.argMax(1);
            const action = actionTensor.dataSync()[0];

            prediction.dispose();
            actionTensor.dispose();

            return action;
        }
    }

    remember(state, action, reward, nextState, done) {
        if (this.memory.length >= this.memorySize) {
            this.memory.shift();
        }
        this.memory.push([state, action, reward, nextState, done]);

        this.totalSteps++;
        this.stepsSinceUpdate++;
        this.updateEpsilon();

        if (this.stepsSinceUpdate >= this.updateFrequency) {
            this.updateTargetModel();
            this.stepsSinceUpdate = 0;
        }
    }

    updateTargetModel() {
        this.targetModel.model.setWeights(this.model.model.getWeights());
    }

    updateEpsilon() {
        if (this.episodeCount <= this.fixedEpsilonEpisodes) {
            // Maintain epsilonStart for fixedEpsilonEpisodes
            this.epsilon = this.epsilonStart;
        } else {
            // Linear decay from epsilonStart to epsilonEnd over decayEpsilonEpisodes
            const decayProgress = Math.min(1, (this.episodeCount - this.fixedEpsilonEpisodes) / this.decayEpsilonEpisodes);
            this.epsilon = Math.max(
                this.epsilonEnd,
                this.epsilonStart - (this.epsilonStart - this.epsilonEnd) * decayProgress
            );
        }
    }

    async replay() {
        if (this.memory.length < this.batchSize) return;

        const samples = this.sampleMemory(this.batchSize);

        const states = samples.map(x => x[0]);
        const nextStates = samples.map(x => x[3]);

        const currentQs = this.model.predict(states);
        const futureQs = this.targetModel.predict(nextStates);

        const updatedQValues = samples.map((sample, index) => {
            const [, action, reward, , done] = sample;
            const currentQ = currentQs.arraySync()[index];
            const futureQ = futureQs.arraySync()[index];
            const maxFutureQ = Math.max(...futureQ);
            
            const updatedQ = [...currentQ];
            updatedQ[action] = reward + (done ? 0 : this.gamma * maxFutureQ);
            
            return updatedQ;
        });

        const loss = await this.model.train(states, updatedQValues);

        currentQs.dispose();
        futureQs.dispose();

        return loss;
    }

    sampleMemory(batchSize) {
        const samples = [];
        for (let i = 0; i < batchSize; i++) {
            const randomIndex = Math.floor(Math.random() * this.memory.length);
            samples.push(this.memory[randomIndex]);
        }
        return samples;
    }

    // Add a new method to increment episode count
    incrementEpisode() {
        this.episodeCount++;
        this.updateEpsilon();
    }
}
