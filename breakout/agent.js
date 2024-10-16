class DQNAgent {
    constructor(stateShape, numActions, batchSize = 1000, memorySize = 100000, gamma = 0.99, epsilonStart = 0.7, epsilonEnd = 0.1, epsilonDecaySteps = 100000) {
        this.stateShape = stateShape;
        this.numActions = numActions;
        this.batchSize = batchSize;
        this.memorySize = memorySize;
        this.gamma = gamma;
        this.epsilonStart = epsilonStart;
        this.epsilonEnd = epsilonEnd;
        this.epsilonDecaySteps = epsilonDecaySteps;
        this.epsilon = epsilonStart;
        this.totalSteps = 0;

        this.model = new DQNModel(stateShape, numActions);
        this.targetModel = new DQNModel(stateShape, numActions);
        this.updateTargetModel();

        this.memory = [];
        this.updateFrequency = 10000;
        this.stepsSinceUpdate = 0;
        this.losses = [];
    }

    act(state, training = true) {
        if (training && Math.random() < this.epsilon) {
            return Math.floor(Math.random() * this.numActions);
        } else {
            // Remove tf.tidy() and manage tensors manually
            const stateTensor = tf.tensor3d([state]);
            const prediction = this.model.predict(stateTensor);
            const actionTensor = prediction.argMax(1);
            const action = actionTensor.dataSync()[0];

            // Dispose tensors
            stateTensor.dispose();
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
        this.epsilon = Math.max(
            this.epsilonEnd,
            this.epsilonStart - (this.epsilonStart - this.epsilonEnd) * (this.totalSteps / this.epsilonDecaySteps)
        );
    }

    async replay() {
        if (this.memory.length < this.batchSize) return;

        const samples = this.sampleMemory(this.batchSize);

        const states = tf.tensor3d(samples.map(x => x[0]));
        const nextStates = tf.tensor3d(samples.map(x => x[3]));

        const currentQs = this.model.predict(states);
        const futureQs = this.targetModel.predict(nextStates);

        // Calculate target Q-values
        const maxFutureQs = futureQs.max(1);
        const doneMask = tf.tensor1d(samples.map(s => s[4] ? 0 : 1));

        const rewards = tf.tensor1d(samples.map(s => s[2]));
        const updatedQs = rewards.add(maxFutureQs.mul(this.gamma).mul(doneMask));

        const oneHotActions = tf.oneHot(tf.tensor1d(samples.map(s => s[1]), 'int32'), this.numActions);
        const masks = oneHotActions;

        const targets = currentQs.mul(tf.scalar(1).sub(masks)).add(oneHotActions.mul(tf.expandDims(updatedQs, 1)));

        const loss = await this.model.train(states, targets);

        // Dispose tensors to free memory
        states.dispose();
        nextStates.dispose();
        currentQs.dispose();
        futureQs.dispose();
        maxFutureQs.dispose();
        doneMask.dispose();
        rewards.dispose();
        updatedQs.dispose();
        oneHotActions.dispose();
        masks.dispose();
        targets.dispose();

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
}
