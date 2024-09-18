class RLAgent {
    constructor(env) {
        this.env = env;
        this.stateSize = 4;
        this.actionSize = 3;
        
        this.model = this.createModel();
        this.targetModel = this.createModel();
        this.updateTargetModel();

        this.epsilon = 1.0;
        this.epsilonMin = 0.01;
        this.epsilonDecay = 0.99; // Increased from 0.995 for faster exploration decay
        this.gamma = 0.99; // Increased from 0.95 for slightly more future-reward consideration
        this.learningRate = 0.005; // Increased from 0.001 for faster learning

        this.memory = [];
        this.batchSize = 64; // Increased from 32 for more stable learning
        this.targetUpdateFrequency = 10; // Update target network every 10 episodes
    }

    createModel() {
        const model = tf.sequential();
        model.add(tf.layers.dense({ units: 64, activation: 'relu', inputShape: [this.stateSize] }));
        model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
        model.add(tf.layers.dense({ units: this.actionSize, activation: 'linear' }));
        model.compile({ optimizer: tf.train.adam(this.learningRate), loss: 'meanSquaredError' });
        return model;
    }

    updateTargetModel() {
        this.targetModel.setWeights(this.model.getWeights());
    }

    async selectAction(state, testing = false) {
        if (!testing && Math.random() < this.epsilon) {
            return Math.floor(Math.random() * this.actionSize);
        } else {
            const stateTensor = tf.tensor2d([state]);
            const prediction = await this.model.predict(stateTensor).array();
            return prediction[0].indexOf(Math.max(...prediction[0]));
        }
    }

    remember(state, action, reward, nextState, done) {
        this.memory.push([state, action, reward, nextState, done]);
        if (this.memory.length > 10000) {
            this.memory.shift();
        }
    }

    async update(state, action, reward, nextState, done) {
        this.remember(state, action, reward, nextState, done);
        
        if (this.memory.length >= this.batchSize) {
            await this.replay();
        }

        if (done) {
            this.epsilon = Math.max(this.epsilonMin, this.epsilon * this.epsilonDecay);
            if (this.memory.length % this.targetUpdateFrequency === 0) {
                this.updateTargetModel();
            }
        }
    }

    async replay() {
        const batch = this.getBatch();
        const states = batch.map(exp => exp[0]);
        const actions = batch.map(exp => exp[1]);
        const rewards = batch.map(exp => exp[2]);
        const nextStates = batch.map(exp => exp[3]);
        const dones = batch.map(exp => exp[4]);

        const currentQs = await this.model.predict(tf.tensor2d(states)).array();
        const nextQs = await this.targetModel.predict(tf.tensor2d(nextStates)).array();

        const updatedQs = currentQs.map((qs, i) => {
            const targetQ = dones[i]
                ? rewards[i]
                : rewards[i] + this.gamma * Math.max(...nextQs[i]);
            qs[actions[i]] = targetQ;
            return qs;
        });

        await this.model.fit(tf.tensor2d(states), tf.tensor2d(updatedQs), {
            epochs: 1,
            verbose: 0
        });

        if (this.memory.length % 100 === 0) {
            this.updateTargetModel();
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

    reset() {
        this.epsilon = 1.0;
    }
}