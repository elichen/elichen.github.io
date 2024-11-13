class DQNAgent {
    constructor() {
        this.policy = new DQNNetwork();
        this.memory = new Memory();
        this.batchSize = 32;
        this.epsilon = 1.0;
        this.epsilonMin = 0.01;
        this.epsilonDecay = 0.95;
        this.frameCount = 0;
        this.updateFrequency = 4;      // Train every 4 frames
        this.targetUpdateFrequency = 1000;  // Update target network every 1000 frames
    }

    selectAction(state) {
        // Epsilon-greedy action selection
        if (Math.random() < this.epsilon) {
            return Math.floor(Math.random() * 3);
        }

        return tf.tidy(() => {
            const stateTensor = tf.tensor2d([state]);
            const qValues = this.policy.predict(stateTensor);
            return qValues.argMax(1).dataSync()[0];
        });
    }

    async update() {
        this.frameCount++;

        // Decay epsilon more slowly
        if (this.frameCount % 100 === 0) {  // Only decay every 100 frames
            this.epsilon = Math.max(this.epsilonMin, this.epsilon * this.epsilonDecay);
        }

        // Only train every updateFrequency frames
        if (this.frameCount % this.updateFrequency !== 0 || 
            this.memory.states.length < this.batchSize) {
            return;
        }

        // Sample batch from memory
        const batchIndices = [];
        for (let i = 0; i < this.batchSize; i++) {
            batchIndices.push(Math.floor(Math.random() * this.memory.states.length));
        }

        // Prepare batch tensors
        const stateBatch = tf.tensor2d(
            batchIndices.map(idx => this.memory.states[idx])
        );
        const actionBatch = tf.tensor1d(
            batchIndices.map(idx => this.memory.actions[idx]), 'int32'
        );
        const rewardBatch = tf.tensor1d(
            batchIndices.map(idx => this.memory.rewards[idx])
        );
        const nextStateBatch = tf.tensor2d(
            batchIndices.map(idx => this.memory.nextStates[idx])
        );
        const doneBatch = tf.tensor1d(
            batchIndices.map(idx => this.memory.dones[idx] ? 1 : 0)
        );

        try {
            // Train on batch
            await this.policy.train(
                stateBatch, 
                actionBatch, 
                rewardBatch, 
                nextStateBatch, 
                doneBatch
            );

            // Update target network periodically
            if (this.frameCount % this.targetUpdateFrequency === 0) {
                console.log("Updating target network");
                this.policy.updateTargetNetwork();
            }
        } finally {
            // Clean up tensors
            stateBatch.dispose();
            actionBatch.dispose();
            rewardBatch.dispose();
            nextStateBatch.dispose();
            doneBatch.dispose();
        }
    }

    store(state, action, reward, nextState, done) {
        this.memory.store(state, action, reward, nextState, done);
    }
} 