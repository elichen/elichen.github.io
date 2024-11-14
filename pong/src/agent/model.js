class DQNNetwork {
    constructor() {
        this.model = this.createModel();
        // Create target network by cloning the architecture
        this.targetModel = this.createModel();

        this.optimizer = tf.train.adam(0.001);
        this.model.compile({ optimizer: this.optimizer, loss: 'meanSquaredError' });
        this.targetModel.compile({ optimizer: this.optimizer, loss: 'meanSquaredError' });
        
        // Sync target network initially
        this.updateTargetNetwork();
    }

    createModel() {
        return tf.sequential({
            layers: [
                tf.layers.dense({
                    inputShape: [6],
                    units: 128,
                    activation: 'relu',
                    kernelInitializer: 'glorotNormal'
                }),
                tf.layers.dense({
                    units: 64,
                    activation: 'relu',
                    kernelInitializer: 'glorotNormal'
                }),
                tf.layers.dense({
                    units: 3,  // 3 possible actions
                    activation: 'linear',
                    kernelInitializer: 'glorotNormal'
                })
            ]
        });
    }

    predict(stateTensor) {
        return tf.tidy(() => {
            return this.model.predict(stateTensor);
        });
    }

    predictTarget(stateTensor) {
        return tf.tidy(() => {
            return this.targetModel.predict(stateTensor);
        });
    }

    async train(states, actions, rewards, nextStates, dones) {
        try {
            // Get Q-values for next states using target network
            const nextQValues = this.targetModel.predict(nextStates);
            const maxNextQ = nextQValues.max(1);
            
            // Calculate target Q-values
            const targetQValues = rewards.add(
                tf.scalar(0.99).mul(maxNextQ).mul(tf.scalar(1).sub(dones))
            );

            // Get current Q-values and update them with new targets
            const currentQValues = this.model.predict(states);
            const actionMask = tf.oneHot(actions, 3);
            const updatedQValues = currentQValues.mul(tf.scalar(1).sub(actionMask))
                .add(actionMask.mul(targetQValues.expandDims(1)));

            // Train the model
            const result = await this.model.trainOnBatch(states, updatedQValues);

            // Cleanup intermediate tensors
            nextQValues.dispose();
            maxNextQ.dispose();
            targetQValues.dispose();
            currentQValues.dispose();
            actionMask.dispose();
            updatedQValues.dispose();

            return result;
        } catch (error) {
            console.error("Error in DQN training:", error);
            throw error;
        }
    }

    updateTargetNetwork() {
        const weights = this.model.getWeights();
        this.targetModel.setWeights(weights);
    }
} 