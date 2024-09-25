class SnakeModel {
    constructor(inputSize, hiddenSize, outputSize) {
        // Policy Network
        this.model = tf.sequential();
        this.model.add(tf.layers.dense({ units: hiddenSize, activation: 'relu', inputShape: [inputSize] }));
        this.model.add(tf.layers.dense({ units: hiddenSize, activation: 'relu' }));
        this.model.add(tf.layers.dense({ units: outputSize, activation: 'linear' }));

        // Target Network (copy of the policy network)
        this.targetModel = tf.sequential();
        this.targetModel.add(tf.layers.dense({ units: hiddenSize, activation: 'relu', inputShape: [inputSize] }));
        this.targetModel.add(tf.layers.dense({ units: hiddenSize, activation: 'relu' }));
        this.targetModel.add(tf.layers.dense({ units: outputSize, activation: 'linear' }));

        this.optimizer = tf.train.adam(0.001);

        // Sync the target network weights with the policy network initially
        this.updateTargetNetwork();
    }

    // Function to update the target network with the weights of the policy network
    updateTargetNetwork() {
        const policyWeights = this.model.getWeights();
        this.targetModel.setWeights(policyWeights);
    }

    predict(state) {
        return tf.tidy(() => {
            const stateTensor = tf.tensor2d(state);
            return this.model.predict(stateTensor);
        });
    }

    predictTarget(state) {
        return tf.tidy(() => {
            const stateTensor = tf.tensor2d(state);
            return this.targetModel.predict(stateTensor);
        });
    }

    async train(states, targets) {
        const statesTensor = tf.tensor2d(states);
        const targetsTensor = tf.tensor2d(targets);

        const loss = () => tf.tidy(() => {
            const predictions = this.model.predict(statesTensor);
            return predictions.sub(targetsTensor).square().mean();
        });

        await this.optimizer.minimize(loss, true);

        tf.dispose([statesTensor, targetsTensor]);
    }

    getWeights() {
        return this.model.getWeights();
    }

    setWeights(weights) {
        this.model.setWeights(weights);
    }
}