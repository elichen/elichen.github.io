class SnakeModel {
    constructor(inputSize, hiddenSize, outputSize) {
        this.model = tf.sequential();
        this.model.add(tf.layers.dense({ units: hiddenSize, activation: 'relu', inputShape: [inputSize] }));
        this.model.add(tf.layers.dense({ units: hiddenSize, activation: 'relu' }));
        this.model.add(tf.layers.dense({ units: outputSize, activation: 'linear' }));

        this.optimizer = tf.train.adam(0.001);
    }

    predict(state) {
        return tf.tidy(() => {
            const stateTensor = Array.isArray(state[0]) ? tf.tensor2d(state) : tf.tensor2d([state]);
            return this.model.predict(stateTensor);
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