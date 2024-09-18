class SnakeModel {
    constructor(inputSize, hiddenSize, outputSize) {
        this.model = tf.sequential();
        this.model.add(tf.layers.dense({ units: hiddenSize, activation: 'relu', inputShape: [inputSize] }));
        this.model.add(tf.layers.dense({ units: hiddenSize, activation: 'relu' }));
        this.model.add(tf.layers.dense({ units: outputSize, activation: 'linear' }));

        this.model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'meanSquaredError'
        });
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
        await this.model.fit(statesTensor, targetsTensor, {
            epochs: 1,
            shuffle: true,
            batchSize: 32
        });
        tf.dispose([statesTensor, targetsTensor]);
    }
}