class DQNModel {
    constructor(inputSize, numActions) {
        this.model = tf.sequential({
            layers: [
                tf.layers.dense({inputShape: [inputSize], units: 128, activation: 'relu'}),
                tf.layers.dense({units: 256, activation: 'relu'}),
                tf.layers.dense({units: 256, activation: 'relu'}),
                tf.layers.dense({units: 256, activation: 'relu'}),
                tf.layers.dense({units: 256, activation: 'relu'}),
                tf.layers.dense({units: numActions, activation: 'linear'})
            ]
        });
        
        this.model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'meanSquaredError'
        });
    }

    predict(state) {
        return tf.tidy(() => {
            let stateTensor;
            if (Array.isArray(state)) {
                if (state.length === 0) {
                    throw new Error('Empty state array');
                }
                if (Array.isArray(state[0])) {
                    // Batch of states
                    stateTensor = tf.tensor2d(state);
                } else {
                    // Single state
                    stateTensor = tf.tensor2d([state]);
                }
            } else {
                throw new Error('State must be an array');
            }
            return this.model.predict(stateTensor);
        });
    }

    async train(states, targets) {
        const statesTensor = tf.tensor2d(states);
        const targetsTensor = tf.tensor2d(targets);
        
        try {
            const result = await this.model.fit(statesTensor, targetsTensor, {
                epochs: 1,
                batchSize: states.length
            });

            return result.history.loss[0];
        } finally {
            statesTensor.dispose();
            targetsTensor.dispose();
        }
    }
}
