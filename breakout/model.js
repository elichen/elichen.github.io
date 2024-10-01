class DQNModel {
    constructor(inputShape, numActions) {
        this.model = tf.sequential();
        
        // Flatten the input
        const flattenedInputSize = inputShape[0] * inputShape[1];
        
        // Input layer
        this.model.add(tf.layers.dense({
            inputShape: [flattenedInputSize],
            units: 256,
            activation: 'relu'
        }));
        
        // Hidden layer 1
        this.model.add(tf.layers.dense({
            units: 128,
            activation: 'relu'
        }));
        
        // Hidden layer 2
        this.model.add(tf.layers.dense({
            units: 64,
            activation: 'relu'
        }));
        
        // Output layer
        this.model.add(tf.layers.dense({
            units: numActions,
            activation: 'linear'
        }));
        
        this.model.compile({
            optimizer: tf.train.adam(),
            loss: 'meanSquaredError'
        });
    }

    predict(state) {
        return this.model.predict(state);
    }

    async train(states, targets) {
        await this.model.fit(states, targets, {
            epochs: 1,
            batchSize: states.shape[0]
        });
    }
}