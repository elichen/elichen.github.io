class DQNModel {
    constructor(inputShape, numActions) {
        this.model = tf.sequential();
        
        // Ensure inputShape is correct: [height, width, channels]
        const [height, width, channels] = inputShape;
        
        // Convolutional layers
        this.model.add(tf.layers.conv2d({
            inputShape: [height, width, channels],
            filters: 32,
            kernelSize: 8,
            strides: 4,
            activation: 'relu'
        }));
        
        this.model.add(tf.layers.conv2d({
            filters: 64,
            kernelSize: 4,
            strides: 2,
            activation: 'relu'
        }));
        
        this.model.add(tf.layers.conv2d({
            filters: 64,
            kernelSize: 3,
            strides: 1,
            activation: 'relu'
        }));
        
        // Flatten the output from convolutional layers
        this.model.add(tf.layers.flatten());
        
        // Dense layers
        this.model.add(tf.layers.dense({
            units: 512,
            activation: 'relu'
        }));
        
        this.model.add(tf.layers.dense({
            units: numActions,
            activation: 'linear'
        }));
        
        this.model.compile({
            optimizer: tf.train.adam(0.003),
            loss: 'meanSquaredError'
        });
    }

    predict(state) {
        // Ensure the input state has the correct shape
        return tf.tidy(() => {
            const reshapedState = tf.reshape(state, [-1, ...this.model.inputs[0].shape.slice(1)]);
            return this.model.predict(reshapedState);
        });
    }

    async train(states, targets) {
        // Reshape states
        const reshapedStates = tf.reshape(states, [-1, ...this.model.inputs[0].shape.slice(1)]);

        const result = await this.model.fit(reshapedStates, targets, {
            epochs: 1,
            batchSize: states.shape[0]
        });

        reshapedStates.dispose(); // Dispose reshapedStates

        return result.history.loss[0];
    }
}
