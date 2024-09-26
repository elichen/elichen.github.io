class DQNModel {
    constructor(inputShape, numActions) {
        this.model = tf.sequential();
        
        this.model.add(tf.layers.conv2d({
            inputShape: inputShape,
            filters: 16,  // Reduced from 32
            kernelSize: 4,  // Reduced from 8
            strides: 2,  // Reduced from 4
            activation: 'relu'
        }));
        
        this.model.add(tf.layers.conv2d({
            filters: 32,  // Reduced from 64
            kernelSize: 3,  // Reduced from 4
            strides: 1,  // Reduced from 2
            activation: 'relu'
        }));
        
        // Removed one convolutional layer
        
        this.model.add(tf.layers.flatten());
        
        this.model.add(tf.layers.dense({
            units: 256,  // Reduced from 512
            activation: 'relu'
        }));
        
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
        await this.model.fit(states, targets, {epochs:1});
    }
}