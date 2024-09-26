class DQNModel {
    constructor(inputShape, numActions) {
        this.model = tf.sequential();
        
        this.model.add(tf.layers.conv2d({
            inputShape: inputShape,
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
        
        this.model.add(tf.layers.flatten());
        
        this.model.add(tf.layers.dense({
            units: 512,
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