class PolicyNetwork {
    constructor() {
        this.model = tf.sequential({
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
                    units: 3,
                    activation: 'softmax',
                    kernelInitializer: 'glorotNormal'
                })
            ]
        });

        this.valueHead = tf.sequential({
            layers: [
                tf.layers.dense({
                    inputShape: [64],
                    units: 1,
                    activation: 'linear',
                    kernelInitializer: 'glorotNormal'
                })
            ]
        });

        // Compile both models to ensure variables are created
        this.model.compile({optimizer: 'adam', loss: 'categoricalCrossentropy'});
        this.valueHead.compile({optimizer: 'adam', loss: 'meanSquaredError'});
        
        this.optimizer = tf.train.adam(3e-4);
    }

    predict(state) {
        return tf.tidy(() => {
            const stateTensor = tf.tensor2d([state]);
            const [actionProbs, value] = this.forward(stateTensor);
            return {
                actionProbs: actionProbs.dataSync(),
                value: value.dataSync()[0]
            };
        });
    }

    forward(stateTensor) {
        const hidden = this.model.layers[1].apply(
            this.model.layers[0].apply(stateTensor)
        );
        const actionProbs = this.model.layers[2].apply(hidden);
        const value = this.valueHead.apply(hidden);
        return [actionProbs, value];
    }

    sampleAction(actionProbs) {
        return tf.tidy(() => {
            const action = tf.multinomial(tf.log(actionProbs), 1);
            return action.dataSync()[0];
        });
    }
} 