class PolicyNetwork {
    constructor() {
        // Create shared base network
        this.baseNetwork = tf.sequential({
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
                })
            ]
        });

        // Policy head (actor)
        this.policyHead = tf.sequential({
            layers: [
                tf.layers.dense({
                    inputShape: [64],
                    units: 3,
                    activation: 'softmax',
                    kernelInitializer: 'glorotNormal'
                })
            ]
        });

        // Value head (critic)
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

        // Compile models to ensure variables are created
        this.baseNetwork.compile({ optimizer: 'adam', loss: 'meanSquaredError' });
        this.policyHead.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy' });
        this.valueHead.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

        this.optimizer = tf.train.adam(3e-4);
    }

    getTrainableVariables() {
        const baseVars = this.baseNetwork.trainableWeights;
        const policyVars = this.policyHead.trainableWeights;
        const valueVars = this.valueHead.trainableWeights;
        return [...baseVars, ...policyVars, ...valueVars];
    }

    predict(stateTensor) {
        return tf.tidy(() => {
            const hidden = this.baseNetwork.predict(stateTensor);
            const actionProbs = this.policyHead.predict(hidden);
            const value = this.valueHead.predict(hidden);
            return {
                actionProbs: actionProbs.dataSync(),
                value: value.dataSync()[0]
            };
        });
    }

    forward(stateTensor) {
        const hidden = this.baseNetwork.predict(stateTensor);
        const actionProbs = this.policyHead.predict(hidden);
        const value = this.valueHead.predict(hidden);
        return [actionProbs, value];
    }

    sampleAction(actionProbs) {
        const probsTensor = tf.tensor1d(actionProbs);
        const action = tf.multinomial(tf.log(probsTensor), 1).dataSync()[0];
        probsTensor.dispose();
        return action;
    }
} 