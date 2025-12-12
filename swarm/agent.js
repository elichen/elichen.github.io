/**
 * PPO Agent for Starling Murmuration
 * Loads trained weights and runs inference with TensorFlow.js
 */

class MurmurationAgent {
    constructor() {
        this.model = null;
        this.weights = null;
        this.numActions = 9;  // 8 directions + stay
        this.numNeighbors = 7;
        // obs_dim = velocity(2) + attractor_dir(2) + neighbors(7*4) = 32
        this.obsDim = 32;
        this.hiddenDim = 64;
    }

    async load(weightsUrl = 'murmuration_weights.json') {
        const response = await fetch(weightsUrl);
        this.weights = await response.json();
        this.buildModel();
        console.log('Murmuration PPO model loaded');
    }

    buildModel() {
        this.model = tf.sequential({
            layers: [
                tf.layers.dense({
                    inputShape: [this.obsDim],
                    units: this.hiddenDim,
                    activation: 'tanh',
                    kernelInitializer: 'zeros',
                    biasInitializer: 'zeros'
                }),
                tf.layers.dense({
                    units: this.hiddenDim,
                    activation: 'tanh',
                    kernelInitializer: 'zeros',
                    biasInitializer: 'zeros'
                }),
                tf.layers.dense({
                    units: this.numActions,
                    activation: 'linear',
                    kernelInitializer: 'zeros',
                    biasInitializer: 'zeros'
                })
            ]
        });
        this.loadWeights();
    }

    loadWeights() {
        const w = this.weights;
        const w0 = tf.tensor2d(w['actor.0.weight']).transpose();
        const b0 = tf.tensor1d(w['actor.0.bias']);
        const w1 = tf.tensor2d(w['actor.2.weight']).transpose();
        const b1 = tf.tensor1d(w['actor.2.bias']);
        const w2 = tf.tensor2d(w['actor.4.weight']).transpose();
        const b2 = tf.tensor1d(w['actor.4.bias']);

        this.model.layers[0].setWeights([w0, b0]);
        this.model.layers[1].setWeights([w1, b1]);
        this.model.layers[2].setWeights([w2, b2]);
    }

    predict(observations) {
        return tf.tidy(() => {
            const obsTensor = tf.tensor2d(observations);
            const logits = this.model.predict(obsTensor);
            const actions = logits.argMax(1);
            return Array.from(actions.dataSync());
        });
    }

    sample(observations) {
        return tf.tidy(() => {
            const obsTensor = tf.tensor2d(observations);
            const logits = this.model.predict(obsTensor);
            // Sample from categorical distribution
            const probs = tf.softmax(logits);
            const probsArray = probs.arraySync();

            const actions = probsArray.map(p => {
                const r = Math.random();
                let cumsum = 0;
                for (let i = 0; i < p.length; i++) {
                    cumsum += p[i];
                    if (r < cumsum) return i;
                }
                return p.length - 1;
            });
            return actions;
        });
    }
}

window.MurmurationAgent = MurmurationAgent;
