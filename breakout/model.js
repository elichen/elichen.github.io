class DQNModel {
    constructor() {
        this.model = null;
    }

    async loadModel(modelUrl) {
        console.log('Loading model from:', modelUrl);
        this.model = await tf.loadLayersModel(modelUrl);
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
}
