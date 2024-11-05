class DQNAgent {
    constructor() {
        this.model = new DQNModel();
    }

    async loadModel(modelUrl) {
        await this.model.loadModel(modelUrl);
    }

    act(state) {
        return tf.tidy(() => {
            const prediction = this.model.predict(state);
            return prediction.dataSync()[0];
        });
    }
}
