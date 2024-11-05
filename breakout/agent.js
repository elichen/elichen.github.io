class DQNAgent {
    constructor() {
        this.model = new DQNModel();
    }

    async loadModel(modelUrl) {
        await this.model.loadModel(modelUrl);
    }

    act(state) {
        return tf.tidy(() => {
            // Get prediction from model
            const prediction = this.model.predict(state);
            
            // Cast to float32 before applying softmax
            const predictionFloat = tf.cast(prediction, 'float32');
            
            // PPO outputs logits/probabilities for each action
            const actionProbs = tf.softmax(predictionFloat);
            const actionTensor = actionProbs.argMax(0);
            const action = actionTensor.dataSync()[0];
            
            // Map to valid actions: 0 = no action, 1 = left, 2 = right
            return action;
        });
    }
}
