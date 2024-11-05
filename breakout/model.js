class DQNModel {
    constructor() {
        this.model = null;
    }

    async loadModel(modelUrl) {
        try {
            this.model = await tf.loadGraphModel(modelUrl);
            console.log('Model loaded successfully');
        } catch (error) {
            console.error('Error loading model:', error);
            throw error;
        }
    }

    predict(state) {
        return tf.tidy(() => {
            const inputs = {
                'step_type': tf.tensor1d([0], 'int32'),
                'reward': tf.tensor1d([0], 'float32'),
                'discount': tf.tensor1d([1], 'float32'),
                'observation': tf.tensor2d([state], [1, 159], 'float32')
            };
            
            return this.model.execute(inputs);
        });
    }
}
