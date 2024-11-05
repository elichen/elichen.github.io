class DQNModel {
    constructor() {
        this.model = null;
    }

    async loadModel(modelUrl) {
        console.log('Loading model from:', modelUrl);
        try {
            this.model = await tf.loadGraphModel(modelUrl);
            console.log('Model loaded successfully');
            
            // Validate model inputs
            const inputShapes = this.model.inputs.map(input => ({
                name: input.name,
                shape: input.shape,
                dtype: input.dtype
            }));
            console.log('Model input shapes:', inputShapes);
            
            // Validate model outputs
            const outputShapes = this.model.outputs.map(output => ({
                name: output.name,
                shape: output.shape,
                dtype: output.dtype
            }));
            console.log('Model output shapes:', outputShapes);
            
        } catch (error) {
            console.error('Error loading model:', error);
            throw error;
        }
    }

    predict(state) {
        return tf.tidy(() => {
            // Ensure state has correct shape
            if (state.length !== 159) {
                console.error(`Invalid state length. Expected 159, got ${state.length}`);
                // Pad with zeros if necessary
                while (state.length < 159) {
                    state.push(0.0);
                }
            }

            // Create tensors for all required inputs
            const stateTensor = tf.tensor2d([state], [1, 159]);  // Explicitly specify shape
            const rewardTensor = tf.tensor1d([0]); // Dummy reward for prediction
            const stepTypeTensor = tf.tensor1d([0], 'int32'); // 0 represents FIRST or MID step
            const discountTensor = tf.tensor1d([1]); // Full discount for prediction
            
            // Make prediction with all required inputs
            const prediction = this.model.execute({
                'observation': stateTensor,
                'reward': rewardTensor,
                'step_type': stepTypeTensor,
                'discount': discountTensor
            });
            
            return prediction;
        });
    }
}
