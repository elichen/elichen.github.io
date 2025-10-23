/**
 * SB3 Pre-trained Model Loader for TensorFlow.js
 *
 * Loads weights from a trained Stable Baselines3 DQN model
 * and provides inference using TensorFlow.js
 */
class SB3Model {
    constructor() {
        this.model = null;
        this.weights = null;
        this.architecture = null;
        this.loaded = false;
    }

    /**
     * Load model weights from JSON file
     */
    async load(weightsPath = 'models/sb3_weights.json') {
        try {
            const response = await fetch(weightsPath);
            if (!response.ok) {
                throw new Error(`Failed to load weights from ${weightsPath}`);
            }

            const data = await response.json();
            this.weights = data.weights;
            this.architecture = data.architecture;

            console.log('Loaded SB3 model:');
            console.log('  Architecture:', this.architecture);
            console.log('  Number of layers:', this.weights.length);

            // Build TensorFlow.js model
            await this.buildModel();

            this.loaded = true;
            return true;
        } catch (error) {
            console.error('Error loading SB3 model:', error);
            return false;
        }
    }

    /**
     * Build TensorFlow.js model from loaded weights
     */
    async buildModel() {
        // Create a sequential model
        const layers = [];

        // Input layer
        layers.push(tf.layers.inputLayer({
            inputShape: [this.architecture.input_size]
        }));

        // Hidden layers + output layer
        for (let i = 0; i < this.weights.length; i++) {
            const layer = this.weights[i];
            const isLastLayer = i === this.weights.length - 1;

            // Create dense layer
            const denseLayer = tf.layers.dense({
                units: layer.output_size,
                useBias: true,
                activation: isLastLayer ? 'linear' : 'relu',  // ReLU for hidden, linear for output
                name: `dense_${i}`
            });

            layers.push(denseLayer);
        }

        // Build the model
        this.model = tf.sequential({
            layers: layers
        });

        // Set the weights
        for (let i = 0; i < this.weights.length; i++) {
            const layer = this.weights[i];
            const tfLayer = this.model.layers[i + 1]; // +1 because of input layer

            // Convert weights to tensors
            // Weights are already in the correct shape (in_features, out_features)
            const weightTensor = tf.tensor2d(layer.weight);
            const biasTensor = tf.tensor1d(layer.bias);

            // Set the layer weights
            tfLayer.setWeights([weightTensor, biasTensor]);
        }

        console.log('TensorFlow.js model built successfully');
        this.model.summary();
    }

    /**
     * Predict Q-values for a given state
     */
    predict(state) {
        if (!this.loaded || !this.model) {
            console.error('Model not loaded yet');
            return null;
        }

        // Convert state to tensor
        const stateTensor = tf.tensor2d([state], [1, state.length]);

        // Get Q-values
        const qValues = this.model.predict(stateTensor);

        // Convert to array and dispose tensors
        const qValuesArray = Array.from(qValues.dataSync());
        stateTensor.dispose();
        qValues.dispose();

        return qValuesArray;
    }

    /**
     * Select best action (greedy policy)
     */
    selectAction(state) {
        const qValues = this.predict(state);
        if (!qValues) return 0;

        // Return index of maximum Q-value
        let maxIdx = 0;
        let maxVal = qValues[0];

        for (let i = 1; i < qValues.length; i++) {
            if (qValues[i] > maxVal) {
                maxVal = qValues[i];
                maxIdx = i;
            }
        }

        return maxIdx;
    }

    /**
     * Dispose of TensorFlow.js resources
     */
    dispose() {
        if (this.model) {
            this.model.dispose();
            this.model = null;
        }
        this.loaded = false;
    }
}


/**
 * SB3 Agent wrapper that implements the same interface as DQNAgent
 */
class SB3Agent {
    constructor(env) {
        this.env = env;
        this.model = new SB3Model();
        this.totalSteps = 0;
        this.ready = false;

        // Create simple action space for compatibility
        this.actionSpace = {
            n: env.numActions || 3,
            sample: () => Math.floor(Math.random() * (env.numActions || 3))
        };
    }

    /**
     * Load the pre-trained model
     */
    async loadModel(weightsPath = 'models/sb3_weights.json') {
        console.log('Loading SB3 pre-trained model...');
        const success = await this.model.load(weightsPath);

        if (success) {
            this.ready = true;
            console.log('SB3 model loaded and ready!');
        } else {
            console.error('Failed to load SB3 model');
        }

        return success;
    }

    /**
     * Select action using the pre-trained policy (greedy)
     */
    act(state, training = false) {
        this.totalSteps++;

        if (!this.ready) {
            console.warn('Model not ready, returning random action');
            return this.actionSpace.sample();
        }

        // Always use greedy policy (no exploration)
        return this.model.selectAction(state);
    }

    /**
     * No-op methods for compatibility with training agents
     */
    remember(state, action, reward, nextState, done) {
        // Pre-trained model doesn't train
    }

    train() {
        // Pre-trained model doesn't train
        return null;
    }

    /**
     * Get stats for display
     */
    getStats() {
        return {
            totalSteps: this.totalSteps,
            modelReady: this.ready,
            modelType: 'SB3 DQN (Pre-trained)'
        };
    }

    /**
     * Save/load not implemented for pre-trained model
     */
    save() {
        return null;
    }

    load(state) {
        // Not applicable for pre-trained model
    }

    /**
     * Clean up resources
     */
    dispose() {
        if (this.model) {
            this.model.dispose();
        }
    }
}
