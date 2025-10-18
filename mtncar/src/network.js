class LayerNorm extends tf.layers.Layer {
    constructor(config) {
        super({
            name: config?.name || null,
            trainable: false,
            dtype: config?.dtype || 'float32'
        });
        this.normalizedShape = config.normalizedShape;
        this.epsilon = 1e-5;
        this.supportsMasking = true;
    }

    build(inputShape) {
        // No learnable parameters
    }

    call(inputs) {
        return tf.tidy(() => {
            let x = Array.isArray(inputs) ? inputs[0] : inputs;
            x = tf.cast(x, 'float32');
            
            // Compute moments over normalized_shape dimensions
            const moments = tf.moments(x, -1, true);
            return x.sub(moments.mean).div(tf.sqrt(moments.variance.add(this.epsilon)));
        });
    }

    computeOutputShape(inputShape) {
        return inputShape;
    }

    getConfig() {
        const config = super.getConfig();
        Object.assign(config, {
            epsilon: this.epsilon,
            normalizedShape: this.normalizedShape
        });
        return config;
    }

    static className = 'LayerNorm';
}
tf.serialization.registerClass(LayerNorm);

class StreamingNetwork {
    constructor(inputSize = 4, hiddenSize = 32, numActions = 2, config = {}) {
        // Support both scalar hiddenSize and array of sizes
        if (Array.isArray(hiddenSize)) {
            this.hiddenSizes = hiddenSize;
        } else {
            this.hiddenSizes = [hiddenSize, hiddenSize];
        }

        this.inputSize = inputSize;
        this.numActions = numActions;
        this.config = config;

        this.model = this.buildModel(inputSize, this.hiddenSizes, numActions);

        // Only apply sparse init if explicitly requested (default for old behavior)
        if (config.sparseInit !== false && !Array.isArray(hiddenSize)) {
            this.sparseInit();
        } else if (config.sparseInit === true) {
            this.sparseInit();
        }

        // Track gradients for manual backprop
        this.gradients = {};
    }

    buildModel(inputSize, hiddenSizes, numActions) {
        const model = tf.sequential();

        // Add hidden layers with layer normalization
        for (let i = 0; i < hiddenSizes.length; i++) {
            const hiddenSize = hiddenSizes[i];

            model.add(tf.layers.dense({
                units: hiddenSize,
                inputShape: i === 0 ? [inputSize] : undefined,
                activation: 'linear',
                name: `fc${i + 1}`,
                trainable: true
            }));

            // Add layer norm if enabled
            if (this.config.layerNorm !== false) {
                model.add(new LayerNorm({name: `norm${i + 1}`, normalizedShape: [hiddenSize]}));
            }

            // Add activation
            if (this.config.activation === 'relu') {
                model.add(tf.layers.reLU());
            } else {
                model.add(tf.layers.leakyReLU({alpha: 0.01}));
            }
        }

        // Output layer
        model.add(tf.layers.dense({
            units: numActions,
            activation: 'linear',
            name: 'output',
            trainable: true
        }));

        model.compile({
            optimizer: tf.train.sgd(0.1),
            loss: 'meanSquaredError'
        });

        return model;
    }

    predict(state) {
        return tf.tidy(() => {
            const stateTensor = tf.tensor2d([state], [1, state.length]);
            return this.model.predict(stateTensor);
        });
    }

    /**
     * Forward pass - returns array of Q-values
     */
    forward(state) {
        const result = tf.tidy(() => {
            const stateTensor = tf.tensor2d([state], [1, state.length]);
            const output = this.model.predict(stateTensor);
            return output.arraySync()[0];
        });
        return result;
    }

    /**
     * Backward pass - accumulate gradients (simplified for DQN)
     */
    backward(gradOutput) {
        // Store gradients for optimizer to use
        // This is a simplified implementation - real backprop would need more complexity
        if (!this.gradients.output) {
            this.gradients.output = [];
        }
        this.gradients.output.push(gradOutput);
    }

    /**
     * Reset accumulated gradients
     */
    zeroGrad() {
        this.gradients = {};
    }

    /**
     * Get all network parameters as a dictionary
     */
    getParameters() {
        const params = {};
        const layers = this.model.layers.filter(layer => layer.getClassName() === 'Dense');

        layers.forEach((layer, idx) => {
            const weights = layer.getWeights();
            params[`layer${idx}_weight`] = weights[0].arraySync();
            params[`layer${idx}_bias`] = weights[1].arraySync();
        });

        return params;
    }

    /**
     * Set network parameters from a dictionary
     */
    setParameters(params) {
        const layers = this.model.layers.filter(layer => layer.getClassName() === 'Dense');

        layers.forEach((layer, idx) => {
            const weightKey = `layer${idx}_weight`;
            const biasKey = `layer${idx}_bias`;

            if (params[weightKey] && params[biasKey]) {
                const weight = tf.tensor(params[weightKey]);
                const bias = tf.tensor(params[biasKey]);
                layer.setWeights([weight, bias]);
            }
        });
    }

    getTrainableVariables() {
        return this.model.trainableWeights;
    }

    async saveModel() {
        await this.model.save('localstorage://streaming-cartpole');
    }

    async loadModel() {
        try {
            this.model = await tf.loadLayersModel('localstorage://streaming-cartpole');
        } catch (error) {
            console.error('Error loading model:', error);
        }
    }

    async sparseInit() {
        const layers = this.model.layers.filter(layer => layer.getClassName() === 'Dense');
        
        for (const layer of layers) {
            const weights = layer.getWeights();
            const w = weights[0];
            const expectedShape = w.shape;
            
            const newWeights = tf.tidy(() => {
                const [inputSize, outputSize] = expectedShape;
                const weights = tf.randomUniform(expectedShape, -1.0/Math.sqrt(inputSize), 1.0/Math.sqrt(inputSize));
                
                const sparsity = 0.9;
                const numZeros = Math.ceil(sparsity * inputSize);
                
                const permutation = Array.from({length: inputSize}, (_, i) => i);
                for (let i = permutation.length - 1; i > 0; i--) {
                    const j = Math.floor(Math.random() * (i + 1));
                    [permutation[i], permutation[j]] = [permutation[j], permutation[i]];
                }
                
                const zeroIndices = new Set(permutation.slice(0, numZeros));
                
                const maskData = new Float32Array(inputSize * outputSize);
                for (let j = 0; j < outputSize; j++) {
                    for (let i = 0; i < inputSize; i++) {
                        maskData[i * outputSize + j] = zeroIndices.has(i) ? 0 : 1;
                    }
                }
                
                const mask = tf.tensor2d(maskData, expectedShape);
                return tf.mul(weights, mask);
            });

            const zeroBias = tf.zeros([expectedShape[1]]);
            await layer.setWeights([newWeights, zeroBias]);
            newWeights.dispose();
        }
    }
} 