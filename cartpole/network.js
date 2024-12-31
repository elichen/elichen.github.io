class StreamingNetwork {
    constructor(inputSize = 4, hiddenSize = 32, numActions = 2) {
        this.model = this.buildModel(inputSize, hiddenSize, numActions);
        this.sparseInit(0.9); // 90% sparsity as in the original code
    }

    buildModel(inputSize, hiddenSize, numActions) {
        const model = tf.sequential();
        
        // First layer with layer normalization
        model.add(tf.layers.dense({
            units: hiddenSize,
            inputShape: [inputSize],
            activation: 'linear',
            name: 'fc1',
            trainable: true
        }));
        model.add(tf.layers.layerNormalization({
            axis: -1,  // Normalize over the last axis (features)
            epsilon: 1e-5,  // Match PyTorch's default
            center: true,  // Use beta
            scale: true,   // Use gamma
            beta_initializer: 'zeros',
            gamma_initializer: 'ones'
        }));
        model.add(tf.layers.leakyReLU({alpha: 0.01}));  // Match PyTorch's default

        // Hidden layer with layer normalization
        model.add(tf.layers.dense({
            units: hiddenSize,
            activation: 'linear',
            name: 'hidden',
            trainable: true
        }));
        model.add(tf.layers.layerNormalization({
            axis: -1,
            epsilon: 1e-5,
            center: true,
            scale: true,
            beta_initializer: 'zeros',
            gamma_initializer: 'ones'
        }));
        model.add(tf.layers.leakyReLU({alpha: 0.01}));

        // Output layer
        model.add(tf.layers.dense({
            units: numActions,
            activation: 'linear',
            name: 'output',
            trainable: true
        }));

        // Compile the model to initialize variables
        model.compile({
            optimizer: tf.train.sgd(0.1),  // Dummy optimizer, we'll use our custom one
            loss: 'meanSquaredError'
        });

        return model;
    }

    async sparseInit(sparsity) {
        // Implement sparse initialization for each dense layer
        const layers = this.model.layers.filter(layer => layer.getClassName() === 'Dense');
        
        for (const layer of layers) {
            const weights = layer.getWeights();
            const w = weights[0];
            const shape = w.shape;
            const fanIn = shape[1];
            
            // Create new weights with uniform distribution
            const newWeights = tf.tidy(() => {
                const scale = Math.sqrt(1.0 / fanIn);
                const weights = tf.randomUniform(shape, -scale, scale);
                
                // Create sparse mask
                const mask = tf.tidy(() => {
                    const random = tf.randomUniform(shape);
                    return tf.greater(random, sparsity).asType('float32');
                });
                
                // Apply mask to weights
                return tf.mul(weights, mask);
            });

            // Set the new weights
            await layer.setWeights([newWeights, weights[1]]);
        }
    }

    predict(state) {
        return tf.tidy(() => {
            const stateTensor = tf.tensor2d([state], [1, state.length]);
            return this.model.predict(stateTensor);
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
            console.log('Model loaded successfully');
        } catch (error) {
            console.error('Error loading model:', error);
        }
    }
} 