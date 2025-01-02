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
            axis: [1],  // Normalize over the input features
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
            axis: [1],  // Normalize over the hidden features
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
            const [fanOut, fanIn] = shape;
            
            // Create new weights with uniform distribution
            const newWeights = tf.tidy(() => {
                const scale = Math.sqrt(1.0 / fanIn);
                const weights = tf.randomUniform(shape, -scale, scale);
                
                // Create sparse mask per output neuron
                const numZeros = Math.ceil(sparsity * fanIn);
                const mask = tf.buffer(shape);
                
                // Fill mask with ones initially
                for (let i = 0; i < fanOut; i++) {
                    for (let j = 0; j < fanIn; j++) {
                        mask.set(1, i, j);
                    }
                }
                
                // Zero out random inputs for each output independently
                for (let outIdx = 0; outIdx < fanOut; outIdx++) {
                    const indices = Array.from({length: fanIn}, (_, i) => i);
                    for (let i = indices.length - 1; i > 0; i--) {
                        const j = Math.floor(Math.random() * (i + 1));
                        [indices[i], indices[j]] = [indices[j], indices[i]];
                    }
                    const zeroIndices = indices.slice(0, numZeros);
                    for (const idx of zeroIndices) {
                        mask.set(0, outIdx, idx);
                    }
                }
                
                // Apply mask to weights
                return tf.mul(weights, mask.toTensor());
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