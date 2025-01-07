class StreamingNetwork {
    constructor(inputSize = 4, hiddenSize = 32, numActions = 2) {
        this.model = this.buildModel(inputSize, hiddenSize, numActions);
        this.sparseInit();
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

    async sparseInit() {
        // Implement sparse initialization for each dense layer
        const layers = this.model.layers.filter(layer => layer.getClassName() === 'Dense');
        
        for (const layer of layers) {
            const weights = layer.getWeights();
            const w = weights[0];
            const expectedShape = w.shape;
            
            // Create new weights with LeCun initialization
            const newWeights = tf.tidy(() => {
                // In TensorFlow.js dense layers, shape is [inputSize, outputSize]
                const [inputSize, outputSize] = expectedShape;
                
                // Algorithm 1: Wi,j ~ U[-1/√fan_in, 1/√fan_in], ∀i,j
                const weights = tf.randomUniform(expectedShape, -1.0/Math.sqrt(inputSize), 1.0/Math.sqrt(inputSize));
                
                // Algorithm 1: n ← s × fan_in
                const sparsity = 0.9;  // Paper specifies s = 0.9
                const numZeros = Math.ceil(sparsity * inputSize);
                
                // Algorithm 1: Permutation set P of size fan_in
                const permutation = Array.from({length: inputSize}, (_, i) => i);
                for (let i = permutation.length - 1; i > 0; i--) {
                    const j = Math.floor(Math.random() * (i + 1));
                    [permutation[i], permutation[j]] = [permutation[j], permutation[i]];
                }
                
                // Algorithm 1: Index set I of size n (subset of P)
                const zeroIndices = new Set(permutation.slice(0, numZeros));
                
                // Algorithm 1: Wi,j ← 0, ∀i∈I, ∀j
                const maskData = new Float32Array(inputSize * outputSize);
                for (let j = 0; j < outputSize; j++) {
                    for (let i = 0; i < inputSize; i++) {
                        maskData[i * outputSize + j] = zeroIndices.has(i) ? 0 : 1;
                    }
                }
                
                // Apply mask to weights
                const mask = tf.tensor2d(maskData, expectedShape);
                return tf.mul(weights, mask);
            });

            // Set the new weights and zero biases (Algorithm 1: bi ← 0, ∀i)
            const zeroBias = tf.zeros([expectedShape[1]]);  // bias size should match output dimension
            await layer.setWeights([newWeights, zeroBias]);
            newWeights.dispose();
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