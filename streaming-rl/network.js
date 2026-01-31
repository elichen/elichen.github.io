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
    constructor(inputSize = 4, hiddenSize = 32, numActions = 2) {
        this.model = this.buildModel(inputSize, hiddenSize, numActions);
        // Store the promise so we can wait for it before loading pretrained weights
        this.initPromise = this.sparseInit();
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
        model.add(new LayerNorm({name: 'norm1', normalizedShape: [hiddenSize]}));
        model.add(tf.layers.leakyReLU({alpha: 0.01}));

        // Hidden layer with layer normalization
        model.add(tf.layers.dense({
            units: hiddenSize,
            activation: 'linear',
            name: 'hidden',
            trainable: true
        }));
        model.add(new LayerNorm({name: 'norm2', normalizedShape: [hiddenSize]}));
        model.add(tf.layers.leakyReLU({alpha: 0.01}));

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

    async loadPretrainedWeights(weightsJson) {
        // Wait for sparseInit to complete first (avoid race condition)
        if (this.initPromise) {
            await this.initPromise;
        }

        // Load weights from JSON format (overwrites sparseInit weights)
        for (const layer of this.model.layers) {
            if (layer.getClassName() === 'Dense' && weightsJson[layer.name]) {
                const data = weightsJson[layer.name];
                const kernel = tf.tensor(data.kernel, data.kernelShape);
                const bias = tf.tensor(data.bias, data.biasShape);
                await layer.setWeights([kernel, bias]);
                kernel.dispose();
                bias.dispose();
            }
        }
        console.log('Pretrained weights loaded successfully');
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
                // Ensure at least 1 input remains active (don't zero everything)
                const numZeros = Math.min(Math.ceil(sparsity * inputSize), inputSize - 1);
                
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