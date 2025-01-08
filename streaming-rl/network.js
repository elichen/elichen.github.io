class LayerNorm extends tf.layers.Layer {
    constructor(config) {
        super({
            name: config?.name || null,
            trainable: false,
            dtype: config?.dtype || 'float32'
        });
        this.epsilon = 1e-5;
        this.supportsMasking = true;
    }

    className = 'LayerNorm';

    call(inputs) {
        return tf.tidy(() => {
            // Handle array of tensors case
            let x = Array.isArray(inputs) ? inputs[0] : inputs;
            
            // Ensure input is a tensor
            x = tf.cast(x, 'float32');
            
            // Get feature dimension and ensure rank 2
            const inputShape = x.shape;
            const featureDim = inputShape[inputShape.length - 1];
            if (x.rank === 1) {
                x = x.reshape([1, featureDim]);
            }
            
            // Compute statistics along feature axis and normalize
            const moments = tf.moments(x, -1, true);
            return x.sub(moments.mean).div(tf.sqrt(moments.variance.add(this.epsilon)));
        });
    }

    computeOutputShape(inputShape) {
        return Array.isArray(inputShape) && inputShape.length === 1 ? inputShape[0] : inputShape;
    }

    getConfig() {
        const config = super.getConfig();
        Object.assign(config, {
            epsilon: this.epsilon
        });
        return config;
    }

    static className = 'LayerNorm';
}
tf.serialization.registerClass(LayerNorm);

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
        model.add(new LayerNorm({name: 'norm1'}));
        model.add(tf.layers.leakyReLU({alpha: 0.01}));

        // Hidden layer with layer normalization
        model.add(tf.layers.dense({
            units: hiddenSize,
            activation: 'linear',
            name: 'hidden',
            trainable: true
        }));
        model.add(new LayerNorm({name: 'norm2'}));
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