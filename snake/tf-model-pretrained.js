class SnakeModel {
    constructor(inputSize, hiddenSize, outputSize) {
        // Policy Network (matching PPO export: tanh activation)
        this.model = tf.sequential();
        this.model.add(tf.layers.dense({ units: hiddenSize, activation: 'tanh', inputShape: [inputSize] }));
        this.model.add(tf.layers.dense({ units: hiddenSize, activation: 'tanh' }));
        this.model.add(tf.layers.dense({ units: outputSize, activation: 'linear' }));

        // Target Network (copy of the policy network)
        this.targetModel = tf.sequential();
        this.targetModel.add(tf.layers.dense({ units: hiddenSize, activation: 'tanh', inputShape: [inputSize] }));
        this.targetModel.add(tf.layers.dense({ units: hiddenSize, activation: 'tanh' }));
        this.targetModel.add(tf.layers.dense({ units: outputSize, activation: 'linear' }));

        this.optimizer = tf.train.adam(0.001);

        // Flag to track if pre-trained weights are loaded
        this.isPreTrained = false;

        // Sync the target network weights with the policy network initially
        this.updateTargetNetwork();
    }

    // Function to update the target network with the weights of the policy network
    updateTargetNetwork() {
        const policyWeights = this.model.getWeights();
        this.targetModel.setWeights(policyWeights);
    }

    predict(state) {
        return tf.tidy(() => {
            // state comes as [[f1, f2, ...f24]] from caller
            // Flatten and reshape to ensure proper tensor
            const features = Array.isArray(state[0]) ? state[0] : state;
            const stateTensor = tf.tensor(features, [1, 24], 'float32');
            return this.model.predict(stateTensor);
        });
    }

    predictTarget(state) {
        return tf.tidy(() => {
            const features = Array.isArray(state[0]) ? state[0] : state;
            const stateTensor = tf.tensor(features, [1, 24], 'float32');
            return this.targetModel.predict(stateTensor);
        });
    }

    async train(states, targets) {
        const statesTensor = tf.tensor2d(states);
        const targetsTensor = tf.tensor2d(targets);

        const loss = () => tf.tidy(() => {
            const predictions = this.model.predict(statesTensor);
            return predictions.sub(targetsTensor).square().mean();
        });

        await this.optimizer.minimize(loss, true);

        tf.dispose([statesTensor, targetsTensor]);
    }

    getWeights() {
        return this.model.getWeights();
    }

    setWeights(weights) {
        this.model.setWeights(weights);
    }
    
    async loadPreTrainedWeights() {
        try {

            // Load weights directly from the binary file
            const weightsManifest = await fetch('tfjs_model/model.json').then(r => r.json());
            const weightSpecs = weightsManifest.weightsManifest[0].weights;
            const weightData = await fetch('tfjs_model/' + weightsManifest.weightsManifest[0].paths[0])
                .then(r => r.arrayBuffer());

            // Parse weights
            const weightValues = new Float32Array(weightData);
            let offset = 0;
            const weights = [];

            for (const spec of weightSpecs) {
                const size = spec.shape.reduce((a, b) => a * b, 1);
                const values = weightValues.slice(offset, offset + size);
                const tensor = tf.tensor(Array.from(values), spec.shape, 'float32');
                weights.push(tensor);
                offset += size;
            }

            // Set weights to both policy and target networks
            this.model.setWeights(weights);
            this.targetModel.setWeights(weights);

            this.isPreTrained = true;

            return true;
        } catch (error) {
            console.error('Error loading pre-trained weights:', error);
            return false;
        }
    }
}