class SnakeModel {
    constructor(inputSize, hiddenSize, outputSize) {
        // Policy Network
        this.model = tf.sequential();
        this.model.add(tf.layers.dense({ units: hiddenSize, activation: 'relu', inputShape: [inputSize] }));
        this.model.add(tf.layers.dense({ units: hiddenSize, activation: 'relu' }));
        this.model.add(tf.layers.dense({ units: outputSize, activation: 'linear' }));

        // Target Network (copy of the policy network)
        this.targetModel = tf.sequential();
        this.targetModel.add(tf.layers.dense({ units: hiddenSize, activation: 'relu', inputShape: [inputSize] }));
        this.targetModel.add(tf.layers.dense({ units: hiddenSize, activation: 'relu' }));
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
            const stateTensor = tf.tensor2d(state);
            return this.model.predict(stateTensor);
        });
    }

    predictTarget(state) {
        return tf.tidy(() => {
            const stateTensor = tf.tensor2d(state);
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
            console.log('Loading pre-trained weights...');
            
            // Load the weights from the JSON file
            const response = await fetch('web_model/model_weights.json');
            const weightsData = await response.json();
            
            // Convert PyTorch weights to TensorFlow.js format
            // PyTorch uses different naming convention: fc1.weight, fc1.bias, etc.
            const tfWeights = [];
            
            // Layer 1 weights and bias
            if (weightsData['fc1.weight'] && weightsData['fc1.bias']) {
                // PyTorch stores weights as [out_features, in_features], TF.js expects [in_features, out_features]
                const weight1 = tf.tensor2d(weightsData['fc1.weight']).transpose();
                const bias1 = tf.tensor1d(weightsData['fc1.bias']);
                tfWeights.push(weight1, bias1);
            }
            
            // Layer 2 weights and bias
            if (weightsData['fc2.weight'] && weightsData['fc2.bias']) {
                const weight2 = tf.tensor2d(weightsData['fc2.weight']).transpose();
                const bias2 = tf.tensor1d(weightsData['fc2.bias']);
                tfWeights.push(weight2, bias2);
            }
            
            // Layer 3 weights and bias
            if (weightsData['fc3.weight'] && weightsData['fc3.bias']) {
                const weight3 = tf.tensor2d(weightsData['fc3.weight']).transpose();
                const bias3 = tf.tensor1d(weightsData['fc3.bias']);
                tfWeights.push(weight3, bias3);
            }
            
            // Set weights to both policy and target networks
            this.model.setWeights(tfWeights);
            this.targetModel.setWeights(tfWeights);
            
            this.isPreTrained = true;
            console.log('Pre-trained weights loaded successfully!');
            
            return true;
        } catch (error) {
            console.error('Error loading pre-trained weights:', error);
            console.log('Continuing with random initialization...');
            return false;
        }
    }
}