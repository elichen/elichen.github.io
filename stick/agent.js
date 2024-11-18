class PolicyGradientAgent {
    constructor(env) {
        this.env = env;
        this.stateSize = 4;
        this.actionSize = 3;
        
        // Hyperparameters
        this.gamma = 0.99;        // Discount factor
        this.learningRate = 0.01; // Learning rate for policy network
        this.episodeMemory = [];  // Store episode trajectories
        
        // Create policy network
        this.policyModel = this.createPolicyNetwork();
    }

    createPolicyNetwork() {
        const model = tf.sequential();
        
        // Hidden layers
        model.add(tf.layers.dense({
            units: 64,
            activation: 'relu',
            inputShape: [this.stateSize]
        }));
        model.add(tf.layers.dense({
            units: 32,
            activation: 'relu'
        }));
        
        // Output layer with softmax for action probabilities
        model.add(tf.layers.dense({
            units: this.actionSize,
            activation: 'softmax'
        }));

        const optimizer = tf.train.adam(this.learningRate);
        model.compile({ optimizer: optimizer, loss: 'categoricalCrossentropy' });
        
        return model;
    }

    async selectAction(state, testing = false) {
        const stateTensor = tf.tensor2d([state]);
        let actionProbs = await this.policyModel.predict(stateTensor).array();
        
        if (!testing) {
            // Add exploration noise
            const noise = 0.1; // Adjust this value to control the amount of noise
            actionProbs[0] = actionProbs[0].map(prob => prob + noise * (Math.random() - 0.5));
            
            // Normalize to ensure they sum to 1
            const sum = actionProbs[0].reduce((a, b) => a + b, 0);
            actionProbs[0] = actionProbs[0].map(prob => prob / sum);
        }

        // During testing, just take the most probable action
        if (testing) {
            return actionProbs[0].indexOf(Math.max(...actionProbs[0]));
        }
        
        // Sample action from probability distribution
        return this.sampleAction(actionProbs[0]);
    }

    sampleAction(probabilities) {
        const cumSum = probabilities.reduce((acc, prob, i) => {
            acc.push((acc[i-1] || 0) + prob);
            return acc;
        }, []);
        
        const random = Math.random();
        return cumSum.findIndex(sum => random < sum);
    }

    remember(state, action, reward) {
        this.episodeMemory.push({state, action, reward});
    }

    async update(state, action, reward, nextState, done) {
        this.remember(state, action, reward);
        
        if (done) {
            await this.trainOnEpisode();
            this.episodeMemory = []; // Clear memory after update
        }
    }

    async trainOnEpisode() {
        if (this.episodeMemory.length === 0) return;

        // Calculate discounted returns
        const returns = this.calculateReturns();
        
        // Calculate advantages by subtracting the mean return
        const meanReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
        const advantages = returns.map(r => r - meanReturn);
        
        // Prepare training data
        const states = this.episodeMemory.map(exp => exp.state);
        const actions = this.episodeMemory.map(exp => exp.action);
        
        // Convert to tensors
        const statesTensor = tf.tensor2d(states);
        const actionsTensor = tf.tensor1d(actions, 'int32');
        const advantagesTensor = tf.tensor1d(advantages);

        // Define the training function
        const trainStep = () => {
            // Forward pass through the model
            const actionProbs = this.policyModel.predict(statesTensor);
            
            // Create one-hot encoded actions
            const actionMasks = tf.oneHot(actionsTensor, this.actionSize);
            
            // Calculate log probabilities
            const logProbs = tf.log(tf.sum(
                actionMasks.mul(actionProbs).add(1e-10),
                -1
            ));
            
            // Calculate loss using advantages
            return advantagesTensor.mul(logProbs).mean().mul(-1);
        };

        // Perform optimization step
        const optimizer = tf.train.adam(this.learningRate);
        optimizer.minimize(trainStep);

        // Clean up tensors
        statesTensor.dispose();
        actionsTensor.dispose();
        advantagesTensor.dispose();
    }

    calculateReturns() {
        const returns = new Array(this.episodeMemory.length);
        let cumReturn = 0;
        
        // Calculate returns from back to front
        for (let t = this.episodeMemory.length - 1; t >= 0; t--) {
            cumReturn = this.episodeMemory[t].reward + this.gamma * cumReturn;
            returns[t] = cumReturn;
        }
        
        // Normalize returns
        const mean = returns.reduce((a, b) => a + b) / returns.length;
        const std = Math.sqrt(
            returns.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / returns.length
        );
        
        return returns.map(r => (r - mean) / (std + 1e-8));
    }

    reset() {
        this.episodeMemory = [];
    }
}