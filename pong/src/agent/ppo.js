class PPOAgent {
    constructor() {
        this.policy = new PolicyNetwork();
        this.memory = new Memory();
        this.clipRatio = 0.2;
        this.batchSize = 64;
        this.epochsPerUpdate = 10;
    }

    async update() {
        console.log("Starting PPO update");
        
        // Compute advantages and returns
        const { advantages, returns } = this.memory.computeGAE();
        console.log("GAE computed, advantages mean:", tf.mean(tf.tensor1d(advantages)).dataSync()[0]);

        // Convert to tensors
        const states = tf.tensor2d(this.memory.states);
        const actions = tf.tensor1d(this.memory.actions, 'int32');
        const oldLogProbs = tf.tensor1d(this.memory.logProbs);
        const advTensor = tf.tensor1d(advantages);
        const retTensor = tf.tensor1d(returns);

        try {
            for (let epoch = 0; epoch < this.epochsPerUpdate; epoch++) {
                const totalLoss = tf.tidy(() => {
                    // Define loss function for gradient calculation
                    const lossFn = () => {
                        // Forward pass through the policy network
                        const [actionProbs, values] = this.policy.forward(states);
                        
                        // Calculate new log probabilities
                        const actionMask = tf.oneHot(actions, 3);
                        const selectedProbs = tf.sum(tf.mul(actionProbs, actionMask), -1);
                        const newLogProbs = tf.log(tf.maximum(selectedProbs, 1e-10)); // Prevent log(0)
                        
                        // Calculate probability ratios
                        const ratio = tf.exp(tf.sub(newLogProbs, oldLogProbs));
                        
                        // Calculate surrogate losses
                        const surr1 = tf.mul(ratio, advTensor);
                        const surr2 = tf.mul(
                            tf.clipByValue(ratio, 1 - this.clipRatio, 1 + this.clipRatio),
                            advTensor
                        );
                        
                        // Calculate actor and critic losses
                        const actorLoss = tf.neg(tf.mean(tf.minimum(surr1, surr2)));
                        const criticLoss = tf.mean(tf.square(tf.sub(values, retTensor)));
                        
                        // Combine losses with value function coefficient
                        return tf.add(actorLoss, tf.mul(0.5, criticLoss));
                    };
                    
                    // Calculate gradients using the loss function
                    const {value, grads} = tf.variableGrads(lossFn);
                    
                    // Apply gradients using the optimizer
                    this.policy.optimizer.applyGradients(grads);
                    
                    return value;
                });

                // Log the loss value
                const lossValue = await totalLoss.data();
                console.log(`Epoch ${epoch + 1}/${this.epochsPerUpdate}, Loss: ${lossValue[0]}`);
                totalLoss.dispose();
            }
        } finally {
            // Cleanup tensors
            states.dispose();
            actions.dispose();
            oldLogProbs.dispose();
            advTensor.dispose();
            retTensor.dispose();
        }

        this.memory.clear();
        console.log("PPO update completed");
    }

    selectAction(state) {
        return tf.tidy(() => {
            const stateTensor = tf.tensor2d([state]);
            const { actionProbs, value } = this.policy.predict(stateTensor);
            const action = this.policy.sampleAction(actionProbs);
            const logProb = Math.log(Math.max(actionProbs[action], 1e-10));
            return { action, value, logProb };
        });
    }
} 