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
        console.log("Memory size:", this.memory.states.length);
        const { advantages, returns } = this.memory.computeGAE();
        console.log("GAE computed, advantages mean:", tf.mean(tf.tensor1d(advantages)).dataSync()[0]);

        const states = tf.tensor2d(this.memory.states);
        const actions = tf.tensor1d(this.memory.actions, 'int32');
        const oldLogProbs = tf.tensor1d(this.memory.logProbs);
        const advTensor = tf.tensor1d(advantages);
        const retTensor = tf.tensor1d(returns);

        for (let epoch = 0; epoch < this.epochsPerUpdate; epoch++) {
            tf.tidy(() => {
                // Define the training function that returns the loss
                const f = () => {
                    // Forward pass through the policy network
                    const [actionProbs, values] = this.policy.forward(states);
                    
                    // Calculate new log probabilities
                    const actionMask = tf.oneHot(actions, 3);
                    const newLogProbs = tf.log(tf.sum(tf.mul(actionProbs, actionMask), -1));
                    
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
                    
                    // Combine losses
                    return tf.add(actorLoss, tf.mul(criticLoss, 0.5));
                };

                // Calculate gradients
                const {value, grads} = tf.variableGrads(f);
                
                // Apply gradients
                this.policy.optimizer.applyGradients(grads);
                
                // Return the loss value
                return value;
            });
        }

        // Cleanup
        states.dispose();
        actions.dispose();
        oldLogProbs.dispose();
        advTensor.dispose();
        retTensor.dispose();

        this.memory.clear();

        console.log("PPO update completed");
    }

    selectAction(state) {
        const { actionProbs, value } = this.policy.predict(state);
        const action = this.policy.sampleAction(actionProbs);
        const logProb = Math.log(actionProbs[action]);
        return { action, value, logProb };
    }
} 