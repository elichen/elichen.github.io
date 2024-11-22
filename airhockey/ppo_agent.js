const LOG_2PI = Math.log(2 * Math.PI);

class PPOAgent {
    constructor(stateSize, actionSize) {
        this.stateSize = stateSize;
        this.actionSize = actionSize;
        
        this.gamma = 0.99;
        this.lambda = 0.95;
        this.epsilon = 0.2; // PPO clipping parameter
        this.learningRate = 0.0005;
        
        // Create actor (policy) and critic (value) networks
        this.actor = this.createActorNetwork();
        this.critic = this.createCriticNetwork();
        
        // **Add optimizers for actor and critic**
        this.actorOptimizer = tf.train.adam(this.learningRate);
        this.criticOptimizer = tf.train.adam(this.learningRate);
        
        // Experience buffer
        this.states = [];
        this.actions = [];
        this.rewards = [];
        this.values = [];
        this.logProbs = [];
        this.dones = [];
        
        this.batchSize = 64;
        this.epochs = 10;
        this.frameCount = 0;
    }

    createActorNetwork() {
        const input = tf.input({shape: [this.stateSize]});
        
        const hidden1 = tf.layers.dense({
            units: 256,
            activation: 'relu'
        }).apply(input);
        
        const hidden2 = tf.layers.dense({
            units: 256,
            activation: 'relu'
        }).apply(hidden1);
        
        const hidden3 = tf.layers.dense({
            units: 256,
            activation: 'relu'
        }).apply(hidden2);
        
        const hidden4 = tf.layers.dense({
            units: 256,
            activation: 'relu'
        }).apply(hidden3);
        
        // Mean output uses tanh for bounded actions
        const actionMean = tf.layers.dense({
            units: this.actionSize,
            activation: 'tanh',
            name: 'mean'
        }).apply(hidden4);
        
        // Standard deviation uses softplus for positive values
        const actionStd = tf.layers.dense({
            units: this.actionSize,
            activation: 'softplus',
            name: 'std'
        }).apply(hidden4);
        
        // Concatenate mean and std
        const output = tf.layers.concatenate().apply([actionMean, actionStd]);
        
        const model = tf.model({inputs: input, outputs: output});
        
        // **Remove model.compile() call, since we will use custom optimizer**
        // model.compile({
        //     optimizer: tf.train.adam(this.learningRate),
        //     loss: 'meanSquaredError'
        // });
        
        return model;
    }

    createCriticNetwork() {
        const model = tf.sequential();
        
        model.add(tf.layers.dense({
            units: 256,
            activation: 'relu',
            inputShape: [this.stateSize]
        }));
        
        model.add(tf.layers.dense({
            units: 256,
            activation: 'relu'
        }));
        
        model.add(tf.layers.dense({
            units: 256,
            activation: 'relu'
        }));
        
        model.add(tf.layers.dense({
            units: 256,
            activation: 'relu'
        }));
        
        model.add(tf.layers.dense({
            units: 1,
            activation: 'linear'
        }));
        
        // **Remove model.compile() call**
        // model.compile({
        //     optimizer: tf.train.adam(this.learningRate),
        //     loss: 'meanSquaredError'
        // });
        
        return model;
    }

    sampleAction(mean, stddev) {
        return tf.tidy(() => {
            // Box-Muller transform for normal distribution sampling
            const u1 = tf.randomUniform(mean.shape);
            const u2 = tf.randomUniform(mean.shape);
            
            const z = tf.sqrt(tf.mul(-2, tf.log(u1)))
                .mul(tf.cos(tf.mul(2 * Math.PI, u2)));
            
            return mean.add(tf.mul(z, stddev));
        });
    }

    act(state) {
        return tf.tidy(() => {
            const stateTensor = tf.tensor2d([state], [1, this.stateSize]);
            const actionParams = this.actor.predict(stateTensor);
            
            // Split into mean and stddev
            const mean = actionParams.slice([0, 0], [-1, this.actionSize]);
            // Use softplus to ensure positive standard deviation
            const stddev = tf.softplus(actionParams.slice([0, this.actionSize], [-1, this.actionSize]))
                .add(1e-5); // Add small constant for numerical stability
            
            // Sample action
            const action = this.sampleAction(mean, stddev);
            
            // Calculate log probability
            const logProb = this.calculateLogProb(action, mean, stddev);
            
            // Get value estimate
            const value = this.critic.predict(stateTensor);
            
            return {
                action: action.dataSync(),
                value: value.dataSync()[0],
                logProb: logProb.dataSync()[0]
            };
        });
    }

    calculateLogProb(action, mean, stddev) {
        return tf.tidy(() => {
            const variance = tf.square(stddev);

            // Correct calculation of the log probability for a Gaussian distribution
            const logVariance = tf.log(variance);
            const logScale = tf.add(logVariance, LOG_2PI);
            const squaredDifference = tf.square(action.sub(mean));

            const logProbs = tf.mul(tf.add(logScale, squaredDifference.div(variance)), -0.5);

            // Sum over action dimensions if action space is multidimensional
            return logProbs.sum(-1);
        });
    }

    remember(state, action, reward, value, logProb, done) {
        // Ensure action is an array
        const actionArray = Array.from(action);
        
        this.states.push(state);
        this.actions.push(actionArray);
        this.rewards.push(reward);
        this.values.push(value);
        this.logProbs.push(logProb);
        this.dones.push(done);
    }

    async train() {
        const returns = this.computeReturns();
        const advantages = this.computeAdvantages(returns);
        
        // Increase epochs for smaller batches to ensure sufficient learning
        const adaptiveEpochs = Math.min(20, Math.max(10, Math.floor(1000 / this.states.length)));
        
        // Convert to tensors with proper shapes
        const statesTensor = tf.tensor2d(this.states);
        const actionsTensor = tf.tensor2d(this.actions, [this.actions.length, this.actionSize]);
        const oldLogProbsTensor = tf.tensor1d(this.logProbs);
        const advantagesTensor = tf.tensor1d(advantages);
        const returnsTensor = tf.tensor1d(returns);

        try {
            // PPO training loop with adaptive epochs
            for (let epoch = 0; epoch < adaptiveEpochs; epoch++) {
                await this.trainStep(
                    statesTensor,
                    actionsTensor,
                    oldLogProbsTensor,
                    advantagesTensor,
                    returnsTensor
                );
            }
        } finally {
            // Clear memory
            this.states = [];
            this.actions = [];
            this.rewards = [];
            this.values = [];
            this.logProbs = [];
            this.dones = [];

            // Cleanup tensors
            statesTensor.dispose();
            actionsTensor.dispose();
            oldLogProbsTensor.dispose();
            advantagesTensor.dispose();
            returnsTensor.dispose();
        }
    }

    async trainStep(states, actions, oldLogProbs, advantages, returns) {
        // Normalize advantages (this is key!)
        const normalizedAdvantages = tf.tidy(() => {
            const mean = tf.mean(advantages);
            const std = tf.sqrt(tf.mean(tf.square(tf.sub(advantages, mean))).add(1e-8));
            return tf.div(tf.sub(advantages, mean), std);
        });

        const entropyCoef = 0.01; // Coefficient for entropy bonus

        // Update actor network
        const actorLoss = await this.actorOptimizer.minimize(() => {
            return tf.tidy(() => {
                const actionParams = this.actor.predict(states);
                const mean = actionParams.slice([0, 0], [-1, this.actionSize]);
                const stddev = tf.softplus(actionParams.slice([0, this.actionSize], [-1, this.actionSize]))
                    .add(1e-5);
                
                const newLogProbs = this.calculateLogProb(actions, mean, stddev);
                const ratio = tf.exp(newLogProbs.sub(oldLogProbs));
                
                const surr1 = ratio.mul(normalizedAdvantages);
                const surr2 = tf.clipByValue(ratio, 1 - this.epsilon, 1 + this.epsilon)
                                   .mul(normalizedAdvantages);
                
                const policyLoss = tf.mean(tf.minimum(surr1, surr2)).neg();

                // **Add entropy bonus to encourage exploration**
                const entropy = stddev.log()
                    .add(0.5 * Math.log(2 * Math.PI * Math.E))
                    .sum(-1)
                    .mean();
                const totalLoss = policyLoss.sub(entropy.mul(entropyCoef));
                
                return totalLoss;
            });
        }, true);

        const criticLoss = await this.criticOptimizer.minimize(() => {
            return tf.tidy(() => {
                const valuesPredicted = this.critic.predict(states);
                const returnsReshaped = returns.reshape([-1, 1]);
                return tf.losses.meanSquaredError(returnsReshaped, valuesPredicted);
            });
        }, true);

        // Clean up
        normalizedAdvantages.dispose();

        return { actorLoss, criticLoss };
    }

    computeReturns() {
        const returns = new Array(this.rewards.length);
        let lastReturn = 0;
        
        for (let t = this.rewards.length - 1; t >= 0; t--) {
            if (this.dones[t]) lastReturn = 0;
            lastReturn = this.rewards[t] + this.gamma * lastReturn;
            returns[t] = lastReturn;
        }
        
        return returns;
    }

    computeAdvantages(returns) {
        const advantages = new Array(this.rewards.length);
        let lastGAE = 0;
        
        for (let t = this.rewards.length - 1; t >= 0; t--) {
            if (this.dones[t]) {
                lastGAE = 0; // Reset GAE at the end of each episode
            }
            
            const currentValue = this.values[t];
            const nextValue = t + 1 < this.values.length
                ? this.values[t + 1]
                : 0;
            
            const delta = this.rewards[t] + this.gamma * nextValue * (1 - this.dones[t]) - currentValue;
            
            lastGAE = delta + this.gamma * this.lambda * (1 - this.dones[t]) * lastGAE;
            advantages[t] = lastGAE;
        }
        
        return advantages;
    }

    getState(puck, playerPaddle, aiPaddle, isTopPlayer = false, canvasWidth, canvasHeight) {
        const ownPaddle = isTopPlayer ? aiPaddle : playerPaddle;
        const oppPaddle = isTopPlayer ? playerPaddle : aiPaddle;
        
        // Flip the y-axis for the top player to standardize the coordinate system
        const ownY = isTopPlayer ? canvasHeight - ownPaddle.y : ownPaddle.y;
        const oppY = isTopPlayer ? canvasHeight - oppPaddle.y : oppPaddle.y;
        const puckY = isTopPlayer ? canvasHeight - puck.y : puck.y;
        const puckDy = isTopPlayer ? -puck.dy : puck.dy;

        // Normalize positions and velocities
        const normX = canvasWidth; // Normalize x positions by canvas width
        const normY = canvasHeight; // Normalize y positions by canvas height
        const normSpeed = maxSpeed; // Normalize velocities by max speed

        // Relative positions (normalized between -1 and 1)
        let relativeX = ((puck.x - ownPaddle.x) / normX) * 2;
        let relativeY = ((puckY - ownY) / normY) * 2;

        // Relative velocities (normalized between -1 and 1)
        let relativeDx = (puck.dx / normSpeed) * 2;
        let relativeDy = (puckDy / normSpeed) * 2;

        // Opponent position relative to own paddle (normalized between -1 and 1)
        let relativeOppX = ((oppPaddle.x - ownPaddle.x) / normX) * 2;
        let relativeOppY = ((oppY - ownY) / normY) * 2;
        
        // Distance to the puck (normalized between 0 and 1)
        const distance = Math.sqrt(relativeX * relativeX + relativeY * relativeY) / Math.sqrt(2);

        // Angle from paddle to puck (normalized between -1 and 1)
        const angle = Math.atan2(relativeY, relativeX) / Math.PI; // Normalized angle

        // Is puck behind paddle relative to goal
        const isPuckBehind = puckY > ownY ? 1 : 0;
        
        // Distance to own goal (normalized between -1 and 1)
        const distanceToGoal = ((canvasHeight - ownY) / (canvasHeight / 2)) - 1;
        
        // Distance of puck to goal (normalized between -1 and 1)
        const puckToGoal = ((canvasHeight - puckY) / (canvasHeight / 2)) - 1;
        
        // Normalize paddle speeds (own paddle)
        const ownDx = ((ownPaddle.dx || 0) / normSpeed) * 2;
        const ownDy = ((ownPaddle.dy || 0) / normSpeed) * 2;

        // Normalize puck speed
        const puckSpeed = Math.sqrt(puck.dx * puck.dx + puck.dy * puck.dy) / normSpeed;

        return [
            relativeX,
            relativeY,
            relativeDx,
            relativeDy,
            relativeOppX,
            relativeOppY,
            distance,
            angle,
            isPuckBehind,
            distanceToGoal,
            puckToGoal,
            ownDx,
            ownDy,
            puckSpeed
        ];
    }
} 