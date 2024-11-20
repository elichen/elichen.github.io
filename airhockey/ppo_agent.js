const LOG_2PI = Math.log(2 * Math.PI);

class PPOAgent {
    constructor(stateSize, actionSize) {
        this.stateSize = stateSize;
        this.actionSize = actionSize;
        
        this.gamma = 0.99;
        this.lambda = 0.95;
        this.epsilon = 0.2; // PPO clipping parameter
        this.learningRate = 0.0003;
        
        // Create actor (policy) and critic (value) networks
        this.actor = this.createActorNetwork();
        this.critic = this.createCriticNetwork();
        
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
            units: 64,
            activation: 'relu'
        }).apply(input);
        
        const hidden2 = tf.layers.dense({
            units: 64,
            activation: 'relu'
        }).apply(hidden1);
        
        // Mean output uses tanh for bounded actions
        const actionMean = tf.layers.dense({
            units: this.actionSize,
            activation: 'tanh',
            name: 'mean'
        }).apply(hidden2);
        
        // Standard deviation uses softplus for positive values
        const actionStd = tf.layers.dense({
            units: this.actionSize,
            activation: 'softplus',
            name: 'std'
        }).apply(hidden2);
        
        // Concatenate mean and std
        const output = tf.layers.concatenate().apply([actionMean, actionStd]);
        
        const model = tf.model({inputs: input, outputs: output});
        
        model.compile({
            optimizer: tf.train.adam(this.learningRate),
            loss: 'meanSquaredError'
        });
        
        return model;
    }

    createCriticNetwork() {
        const model = tf.sequential();
        
        model.add(tf.layers.dense({
            units: 64,
            activation: 'relu',
            inputShape: [this.stateSize]
        }));
        
        model.add(tf.layers.dense({
            units: 64,
            activation: 'relu'
        }));
        
        model.add(tf.layers.dense({
            units: 1,
            activation: 'linear'
        }));
        
        model.compile({
            optimizer: tf.train.adam(this.learningRate),
            loss: 'meanSquaredError'
        });
        
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
            // Calculate log probability manually
            // log(P(x)) = -0.5 * (log(2π) + log(σ²) + ((x-μ)²/σ²))
            const variance = tf.square(stddev);
            const diff = action.sub(mean);
            const squaredDiff = tf.square(diff);
            const logProbs = squaredDiff.div(variance.mul(2))
                .add(tf.log(stddev))
                .add(LOG_2PI / 2)
                .mul(-1);
            
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
        if (this.states.length < this.batchSize) return;

        const returns = this.computeReturns();
        const advantages = this.computeAdvantages(returns);

        // Convert to tensors with proper shapes
        const statesTensor = tf.tensor2d(this.states);
        const actionsTensor = tf.tensor2d(this.actions, [this.actions.length, this.actionSize]);
        const oldLogProbsTensor = tf.tensor1d(this.logProbs);
        const advantagesTensor = tf.tensor1d(advantages);
        const returnsTensor = tf.tensor1d(returns);

        try {
            // PPO training loop
            for (let epoch = 0; epoch < this.epochs; epoch++) {
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
        // Update actor network
        const actorLoss = await this.actor.optimizer.minimize(() => {
            return tf.tidy(() => {
                const actionParams = this.actor.predict(states);
                const mean = actionParams.slice([0, 0], [-1, this.actionSize]);
                const stddev = tf.softplus(actionParams.slice([0, this.actionSize], [-1, this.actionSize]))
                    .add(1e-5);
                
                const newLogProbs = this.calculateLogProb(actions, mean, stddev);
                const ratio = tf.exp(newLogProbs.sub(oldLogProbs));
                
                const surr1 = ratio.mul(advantages);
                const surr2 = tf.clipByValue(ratio, 1 - this.epsilon, 1 + this.epsilon).mul(advantages);
                
                return tf.mean(tf.minimum(surr1, surr2)).neg();
            });
        });

        // Update critic network
        const criticLoss = await this.critic.optimizer.minimize(() => {
            return tf.tidy(() => {
                const valuesPredicted = this.critic.predict(states);
                const returnsReshaped = returns.reshape([-1, 1]);
                return tf.losses.meanSquaredError(returnsReshaped, valuesPredicted);
            });
        });

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
        return returns.map((ret, i) => ret - this.values[i]);
    }

    getState(puck, playerPaddle, aiPaddle, isTopPlayer = false, canvasWidth, canvasHeight) {
        const ownPaddle = isTopPlayer ? aiPaddle : playerPaddle;
        const oppPaddle = isTopPlayer ? playerPaddle : aiPaddle;
        
        // Convert everything to relative coordinates from paddle's perspective
        let relativeX = (puck.x - ownPaddle.x) / canvasWidth;
        let relativeY = (puck.y - ownPaddle.y) / canvasHeight;
        
        // Velocities relative to paddle
        let relativeDx = puck.dx / maxSpeed;
        let relativeDy = puck.dy / maxSpeed;
        
        // Opponent position relative to own paddle
        let relativeOppX = (oppPaddle.x - ownPaddle.x) / canvasWidth;
        let relativeOppY = (oppPaddle.y - ownPaddle.y) / canvasHeight;
        
        // Distance is already relative
        const distance = Math.sqrt(relativeX * relativeX + relativeY * relativeY);
        
        // Angle from paddle to puck (relative to vertical)
        const angle = Math.atan2(relativeX, relativeY) / Math.PI;
        
        // Is puck behind paddle relative to goal
        const isPuckBehind = isTopPlayer ? 
            (puck.y < ownPaddle.y) : 
            (puck.y > ownPaddle.y);
        
        // Distance to own goal (normalized)
        const distanceToGoal = isTopPlayer ?
            ownPaddle.y / (canvasHeight/2) :
            (canvasHeight - ownPaddle.y) / (canvasHeight/2);
        
        // Distance of puck to goal (normalized)
        const puckToGoal = isTopPlayer ?
            puck.y / (canvasHeight/2) :
            (canvasHeight - puck.y) / (canvasHeight/2);
        
        return [
            relativeX,
            relativeY,
            relativeDx,
            relativeDy,
            relativeOppX,
            relativeOppY,
            distance,
            angle,
            isPuckBehind ? 1 : 0,
            distanceToGoal,
            puckToGoal
        ];
    }
} 