/**
 * Deep Q-Network (DQN) Agent with Experience Replay and Target Network
 * More sample-efficient than streaming Q-learning
 */
class DQNAgent {
    constructor(env, config = {}) {
        // Environment
        this.env = env;

        // Get environment dimensions
        const testState = env.getState();
        this.stateSize = testState.length;
        this.numActions = env.numActions || 3; // Mountain car has 3 actions

        // Create simple action space for compatibility
        this.actionSpace = {
            n: this.numActions,
            sample: () => Math.floor(Math.random() * this.numActions)
        };

        // Hyperparameters with defaults
        this.config = {
            // Network architecture
            hiddenSizes: config.hiddenSizes || [64, 64], // Larger network for better representation
            activation: config.activation || 'leakyReLU',
            layerNorm: config.layerNorm !== false, // Default to true

            // Q-learning parameters
            gamma: config.gamma || 0.99,
            learningRate: config.learningRate || 0.001,

            // Experience replay
            bufferSize: config.bufferSize || 100000,
            batchSize: config.batchSize || 32,
            minBufferSize: config.minBufferSize || 1000, // Start learning after this many samples
            usePrioritizedReplay: config.usePrioritizedReplay !== false, // Default to true

            // Target network
            targetUpdateFreq: config.targetUpdateFreq || 100, // Update target network every N steps
            tau: config.tau || 1.0, // Soft update parameter (1.0 = hard update)

            // Exploration
            explorationMode: config.explorationMode || 'epsilon', // 'epsilon', 'ucb', 'noisy', 'boltzmann'
            epsilonStart: config.epsilonStart || 1.0,
            epsilonEnd: config.epsilonEnd || 0.01,
            epsilonDecaySteps: config.epsilonDecaySteps || 50000, // Faster decay for better efficiency
            useNoisyNets: config.useNoisyNets || false, // Alternative to epsilon-greedy
            ucbC: config.ucbC || 2.0, // UCB exploration parameter
            boltzmannTemp: config.boltzmannTemp || 1.0, // Initial Boltzmann temperature
            boltzmannTempEnd: config.boltzmannTempEnd || 0.1, // Final Boltzmann temperature

            // N-step returns
            nSteps: config.nSteps || 3, // Use 3-step returns by default

            // Double DQN
            useDoubleDQN: config.useDoubleDQN !== false, // Default to true

            // Training
            updateFreq: config.updateFreq || 4, // Update every 4 steps
            gradientsPerStep: config.gradientsPerStep || 1, // Number of gradient steps per update
        };

        // Initialize networks
        this.qNetwork = new StreamingNetwork(
            this.stateSize,              // inputSize (2 for mountain car)
            this.config.hiddenSizes,      // hiddenSize ([64, 64])
            this.numActions,              // numActions (3)
            {
                activation: this.config.activation,
                layerNorm: this.config.layerNorm,
                sparseInit: false // Don't use sparse init for better convergence
            }
        );

        // Target network for stable Q-targets
        this.targetNetwork = new StreamingNetwork(
            this.stateSize,              // inputSize (2 for mountain car)
            this.config.hiddenSizes,      // hiddenSize ([64, 64])
            this.numActions,              // numActions (3)
            {
                activation: this.config.activation,
                layerNorm: this.config.layerNorm,
                sparseInit: false
            }
        );

        // Copy initial weights to target network
        this.updateTargetNetwork(1.0);

        // Initialize replay buffer
        if (this.config.usePrioritizedReplay) {
            this.replayBuffer = new PrioritizedReplayBuffer(
                this.config.bufferSize,
                0.6,  // alpha
                0.4,  // beta
                this.config.epsilonDecaySteps  // betaSchedule
            );
        } else {
            this.replayBuffer = new ReplayBuffer(this.config.bufferSize);
        }

        // N-step buffer for multi-step returns
        this.nStepBuffer = [];

        // Optimizer
        this.optimizer = new AdamOptimizer(this.config.learningRate);

        // Training statistics
        this.totalSteps = 0;
        this.episodeSteps = 0;
        this.updateCount = 0;
        this.epsilon = this.config.epsilonStart;

        // UCB exploration tracking
        this.stateActionCounts = new Map(); // Track state-action visit counts for UCB
        this.stateVisits = new Map(); // Track state visit counts

        // Logging
        this.lossHistory = [];
        this.tdErrorHistory = [];
    }

    /**
     * Select action using various exploration strategies
     */
    act(state, training = true) {
        this.totalSteps++;

        // Get Q-values
        const qValues = this.qNetwork.forward(state);

        // Handle NaN values
        if (qValues.some(isNaN)) {
            console.warn('NaN in Q-values, returning random action');
            return this.actionSpace.sample();
        }

        if (!training) {
            // Greedy action selection during evaluation
            return qValues.indexOf(Math.max(...qValues));
        }

        // Select exploration strategy
        switch (this.config.explorationMode) {
            case 'epsilon':
                return this.epsilonGreedyAction(qValues);

            case 'ucb':
                return this.ucbAction(state, qValues);

            case 'boltzmann':
                return this.boltzmannAction(qValues);

            case 'noisy':
                // Noisy nets add parameter noise, just use greedy here
                return qValues.indexOf(Math.max(...qValues));

            default:
                return this.epsilonGreedyAction(qValues);
        }
    }

    /**
     * Epsilon-greedy action selection
     */
    epsilonGreedyAction(qValues) {
        this.epsilon = this.getEpsilon();

        if (Math.random() < this.epsilon) {
            return this.actionSpace.sample();
        }

        return qValues.indexOf(Math.max(...qValues));
    }

    /**
     * Upper Confidence Bound (UCB) action selection
     */
    ucbAction(state, qValues) {
        // Convert state to string key for tracking
        const stateKey = state.join(',');

        // Initialize visit counts if first visit
        if (!this.stateVisits.has(stateKey)) {
            this.stateVisits.set(stateKey, 0);
            for (let a = 0; a < qValues.length; a++) {
                const key = `${stateKey}_${a}`;
                this.stateActionCounts.set(key, 0);
            }
        }

        const stateVisitCount = this.stateVisits.get(stateKey);

        // If state rarely visited, explore randomly
        if (stateVisitCount < qValues.length) {
            return this.actionSpace.sample();
        }

        // Calculate UCB values
        const ucbValues = qValues.map((q, a) => {
            const key = `${stateKey}_${a}`;
            const actionCount = this.stateActionCounts.get(key) || 0;

            if (actionCount === 0) {
                return Infinity; // Prioritize unvisited actions
            }

            // UCB formula: Q(s,a) + c * sqrt(ln(N(s)) / N(s,a))
            const exploration = this.config.ucbC * Math.sqrt(
                Math.log(stateVisitCount) / actionCount
            );

            return q + exploration;
        });

        // Select action with highest UCB value
        const action = ucbValues.indexOf(Math.max(...ucbValues));

        // Update counts
        this.stateVisits.set(stateKey, stateVisitCount + 1);
        const actionKey = `${stateKey}_${action}`;
        this.stateActionCounts.set(actionKey,
            (this.stateActionCounts.get(actionKey) || 0) + 1);

        return action;
    }

    /**
     * Boltzmann (softmax) exploration
     */
    boltzmannAction(qValues) {
        // Get current temperature with annealing
        const progress = Math.min(1.0, this.totalSteps / this.config.epsilonDecaySteps);
        const temperature = this.config.boltzmannTemp +
            (this.config.boltzmannTempEnd - this.config.boltzmannTemp) * progress;

        // Compute softmax probabilities
        const expValues = qValues.map(q => Math.exp(q / temperature));
        const sumExp = expValues.reduce((a, b) => a + b, 0);
        const probs = expValues.map(e => e / sumExp);

        // Sample from distribution
        const rand = Math.random();
        let cumSum = 0;
        for (let i = 0; i < probs.length; i++) {
            cumSum += probs[i];
            if (rand < cumSum) {
                return i;
            }
        }

        return probs.length - 1;
    }

    /**
     * Store transition in replay buffer (handles n-step returns)
     */
    remember(state, action, reward, nextState, done) {
        // Add to n-step buffer
        this.nStepBuffer.push({
            state: state,
            action: action,
            reward: reward,
            nextState: nextState,
            done: done
        });

        // If buffer is full or episode ended, compute n-step return and store
        if (this.nStepBuffer.length >= this.config.nSteps || done) {
            const transition = this.computeNStepTransition();
            if (transition) {
                // Calculate initial TD error for prioritized replay
                let tdError = null;
                if (this.config.usePrioritizedReplay) {
                    tdError = this.computeTDError(
                        transition.state,
                        transition.action,
                        transition.reward,
                        transition.nextState,
                        transition.done
                    );
                }

                this.replayBuffer.push(
                    transition.state,
                    transition.action,
                    transition.reward,
                    transition.nextState,
                    transition.done,
                    tdError
                );
            }

            // Clear buffer on episode end
            if (done) {
                this.nStepBuffer = [];
            }
        }

        this.episodeSteps++;
        if (done) {
            this.episodeSteps = 0;
        }
    }

    /**
     * Compute n-step return from buffer
     */
    computeNStepTransition() {
        if (this.nStepBuffer.length === 0) return null;

        const first = this.nStepBuffer[0];
        const last = this.nStepBuffer[this.nStepBuffer.length - 1];

        // Compute n-step discounted reward
        let reward = 0;
        let discount = 1.0;
        for (let i = 0; i < this.nStepBuffer.length; i++) {
            reward += discount * this.nStepBuffer[i].reward;
            discount *= this.config.gamma;
        }

        // Remove first transition from buffer
        this.nStepBuffer.shift();

        return {
            state: first.state,
            action: first.action,
            reward: reward,
            nextState: last.done ? null : last.nextState,
            done: last.done
        };
    }

    /**
     * Perform one training step
     */
    train() {
        // Only train if we have enough samples and it's time to update
        if (this.replayBuffer.size() < this.config.minBufferSize) {
            return null;
        }

        if (this.totalSteps % this.config.updateFreq !== 0) {
            return null;
        }

        let totalLoss = 0;

        // Perform multiple gradient steps
        for (let g = 0; g < this.config.gradientsPerStep; g++) {
            // Sample batch from replay buffer
            let batch, indices, weights;

            if (this.config.usePrioritizedReplay) {
                const sample = this.replayBuffer.sample(this.config.batchSize);
                if (!sample) return null;
                batch = sample.batch;
                indices = sample.indices;
                weights = sample.weights;
            } else {
                batch = this.replayBuffer.sample(this.config.batchSize);
                if (!batch) return null;
                weights = new Array(batch.length).fill(1.0); // Uniform weights
            }

            // Compute loss and gradients for batch
            const { loss, tdErrors } = this.computeBatchLoss(batch, weights);
            totalLoss += loss;

            // Update priorities if using prioritized replay
            if (this.config.usePrioritizedReplay && indices) {
                this.replayBuffer.updatePriorities(indices, tdErrors);
            }

            // Update network parameters
            this.optimizer.step();
        }

        this.updateCount++;

        // Update target network
        if (this.updateCount % this.config.targetUpdateFreq === 0) {
            this.updateTargetNetwork(this.config.tau);
        }

        // Store statistics
        const avgLoss = totalLoss / this.config.gradientsPerStep;
        this.lossHistory.push(avgLoss);

        return avgLoss;
    }

    /**
     * Compute loss for a batch of transitions
     */
    computeBatchLoss(batch, weights) {
        const tdErrors = [];
        let totalLoss = 0;

        // Reset gradients
        this.qNetwork.zeroGrad();

        for (let i = 0; i < batch.length; i++) {
            const { state, action, reward, nextState, done } = batch[i];

            // Compute TD error
            const tdError = this.computeTDError(state, action, reward, nextState, done);
            tdErrors.push(tdError);

            // Weighted loss (for importance sampling)
            const loss = weights[i] * Math.pow(tdError, 2);
            totalLoss += loss;

            // Compute gradients
            const qValues = this.qNetwork.forward(state);
            const grad = new Array(qValues.length).fill(0);
            grad[action] = -2 * weights[i] * tdError / batch.length;

            // Backward pass
            this.qNetwork.backward(grad);
        }

        return { loss: totalLoss / batch.length, tdErrors: tdErrors };
    }

    /**
     * Compute TD error for a single transition
     */
    computeTDError(state, action, reward, nextState, done) {
        // Current Q-value
        const qValues = this.qNetwork.forward(state);
        const qValue = qValues[action];

        // Target Q-value
        let targetValue = reward;

        if (!done && nextState) {
            if (this.config.useDoubleDQN) {
                // Double DQN: use online network to select action, target network to evaluate
                const nextQValues = this.qNetwork.forward(nextState);
                const nextAction = nextQValues.indexOf(Math.max(...nextQValues));
                const targetQValues = this.targetNetwork.forward(nextState);
                targetValue += this.config.gamma * targetQValues[nextAction];
            } else {
                // Standard DQN
                const targetQValues = this.targetNetwork.forward(nextState);
                targetValue += this.config.gamma * Math.max(...targetQValues);
            }
        }

        return targetValue - qValue;
    }

    /**
     * Update target network weights
     */
    updateTargetNetwork(tau = null) {
        if (tau === null) tau = this.config.tau;

        // Get parameters from both networks
        const qParams = this.qNetwork.getParameters();
        const targetParams = this.targetNetwork.getParameters();

        // Soft update: target = tau * q + (1-tau) * target
        const updatedParams = {};
        for (const key in qParams) {
            if (tau >= 1.0) {
                // Hard update
                updatedParams[key] = qParams[key].slice();
            } else {
                // Soft update
                updatedParams[key] = targetParams[key].map((v, i) =>
                    tau * qParams[key][i] + (1 - tau) * v
                );
            }
        }

        this.targetNetwork.setParameters(updatedParams);
    }

    /**
     * Get current epsilon value
     */
    getEpsilon() {
        const decaySteps = this.config.epsilonDecaySteps;
        const progress = Math.min(1.0, this.totalSteps / decaySteps);

        // Linear decay
        return this.config.epsilonStart +
               (this.config.epsilonEnd - this.config.epsilonStart) * progress;
    }

    /**
     * Get training statistics
     */
    getStats() {
        return {
            totalSteps: this.totalSteps,
            bufferSize: this.replayBuffer.size(),
            epsilon: this.epsilon,
            updateCount: this.updateCount,
            avgLoss: this.lossHistory.length > 0 ?
                this.lossHistory.slice(-10).reduce((a, b) => a + b, 0) /
                Math.min(10, this.lossHistory.length) : 0
        };
    }

    /**
     * Save agent state
     */
    save() {
        return {
            qNetwork: this.qNetwork.getParameters(),
            targetNetwork: this.targetNetwork.getParameters(),
            totalSteps: this.totalSteps,
            updateCount: this.updateCount
        };
    }

    /**
     * Load agent state
     */
    load(state) {
        this.qNetwork.setParameters(state.qNetwork);
        this.targetNetwork.setParameters(state.targetNetwork);
        this.totalSteps = state.totalSteps;
        this.updateCount = state.updateCount;
    }
}

/**
 * Adam optimizer for neural network training
 */
class AdamOptimizer {
    constructor(learningRate = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8) {
        this.lr = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.t = 0; // Timestep
        this.m = {}; // First moment estimates
        this.v = {}; // Second moment estimates
    }

    step() {
        this.t++;
        // Implementation would go here - integrating with network gradients
    }
}