class StreamQ {
    constructor(config = {}) {
        this.numActions = config.numActions || 2;
        this.gamma = config.gamma || 0.99;
        this.epsilonStart = config.epsilonStart;
        this.epsilonTarget = config.epsilonTarget;
        this.decaySteps = config.totalSteps;
        this.timeStep = 0;
        this.epsilon = this.epsilonStart;

        // Validate required user-facing configs
        if (!this.epsilonStart) {
            throw new Error('epsilonStart must be provided in config');
        }
        if (!this.epsilonTarget) {
            throw new Error('epsilonTarget must be provided in config');
        }
        if (!this.decaySteps) {
            throw new Error('totalSteps (decay steps) must be provided in config');
        }

        // Get input size from environment by doing a reset
        const initialState = config.env.reset();
        const inputSize = initialState.length;

        // Initialize network and optimizer
        this.network = new StreamingNetwork(
            inputSize,
            config.hiddenSize || 32,
            this.numActions
        );

        this.optimizer = new StreamingOptimizer(
            this.network.getTrainableVariables(),
            config.learningRate || 3e-4,
            this.gamma,
            config.lambda || 0.8,
            config.beta || 0.99995,
            config.warmupSteps ?? 1000
        );
    }

    linearSchedule(t) {
        const slope = (this.epsilonTarget - this.epsilonStart) / this.decaySteps;
        return Math.max(slope * t + this.epsilonStart, this.epsilonTarget);
    }

    async sampleAction(state) {
        this.timeStep++;
        this.epsilon = this.linearSchedule(this.timeStep);

        const qValues = await this.network.predict(state);
        const qArray = await qValues.array();
        qValues.dispose();

        const greedyAction = qArray[0].indexOf(Math.max(...qArray[0]));

        if (Math.random() < this.epsilon) {
            // When exploring, randomly select an action
            const randomAction = Math.floor(Math.random() * this.numActions);
            // If random action matches greedy action, it's not considered non-greedy
            return {
                action: randomAction,
                isNonGreedy: randomAction !== greedyAction
            };
        } else {
            // When exploiting, use the greedy action
            return {
                action: greedyAction,
                isNonGreedy: false
            };
        }
    }

    async update(state, action, reward, nextState, done, isNonGreedy, truncated = false) {
        const stateTensor = tf.tensor2d([state], [1, state.length]);
        const nextStateTensor = tf.tensor2d([nextState], [1, nextState.length]);

        try {
            // 1. Compute TD target. A step-limit cutoff still bootstraps: the time
            // input never reaches the limit, so the network can't see it coming
            const nextQValues = this.network.model.predict(nextStateTensor);
            const maxNextQ = nextQValues.max(1);
            const doneMask = done && !truncated ? 0 : 1;
            const tdTarget = tf.tidy(() => tf.scalar(reward).add(maxNextQ.mul(tf.scalar(this.gamma * doneMask))));
            
            // 2. Compute current Q-value and gradients
            const {value: qsa, grads} = tf.variableGrads(() => {
                const qValues = this.network.model.predict(stateTensor);
                const actionMask = tf.oneHot([action], this.numActions);
                const selectedQ = tf.sum(tf.mul(qValues, actionMask));
                return selectedQ.neg();
            });

            // 3. Compute TD error: δ = R + γ max_a q̂(S', a) - q̂(S, A)
            const selectedQ = qsa.neg();
            const tdError = tdTarget.sub(selectedQ);
            const tdErrorValue = await tdError.data();

            // 4. Update parameters
            await this.optimizer.step(
                tdErrorValue[0],
                Object.values(grads),
                done || isNonGreedy
            );

            // Cleanup tensors
            tf.dispose([
                stateTensor,
                nextStateTensor,
                nextQValues,
                maxNextQ,
                tdTarget,
                qsa,
                selectedQ,
                tdError,
                ...Object.values(grads)
            ]);
        } catch (error) {
            console.error('Error in update:', error);
            throw error;
        }
    }
} 