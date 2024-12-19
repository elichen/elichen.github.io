class StreamQ {
    constructor(config = {}) {
        this.numActions = config.numActions || 2;
        this.gamma = config.gamma || 0.99;
        this.epsilonStart = config.epsilonStart || 1.0;
        this.epsilonTarget = config.epsilonTarget || 0.01;
        this.explorationFraction = config.explorationFraction || 0.1;
        this.totalSteps = config.totalSteps || 500000;
        this.timeStep = 0;
        this.epsilon = this.epsilonStart;

        // Initialize network and optimizer
        this.network = new StreamingNetwork(
            config.inputSize || 4,
            config.hiddenSize || 32,
            this.numActions
        );

        this.optimizer = new ObGD(
            this.network.getTrainableVariables(),
            config.learningRate || 1.0,
            this.gamma,
            config.lambda || 0.8,
            config.kappa || 2.0
        );
    }

    linearSchedule(t) {
        const duration = this.explorationFraction * this.totalSteps;
        const slope = (this.epsilonTarget - this.epsilonStart) / duration;
        return Math.max(slope * t + this.epsilonStart, this.epsilonTarget);
    }

    async sampleAction(state) {
        this.timeStep++;
        this.epsilon = this.linearSchedule(this.timeStep);

        const qValues = await this.network.predict(state);
        const qArray = await qValues.array();
        qValues.dispose();

        if (Math.random() < this.epsilon) {
            const greedyAction = qArray[0].indexOf(Math.max(...qArray[0]));
            const randomAction = Math.floor(Math.random() * this.numActions);
            return {
                action: randomAction,
                isNonGreedy: randomAction !== greedyAction
            };
        } else {
            return {
                action: qArray[0].indexOf(Math.max(...qArray[0])),
                isNonGreedy: false
            };
        }
    }

    async update(state, action, reward, nextState, done, isNonGreedy) {
        return tf.tidy(async () => {
            const stateTensor = tf.tensor2d([state], [1, state.length]);
            const nextStateTensor = tf.tensor2d([nextState], [1, nextState.length]);
            
            // Get current Q-values
            const qValues = this.network.model.predict(stateTensor);
            const qValueAtAction = qValues.gather([action]).asScalar();

            // Get max Q-value for next state
            const nextQValues = this.network.model.predict(nextStateTensor);
            const maxNextQ = nextQValues.max();
            
            // Calculate TD target and error
            const doneMask = done ? 0 : 1;
            const tdTarget = reward + this.gamma * maxNextQ.mul(doneMask);
            const delta = tdTarget.sub(qValueAtAction);

            // Get gradients
            const grads = tf.variableGrads(() => qValueAtAction.neg());
            
            // Update parameters
            await this.optimizer.step(
                delta.arraySync(),
                Object.values(grads.grads),
                done || isNonGreedy
            );

            return delta.arraySync();
        });
    }

    async saveAgent() {
        await this.network.saveModel();
    }

    async loadAgent() {
        await this.network.loadModel();
    }

    dispose() {
        this.optimizer.dispose();
    }
} 