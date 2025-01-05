class StreamQ {
    constructor(config = {}) {
        this.numActions = config.numActions || 2;
        this.gamma = config.gamma || 0.99;
        this.epsilonStart = config.epsilonStart || 1.0;
        this.epsilonTarget = config.epsilonTarget || 0.01;
        this.explorationFraction = config.explorationFraction || 0.05;
        this.totalSteps = config.totalSteps || 100000;
        this.timeStep = 0;
        this.epsilon = this.epsilonStart;

        // Get input size from environment by doing a reset
        const initialState = config.env.reset();
        const inputSize = initialState.length;

        // Initialize network and optimizer
        this.network = new StreamingNetwork(
            inputSize,
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

    async update(state, action, reward, nextState, done, isNonGreedy) {
        const stateTensor = tf.tensor2d([state], [1, state.length]);
        const nextStateTensor = tf.tensor2d([nextState], [1, nextState.length]);
        
        try {
            // Compute target Q-value outside of gradient tape
            const nextQValues = this.network.model.predict(nextStateTensor);
            const maxNextQ = nextQValues.max(1);  // Max over actions dimension
            const doneMask = done ? 0 : 1;
            const tdTarget = tf.scalar(reward).add(maxNextQ.mul(tf.scalar(this.gamma * doneMask)));
            
            // Compute gradients with respect to -q_sa
            const {value: qValueAtAction, grads} = tf.variableGrads(() => {
                const qValues = this.network.model.predict(stateTensor);
                const actionMask = tf.oneHot([action], this.numActions);
                const qsa = tf.sum(tf.mul(qValues, actionMask));
                return tf.neg(qsa);  // -q_sa like in Python
            });

            // Compute TD error
            const delta = tdTarget.sub(qValueAtAction);  // Remove unnecessary negation
            const deltaValue = await delta.data();

            await this.optimizer.step(
                deltaValue[0],
                Object.values(grads),
                done || isNonGreedy
            );

            // Clean up tensors
            stateTensor.dispose();
            nextStateTensor.dispose();
            nextQValues.dispose();
            maxNextQ.dispose();
            tdTarget.dispose();
            delta.dispose();
            Object.values(grads).forEach(g => g && g.dispose());
            
            return deltaValue[0];
        } catch (error) {
            console.error('Error in update:', error);
            stateTensor.dispose();
            nextStateTensor.dispose();
            throw error;
        }
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