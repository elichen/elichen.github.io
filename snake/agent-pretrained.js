class SnakeAgent {
    constructor(gridSize) {
        this.gridSize = gridSize;
        this.inputSize = 12; // 12 binary inputs for one-hot encoded states
        this.hiddenSize = 256;
        this.outputSize = 4; // 4 possible actions (up, down, left, right)
        this.model = new SnakeModel(this.inputSize, this.hiddenSize, this.outputSize);
        this.steps = 0;
        this.epsilon = 0.0; // Start with no exploration for pre-trained model
        this.epsilonMin = 0.0;
        this.epsilonDecay = 0.995;
        this.gamma = 0.95;
        this.replayBufferSize = 100000;
        this.replayBuffer = new ReplayBuffer(this.replayBufferSize);
        this.batchSize = 1000;
        this.testingMode = true; // Start in testing mode
        this.episodeCount = 0;
        this.targetUpdateFrequency = 100;
        this.isPreTrained = false;
    }

    async loadPreTrainedModel() {
        // Load the pre-trained weights
        this.isPreTrained = await this.model.loadPreTrainedWeights();
        if (this.isPreTrained) {
            console.log('Pre-trained model loaded successfully!');
            // Set epsilon to 0 for pre-trained model (no exploration)
            this.epsilon = 0.0;
            this.epsilonMin = 0.0;
        }
        return this.isPreTrained;
    }

    getState(game) {
        const head = game.snake[0];
        const food = game.food;

        // One-hot encode snake direction (NSEW)
        const snakeDirection = [0, 0, 0, 0];
        if (game.direction.y === -1) snakeDirection[0] = 1; // North
        else if (game.direction.y === 1) snakeDirection[1] = 1; // South
        else if (game.direction.x === 1) snakeDirection[2] = 1; // East
        else if (game.direction.x === -1) snakeDirection[3] = 1; // West

        // One-hot encode food direction (NSEW)
        const foodDirection = [0, 0, 0, 0];
        if (food.y < head.y) foodDirection[0] = 1; // North
        else if (food.y > head.y) foodDirection[1] = 1; // South
        if (food.x > head.x) foodDirection[2] = 1; // East
        else if (food.x < head.x) foodDirection[3] = 1; // West

        // One-hot encode immediate danger (NSEW)
        const danger = [0, 0, 0, 0];
        const checkDanger = (x, y) => {
            return x < 0 || x >= this.gridSize || y < 0 || y >= this.gridSize ||
                   game.snake.some(segment => segment.x === x && segment.y === y);
        };
        if (checkDanger(head.x, head.y - 1)) danger[0] = 1; // North
        if (checkDanger(head.x, head.y + 1)) danger[1] = 1; // South
        if (checkDanger(head.x + 1, head.y)) danger[2] = 1; // East
        if (checkDanger(head.x - 1, head.y)) danger[3] = 1; // West

        // Combine all one-hot encoded vectors
        return [...snakeDirection, ...foodDirection, ...danger];
    }

    setTestingMode(isTestingMode) {
        this.testingMode = isTestingMode;
        if (isTestingMode && this.isPreTrained) {
            // No exploration in testing mode with pre-trained model
            this.epsilon = 0.0;
        }
    }

    getAction(state) {
        if (!this.testingMode && Math.random() > this.epsilon) {
            // Exploit: Use the model to predict the best action
            return tf.tidy(() => {
                const prediction = this.model.predict([state]);
                const action = tf.argMax(prediction, 1).dataSync()[0];
                return action;
            });
        } else if (!this.testingMode && Math.random() <= this.epsilon) {
            // Explore: Choose a random action
            return Math.floor(Math.random() * 4);
        } else {
            // Testing mode: Always use the model
            return tf.tidy(() => {
                const prediction = this.model.predict([state]);
                const action = tf.argMax(prediction, 1).dataSync()[0];
                return action;
            });
        }
    }

    remember(state, action, reward, nextState, done) {
        this.replayBuffer.add([state, action, reward, nextState, done]);
    }

    async trainShortTerm(state, action, reward, nextState, done) {
        let target = reward;
        if (!done) {
            const predictions = this.model.predict([nextState]);
            const maxQ = tf.max(predictions, 1).dataSync()[0];
            target += this.gamma * maxQ;
        }

        const qValues = this.getQValues(state);
        qValues[action] = target;

        await this.model.train([state], [qValues]);

        // Update the target network periodically
        if (this.steps % this.targetUpdateFrequency === 0) {
            this.model.updateTargetNetwork();
        }
        this.steps++;
    }

    getQValues(state) {
        const prediction = this.model.predict([state]);
        return prediction.arraySync()[0];
    }

    async replay() {
        if (this.replayBuffer.size() < this.batchSize) return;

        const batch = this.replayBuffer.sample(this.batchSize);

        const states = batch.map(exp => exp[0]);
        const actions = batch.map(exp => exp[1]);
        const rewards = batch.map(exp => exp[2]);
        const nextStates = batch.map(exp => exp[3]);
        const dones = batch.map(exp => exp[4]);

        let updatedQs = [];

        // Use tf.tidy to manage tensor memory for synchronous operations
        tf.tidy(() => {
            const currentQs = this.model.predict(states);
            const nextQs = this.model.predict(nextStates);
            const targetQs = this.model.predictTarget(nextStates);

            const currentQsData = currentQs.arraySync();
            const nextQsData = nextQs.arraySync();
            const targetQsData = targetQs.arraySync();

            updatedQs = currentQsData.map(q => q.slice()); // Deep copy

            for (let i = 0; i < this.batchSize; i++) {
                let newQ = rewards[i];
                if (!dones[i]) {
                    const bestAction = nextQsData[i].indexOf(Math.max(...nextQsData[i]));
                    const targetQ = targetQsData[i][bestAction];
                    newQ += this.gamma * targetQ;
                }
                updatedQs[i][actions[i]] = newQ;
            }
        });

        // Train the model outside of tf.tidy to handle asynchronous operations
        await this.model.train(states, updatedQs);
    }

    incrementEpisodeCount() {
        this.episodeCount++;

        // Only decay epsilon if not using pre-trained model or continuing training
        if (!this.isPreTrained || !this.testingMode) {
            if (this.epsilon > this.epsilonMin) {
                this.epsilon *= this.epsilonDecay;
            }
        }
    }
}

class ReplayBuffer {
    constructor(maxSize) {
        this.maxSize = maxSize;
        this.buffer = [];
    }

    add(experience) {
        if (this.buffer.length >= this.maxSize) {
            this.buffer.shift();
        }
        this.buffer.push(experience);
    }

    sample(batchSize) {
        const samples = [];
        for (let i = 0; i < batchSize; i++) {
            const index = Math.floor(Math.random() * this.buffer.length);
            samples.push(this.buffer[index]);
        }
        return samples;
    }

    size() {
        return this.buffer.length;
    }
}