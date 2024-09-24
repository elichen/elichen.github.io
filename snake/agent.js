class SnakeAgent {
    constructor(gridSize) {
        this.gridSize = gridSize;
        this.inputSize = 15; // 11 existing + 4 for direction
        this.hiddenSize = 128;
        this.outputSize = 4; // 4 possible actions (up, down, left, right)
        this.model = new SnakeModel(this.inputSize, this.hiddenSize, this.outputSize);
        this.steps = 0;
        this.epsilon = 1.0;
        this.epsilonMin = 0.0;
        this.epsilonDecay = 0.995;
        this.gamma = 0.95;
        this.replayBufferSize = 100000; // Define replayBufferSize
        this.replayBuffer = new ReplayBuffer(this.replayBufferSize);
        this.batchSize = 1000;
        this.testingMode = false;
        this.episodeCount = 0; // Keep track of episodes
    }

    getState(game) {
        const head = game.snake[0];
        const food = game.food;

        // Check 8 directions
        const directions = [
            { x: 0, y: -1 }, { x: 1, y: -1 }, { x: 1, y: 0 }, { x: 1, y: 1 },
            { x: 0, y: 1 }, { x: -1, y: 1 }, { x: -1, y: 0 }, { x: -1, y: -1 }
        ];

        const state = directions.map(dir => {
            let x = head.x;
            let y = head.y;
            let distance = 0;
            while (true) {
                x += dir.x;
                y += dir.y;
                distance++;
                if (x < 0 || x >= this.gridSize || y < 0 || y >= this.gridSize) {
                    return 1 / distance; // Wall
                }
                if (game.snake.some(segment => segment.x === x && segment.y === y)) {
                    return 1 / distance; // Snake body
                }
                if (x === food.x && y === food.y) {
                    return -1; // Food
                }
            }
        });

        // Add relative food position
        state.push((food.x - head.x) / this.gridSize);
        state.push((food.y - head.y) / this.gridSize);
        state.push(game.snake.length / (this.gridSize * this.gridSize));

        // **Add current direction as one-hot encoding**
        const direction = game.direction; // {x, y}
        let directionOneHot = [0, 0, 0, 0]; // Up, Right, Down, Left
        if (direction.x === 0 && direction.y === -1) directionOneHot[0] = 1; // Up
        else if (direction.x === 1 && direction.y === 0) directionOneHot[1] = 1; // Right
        else if (direction.x === 0 && direction.y === 1) directionOneHot[2] = 1; // Down
        else if (direction.x === -1 && direction.y === 0) directionOneHot[3] = 1; // Left

        state.push(...directionOneHot); // Add the one-hot direction to the state

        return state;
    }

    setTestingMode(isTestingMode) {
        this.testingMode = isTestingMode;
    }

    getAction(state) {
        if (this.testingMode || Math.random() > this.epsilon) {
            // Exploit: Use the model to predict the best action
            return tf.tidy(() => {
                const prediction = this.model.predict([state]);
                const action = tf.argMax(prediction, 1).dataSync()[0];
                return action;
            });
        } else {
            // Explore: Choose a random action
            return Math.floor(Math.random() * 4);
        }
    }

    /**
     * Add experience to replay buffer without triggering training.
     * @param {Array} state 
     * @param {number} action 
     * @param {number} reward 
     * @param {Array} nextState 
     * @param {boolean} done 
     */
    remember(state, action, reward, nextState, done) {
        this.replayBuffer.add([state, action, reward, nextState, done]);
    }

    /**
     * Train the model immediately on the latest experience (short-term memory).
     * @param {Array} state 
     * @param {number} action 
     * @param {number} reward 
     * @param {Array} nextState 
     * @param {boolean} done 
     */
    async trainShortTerm(state, action, reward, nextState, done) {
        let target = reward;
        if (!done) {
            const predictions = this.model.predict([nextState]); // Correct input shape
            const maxQ = tf.max(predictions, 1).dataSync()[0];
            target += this.gamma * maxQ;
        }

        const qValues = this.getQValues(state); // Array of Q-values [q1, q2, q3, q4]
        qValues[action] = target;

        await this.model.train([state], [qValues]);
    }

    /**
     * Retrieve current Q-values for a given state.
     * @param {Array} state 
     * @returns {Array} Q-values
     */
    getQValues(state) {
        const prediction = this.model.predict([state]);
        return prediction.arraySync()[0];
    }

    /**
     * Replay experiences from the replay buffer for long-term training.
     */
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

            const currentQsData = currentQs.arraySync();
            const nextQsData = nextQs.arraySync();

            updatedQs = currentQsData.map(q => q.slice()); // Deep copy

            for (let i = 0; i < this.batchSize; i++) {
                let newQ = rewards[i];
                if (!dones[i]) {
                    const nextQ = Math.max(...nextQsData[i]);
                    newQ += this.gamma * nextQ;
                }
                updatedQs[i][actions[i]] = newQ;
            }
        });

        // Train the model outside of tf.tidy to handle asynchronous operations
        await this.model.train(states, updatedQs);
    }

    // Method to increment the episode count
    incrementEpisodeCount() {
        this.episodeCount++;

        // Decrease epsilon after each episode
        if (this.epsilon > this.epsilonMin) {
            this.epsilon *= this.epsilonDecay;
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
            this.buffer.shift(); // Remove the oldest experience
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