class SnakeAgent {
    constructor(gridSize) {
        this.gridSize = gridSize;
        this.inputSize = 15; // 11 existing + 4 for direction
        this.hiddenSize = 128;
        this.outputSize = 4; // 4 possible actions (up, down, left, right)
        this.model = new SnakeModel(this.inputSize, this.hiddenSize, this.outputSize);
        this.steps = 0;
        this.epsilon = 1.0;
        this.epsilonMin = 0.01;
        this.epsilonDecay = 0.995;
        this.gamma = 0.95;
        this.replayBufferSize = 10000; // Define replayBufferSize
        this.replayBuffer = new PrioritizedReplayBuffer(this.replayBufferSize);
        this.alpha = 0.6; // Priority exponent
        this.beta = 0.4; // Initial importance-sampling weight
        this.betaIncrement = 0.001; // Beta increment per replay
        this.batchSize = 32;
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

    remember(state, action, reward, nextState, done) {
        const priority = 1.0; // Start with max priority for new experiences
        this.replayBuffer.add(priority, [state, action, reward, nextState, done]);
    }

    async replay() {
        if (this.replayBuffer.size() < this.batchSize) return;

        this.beta = Math.min(1.0, this.beta + this.betaIncrement);
        const [batch, indices, importanceWeights] = this.replayBuffer.sample(this.batchSize, this.beta);

        const states = batch.map(exp => exp[0]);
        const actions = batch.map(exp => exp[1]);
        const rewards = batch.map(exp => exp[2]);
        const nextStates = batch.map(exp => exp[3]);
        const dones = batch.map(exp => exp[4]);

        let updatedQs = [];
        let errors = [];
        let normalizedWeights = [];

        // Use tf.tidy to manage tensor memory for synchronous operations
        tf.tidy(() => {
            const currentQs = this.model.predict(states);
            const nextQs = this.model.predict(nextStates);

            const currentQsData = currentQs.arraySync();
            const nextQsData = nextQs.arraySync();

            updatedQs = currentQsData.map(q => q.slice()); // Deep copy
            errors = [];

            for (let i = 0; i < this.batchSize; i++) {
                let newQ = rewards[i];
                if (!dones[i]) {
                    const nextQ = Math.max(...nextQsData[i]);
                    newQ += this.gamma * nextQ;
                }
                const error = Math.abs(newQ - currentQsData[i][actions[i]]);
                errors.push(error);
                updatedQs[i][actions[i]] = newQ;
            }

            // Update priorities in the replay buffer
            for (let i = 0; i < this.batchSize; i++) {
                const priority = Math.pow(errors[i] + 0.01, this.alpha);
                this.replayBuffer.updatePriority(indices[i], priority);
            }

            // Normalize importance weights
            const maxWeight = Math.max(...importanceWeights);
            normalizedWeights = importanceWeights.map(w => w / maxWeight);
        });

        // Train the model outside of tf.tidy to handle asynchronous operations
        await this.model.train(states, updatedQs, normalizedWeights);
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

class PrioritizedReplayBuffer {
    constructor(maxSize) {
        this.maxSize = maxSize;
        this.buffer = [];
        this.priorities = [];
        this.currentSize = 0; // Rename 'size' to 'currentSize'
    }

    add(priority, experience) {
        if (this.currentSize < this.maxSize) { // Use 'currentSize'
            this.buffer.push(experience);
            this.priorities.push(priority);
            this.currentSize++; // Increment 'currentSize'
        } else {
            const index = this.currentSize % this.maxSize; // Calculate index
            this.buffer[index] = experience;
            this.priorities[index] = priority;
            // Cap the currentSize to maxSize to prevent it from exceeding
            this.currentSize = this.maxSize;
        }
    }

    sample(batchSize, beta) {
        const bufferSize = this.currentSize; // Ensure we use the capped currentSize
        const total = this.priorities.slice(0, bufferSize).reduce((a, b) => a + b, 0);
        const probabilities = this.priorities.slice(0, bufferSize).map(p => p / total);
        const indices = [];
        const batch = [];
        const importanceWeights = [];

        for (let i = 0; i < batchSize; i++) {
            const r = Math.random();
            let cumSum = 0;
            for (let j = 0; j < bufferSize; j++) { // Use 'bufferSize'
                cumSum += probabilities[j];
                if (r < cumSum) {
                    indices.push(j);
                    batch.push(this.buffer[j]);
                    break;
                }
            }
        }

        const maxWeight = Math.max(...probabilities.slice(0, bufferSize)) ** -beta;
        for (const index of indices) {
            const weight = Math.pow(probabilities[index] * bufferSize, -beta); // Use 'bufferSize'
            importanceWeights.push(weight / maxWeight);
        }

        return [batch, indices, importanceWeights];
    }

    updatePriority(index, priority) {
        this.priorities[index] = priority;
    }

    size() {
        return this.currentSize; // Return 'currentSize'
    }
}