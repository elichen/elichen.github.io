class SnakeAgent {
    constructor(gridSize) {
        this.gridSize = gridSize;
        this.inputSize = 11; // 8 directions + 3 (food relative position)
        this.hiddenSize = 128;
        this.outputSize = 4; // 4 possible actions (up, down, left, right)
        this.model = new SnakeModel(this.inputSize, this.hiddenSize, this.outputSize);
        this.targetModel = new SnakeModel(this.inputSize, this.hiddenSize, this.outputSize);
        this.updateTargetModel();
        this.updateFrequency = 1000; // Update target network every 1000 steps
        this.steps = 0;
        this.epsilon = 1.0;
        this.epsilonMin = 0.01;
        this.epsilonDecay = 0.995;
        this.gamma = 0.95;
        this.replayBufferSize = 10000; // Add this line to define replayBufferSize
        this.replayBuffer = new PrioritizedReplayBuffer(this.replayBufferSize);
        this.alpha = 0.6; // Priority exponent
        this.beta = 0.4; // Initial importance-sampling weight
        this.betaIncrement = 0.001; // Beta increment per replay
        this.batchSize = 32;
        this.testingMode = false;
        this.episodeCount = 0; // Add this line to keep track of episodes
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

        return state;
    }

    setTestingMode(isTestingMode) {
        this.testingMode = isTestingMode;
    }

    getAction(state) {
        if (this.testingMode || Math.random() > this.epsilon) {
            // Exploit: Use the model to predict the best action
            const prediction = this.model.predict(state);
            return tf.tidy(() => tf.argMax(prediction, 1).dataSync()[0]);
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

        const currentQs = this.model.predict(states);
        const nextQs = this.model.predict(nextStates);
        const nextTargetQs = this.targetModel.predict(nextStates);

        const updatedQs = currentQs.arraySync();
        const errors = [];

        for (let i = 0; i < this.batchSize; i++) {
            let newQ = rewards[i];
            if (!dones[i]) {
                const nextQ = nextTargetQs.arraySync()[i];
                const bestAction = tf.argMax(nextQs.arraySync()[i]).dataSync()[0];
                newQ += this.gamma * nextQ[bestAction];
            }
            const error = Math.abs(newQ - updatedQs[i][actions[i]]);
            errors.push(error);
            updatedQs[i][actions[i]] = newQ;
        }

        // Update priorities in the replay buffer
        for (let i = 0; i < this.batchSize; i++) {
            const priority = (errors[i] + 0.01) ** this.alpha;
            this.replayBuffer.updatePriority(indices[i], priority);
        }

        // Normalize importance weights
        const maxWeight = Math.max(...importanceWeights);
        const normalizedWeights = importanceWeights.map(w => w / maxWeight);

        await this.model.train(states, updatedQs, normalizedWeights);

        // Update target network
        this.steps++;
        if (this.steps % this.updateFrequency === 0) {
            this.updateTargetModel();
        }
    }

    updateTargetModel() {
        this.targetModel.model.setWeights(this.model.model.getWeights());
    }

    // Add this new method to increment the episode count
    incrementEpisodeCount() {
        this.episodeCount++;
        
        // Only decrease epsilon after 20 episodes
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
            this.currentSize++; // Use 'currentSize'
        } else {
            const index = this.currentSize % this.maxSize; // Use 'currentSize'
            this.buffer[index] = experience;
            this.priorities[index] = priority;
        }
    }

    sample(batchSize, beta) {
        const total = this.priorities.reduce((a, b) => a + b, 0);
        const probabilities = this.priorities.map(p => p / total);
        const indices = [];
        const batch = [];
        const importanceWeights = [];

        for (let i = 0; i < batchSize; i++) {
            const r = Math.random();
            let cumSum = 0;
            for (let j = 0; j < this.currentSize; j++) { // Use 'currentSize'
                cumSum += probabilities[j];
                if (r < cumSum) {
                    indices.push(j);
                    batch.push(this.buffer[j]);
                    break;
                }
            }
        }

        const maxWeight = Math.max(...probabilities) ** -beta;
        for (const index of indices) {
            const weight = (probabilities[index] * this.currentSize) ** -beta; // Use 'currentSize'
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