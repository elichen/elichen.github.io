class SnakeAgent {
    constructor(gridSize) {
        this.gridSize = gridSize;
        this.inputSize = 24; // 24 features (PPO state representation)
        this.hiddenSize = 64;
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
            // Set epsilon to 0 for pre-trained model (no exploration)
            this.epsilon = 0.0;
            this.epsilonMin = 0.0;
        }
        return this.isPreTrained;
    }

    getState(game) {
        const head = game.snake[0];
        const food = game.food;
        const features = [];

        // 1. Direction one-hot (4 features): [up, right, down, left]
        const directionOnehot = [0, 0, 0, 0];
        if (game.direction.y === -1) directionOnehot[0] = 1; // Up
        else if (game.direction.x === 1) directionOnehot[1] = 1; // Right
        else if (game.direction.y === 1) directionOnehot[2] = 1; // Down
        else if (game.direction.x === -1) directionOnehot[3] = 1; // Left
        features.push(...directionOnehot);

        // 2. Food direction (2 features): normalized dx, dy
        const foodDx = (food.x - head.x) / this.gridSize;
        const foodDy = (food.y - head.y) / this.gridSize;
        features.push(foodDx, foodDy);

        // 3. Danger detection in 4 directions (4 features): [up, right, down, left]
        const checkDanger = (x, y) => {
            return x < 0 || x >= this.gridSize || y < 0 || y >= this.gridSize ||
                   game.snake.some(segment => segment.x === x && segment.y === y);
        };
        const dangers = [
            checkDanger(head.x, head.y - 1) ? 1 : 0, // Up
            checkDanger(head.x + 1, head.y) ? 1 : 0, // Right
            checkDanger(head.x, head.y + 1) ? 1 : 0, // Down
            checkDanger(head.x - 1, head.y) ? 1 : 0  // Left
        ];
        features.push(...dangers);

        // 4. Distance to walls (4 features): [top, right, bottom, left]
        const wallDistances = [
            head.y / this.gridSize,
            (this.gridSize - 1 - head.x) / this.gridSize,
            (this.gridSize - 1 - head.y) / this.gridSize,
            head.x / this.gridSize
        ];
        features.push(...wallDistances);

        // 5. Snake length (1 feature): normalized
        const maxLength = this.gridSize * this.gridSize;
        features.push(game.snake.length / maxLength);

        // 6. Grid fill ratio (1 feature)
        features.push(game.score / (this.gridSize * this.gridSize));

        // 7. Food distance (1 feature): manhattan distance normalized
        const manhattanDist = Math.abs(head.x - food.x) + Math.abs(head.y - food.y);
        features.push(manhattanDist / (2 * this.gridSize));

        // 8. Body pattern features (3 features)
        if (game.snake.length >= 3) {
            // Simplified body patterns for web
            const straightness = 0.5; // Placeholder
            const turns = 0.5; // Placeholder
            const loopiness = 0.0; // Placeholder
            features.push(straightness, turns, loopiness);
        } else {
            features.push(0, 0, 0);
        }

        // 9. Connectivity features (4 features): accessible cells in each direction
        const connectivity = this.calculateConnectivity(game, head);
        features.push(...connectivity);

        return features;
    }

    calculateConnectivity(game, head) {
        // Calculate accessible cells in each direction using BFS (depth-limited to 5)
        // Matches Python: snake_gym_env.py:372-411
        const connectivity = [];
        const maxDepth = 5;

        // Check each direction: up, right, down, left
        const directions = [
            [0, -1], // Up
            [1, 0],  // Right
            [0, 1],  // Down
            [-1, 0]  // Left
        ];

        for (const [dx, dy] of directions) {
            const visited = new Set();
            const queue = [{ x: head.x + dx, y: head.y + dy, depth: 0 }];
            let accessible = 0;

            while (queue.length > 0) {
                const current = queue.shift();

                if (current.depth > maxDepth) continue;

                const key = `${current.x},${current.y}`;
                if (visited.has(key)) continue;

                // Check boundaries and obstacles
                if (current.x < 0 || current.x >= this.gridSize ||
                    current.y < 0 || current.y >= this.gridSize) {
                    continue;
                }

                if (game.snake.some(seg => seg.x === current.x && seg.y === current.y)) {
                    continue;
                }

                visited.add(key);
                accessible++;

                // Add neighbors with increased depth
                const newDepth = current.depth + 1;
                queue.push({ x: current.x, y: current.y + 1, depth: newDepth });
                queue.push({ x: current.x + 1, y: current.y, depth: newDepth });
                queue.push({ x: current.x, y: current.y - 1, depth: newDepth });
                queue.push({ x: current.x - 1, y: current.y, depth: newDepth });
            }

            // Normalize by max possible (matching Python exactly)
            const maxAccessible = Math.min(maxDepth * 4, this.gridSize * this.gridSize);
            connectivity.push(accessible / maxAccessible);
        }

        return connectivity;
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