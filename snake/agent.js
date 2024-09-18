class SnakeAgent {
    constructor(gridSize) {
        this.gridSize = gridSize;
        this.inputSize = 11; // 8 directions + 3 (food relative position)
        this.hiddenSize = 128;
        this.outputSize = 4; // 4 possible actions (up, down, left, right)
        this.model = new SnakeModel(this.inputSize, this.hiddenSize, this.outputSize);
        this.epsilon = 1.0;
        this.epsilonMin = 0.01;
        this.epsilonDecay = 0.995;
        this.gamma = 0.95;
        this.replayBuffer = [];
        this.batchSize = 32;
        this.replayBufferSize = 10000;
        this.testingMode = false;
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
        this.replayBuffer.push([state, action, reward, nextState, done]);
        if (this.replayBuffer.length > this.replayBufferSize) {
            this.replayBuffer.shift();
        }
    }

    async replay() {
        if (this.replayBuffer.length < this.batchSize) return;

        const batch = this.getRandomBatch();
        const states = batch.map(exp => exp[0]);
        const actions = batch.map(exp => exp[1]);
        const rewards = batch.map(exp => exp[2]);
        const nextStates = batch.map(exp => exp[3]);
        const dones = batch.map(exp => exp[4]);

        const currentQs = this.model.predict(states);
        const nextQs = this.model.predict(nextStates);

        const updatedQs = currentQs.arraySync();

        for (let i = 0; i < this.batchSize; i++) {
            let newQ = rewards[i];
            if (!dones[i]) {
                newQ += this.gamma * Math.max(...nextQs.arraySync()[i]);
            }
            updatedQs[i][actions[i]] = newQ;
        }

        await this.model.train(states, updatedQs);

        if (this.epsilon > this.epsilonMin) {
            this.epsilon *= this.epsilonDecay;
        }
    }

    getRandomBatch() {
        const batch = [];
        for (let i = 0; i < this.batchSize; i++) {
            const index = Math.floor(Math.random() * this.replayBuffer.length);
            batch.push(this.replayBuffer[index]);
        }
        return batch;
    }
}