class PPOWebAgent {
    constructor(gridSize = 20, stackSize = 4) {
        this.gridSize = gridSize;
        this.stackSize = stackSize;
        this.channels = 4;
        this.statsDim = 6;
        this.boardDepth = this.channels * this.stackSize;
        this.statsDepth = this.statsDim * this.stackSize;
        const plane = gridSize * gridSize;
        this.boardStack = new Float32Array(this.boardDepth * plane);
        this.statsStack = new Float32Array(this.statsDepth);
    }

    async load() {
        this.model = await loadGraphPolicy();
        this.boardInputName = this.model.inputs[0].name;
        this.statsInputName = this.model.inputs[1].name;
        this.logitsOutputName = this.model.outputs[0].name;
    }

    bootstrap(game) {
        const obs = this.computeObservation(game);
        for (let i = 0; i < this.stackSize; i++) {
            this.copyIntoStack(i, obs.board, obs.stats);
        }
    }

    copyIntoStack(index, board, stats) {
        const plane = this.gridSize * this.gridSize;
        const boardOffset = index * this.channels * plane;
        this.boardStack.set(board, boardOffset);

        const statsOffset = index * this.statsDim;
        this.statsStack.set(stats, statsOffset);
    }

    updateStack(board, stats) {
        const plane = this.gridSize * this.gridSize;
        const boardSlice = this.channels * plane;
        const statsSlice = this.statsDim;

        // Shift older frames
        this.boardStack.copyWithin(0, boardSlice);
        this.boardStack.set(board, this.boardStack.length - boardSlice);

        this.statsStack.copyWithin(0, statsSlice);
        this.statsStack.set(stats, this.statsStack.length - statsSlice);
    }

    computeObservation(game) {
        const plane = this.gridSize * this.gridSize;
        const board = new Float32Array(this.channels * plane);

        // Channel 3: walls
        for (let i = 0; i < plane; i++) {
            board[3 * plane + i] = 1;
        }
        for (let y = 0; y < this.gridSize; y++) {
            for (let x = 0; x < this.gridSize; x++) {
                board[3 * plane + y * this.gridSize + x] = 0;
            }
        }

        // Body channel
        for (let i = 1; i < game.snake.length; i++) {
            const segment = game.snake[i];
            board[plane + segment.y * this.gridSize + segment.x] = 1;
        }

        // Head channel
        const head = game.snake[0];
        board[head.y * this.gridSize + head.x] = 1;

        // Food channel
        if (game.food) {
            board[2 * plane + game.food.y * this.gridSize + game.food.x] = 1;
        }

        const stats = new Float32Array(this.statsDim);
        stats[0] = game.snake.length / (this.gridSize * this.gridSize);
        const manhattan = Math.abs(head.x - game.food.x) + Math.abs(head.y - game.food.y);
        stats[1] = manhattan / (2 * this.gridSize);
        const maxSteps = game.maxMovesWithoutFood || 1;
        stats[2] = Math.min(1, game.movesSinceLastFood / maxSteps);
        stats[3] = game.direction.x;
        stats[4] = game.direction.y;
        stats[5] = game.score / (this.gridSize * this.gridSize);

        return { board, stats };
    }

    validActionMask(game) {
        const mask = [true, true, true, true];
        if (game.snake.length > 1) {
            if (game.direction.x === 0 && game.direction.y === -1) mask[2] = false; // can't go down
            if (game.direction.x === 1 && game.direction.y === 0) mask[3] = false; // can't go left
            if (game.direction.x === 0 && game.direction.y === 1) mask[0] = false; // can't go up
            if (game.direction.x === -1 && game.direction.y === 0) mask[1] = false; // can't go right
        }
        return mask;
    }

    predictAction(game) {
        return tf.tidy(() => {
            const boardTensor = tf.tensor(this.boardStack, [1, this.boardDepth, this.gridSize, this.gridSize]);
            const statsTensor = tf.tensor(this.statsStack, [1, this.statsDepth]);
            const output = this.model.execute(
                {
                    [this.boardInputName]: boardTensor,
                    [this.statsInputName]: statsTensor
                },
                [this.logitsOutputName]
            );
            const logits = output.dataSync();
            const mask = this.validActionMask(game);
            const masked = logits.map((val, idx) => (mask[idx] ? val : Number.NEGATIVE_INFINITY));
            const action = masked.indexOf(Math.max(...masked));
            tf.dispose(output);
            return action;
        });
    }
}
