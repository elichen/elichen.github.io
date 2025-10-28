class PPOAgent {
    constructor(stateSize, actionSize) {
        this.stateSize = stateSize;
        this.actionSize = actionSize;
        this.onnxSession = null;
    }

    async loadONNXModel(modelPath) {
        this.onnxSession = await ort.InferenceSession.create(modelPath);
        return true;
    }

    async act(state) {
        const inputTensor = new ort.Tensor('float32', new Float32Array(state), [1, this.stateSize]);
        const feeds = { observation: inputTensor };
        const output = await this.onnxSession.run(feeds);
        let action = Array.from(output.action.data);
        action = action.map(a => Math.max(-1, Math.min(1, a)));
        return { action: action, value: 0, logProb: 0 };
    }

    getState(puck, playerPaddle, aiPaddle, isTopPlayer, canvasWidth, canvasHeight) {
        const ownPaddle = isTopPlayer ? aiPaddle : playerPaddle;
        const maxSpeed = 25;

        // Match Python environment's 8-feature observation space exactly
        if (isTopPlayer) {
            // Player 2 (top): flip perspective to match training
            const paddle_x = ownPaddle.x / canvasWidth;
            const paddle_y = ownPaddle.y / canvasHeight;
            const puck_x = puck.x / canvasWidth;
            const puck_y = puck.y / canvasHeight;
            const paddle_dx = ((ownPaddle.dx || 0) / maxSpeed + 1) / 2;
            const paddle_dy = ((ownPaddle.dy || 0) / maxSpeed + 1) / 2;
            const puck_dx = (puck.dx / maxSpeed + 1) / 2;
            const puck_dy = (puck.dy / maxSpeed + 1) / 2;

            return [paddle_x, paddle_y, puck_x, puck_y, paddle_dx, paddle_dy, puck_dx, puck_dy];
        } else {
            // Player 1 (bottom): use coordinates as-is with flipped Y perspective
            const paddle_x = ownPaddle.x / canvasWidth;
            const paddle_y = (canvasHeight - ownPaddle.y) / canvasHeight;
            const puck_x = puck.x / canvasWidth;
            const puck_y = (canvasHeight - puck.y) / canvasHeight;
            const paddle_dx = ((ownPaddle.dx || 0) / maxSpeed + 1) / 2;
            const paddle_dy = (-(ownPaddle.dy || 0) / maxSpeed + 1) / 2;
            const puck_dx = (puck.dx / maxSpeed + 1) / 2;
            const puck_dy = (-puck.dy / maxSpeed + 1) / 2;

            return [paddle_x, paddle_y, puck_x, puck_y, paddle_dx, paddle_dy, puck_dx, puck_dy];
        }
    }
}