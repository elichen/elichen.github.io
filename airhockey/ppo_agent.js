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
        const oppPaddle = isTopPlayer ? playerPaddle : aiPaddle;

        // Flip coordinates for top player to match training perspective
        const ownY = isTopPlayer ? canvasHeight - ownPaddle.y : ownPaddle.y;
        const oppY = isTopPlayer ? canvasHeight - oppPaddle.y : oppPaddle.y;
        const puckY = isTopPlayer ? canvasHeight - puck.y : puck.y;
        const puckDy = isTopPlayer ? -puck.dy : puck.dy;

        const normSpeed = 25;

        // Match Python observation space exactly
        const relativeX = (puck.x - ownPaddle.x) / canvasWidth;
        const relativeY = (puckY - ownY) / canvasHeight;
        const relativeDx = puck.dx / normSpeed;
        const relativeDy = puckDy / normSpeed;
        const relativeOppX = (oppPaddle.x - ownPaddle.x) / canvasWidth;
        const relativeOppY = (oppY - ownY) / canvasHeight;

        const distPixels = Math.sqrt(Math.pow(puck.x - ownPaddle.x, 2) + Math.pow(puckY - ownY, 2));
        const maxDist = Math.sqrt(canvasWidth * canvasWidth + canvasHeight * canvasHeight);
        const distance = distPixels / maxDist;

        const angle = Math.atan2(puckY - ownY, puck.x - ownPaddle.x) / Math.PI;

        const isPuckBehind = puckY > ownY ? 1.0 : 0.0;
        const distanceToGoal = (canvasHeight - ownY) / canvasHeight * 2 - 1;
        const puckToGoal = (canvasHeight - puckY) / canvasHeight * 2 - 1;
        const ownDx = Math.max(-1, Math.min(1, (ownPaddle.dx || 0) / normSpeed));
        const ownDy = Math.max(-1, Math.min(1, (ownPaddle.dy || 0) / normSpeed));
        const puckSpeed = Math.sqrt(puck.dx * puck.dx + puck.dy * puck.dy) / normSpeed;

        return [relativeX, relativeY, relativeDx, relativeDy, relativeOppX, relativeOppY,
                distance, angle, isPuckBehind, distanceToGoal, puckToGoal, ownDx, ownDy, puckSpeed];
    }
}