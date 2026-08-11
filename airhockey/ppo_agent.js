class PPOAgent {
    constructor(stateSize, actionSize) {
        this.stateSize = stateSize;
        this.actionSize = actionSize;
        this.onnxSession = null;
        this.league = [];
        this.activePolicy = 0;
    }

    async loadONNXModel(modelPath) {
        await this.loadONNXLeague([{ path: modelPath, name: 'Policy', weight: 1 }]);
        return true;
    }

    async loadONNXLeague(models) {
        this.league = await Promise.all(models.map(async model => ({
            ...model,
            weight: Math.max(0, Number(model.weight) || 0),
            session: await ort.InferenceSession.create(model.path)
        })));
        if (!this.league.length || !this.league.some(model => model.weight > 0)) {
            throw new Error('The policy league needs at least one positive-weight model.');
        }
        this.onnxSession = this.league[0].session;
        this.activePolicy = this.samplePolicy();
        return true;
    }

    get policyCount() {
        return this.league.length;
    }

    samplePolicy(exclude = null) {
        const eligible = this.league.map((model, index) => ({ model, index }))
            .filter(({ index }) => index !== exclude || this.league.length === 1);
        const total = eligible.reduce((sum, { model }) => sum + model.weight, 0);
        if (total <= 0) return eligible[Math.floor(Math.random() * eligible.length)].index;
        let draw = Math.random() * total;
        for (const { model, index } of eligible) {
            draw -= model.weight;
            if (draw <= 0) return index;
        }
        return eligible[eligible.length - 1].index;
    }

    getPolicyName(index) {
        return this.league[index]?.name || `Policy ${index + 1}`;
    }

    async act(state, policyIndex = this.activePolicy) {
        const inputTensor = new ort.Tensor('float32', new Float32Array(state), [1, this.stateSize]);
        const feeds = { observation: inputTensor };
        const session = this.league[policyIndex]?.session || this.onnxSession;
        const output = await session.run(feeds);
        let action = Array.from(output.action.data);
        action = action.map(a => Math.max(-1, Math.min(1, a)));
        return { action: action, value: 0, logProb: 0 };
    }

    getState(puck, playerPaddle, aiPaddle, isTopPlayer, canvasWidth, canvasHeight) {
        const ownPaddle = isTopPlayer ? aiPaddle : playerPaddle;
        const opponentPaddle = isTopPlayer ? playerPaddle : aiPaddle;
        const maxSpeed = 25;

        // Match Python's 12 features. Slicing keeps old 8-input models usable.
        if (isTopPlayer) {
            // Player 2 (top): flip perspective to match training
            const paddle_x = ownPaddle.x / canvasWidth;
            const paddle_y = ownPaddle.y / canvasHeight;
            const puck_x = puck.x / canvasWidth;
            const puck_y = puck.y / canvasHeight;
            const paddle_dx = Math.max(-1, Math.min(1, (ownPaddle.dx || 0) / maxSpeed)) * 0.5 + 0.5;
            const paddle_dy = Math.max(-1, Math.min(1, (ownPaddle.dy || 0) / maxSpeed)) * 0.5 + 0.5;
            const puck_dx = Math.max(-1, Math.min(1, puck.dx / maxSpeed)) * 0.5 + 0.5;
            const puck_dy = Math.max(-1, Math.min(1, puck.dy / maxSpeed)) * 0.5 + 0.5;
            const opponent_x = opponentPaddle.x / canvasWidth;
            const opponent_y = opponentPaddle.y / canvasHeight;
            const opponent_dx = Math.max(-1, Math.min(1, (opponentPaddle.dx || 0) / maxSpeed)) * 0.5 + 0.5;
            const opponent_dy = Math.max(-1, Math.min(1, (opponentPaddle.dy || 0) / maxSpeed)) * 0.5 + 0.5;

            return [paddle_x, paddle_y, puck_x, puck_y, paddle_dx, paddle_dy, puck_dx, puck_dy,
                opponent_x, opponent_y, opponent_dx, opponent_dy].slice(0, this.stateSize);
        } else {
            // Player 1 (bottom): use coordinates as-is with flipped Y perspective
            const paddle_x = ownPaddle.x / canvasWidth;
            const paddle_y = (canvasHeight - ownPaddle.y) / canvasHeight;
            const puck_x = puck.x / canvasWidth;
            const puck_y = (canvasHeight - puck.y) / canvasHeight;
            const paddle_dx = Math.max(-1, Math.min(1, (ownPaddle.dx || 0) / maxSpeed)) * 0.5 + 0.5;
            const paddle_dy = Math.max(-1, Math.min(1, -(ownPaddle.dy || 0) / maxSpeed)) * 0.5 + 0.5;
            const puck_dx = Math.max(-1, Math.min(1, puck.dx / maxSpeed)) * 0.5 + 0.5;
            const puck_dy = Math.max(-1, Math.min(1, -puck.dy / maxSpeed)) * 0.5 + 0.5;
            const opponent_x = opponentPaddle.x / canvasWidth;
            const opponent_y = (canvasHeight - opponentPaddle.y) / canvasHeight;
            const opponent_dx = Math.max(-1, Math.min(1, (opponentPaddle.dx || 0) / maxSpeed)) * 0.5 + 0.5;
            const opponent_dy = Math.max(-1, Math.min(1, -(opponentPaddle.dy || 0) / maxSpeed)) * 0.5 + 0.5;

            return [paddle_x, paddle_y, puck_x, puck_y, paddle_dx, paddle_dy, puck_dx, puck_dy,
                opponent_x, opponent_y, opponent_dx, opponent_dy].slice(0, this.stateSize);
        }
    }
}
