class PongEnvironment {
    constructor() {
        this.game = new PongGame();
        this.episodeSteps = 0;
        this.maxSteps = 2000;  // Prevent infinite episodes
        this.currentRally = 0;
        this.maxRally = 0;
    }

    reset() {
        this.episodeSteps = 0;
        this.currentRally = 0;
        const state = this.game.reset();
        return {
            state1: this.getStateForAgent(state, false),
            state2: this.getStateForAgent(state, true)
        };
    }

    // Transform state to agent's perspective
    getStateForAgent(state, isAgent2) {
        if (!isAgent2) {
            return state; // Agent 1's perspective is the default
        }
        // For Agent 2, flip the x coordinates and swap paddle positions
        return [
            1 - state[0],           // Flip ball x position
            state[1],               // Ball y position stays same
            -state[2],              // Flip ball x velocity
            state[3],               // Ball y velocity stays same
            state[5],               // Right paddle becomes "my" paddle
            state[4]                // Left paddle becomes opponent paddle
        ];
    }

    step(action1, action2) {
        this.episodeSteps++;
        const result = this.game.step(action1 - 1, action2 - 1); // Convert [0,1,2] to [-1,0,1]
        
        // Transform state for each agent's perspective
        const state1 = this.getStateForAgent(result.state, false);
        const state2 = this.getStateForAgent(result.state, true);
        
        // Track rally length
        if (this.game.checkPaddleCollision(this.game.leftPaddle) || 
            this.game.checkPaddleCollision(this.game.rightPaddle)) {
            this.currentRally++;
            this.maxRally = Math.max(this.maxRally, this.currentRally);
        }

        // End episode if max steps reached
        if (this.episodeSteps >= this.maxSteps) {
            result.done = true;
        }

        // Reset rally counter on point score
        if (result.done) {
            this.currentRally = 0;
        }

        return {
            state1,
            state2,
            reward1: result.reward1,
            reward2: result.reward2,
            done: result.done
        };
    }

    getStats() {
        return {
            steps: this.episodeSteps,
            currentRally: this.currentRally,
            maxRally: this.maxRally,
            scores: [this.game.leftPaddle.score, this.game.rightPaddle.score]
        };
    }
} 