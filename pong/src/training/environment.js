class PongEnvironment {
    constructor() {
        this.game = new PongGame();
        this.episodeSteps = 0;
        this.maxSteps = 2000;  // Prevent infinite episodes
        this.currentRally = 0;
        this.maxRally = 0;
        this.lastHitBy = null;
        // Store previous distances for reward calculation
        this.prevDist1 = null;
        this.prevDist2 = null;
    }

    reset() {
        this.episodeSteps = 0;
        this.currentRally = 0;
        this.lastHitBy = null;
        const state = this.game.reset();
        
        // Initialize previous distances
        this.prevDist1 = Math.abs(this.game.leftPaddle.y + this.game.leftPaddle.height/2 - this.game.ball.y);
        this.prevDist2 = Math.abs(this.game.rightPaddle.y + this.game.rightPaddle.height/2 - this.game.ball.y);
        
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
        
        // Convert actions from [0,1,2] to [-1,0,1]
        let normalizedAction1 = action1 - 1;
        let normalizedAction2 = action2 - 1;
        
        // Flip action2 since its perspective is flipped
        normalizedAction2 = -normalizedAction2;  // Flip up/down for right paddle
        
        // Store current paddle centers before movement
        const prevCenter1 = this.game.leftPaddle.y + this.game.leftPaddle.height/2;
        const prevCenter2 = this.game.rightPaddle.y + this.game.rightPaddle.height/2;
        
        // Take step in environment
        const result = this.game.step(normalizedAction1, normalizedAction2);
        
        // Calculate new distances to ball
        const newDist1 = Math.abs(this.game.leftPaddle.y + this.game.leftPaddle.height/2 - this.game.ball.y);
        const newDist2 = Math.abs(this.game.rightPaddle.y + this.game.rightPaddle.height/2 - this.game.ball.y);
        
        // Calculate rewards based on distance change
        let reward1 = (this.prevDist1 - newDist1) * 0.1; // Scale factor to keep rewards small
        let reward2 = (this.prevDist2 - newDist2) * 0.1;
        
        // Store new distances for next step
        this.prevDist1 = newDist1;
        this.prevDist2 = newDist2;
        
        // Small penalty for movement to discourage constant motion
        if (normalizedAction1 !== 0) reward1 -= 0.01;
        if (normalizedAction2 !== 0) reward2 -= 0.01;

        // Transform state for each agent's perspective
        const state1 = this.getStateForAgent(result.state, false);
        const state2 = this.getStateForAgent(result.state, true);

        // End episode if max steps reached
        if (this.episodeSteps >= this.maxSteps) {
            result.done = true;
        }

        return {
            state1,
            state2,
            reward1,
            reward2,
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