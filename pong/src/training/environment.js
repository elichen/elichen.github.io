class PongEnvironment {
    constructor() {
        this.game = new PongGame();
        this.episodeSteps = 0;
        this.maxSteps = 2000;  // Prevent infinite episodes
        this.currentRally = 0;
        this.maxRally = 0;
        this.lastHitBy = null; // Track which paddle last hit the ball
    }

    reset() {
        this.episodeSteps = 0;
        this.currentRally = 0;
        this.lastHitBy = null;
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
        
        // Convert actions from [0,1,2] to [-1,0,1]
        let normalizedAction1 = action1 - 1;
        let normalizedAction2 = action2 - 1;
        
        // Flip action2 since its perspective is flipped
        normalizedAction2 = -normalizedAction2;  // Flip up/down for right paddle
        
        const result = this.game.step(normalizedAction1, normalizedAction2);
        
        // Transform state for each agent's perspective
        const state1 = this.getStateForAgent(result.state, false);
        const state2 = this.getStateForAgent(result.state, true);
        
        // Initialize rewards
        let reward1 = 0;
        let reward2 = 0;
        
        // Check paddle hits
        if (this.game.checkPaddleCollision(this.game.leftPaddle)) {
            reward1 = 1.0;  // Reward for successful hit
            this.lastHitBy = 'left';
            this.currentRally++;
            this.maxRally = Math.max(this.maxRally, this.currentRally);
        }
        else if (this.game.checkPaddleCollision(this.game.rightPaddle)) {
            reward2 = 1.0;  // Reward for successful hit
            this.lastHitBy = 'right';
            this.currentRally++;
            this.maxRally = Math.max(this.maxRally, this.currentRally);
        }
        
        // Small negative reward for missing the ball
        if (result.done) {
            if (this.game.ball.x <= 0 && this.lastHitBy !== 'left') {
                reward1 = -0.5;  // Left paddle missed
            } else if (this.game.ball.x >= this.game.width && this.lastHitBy !== 'right') {
                reward2 = -0.5;  // Right paddle missed
            }
            this.currentRally = 0;
            this.lastHitBy = null;
        }

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