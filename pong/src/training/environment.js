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
        return this.game.reset();
    }

    step(action1, action2) {
        this.episodeSteps++;
        const result = this.game.step(action1 - 1, action2 - 1); // Convert [0,1,2] to [-1,0,1]
        
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

        return result;
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