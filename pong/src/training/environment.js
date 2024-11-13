class PongEnvironment {
    constructor() {
        this.game = new PongGame();
        this.episodeSteps = 0;
        this.maxSteps = 2000;
        this.currentRally = 0;
        this.maxRally = 0;
        this.lastHitBy = null;
        this.prevDist1 = null;
        this.prevDist2 = null;
        this.stage = 1;
        this.bestRally = 0;
    }

    reset() {
        this.episodeSteps = 0;
        this.currentRally = 0;
        this.lastHitBy = null;
        const state = this.game.reset();
        
        this.prevDist1 = Math.abs(this.game.leftPaddle.y + this.game.leftPaddle.height/2 - this.game.ball.y);
        this.prevDist2 = Math.abs(this.game.rightPaddle.y + this.game.rightPaddle.height/2 - this.game.ball.y);
        
        return {
            state1: this.getStateForAgent(state, false),
            state2: this.getStateForAgent(state, true)
        };
    }

    getStateForAgent(state, isAgent2) {
        if (!isAgent2) {
            return state;
        }
        return [
            1 - state[0],
            state[1],
            -state[2],
            state[3],
            state[5],
            state[4]
        ];
    }

    step(action1, action2) {
        this.episodeSteps++;
        
        let normalizedAction1 = action1 - 1;
        let normalizedAction2 = action2 - 1;
        normalizedAction2 = -normalizedAction2;
        
        const result = this.game.step(normalizedAction1, normalizedAction2);
        
        let reward1 = 0;
        let reward2 = 0;

        if (result.hitPaddle) {
            this.currentRally++;
            this.maxRally = Math.max(this.maxRally, this.currentRally);
            this.bestRally = Math.max(this.bestRally, this.currentRally);
            
            if (this.stage === 1 && this.bestRally >= 5) {
                console.log("Advancing to Stage 2 - Score-based rewards");
                this.stage = 2;
            }
        }

        if (result.done) {
            this.currentRally = 0;
        }

        if (this.stage === 1) {
            const newDist1 = Math.abs(this.game.leftPaddle.y + this.game.leftPaddle.height/2 - this.game.ball.y);
            const newDist2 = Math.abs(this.game.rightPaddle.y + this.game.rightPaddle.height/2 - this.game.ball.y);
            
            reward1 = (this.prevDist1 - newDist1) * 0.1;
            reward2 = (this.prevDist2 - newDist2) * 0.1;
            
            this.prevDist1 = newDist1;
            this.prevDist2 = newDist2;
            
            if (normalizedAction1 !== 0) reward1 -= 0.01;
            if (normalizedAction2 !== 0) reward2 -= 0.01;
        } else {
            if (result.done) {
                if (this.game.ball.x <= 0) {
                    reward1 = -1;
                    reward2 = 1;
                } else if (this.game.ball.x >= this.game.width) {
                    reward1 = 1;
                    reward2 = -1;
                }
            }
        }

        const state1 = this.getStateForAgent(result.state, false);
        const state2 = this.getStateForAgent(result.state, true);

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
            scores: [this.game.leftPaddle.score, this.game.rightPaddle.score],
            stage: this.stage,
            bestRally: this.bestRally
        };
    }
} 