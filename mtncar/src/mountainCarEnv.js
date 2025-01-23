class MountainCarEnv {
    constructor(mountainCar) {
        this.mountainCar = mountainCar;
        this.numActions = 3; // left, none, right
        this.lastPosition = null;
    }

    reset() {
        this.mountainCar = new MountainCar(this.mountainCar.canvas);
        this.lastPosition = this.mountainCar.position;
        return this.getState();
    }

    getState() {
        return [this.mountainCar.position, this.mountainCar.velocity];
    }

    step(action) {
        // Convert action index to action string
        const actionMap = {
            0: 'left',
            1: 'none', 
            2: 'right'
        };

        const success = this.mountainCar.step(actionMap[action]);
        const state = this.getState();
        
        // Calculate rewards
        let reward = 0;
        
        // Big reward for reaching the goal
        if (success) {
            reward = 100;
        } else {
            // Reward for height gained (potential energy)
            const heightReward = Math.sin(3 * this.mountainCar.position) - 
                               Math.sin(3 * this.lastPosition);
            
            // Reward for velocity in the right direction (kinetic energy)
            const velocityReward = this.mountainCar.velocity * 
                                 Math.sign(this.mountainCar.goalPosition - this.mountainCar.position);
            
            // Small penalty for using the engine (encourage efficiency)
            const actionPenalty = action !== 1 ? -0.1 : 0;
            
            // Combine rewards with appropriate scaling
            reward = heightReward * 10 + velocityReward * 5 + actionPenalty;
        }

        // Update last position for next step
        this.lastPosition = this.mountainCar.position;
        
        const done = success || this.mountainCar.position <= this.mountainCar.minPosition;

        return {
            state: state,
            reward: reward,
            done: done,
            info: { 
                episode: { 
                    r: reward,
                    steps: 1,
                    success: success 
                } 
            }
        };
    }

    render() {
        this.mountainCar.render();
    }
} 