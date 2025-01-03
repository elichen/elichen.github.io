class CartPole {
    constructor() {
        // Physics constants
        this.gravity = 9.8;
        this.cartMass = 1.0;
        this.poleMass = 0.1;
        this.totalMass = this.cartMass + this.poleMass;
        this.length = 0.5;
        this.poleMassLength = this.poleMass * this.length;
        this.forceMag = 10.0;
        this.dt = 0.02;

        // Angle at which to fail the episode (radians)
        this.thetaThresholdRadians = 12 * Math.PI / 180;  // ±12 degrees
        this.xThreshold = 2.4;

        // Display settings
        this.canvas = document.getElementById('cartpoleCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.scale = 200; // pixels per meter

        // Episode statistics
        this.episodeReturn = 0;
        this.episodeLength = 0;

        this.reset();
    }

    reset() {
        // All observations uniformly random in (-0.05, 0.05) like Gym
        this.x = (Math.random() - 0.5) * 0.1;
        this.xDot = (Math.random() - 0.5) * 0.1;
        this.theta = (Math.random() - 0.5) * 0.1;
        this.thetaDot = (Math.random() - 0.5) * 0.1;

        // Reset episode statistics
        this.episodeReturn = 0;
        this.episodeLength = 0;

        return this.getState();
    }

    step(action) {
        const force = action === 1 ? this.forceMag : -this.forceMag;
        
        const cosTheta = Math.cos(this.theta);
        const sinTheta = Math.sin(this.theta);

        const temp = (force + this.poleMassLength * this.thetaDot ** 2 * sinTheta) / this.totalMass;
        const thetaAcc = (this.gravity * sinTheta - cosTheta * temp) / 
                        (this.length * (4.0/3.0 - this.poleMass * cosTheta ** 2 / this.totalMass));
        const xAcc = temp - this.poleMassLength * thetaAcc * cosTheta / this.totalMass;

        // Update state using Euler integration
        this.x += this.dt * this.xDot;
        this.xDot += this.dt * xAcc;
        this.theta += this.dt * this.thetaDot;
        this.thetaDot += this.dt * thetaAcc;

        // Check if episode is done
        const done = this.x < -this.xThreshold || this.x > this.xThreshold || 
                    this.theta < -this.thetaThresholdRadians || this.theta > this.thetaThresholdRadians;
        
        const reward = done ? 0.0 : 1.0;

        // Update episode statistics
        this.episodeReturn += reward;
        this.episodeLength += 1;

        // Include episode info like Python's RecordEpisodeStatistics
        const info = done ? {
            episode: {
                r: this.episodeReturn,
                l: this.episodeLength
            }
        } : {};

        return {
            state: this.getState(),
            reward: reward,
            done: done,
            info: info
        };
    }

    getState() {
        return [this.x, this.xDot, this.theta, this.thetaDot];
    }

    render() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Convert to screen coordinates
        const cartX = this.x * this.scale + this.canvas.width/2;
        const cartY = this.canvas.height/2;
        
        // Draw cart
        this.ctx.fillStyle = '#333';
        this.ctx.fillRect(cartX - 20, cartY - 10, 40, 20);
        
        // Draw pole
        this.ctx.beginPath();
        this.ctx.moveTo(cartX, cartY);
        const poleEndX = cartX + Math.sin(this.theta) * this.length * this.scale;
        const poleEndY = cartY - Math.cos(this.theta) * this.length * this.scale;
        this.ctx.lineTo(poleEndX, poleEndY);
        this.ctx.strokeStyle = '#333';
        this.ctx.lineWidth = 6;
        this.ctx.stroke();
    }
} 