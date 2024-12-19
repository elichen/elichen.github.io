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
        this.thetaThresholdRadians = Math.PI / 2;
        this.xThreshold = 2.4;

        // Display settings
        this.canvas = document.getElementById('cartpoleCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.scale = 200; // pixels per meter

        this.reset();
    }

    reset() {
        // Initial state with small random perturbations
        this.x = 0;
        this.xDot = 0;
        this.theta = (Math.random() - 0.5) * 0.05;
        this.thetaDot = (Math.random() - 0.5) * 0.05;
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

        return {
            state: this.getState(),
            reward: reward,
            done: done
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