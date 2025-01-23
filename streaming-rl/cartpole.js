class CartPole {
    constructor(config = {}) {
        // Physics constants matching Gym exactly
        this.gravity = 9.8;
        this.cartMass = 1.0;
        this.poleMass = 0.1;
        this.totalMass = this.cartMass + this.poleMass;
        this.length = 0.5;  // actually half the pole's length
        this.poleMassLength = this.poleMass * this.length;
        this.forceMag = 10.0;
        this.dt = 0.02;  // seconds between state updates (tau in Gym)
        
        // Boundaries matching Gym exactly
        this.xLimit = 2.4;  // Gym's x_threshold
        this.thetaLimit = 12 * 2 * Math.PI / 360;  // Gym's theta_threshold_radians (12 degrees)

        // Episode management
        this.steps = 0;
        this.maxSteps = 500;
        this.episodeReturn = 0;

        // Rendering setup
        this.canvas = document.getElementById('cartpoleCanvas');
        this.ctx = this.canvas.getContext('2d');
        
        // Rendering dimensions (in pixels)
        this.cartWidth = 50;
        this.cartHeight = 30;
        this.poleWidth = 6;
        this.axleHeight = 8;
        
        // Calculate scale to fit cart's full range of motion
        // Leave 10% margin on each side and account for cart width
        const totalWidth = this.canvas.width - this.cartWidth - 40;  // 40px total margin
        this.scale = totalWidth / (2 * this.xLimit);  // pixels per meter
        
        this.reset();
    }

    reset() {
        // Match Gym's starting state: uniform random in (-0.05, 0.05) for all observations
        this.state = [
            (Math.random() - 0.5) * 0.1,  // Cart Position
            (Math.random() - 0.5) * 0.1,  // Cart Velocity
            (Math.random() - 0.5) * 0.1,  // Pole Angle
            (Math.random() - 0.5) * 0.1   // Pole Angular Velocity
        ];
        this.steps = 0;
        this.episodeReturn = 0;
        return this.getState();
    }

    step(action) {
        this.steps += 1;
        
        // Extract state
        let [x, xDot, theta, thetaDot] = this.state;

        // Get force direction (0: left, 1: right) matching Gym
        let force = (action === 0 ? -1 : 1) * this.forceMag;

        const cosTheta = Math.cos(theta);
        const sinTheta = Math.sin(theta);

        // Equations directly from Gym implementation
        const temp = (
            force + this.poleMassLength * thetaDot ** 2 * sinTheta
        ) / this.totalMass;

        const thetaAcc = (this.gravity * sinTheta - cosTheta * temp) / 
            (this.length * (4.0/3.0 - this.poleMass * cosTheta ** 2 / this.totalMass));

        const xAcc = temp - this.poleMassLength * thetaAcc * cosTheta / this.totalMass;

        // Update state with Euler integration
        x = x + this.dt * xDot;
        xDot = xDot + this.dt * xAcc;
        theta = theta + this.dt * thetaDot;
        thetaDot = thetaDot + this.dt * thetaAcc;

        this.state = [x, xDot, theta, thetaDot];

        // Calculate reward and done flag
        const done = Math.abs(x) >= this.xLimit || 
                    Math.abs(theta) > this.thetaLimit || 
                    this.steps >= this.maxSteps;
        
        const reward = done ? 0.0 : 1.0;  // Gym's reward structure

        this.episodeReturn += reward;

        return {
            state: this.getState(),
            reward: reward,
            done: done,
            info: { 
                episode: { 
                    r: this.episodeReturn,
                    steps: this.steps
                } 
            }
        };
    }

    getState() {
        // Return raw state values just like Gym
        return [...this.state];  // Return copy of state array
    }

    render() {
        const [x, _, theta] = this.state;  // Extract position and angle from state
        
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Convert to screen coordinates
        const cartX = x * this.scale + this.canvas.width/2;
        const cartY = this.canvas.height/2;
        
        // Draw cart
        this.ctx.fillStyle = '#333';
        this.ctx.fillRect(cartX - this.cartWidth/2, cartY - this.cartHeight/2, this.cartWidth, this.cartHeight);
        
        // Draw pole
        this.ctx.beginPath();
        this.ctx.moveTo(cartX, cartY);
        const poleEndX = cartX + Math.sin(theta) * this.length * 2 * this.scale;  // length * 2 because length is half the pole length
        const poleEndY = cartY - Math.cos(theta) * this.length * 2 * this.scale;
        this.ctx.lineTo(poleEndX, poleEndY);
        this.ctx.strokeStyle = '#333';
        this.ctx.lineWidth = this.poleWidth;
        this.ctx.stroke();

        // Optionally draw boundaries
        this.ctx.beginPath();
        this.ctx.moveTo(this.canvas.width/2 - this.xLimit * this.scale, 0);
        this.ctx.lineTo(this.canvas.width/2 - this.xLimit * this.scale, this.canvas.height);
        this.ctx.moveTo(this.canvas.width/2 + this.xLimit * this.scale, 0);
        this.ctx.lineTo(this.canvas.width/2 + this.xLimit * this.scale, this.canvas.height);
        this.ctx.strokeStyle = '#ccc';
        this.ctx.lineWidth = 1;
        this.ctx.stroke();
    }
} 