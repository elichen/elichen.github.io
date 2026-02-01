class CartPoleSwingup {
    constructor(config = {}) {
        // Physics constants matching Gym exactly
        this.gravity = 9.8;
        this.cartMass = 1.0;
        this.poleMass = 0.1;
        this.totalMass = this.cartMass + this.poleMass;
        this.length = 0.5;  // actually half the pole's length
        this.poleMassLength = this.poleMass * this.length;
        this.forceMag = 10.0;
        this.dt = 0.02;  // seconds between state updates

        // Boundaries
        this.xLimit = 2.4;
        // No theta limit for swingup - pole can rotate freely

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
        const totalWidth = this.canvas.width - this.cartWidth - 40;
        this.scale = totalWidth / (2 * this.xLimit);

        this.reset();
    }

    reset() {
        // Start with pole hanging DOWN (θ = π) with small random perturbation
        this.state = [
            (Math.random() - 0.5) * 0.1,  // Cart Position
            0.0,                           // Cart Velocity
            Math.PI + (Math.random() - 0.5) * 0.1,  // Pole Angle (hanging down)
            0.0                            // Pole Angular Velocity
        ];
        this.steps = 0;
        this.episodeReturn = 0;
        return this.getState();
    }

    step(action) {
        this.steps += 1;

        let [x, xDot, theta, thetaDot] = this.state;

        // Get force direction (0: left, 1: right)
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

        // Normalize theta to [-π, π]
        theta = ((theta + Math.PI) % (2 * Math.PI)) - Math.PI;
        if (theta < -Math.PI) theta += 2 * Math.PI;

        this.state = [x, xDot, theta, thetaDot];

        // Reward: height of pole tip = cos(theta)
        // cos(0) = 1 (upright), cos(π) = -1 (down)
        // Shift to [0, 2] range: 1 + cos(theta)
        let reward = 1.0 + Math.cos(theta);

        // Episode ends if cart goes out of bounds or max steps
        const done = Math.abs(x) >= this.xLimit || this.steps >= this.maxSteps;

        // Penalty for going out of bounds
        if (Math.abs(x) >= this.xLimit) {
            reward = 0.0;
        }

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
        return [...this.state];
    }

    render() {
        const [x, _, theta] = this.state;

        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Convert to screen coordinates
        const cartX = x * this.scale + this.canvas.width/2;
        const cartY = this.canvas.height/2;

        // Draw track
        this.ctx.beginPath();
        this.ctx.moveTo(this.canvas.width/2 - this.xLimit * this.scale, cartY + this.cartHeight/2 + 5);
        this.ctx.lineTo(this.canvas.width/2 + this.xLimit * this.scale, cartY + this.cartHeight/2 + 5);
        this.ctx.strokeStyle = '#999';
        this.ctx.lineWidth = 2;
        this.ctx.stroke();

        // Draw cart
        this.ctx.fillStyle = '#333';
        this.ctx.fillRect(cartX - this.cartWidth/2, cartY - this.cartHeight/2, this.cartWidth, this.cartHeight);

        // Draw pole
        this.ctx.beginPath();
        this.ctx.moveTo(cartX, cartY);
        const poleEndX = cartX + Math.sin(theta) * this.length * 2 * this.scale;
        const poleEndY = cartY - Math.cos(theta) * this.length * 2 * this.scale;
        this.ctx.lineTo(poleEndX, poleEndY);

        // Color pole based on height (green when up, red when down)
        const upness = (1 + Math.cos(theta)) / 2;  // 0 = down, 1 = up
        const r = Math.round(255 * (1 - upness));
        const g = Math.round(200 * upness);
        this.ctx.strokeStyle = `rgb(${r}, ${g}, 50)`;
        this.ctx.lineWidth = this.poleWidth;
        this.ctx.stroke();

        // Draw pole tip
        this.ctx.beginPath();
        this.ctx.arc(poleEndX, poleEndY, 8, 0, 2 * Math.PI);
        this.ctx.fillStyle = this.ctx.strokeStyle;
        this.ctx.fill();

        // Draw boundaries
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
