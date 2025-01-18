class CartPole {
    constructor(config = {}) {
        // Physics constants
        this.gravity = 9.8;
        this.cartMass = 1.0;
        this.poleMass = 0.1;
        this.totalMass = this.cartMass + this.poleMass;
        this.length = 0.5;
        this.poleMassLength = this.poleMass * this.length;
        this.forceMag = 10.0;
        this.dt = 0.02;

        // Swing-up mode configuration
        this.swingUp = config.swingUp || false;

        // Episode management
        this.steps = 0;
        this.maxSteps = 1000;
        this.episodeReturn = 0;

        // Rendering
        this.canvas = document.getElementById('cartpoleCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.cartWidth = 50;
        this.cartHeight = 30;
        this.poleWidth = 6;
        this.axleHeight = 8;
        this.scale = 200;  // pixels per meter
        
        // Physical boundaries (in meters)
        this.xLimit = (this.canvas.width/2 - this.cartWidth) / this.scale;
        
        this.reset();
    }

    reset() {
        if (this.swingUp) {
            // Start pole hanging down in swing-up mode
            this.state = [
                0.0,                    // Cart Position
                0.0,                    // Cart Velocity
                Math.PI,                // Pole Angle (hanging down)
                0.0                     // Pole Angular Velocity
            ];
        } else {
            // Start pole upright with small random perturbation in balance mode
            this.state = [
                0.0,                    // Cart Position
                0.0,                    // Cart Velocity
                0.1 * (Math.random() - 0.5),  // Small random angle
                0.0                     // Pole Angular Velocity
            ];
        }
        this.steps = 0;
        this.episodeReturn = 0;
        return this.getState();
    }

    step(action) {
        this.steps += 1;
        
        // Extract state
        let [x, xDot, theta, thetaDot] = this.state;

        // Get force direction
        let force = action === 1 ? this.forceMag : -this.forceMag;

        // Calculate physics
        const cosTheta = Math.cos(theta);
        const sinTheta = Math.sin(theta);

        // Check if we're at a boundary and trying to move further into it
        if ((x <= -this.xLimit && force < 0) || (x >= this.xLimit && force > 0)) {
            force = 0;  // Can't push into wall
        }

        const temp = (force + this.poleMassLength * thetaDot ** 2 * sinTheta) / this.totalMass;
        const thetaAcc = (this.gravity * sinTheta - cosTheta * temp) / (this.length * (4/3 - this.poleMass * cosTheta ** 2 / this.totalMass));
        const xAcc = temp - this.poleMassLength * thetaAcc * cosTheta / this.totalMass;

        // Update state with Euler integration
        x = x + this.dt * xDot;
        xDot = xDot + this.dt * xAcc;
        theta = theta + this.dt * thetaDot;
        thetaDot = thetaDot + this.dt * thetaAcc;

        // Check boundary violations
        const hitBoundary = x < -this.xLimit || x > this.xLimit;

        // Apply boundary constraints
        if (x < -this.xLimit) {
            x = -this.xLimit;
            xDot = 0;
        } else if (x > this.xLimit) {
            x = this.xLimit;
            xDot = 0;
        }

        this.state = [x, xDot, theta, thetaDot];

        // Calculate reward and done flag
        let done = false;
        let reward;

        if (this.swingUp) {
            // Reward based on pole angle (1 when upright, -1 when hanging)
            reward = Math.cos(theta);
            
            // Add penalty and terminate if cart hits boundaries
            if (hitBoundary) {
                reward = -2.0;  // Larger penalty for boundary violation
                done = true;
            } else {
                // Only terminate on max steps if we haven't hit boundaries
                done = this.steps >= this.maxSteps;
            }
        } else {
            // Original balance task termination conditions
            done = theta < -0.21 || theta > 0.21 || hitBoundary || this.steps >= this.maxSteps;
            
            if (hitBoundary) {
                reward = -1.0;  // Penalty for hitting boundary
            } else {
                reward = done ? 0.0 : 1.0;  // Normal reward structure
            }
        }

        this.episodeReturn += reward;

        return {
            state: this.getState(),
            reward: reward,
            done: done,
            info: { 
                episode: { 
                    r: this.episodeReturn,
                    steps: this.steps,
                    mode: this.swingUp ? 'swing-up' : 'balance'
                } 
            }
        };
    }

    getState() {
        const [x, xDot, theta, thetaDot] = this.state;
        
        // Add normalized distances to boundaries (-1 at left boundary, 0 at center, 1 at right boundary)
        const normalizedPosition = x / this.xLimit;
        
        return [
            normalizedPosition,  // Position normalized to boundaries
            xDot,               // Cart velocity
            theta,              // Pole angle
            thetaDot           // Pole angular velocity
        ];
    }

    render() {
        const [x, _, theta] = this.state;  // Extract position and angle from state
        
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Convert to screen coordinates (no clamping needed since physics handles boundaries)
        const cartX = x * this.scale + this.canvas.width/2;
        const cartY = this.canvas.height/2;
        
        // Draw cart
        this.ctx.fillStyle = '#333';
        this.ctx.fillRect(cartX - this.cartWidth/2, cartY - this.cartHeight/2, this.cartWidth, this.cartHeight);
        
        // Draw pole
        this.ctx.beginPath();
        this.ctx.moveTo(cartX, cartY);
        const poleEndX = cartX + Math.sin(theta) * this.length * this.scale;
        const poleEndY = cartY - Math.cos(theta) * this.length * this.scale;
        this.ctx.lineTo(poleEndX, poleEndY);
        this.ctx.strokeStyle = '#333';
        this.ctx.lineWidth = this.poleWidth;
        this.ctx.stroke();
    }
} 