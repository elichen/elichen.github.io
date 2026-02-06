class CartPoleDouble {
    constructor(config = {}) {
        // Physics constants
        this.gravity = config.gravity ?? 9.8;
        this.cartMass = config.cartMass ?? 1.0;       // M: cart mass
        this.m1 = config.m1 ?? 0.1;                   // m1: first segment mass
        this.m2 = config.m2 ?? 0.1;                   // m2: second segment mass
        this.L1 = config.L1 ?? 0.5;                   // L1: first segment half-length
        this.L2 = config.L2 ?? 0.5;                   // L2: second segment half-length
        this.forceMag = config.forceMag ?? 10.0;
        this.dt = config.dt ?? 0.02;                  // seconds between state updates

        // Boundaries
        this.xLimit = config.xLimit ?? 2.4;

        // Episode management
        this.steps = 0;
        this.maxSteps = config.maxSteps ?? 500;
        this.episodeReturn = 0;

        // Rendering setup
        this.canvas = document.getElementById('cartpoleCanvas');
        this.ctx = this.canvas.getContext('2d');

        // Rendering dimensions (in pixels)
        this.cartWidth = 50;
        this.cartHeight = 30;
        this.poleWidth = 6;

        // Calculate scale to fit cart's full range of motion
        const totalWidth = this.canvas.width - this.cartWidth - 40;
        this.scale = totalWidth / (2 * this.xLimit);

        this.reset();
    }

    reset() {
        // Start with both segments hanging DOWN (θ = π) with small random perturbation
        this.state = [
            (Math.random() - 0.5) * 0.1,              // Cart Position x
            0.0,                                       // Cart Velocity x_dot
            Math.PI + (Math.random() - 0.5) * 0.1,   // First segment angle theta1 (hanging down)
            0.0,                                       // First segment angular velocity theta1_dot
            Math.PI + (Math.random() - 0.5) * 0.1,   // Second segment angle theta2 (hanging down)
            0.0                                        // Second segment angular velocity theta2_dot
        ];
        this.steps = 0;
        this.episodeReturn = 0;
        return this.getState();
    }

    step(action) {
        this.steps += 1;

        let [x, xDot, theta1, theta1Dot, theta2, theta2Dot] = this.state;

        // Support both:
        // - discrete actions: 0 (left), 1 (right)
        // - continuous actions: force value in [-forceMag, forceMag]
        let F;
        if (action === 0 || action === 1) {
            F = (action === 0 ? -1 : 1) * this.forceMag;
        } else {
            const a = Number(action);
            F = Math.max(-this.forceMag, Math.min(this.forceMag, Number.isFinite(a) ? a : 0));
        }

        // Double pendulum on cart physics using Lagrangian mechanics
        // Using absolute angles (both measured from vertical)
        const g = this.gravity;
        const M = this.cartMass;
        const m1 = this.m1;
        const m2 = this.m2;
        const l1 = this.L1;  // half-length, so full length is 2*l1
        const l2 = this.L2;

        const c1 = Math.cos(theta1);
        const s1 = Math.sin(theta1);
        const c2 = Math.cos(theta2);
        const s2 = Math.sin(theta2);
        const c12 = Math.cos(theta1 - theta2);
        const s12 = Math.sin(theta1 - theta2);

        // Mass matrix elements for the double pendulum on cart system
        // Using full lengths (2*l1, 2*l2) for moment of inertia
        const L1full = 2 * l1;
        const L2full = 2 * l2;

        // Equations of motion derived from Lagrangian
        // M_total * x_ddot + m1*L1*theta1_ddot*cos(theta1) + m2*L1*theta1_ddot*cos(theta1)
        //                  + m2*L2*theta2_ddot*cos(theta2) = F + m1*L1*theta1_dot^2*sin(theta1)
        //                  + m2*L1*theta1_dot^2*sin(theta1) + m2*L2*theta2_dot^2*sin(theta2)

        // Simplified double pendulum equations (standard formulation)
        const d1 = M + m1 + m2;
        const d2 = (m1/2 + m2) * L1full;
        const d3 = m2 * L2full / 2;
        const d4 = (m1/3 + m2) * L1full * L1full;
        const d5 = m2 * L1full * L2full / 2;
        const d6 = m2 * L2full * L2full / 3;

        // Build mass matrix
        const M11 = d1;
        const M12 = d2 * c1;
        const M13 = d3 * c2;
        const M21 = d2 * c1;
        const M22 = d4;
        const M23 = d5 * c12;
        const M31 = d3 * c2;
        const M32 = d5 * c12;
        const M33 = d6;

        // Build forcing vector (Coriolis/centrifugal + gravity + external force)
        const f1 = F + d2 * theta1Dot * theta1Dot * s1 + d3 * theta2Dot * theta2Dot * s2;
        const f2 = d5 * theta2Dot * theta2Dot * s12 + (m1/2 + m2) * g * L1full * s1;
        const f3 = -d5 * theta1Dot * theta1Dot * s12 + m2 * g * L2full * s2 / 2;

        // Solve 3x3 system: M * [x_ddot, theta1_ddot, theta2_ddot]^T = [f1, f2, f3]^T
        // Using Cramer's rule or direct inversion
        const det = M11 * (M22 * M33 - M23 * M32)
                  - M12 * (M21 * M33 - M23 * M31)
                  + M13 * (M21 * M32 - M22 * M31);

        // Avoid division by zero
        if (Math.abs(det) < 1e-10) {
            // Fallback: small perturbation
            return this.stepFallback(F);
        }

        // Compute accelerations using Cramer's rule
        const xAcc = (f1 * (M22 * M33 - M23 * M32)
                    - M12 * (f2 * M33 - M23 * f3)
                    + M13 * (f2 * M32 - M22 * f3)) / det;

        const theta1Acc = (M11 * (f2 * M33 - M23 * f3)
                         - f1 * (M21 * M33 - M23 * M31)
                         + M13 * (M21 * f3 - f2 * M31)) / det;

        const theta2Acc = (M11 * (M22 * f3 - f2 * M32)
                         - M12 * (M21 * f3 - f2 * M31)
                         + f1 * (M21 * M32 - M22 * M31)) / det;

        // Update state with Euler integration
        x = x + this.dt * xDot;
        xDot = xDot + this.dt * xAcc;
        theta1 = theta1 + this.dt * theta1Dot;
        theta1Dot = theta1Dot + this.dt * theta1Acc;
        theta2 = theta2 + this.dt * theta2Dot;
        theta2Dot = theta2Dot + this.dt * theta2Acc;

        // Normalize angles to [-π, π]
        theta1 = this.normalizeAngle(theta1);
        theta2 = this.normalizeAngle(theta2);

        this.state = [x, xDot, theta1, theta1Dot, theta2, theta2Dot];

        // Reward: height-based for both segments
        // cos(0) = 1 (upright), cos(π) = -1 (down)
        // Range [0, 2] total: 1 + 0.5*cos(theta1) + 0.5*cos(theta2)
        let reward = 1.0 + 0.5 * Math.cos(theta1) + 0.5 * Math.cos(theta2);

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

    stepFallback(action) {
        // Simple fallback if matrix is singular
        this.state[1] += Math.sign(action) * 0.1;
        const done = Math.abs(this.state[0]) >= this.xLimit || this.steps >= this.maxSteps;
        const reward = 1.0 + 0.5 * Math.cos(this.state[2]) + 0.5 * Math.cos(this.state[4]);
        this.episodeReturn += reward;
        return {
            state: this.getState(),
            reward: reward,
            done: done,
            info: { episode: { r: this.episodeReturn, steps: this.steps } }
        };
    }

    normalizeAngle(theta) {
        theta = ((theta + Math.PI) % (2 * Math.PI)) - Math.PI;
        if (theta < -Math.PI) theta += 2 * Math.PI;
        return theta;
    }

    getState() {
        return [...this.state];
    }

    render() {
        const [x, _, theta1, __, theta2] = this.state;

        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Convert to screen coordinates
        const cartX = x * this.scale + this.canvas.width / 2;
        const cartY = this.canvas.height / 2;

        // Draw track
        this.ctx.beginPath();
        this.ctx.moveTo(this.canvas.width / 2 - this.xLimit * this.scale, cartY + this.cartHeight / 2 + 5);
        this.ctx.lineTo(this.canvas.width / 2 + this.xLimit * this.scale, cartY + this.cartHeight / 2 + 5);
        this.ctx.strokeStyle = '#999';
        this.ctx.lineWidth = 2;
        this.ctx.stroke();

        // Draw cart
        this.ctx.fillStyle = '#333';
        this.ctx.fillRect(cartX - this.cartWidth / 2, cartY - this.cartHeight / 2, this.cartWidth, this.cartHeight);

        // First segment endpoint (joint)
        const joint1X = cartX + Math.sin(theta1) * this.L1 * 2 * this.scale;
        const joint1Y = cartY - Math.cos(theta1) * this.L1 * 2 * this.scale;

        // Second segment endpoint (tip)
        const tipX = joint1X + Math.sin(theta2) * this.L2 * 2 * this.scale;
        const tipY = joint1Y - Math.cos(theta2) * this.L2 * 2 * this.scale;

        // Calculate upness for color (0=down, 1=up)
        const upness1 = (1 + Math.cos(theta1)) / 2;
        const upness2 = (1 + Math.cos(theta2)) / 2;

        // Draw first segment
        this.ctx.beginPath();
        this.ctx.moveTo(cartX, cartY);
        this.ctx.lineTo(joint1X, joint1Y);
        const r1 = Math.round(255 * (1 - upness1));
        const g1 = Math.round(200 * upness1);
        this.ctx.strokeStyle = `rgb(${r1}, ${g1}, 50)`;
        this.ctx.lineWidth = this.poleWidth;
        this.ctx.stroke();

        // Draw joint between segments
        this.ctx.beginPath();
        this.ctx.arc(joint1X, joint1Y, 6, 0, 2 * Math.PI);
        this.ctx.fillStyle = '#666';
        this.ctx.fill();

        // Draw second segment
        this.ctx.beginPath();
        this.ctx.moveTo(joint1X, joint1Y);
        this.ctx.lineTo(tipX, tipY);
        const r2 = Math.round(255 * (1 - upness2));
        const g2 = Math.round(200 * upness2);
        this.ctx.strokeStyle = `rgb(${r2}, ${g2}, 50)`;
        this.ctx.lineWidth = this.poleWidth;
        this.ctx.stroke();

        // Draw pole tip
        this.ctx.beginPath();
        this.ctx.arc(tipX, tipY, 8, 0, 2 * Math.PI);
        this.ctx.fillStyle = this.ctx.strokeStyle;
        this.ctx.fill();

        // Draw boundaries
        this.ctx.beginPath();
        this.ctx.moveTo(this.canvas.width / 2 - this.xLimit * this.scale, 0);
        this.ctx.lineTo(this.canvas.width / 2 - this.xLimit * this.scale, this.canvas.height);
        this.ctx.moveTo(this.canvas.width / 2 + this.xLimit * this.scale, 0);
        this.ctx.lineTo(this.canvas.width / 2 + this.xLimit * this.scale, this.canvas.height);
        this.ctx.strokeStyle = '#ccc';
        this.ctx.lineWidth = 1;
        this.ctx.stroke();
    }
}
