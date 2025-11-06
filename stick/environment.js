class StickBalancingEnv {
    constructor() {
        // Environment parameters
        this.gravity = 9.81;
        this.railLength = 4.8;
        this.stickLength = 1.8;
        this.dt = 0.02;
        this.maxPosition = this.railLength / 2;

        // Mass parameters
        this.stickMass = 0.1;
        this.weightMass = 0.3;

        // Calculate center of mass and moment of inertia
        this.totalMass = this.stickMass + this.weightMass;
        // Center of mass from pivot point
        this.centerOfMass = (this.stickMass * this.stickLength/2 + this.weightMass * this.stickLength) / this.totalMass;
        // Moment of inertia: rod (1/3 mL^2) + point mass (mL^2)
        this.momentOfInertia = (this.stickMass * this.stickLength * this.stickLength / 3) +
                               (this.weightMass * this.stickLength * this.stickLength);

        // Damping coefficient for pendulum
        this.angularDamping = 0.10;

        this.reset();
    }

    reset() {
        // Randomize position and velocity with small values
        this.position = (Math.random() - 0.5) * 0.1;
        this.velocity = (Math.random() - 0.5) * 0.1;

        // Start with stick pointing down (π radians) with some variation
        this.angle = Math.PI + (Math.random() - 0.5) * 0.3;

        // Small random initial angular velocity
        this.angularVelocity = (Math.random() - 0.5) * 0.1;

        return this.getState();
    }

    step(action) {
        // Direct cart velocity control for responsive motion
        const targetVelocity = (action - 1) * 5.0;  // -5, 0, or 5 m/s target velocity

        // Check if we're at a boundary and trying to move further into it
        const atLeftBoundary = this.position <= -this.maxPosition && targetVelocity < 0;
        const atRightBoundary = this.position >= this.maxPosition && targetVelocity > 0;

        let cartAcceleration = 0;

        if (!atLeftBoundary && !atRightBoundary) {
            // Only accelerate if not pushing into a boundary
            cartAcceleration = (targetVelocity - this.velocity) * 25.0;
        } else if ((atLeftBoundary && targetVelocity > 0) || (atRightBoundary && targetVelocity < 0)) {
            // Allow moving away from boundary
            cartAcceleration = (targetVelocity - this.velocity) * 25.0;
        } else {
            // At boundary and trying to push into it - stop completely
            this.velocity = 0;
        }

        // Update cart position and velocity
        this.velocity += cartAcceleration * this.dt;
        this.velocity = Math.max(-6, Math.min(6, this.velocity));  // Limit max velocity
        this.position += this.velocity * this.dt;

        // Hard stop at rail boundaries
        if (this.position < -this.maxPosition) {
            this.position = -this.maxPosition;
            this.velocity = Math.max(0, this.velocity);  // Only allow positive velocity
        } else if (this.position > this.maxPosition) {
            this.position = this.maxPosition;
            this.velocity = Math.min(0, this.velocity);  // Only allow negative velocity
        }

        // Realistic pendulum physics
        const sinTheta = Math.sin(this.angle);
        const cosTheta = Math.cos(this.angle);

        // Torque from gravity acting on center of mass
        const gravityTorque = this.totalMass * this.gravity * this.centerOfMass * sinTheta;

        // Torque from cart acceleration (pseudo-force in accelerating frame)
        const accelerationTorque = -this.totalMass * this.centerOfMass * cartAcceleration * cosTheta;

        // Total torque
        const totalTorque = gravityTorque + accelerationTorque;

        // Angular acceleration = torque / moment of inertia
        const angularAcceleration = totalTorque / this.momentOfInertia;

        // Apply damping (proportional to angular velocity)
        const dampingAcceleration = -this.angularDamping * this.angularVelocity;
        const totalAngularAcceleration = angularAcceleration + dampingAcceleration;

        // Update angle and angular velocity
        this.angularVelocity += totalAngularAcceleration * this.dt;
        this.angle += this.angularVelocity * this.dt;

        // Normalize angle to [-π, π]
        while (this.angle > Math.PI) {
            this.angle -= 2 * Math.PI;
        }
        while (this.angle < -Math.PI) {
            this.angle += 2 * Math.PI;
        }

        // Continuous environment
        const done = false;

        // Reward structure: cosine gives 1 when upright, -1 when down
        const uprightReward = Math.cos(this.angle);

        // Velocity penalty
        const velocityPenalty = 0.001 * (this.velocity ** 2 + this.angularVelocity ** 2);

        const reward = uprightReward - velocityPenalty;

        return [this.getState(), reward, done];
    }

    getState() {
        return [this.position, this.velocity, this.angle, this.angularVelocity];
    }
}