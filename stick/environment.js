class StickBalancingEnv {
    constructor() {
        // Environment parameters
        this.gravity = 9.81;
        this.cartMass = 1.0;
        this.poleMass = 0.1;
        this.poleLength = 0.5;
        this.dt = 0.02;
        this.forceStrength = 10.0;
        this.maxPosition = 2.4;
        this.maxAngle = 1
        
        // Angular friction
        this.angularFriction = 0.05;

        // Parameters for restart condition
        this.maxStepsDown = 20;
        this.stepsDown = 0; // Counter for steps the stick has been down

        this.reset();
    }

    reset() {
        this.position = (Math.random() - 0.5) * 0.1;
        this.velocity = (Math.random() - 0.5) * 0.1;
        this.angle = (Math.random() - 0.5) * 0.1;
        this.angularVelocity = (Math.random() - 0.5) * 0.1;
        this.stepsDown = 0; // Reset the counter
        return this.getState();
    }

    step(action) {
        const force = (action - 1) * this.forceStrength;
        
        const cosTheta = Math.cos(this.angle);
        const sinTheta = Math.sin(this.angle);

        const temp = (force + this.poleMass * this.poleLength * this.angularVelocity ** 2 * sinTheta) / (this.cartMass + this.poleMass);
        const angularAcceleration = (this.gravity * sinTheta - cosTheta * temp) / (this.poleLength * (4/3 - this.poleMass * cosTheta ** 2 / (this.cartMass + this.poleMass)));
        const acceleration = temp - this.poleMass * this.poleLength * angularAcceleration * cosTheta / (this.cartMass + this.poleMass);

        // Update position and velocity with boundary checks
        this.position += this.velocity * this.dt;
        this.velocity += acceleration * this.dt;

        // Limit position within boundaries
        if (Math.abs(this.position) > this.maxPosition) {
            this.position = Math.sign(this.position) * this.maxPosition;
            this.velocity = 0; // Stop the cart at the boundary
        }

        // Apply angular friction
        const frictionTorque = -this.angularFriction * this.angularVelocity;
        const totalAngularAcceleration = angularAcceleration + frictionTorque / (this.poleMass * this.poleLength ** 2 / 3);

        this.angle += this.angularVelocity * this.dt;
        this.angularVelocity += totalAngularAcceleration * this.dt;

        // Check if the stick is down and update the counter
        if (Math.abs(this.angle) > this.maxAngle) {
            this.stepsDown++;
        } else {
            this.stepsDown = 0; // Reset the counter if the stick is upright
        }

        const done = this.stepsDown >= this.maxStepsDown;
        
        // New reward structure
        let reward;
        if (done) {
            reward = -10;  // Stronger penalty for failure
        } else {
            // Base reward for staying alive, plus bonus for being upright
            reward = 1.0 - Math.abs(this.angle) / this.maxAngle;
        }

        return [this.getState(), reward, done];
    }

    getState() {
        return [this.position, this.velocity, this.angle, this.angularVelocity];
    }
}