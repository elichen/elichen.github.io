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
        this.maxAngle = 1  // About 57.3 degrees
        
        // Angular friction
        this.angularFriction = 0.05;

        // Parameters for restart condition
        this.maxStepsDown = 20;
        this.stepsDown = 0; // Counter for steps the stick has been down

        this.previousAngle = 0;  // Add this to track previous angle
        this.reset();
    }

    reset() {
        // Randomize position and velocity with small values
        this.position = (Math.random() - 0.5) * 0.1;
        this.velocity = (Math.random() - 0.5) * 0.1;
        
        // Increase the range of starting angles
        // Start with angles up to ±30 degrees (±0.52 radians)
        const maxStartAngle = 0.52; // about 30 degrees
        this.angle = Math.random() * maxStartAngle * (Math.random() < 0.5 ? -1 : 1);
        
        // Small random initial angular velocity
        this.angularVelocity = (Math.random() - 0.5) * 0.1;
        
        this.stepsDown = 0; // Reset the counter
        this.previousAngle = this.angle;  // Initialize previous angle
        return this.getState();
    }

    step(action) {
        // Store previous angle before updating
        this.previousAngle = this.angle;

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
        
        // New reward structure based on improvement
        let reward;
        if (done) {
            reward = -10;  // Stronger penalty for failure
        } else {
            // Combine improvement reward with upright position reward
            const previousAbsAngle = Math.abs(this.previousAngle);
            const currentAbsAngle = Math.abs(this.angle);
            const improvement = previousAbsAngle - currentAbsAngle;
            
            // Base reward for being upright (1.0 when vertical, 0 when at maxAngle)
            const uprightReward = 1.0 - (currentAbsAngle / this.maxAngle);
            
            // Combine both rewards
            reward = uprightReward + improvement * 5.0;  // Scale improvement for stronger signal
        }

        return [this.getState(), reward, done];
    }

    getState() {
        return [this.position, this.velocity, this.angle, this.angularVelocity];
    }
}