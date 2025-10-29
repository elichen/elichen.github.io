/**
 * Racing AI Agent using TensorFlow.js
 * Loads PPO model weights and provides inference for the racing game
 */

class RacerAgent {
    constructor() {
        this.model = null;
        this.architecture = null;
        this.weights = null;
        this.ready = false;

        // Ray sensor configuration (must match Python environment)
        this.numRays = 9;
        this.maxRayDistance = 300;
        this.rayAngles = [];
        for (let i = 0; i < this.numRays; i++) {
            this.rayAngles.push(-90 + (180 * i / (this.numRays - 1)));
        }
    }

    /**
     * Load model weights from JSON file
     */
    async loadModel(weightsPath = 'models/ppo_weights.json') {
        try {
            console.log('🤖 Loading racing AI model...');

            // Fetch weights JSON
            const response = await fetch(weightsPath);
            const data = await response.json();

            this.architecture = data.architecture;
            this.weights = data.weights;

            // Build TensorFlow.js model
            await this.buildModel();

            this.ready = true;
            console.log('✅ Racing AI model loaded successfully!');
            console.log(`   Input size: ${this.architecture.input_size}`);
            console.log(`   Output size: ${this.architecture.output_size}`);

        } catch (error) {
            console.error('❌ Failed to load model:', error);
            throw error;
        }
    }

    /**
     * Build TensorFlow.js model from architecture and weights
     */
    async buildModel() {
        const input = tf.input({shape: [this.architecture.input_size]});
        let x = input;

        // Build features extractor layers
        let feLayer = 0;
        for (const layerInfo of this.architecture.features_extractor_layers) {
            if (layerInfo.type === 'linear') {
                const weightKey = `features_extractor_${feLayer}_weight`;
                const biasKey = `features_extractor_${feLayer}_bias`;

                const weights = tf.tensor2d(this.weights[weightKey]);
                const bias = tf.tensor1d(this.weights[biasKey]);

                x = tf.layers.dense({
                    units: layerInfo.out_features,
                    weights: [weights, bias],
                    trainable: false
                }).apply(x);

                feLayer++;
            } else if (layerInfo.type === 'activation' && layerInfo.activation === 'relu') {
                x = tf.layers.reLU().apply(x);
            }
        }

        // Build policy layers
        let policyLayer = 0;
        for (const layerInfo of this.architecture.policy_layers) {
            if (layerInfo.type === 'linear') {
                const weightKey = `policy_${policyLayer}_weight`;
                const biasKey = `policy_${policyLayer}_bias`;

                const weights = tf.tensor2d(this.weights[weightKey]);
                const bias = tf.tensor1d(this.weights[biasKey]);

                x = tf.layers.dense({
                    units: layerInfo.out_features,
                    weights: [weights, bias],
                    trainable: false
                }).apply(x);

                policyLayer++;
            } else if (layerInfo.type === 'activation' && layerInfo.activation === 'relu') {
                x = tf.layers.reLU().apply(x);
            }
        }

        // Action layer
        if (this.architecture.action_layer) {
            const weights = tf.tensor2d(this.weights.action_weight);
            const bias = tf.tensor1d(this.weights.action_bias);

            x = tf.layers.dense({
                units: this.architecture.action_layer.out_features,
                weights: [weights, bias],
                activation: 'tanh',  // Bound actions to [-1, 1]
                trainable: false
            }).apply(x);
        }

        // Create model
        this.model = tf.model({inputs: input, outputs: x});
    }

    /**
     * Cast a ray from the car to detect track boundaries
     */
    castRay(car, track, angleOffset) {
        const rayAngle = car.angle + (angleOffset * Math.PI / 180);
        const dx = Math.cos(rayAngle);
        const dy = Math.sin(rayAngle);

        let distance = 0;
        const stepSize = 5;
        const maxSteps = Math.floor(this.maxRayDistance / stepSize);

        for (let i = 0; i < maxSteps; i++) {
            distance = i * stepSize;
            const rayX = car.x + dx * distance;
            const rayY = car.y + dy * distance;

            if (!track.isPointInsideTrack(rayX, rayY)) {
                break;
            }
        }

        return distance;
    }

    /**
     * Get observation vector from current game state
     */
    getObservation(car, track) {
        const observations = [];

        // Cast rays and normalize distances
        for (const angle of this.rayAngles) {
            const distance = this.castRay(car, track, angle);
            const normalizedDistance = distance / this.maxRayDistance;
            observations.push(normalizedDistance);
        }

        // Add normalized speed
        const normalizedSpeed = car.speed / car.maxSpeed;
        observations.push(Math.max(-1, Math.min(1, normalizedSpeed)));

        // Add angular velocity (approximate from last frame)
        const angularVelocity = car.angularVelocity || 0;
        observations.push(Math.max(-1, Math.min(1, angularVelocity)));

        return observations;
    }

    /**
     * Predict action from current state
     */
    async predict(car, track) {
        if (!this.ready || !this.model) {
            return null;
        }

        // Get observation
        const observation = this.getObservation(car, track);

        // Convert to tensor and predict
        const inputTensor = tf.tensor2d([observation]);
        const outputTensor = this.model.predict(inputTensor);
        const action = await outputTensor.array();

        // Clean up tensors
        inputTensor.dispose();
        outputTensor.dispose();

        // Return steering and throttle
        return {
            steering: action[0][0],  // -1 to 1
            throttle: action[0][1]   // -1 to 1
        };
    }

    /**
     * Draw ray sensors for debugging
     */
    drawRays(ctx, car, track) {
        if (!this.ready) return;

        ctx.save();

        for (const angle of this.rayAngles) {
            const distance = this.castRay(car, track, angle);
            const rayAngle = car.angle + (angle * Math.PI / 180);

            const endX = car.x + Math.cos(rayAngle) * distance;
            const endY = car.y + Math.sin(rayAngle) * distance;

            // Draw ray line
            ctx.strokeStyle = distance < this.maxRayDistance * 0.3 ? 'red' :
                            distance < this.maxRayDistance * 0.6 ? 'yellow' : 'green';
            ctx.lineWidth = 1;
            ctx.globalAlpha = 0.3;

            ctx.beginPath();
            ctx.moveTo(car.x, car.y);
            ctx.lineTo(endX, endY);
            ctx.stroke();

            // Draw hit point
            if (distance < this.maxRayDistance) {
                ctx.fillStyle = 'red';
                ctx.globalAlpha = 0.8;
                ctx.beginPath();
                ctx.arc(endX, endY, 3, 0, Math.PI * 2);
                ctx.fill();
            }
        }

        ctx.restore();
    }

    /**
     * Apply AI action to car controls
     */
    applyAction(action, keys) {
        if (!action) return;

        // Clear all keys first
        keys.ArrowUp = false;
        keys.ArrowDown = false;
        keys.ArrowLeft = false;
        keys.ArrowRight = false;

        // Apply throttle/brake
        if (action.throttle > 0.1) {
            keys.ArrowUp = true;
        } else if (action.throttle < -0.1) {
            keys.ArrowDown = true;
        }

        // Apply steering
        if (action.steering < -0.1) {
            keys.ArrowLeft = true;
        } else if (action.steering > 0.1) {
            keys.ArrowRight = true;
        }
    }
}

// Export for use in game
if (typeof module !== 'undefined' && module.exports) {
    module.exports = RacerAgent;
}