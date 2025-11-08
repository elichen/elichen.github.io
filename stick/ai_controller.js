/**
 * AI Controller for Stick Balancing using TensorFlow.js
 * Loads a trained PPO model and performs inference
 */

class AIController {
    constructor() {
        this.model = null;
        this.normalizationParams = null;
        this.isLoaded = false;
        this.isEnabled = false;
    }

    /**
     * Load the TensorFlow.js model and normalization parameters
     * @param {string} modelPath - Path to the model.json file
     * @param {string} normPath - Path to the normalization parameters JSON
     */
    async loadModel(modelPath = './models/tfjs/model.json', normPath = './models/tfjs/normalization.json') {
        try {
            console.log('Loading AI model...');

            // Load the TensorFlow.js model (layers-model format)
            this.model = await tf.loadLayersModel(modelPath);
            console.log('Model loaded successfully');

            // Load normalization parameters
            const response = await fetch(normPath);
            this.normalizationParams = await response.json();
            console.log('Normalization parameters loaded:', this.normalizationParams);

            this.isLoaded = true;
            return true;
        } catch (error) {
            console.error('Error loading model:', error);
            this.isLoaded = false;
            return false;
        }
    }

    /**
     * Normalize observations using the saved statistics from training
     * @param {Array} observation - Raw observation [position, velocity, angle, angular_velocity]
     * @returns {tf.Tensor} Normalized observation tensor
     */
    normalizeObservation(observation) {
        if (!this.normalizationParams) {
            // If no normalization params, return as-is
            return tf.tensor2d([observation], [1, 4]);
        }

        // Apply normalization: (obs - mean) / std
        const mean = tf.tensor1d(this.normalizationParams.mean);
        const std = tf.tensor1d(this.normalizationParams.std);

        const obsTensor = tf.tensor2d([observation], [1, 4]);
        const normalized = obsTensor.sub(mean).div(std);

        // Clean up intermediate tensors
        mean.dispose();
        std.dispose();

        return normalized;
    }

    /**
     * Get action from the model given the current state
     * @param {Array} observation - Current state [position, velocity, angle, angular_velocity]
     * @returns {number} Continuous action value [-1, 1]
     */
    async getAction(observation) {
        if (!this.isLoaded || !this.isEnabled) {
            return 0.0; // Return "stop" action if not ready
        }

        try {
            // Normalize the observation
            const normalizedObs = this.normalizeObservation(observation);

            // Run inference to get continuous action
            const prediction = this.model.predict(normalizedObs);
            const actionArray = await prediction.array();

            // Clean up tensors
            normalizedObs.dispose();
            prediction.dispose();

            // For continuous PPO, the model outputs mean and log_std
            // We use the mean as the deterministic action
            // The output shape should be [1, 1] for a single continuous action
            const action = actionArray[0][0];

            // Clip action to [-1, 1] range for safety
            return Math.max(-1, Math.min(1, action));
        } catch (error) {
            console.error('Error during inference:', error);
            return 0.0; // Return "stop" action on error
        }
    }

    /**
     * Enable or disable AI control
     * @param {boolean} enabled - Whether to enable AI control
     */
    setEnabled(enabled) {
        if (!this.isLoaded && enabled) {
            console.warn('Cannot enable AI: Model not loaded');
            return false;
        }
        this.isEnabled = enabled;
        return true;
    }

    /**
     * Check if AI is ready and enabled
     * @returns {boolean} Whether AI is active
     */
    isActive() {
        return this.isLoaded && this.isEnabled;
    }

    /**
     * Get status information
     * @returns {Object} Status object
     */
    getStatus() {
        return {
            loaded: this.isLoaded,
            enabled: this.isEnabled,
            active: this.isActive()
        };
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AIController;
}