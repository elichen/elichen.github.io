class ObGD {
    constructor(params, learningRate = 1.0, gamma = 0.99, lambda = 0.8, kappa = 2.0) {
        this.params = params;
        this.lr = learningRate;
        this.gamma = gamma;
        this.lambda = lambda;
        this.kappa = kappa;
        
        // Initialize eligibility traces for each parameter
        this.eligibilityTraces = params.map(param => tf.zeros(param.shape));
    }

    async step(delta, grads, reset = false) {
        return tf.tidy(() => {
            let zSum = tf.scalar(0);
            
            // Update eligibility traces and compute zSum
            this.eligibilityTraces = this.eligibilityTraces.map((trace, i) => {
                const newTrace = tf.add(
                    tf.mul(trace, this.gamma * this.lambda),
                    grads[i]
                );
                zSum = tf.add(zSum, tf.sum(tf.abs(newTrace)));
                return newTrace;
            });

            // Compute step size
            const deltaBar = Math.max(Math.abs(delta), 1.0);
            const dotProduct = deltaBar * zSum.dataSync()[0] * this.lr * this.kappa;
            const stepSize = dotProduct > 1 ? this.lr / dotProduct : this.lr;

            // Update parameters
            this.params.forEach((param, i) => {
                const update = tf.mul(tf.mul(this.eligibilityTraces[i], delta), -stepSize);
                param.assign(tf.add(param, update));
            });

            // Reset eligibility traces if needed
            if (reset) {
                this.eligibilityTraces = this.eligibilityTraces.map(trace => 
                    tf.zeros(trace.shape)
                );
            }
        });
    }

    dispose() {
        this.eligibilityTraces.forEach(trace => trace.dispose());
    }
} 