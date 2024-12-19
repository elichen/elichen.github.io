class ObGD {
    constructor(params, learningRate = 1.0, gamma = 0.99, lambda = 0.8, kappa = 2.0) {
        // Convert params to Variables if they aren't already
        this.params = params.map(param => {
            if (param instanceof tf.Variable) {
                return param;
            }
            const value = param.val || param.value || param;
            return tf.variable(value);
        });

        this.lr = learningRate;
        this.gamma = gamma;
        this.lambda = lambda;
        this.kappa = kappa;
        
        // Initialize eligibility traces as Variables
        this.eligibilityTraces = this.params.map(param => 
            tf.variable(tf.zeros(param.shape))
        );
    }

    async step(delta, grads, reset = false) {
        const gradArray = Array.isArray(grads) ? grads : Object.values(grads);

        return tf.tidy(() => {
            let zSum = 0;

            // Update eligibility traces and compute zSum
            for (let i = 0; i < this.eligibilityTraces.length; i++) {
                const trace = this.eligibilityTraces[i];
                const grad = gradArray[i];
                const param = this.params[i];

                if (!grad || !trace || !param) continue;

                // Create gradient tensor
                const gradTensor = tf.tensor(
                    grad instanceof tf.Tensor ? grad.dataSync() : grad,
                    param.shape
                );

                // Update trace
                const decayFactor = tf.scalar(this.gamma * this.lambda);
                const newTrace = trace.mul(decayFactor).add(gradTensor);
                trace.assign(newTrace);

                // Update zSum
                zSum += tf.sum(tf.abs(trace)).dataSync()[0];
            }

            // Compute step size
            const deltaBar = Math.max(Math.abs(delta), 1.0);
            const dotProduct = deltaBar * zSum * this.lr * this.kappa;
            const stepSize = dotProduct > 1 ? this.lr / dotProduct : this.lr;
            const updateFactor = -stepSize * delta;

            // Update parameters
            for (let i = 0; i < this.params.length; i++) {
                const param = this.params[i];
                const trace = this.eligibilityTraces[i];

                if (!param || !trace) continue;

                const update = trace.mul(tf.scalar(updateFactor));
                
                if (tf.util.arraysEqual(param.shape, update.shape)) {
                    param.assign(param.add(update));
                }
            }

            // Reset eligibility traces if needed
            if (reset) {
                this.eligibilityTraces.forEach(trace => {
                    trace.assign(tf.zeros(trace.shape));
                });
            }
        });
    }

    dispose() {
        this.eligibilityTraces.forEach(trace => {
            if (trace instanceof tf.Variable) {
                trace.dispose();
            }
        });
    }
} 