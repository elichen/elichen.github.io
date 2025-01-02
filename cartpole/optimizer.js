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

        // Store parameters in a group like PyTorch
        this.paramGroup = {
            lr: learningRate,
            gamma: gamma,
            lambda: lambda,
            kappa: kappa,
            params: this.params
        };
        
        // Initialize eligibility traces as Variables
        this.eligibilityTraces = this.params.map(param => 
            tf.variable(tf.zeros(param.shape))
        );
    }

    async step(delta, grads, reset = false) {
        const gradArray = Array.isArray(grads) ? grads : Object.values(grads);

        return tf.tidy(() => {
            let zSum = 0;

            // Cache the gamma * lambda scalar to avoid creating it multiple times
            const gammaLambda = tf.scalar(this.paramGroup.gamma * this.paramGroup.lambda);

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
                
                // Update trace in place: e.mul_(gamma * lambda).add_(grad, alpha=1.0)
                trace.assign(
                    trace.mul(gammaLambda).add(gradTensor)
                );

                zSum += tf.sum(tf.abs(trace)).dataSync()[0];
            }

            const deltaBar = Math.max(Math.abs(delta), 1.0);
            const dotProduct = deltaBar * zSum * this.paramGroup.lr * this.paramGroup.kappa;
            const stepSize = dotProduct > 1 ? this.paramGroup.lr / dotProduct : this.paramGroup.lr;

            // Cache delta scalar to avoid creating it multiple times
            const deltaScalar = tf.scalar(delta);
            const stepSizeScalar = tf.scalar(-stepSize);

            // Update parameters: p.data.add_(delta * e, alpha=-step_size)
            for (let i = 0; i < this.params.length; i++) {
                const param = this.params[i];
                const trace = this.eligibilityTraces[i];

                if (!param || !trace) continue;

                param.assign(
                    param.add(trace.mul(deltaScalar).mul(stepSizeScalar))
                );
            }

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