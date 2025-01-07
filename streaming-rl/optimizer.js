class ObGD {
    constructor(params, learningRate = 1.0, gamma = 0.99, lambda = 0.8, kappa = 2.0) {
        this.lr = learningRate;
        this.gamma = gamma;
        this.lambda = lambda;
        this.kappa = kappa;
        this.traces = new Map();
        this.params = params;
        
        // Initialize eligibility traces for each parameter
        params.forEach((param, index) => {
            if (!param.name) {
                console.error(`Parameter ${index} has no name`);
                return;
            }
            const trace = tf.variable(tf.zeros(param.shape));
            this.traces.set(param.name, trace);
        });
    }

    async step(delta, grads, reset) {
        // First pass: update traces and compute z_sum
        let z_sum = 0.0;
        const gammaLambda = this.gamma * this.lambda;
        
        grads.forEach((grad, index) => {
            if (!grad) return;
            const param = this.params[index];
            if (!param || !param.name) return;
            
            const e = this.traces.get(param.name);
            if (!e) return;
            
            // Update trace: e = γλe + grad
            const newTrace = e.mul(gammaLambda).add(grad);
            e.assign(newTrace);
            
            // Add to z_sum
            const traceSum = e.abs().sum().dataSync()[0];
            z_sum += traceSum;
        });

        // Compute step size
        const deltaBar = Math.max(Math.abs(delta), 1.0);
        const dotProduct = deltaBar * z_sum * this.lr * this.kappa;
        const stepSize = dotProduct > 1 ? this.lr / dotProduct : this.lr;

        // Second pass: update parameters
        grads.forEach((grad, index) => {
            if (!grad) return;
            const param = this.params[index];
            if (!param || !param.name) return;
            
            const e = this.traces.get(param.name);
            if (!e) return;
            
            // Update parameter: w = w - αδe
            const update = e.mul(-stepSize * delta);
            const currentValue = tf.variable(param.read());
            const newValue = currentValue.add(update);
            param.write(newValue);
            
            if (reset) {
                e.assign(tf.zeros(e.shape));
            }
        });
    }
} 