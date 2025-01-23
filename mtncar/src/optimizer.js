class ObGD {
    constructor(params, learningRate = 1.0, gamma = 0.99, lambda = 0.8, kappa = 2.0) {
        this.lr = learningRate;
        this.gamma = gamma;
        this.lambda = lambda;
        this.kappa = kappa;
        this.traces = new Map();
        this.params = params;
        this.lastStats = null;
        
        // Initialize eligibility traces for each parameter
        params.forEach((param, index) => {
            if (!param.name) return;
            const trace = tf.variable(tf.zeros(param.shape));
            this.traces.set(param.name, trace);
        });
    }

    async step(delta, grads, reset) {
        // First pass: update traces and compute z_sum
        let z_sum = 0.0;
        const gammaLambda = this.gamma * this.lambda;
        
        // Collect gradient stats
        const gradStats = [];
        const tensorsToDispose = [];
        
        grads.forEach((grad, index) => {
            if (!grad) return;
            const param = this.params[index];
            if (!param || !param.name) return;
            
            const e = this.traces.get(param.name);
            if (!e) return;
            
            // Update trace: e = γλe + grad
            const newTrace = e.mul(gammaLambda).add(grad);
            tensorsToDispose.push(newTrace);
            e.assign(newTrace);
            
            // Add to z_sum
            const traceSum = e.abs().sum().dataSync()[0];
            z_sum += traceSum;

            // Collect gradient statistics
            const data = grad.dataSync();
            const nonZeros = data.filter(x => x !== 0);
            const mean = nonZeros.length > 0 ? 
                nonZeros.reduce((a, b) => a + b, 0) / nonZeros.length : 0;
            const max = Math.max(...data);
            const min = Math.min(...data);
            const zeroCount = data.length - nonZeros.length;
            
            gradStats.push({
                name: param.name,
                mean,
                max,
                min,
                zeroCount,
                total: data.length
            });
        });

        // Compute step size
        const deltaBar = Math.max(Math.abs(delta), 1.0);
        const dotProduct = deltaBar * z_sum * this.lr * this.kappa;
        const stepSize = dotProduct > 1 ? this.lr / dotProduct : this.lr;

        // Store stats
        this.lastStats = {
            gradients: gradStats,
            obgd: {
                delta,
                deltaBar,
                zSum: z_sum,
                dotProduct,
                stepSize
            }
        };

        // Second pass: update parameters
        grads.forEach((grad, index) => {
            if (!grad) return;
            const param = this.params[index];
            if (!param || !param.name) return;
            
            const e = this.traces.get(param.name);
            if (!e) return;
            
            // Update parameter: w = w - αδe
            const update = e.mul(-stepSize * delta);
            tensorsToDispose.push(update);
            const currentValue = tf.variable(param.read());
            tensorsToDispose.push(currentValue);
            const newValue = currentValue.add(update);
            tensorsToDispose.push(newValue);
            param.write(newValue);
            
            if (reset) {
                e.assign(tf.zeros(e.shape));
            }
        });

        // Cleanup tensors
        tensorsToDispose.forEach(tensor => tensor.dispose());
    }

    getLastStats() {
        if (!this.lastStats) return '';
        
        const gradientText = this.lastStats.gradients.map(g => 
            `Layer: ${g.name}
  Mean (non-zero): ${g.mean.toFixed(6)}
  Max: ${g.max.toFixed(6)}
  Min: ${g.min.toFixed(6)}
  Zero grads: ${g.zeroCount}/${g.total} (${(g.zeroCount/g.total*100).toFixed(2)}%)`
        ).join('\n\n');

        const obgdText = `ObGD Stats:
Delta: ${this.lastStats.obgd.delta.toFixed(6)}
Delta Bar: ${this.lastStats.obgd.deltaBar.toFixed(6)}
Z Sum: ${this.lastStats.obgd.zSum.toFixed(6)}
Dot Product: ${this.lastStats.obgd.dotProduct.toFixed(6)}
Step Size: ${this.lastStats.obgd.stepSize.toFixed(6)}`;

        return `Gradient Flow Stats:\n${gradientText}\n\n${obgdText}`;
    }

    dispose() {
        // Clean up all traces
        for (const trace of this.traces.values()) {
            trace.dispose();
        }
        this.traces.clear();
    }
} 