// Simple network for testing gradient flow
class TestNetwork {
    constructor() {
        this.model = tf.sequential();
        
        // First layer with layer normalization (match main app exactly)
        this.model.add(tf.layers.dense({
            units: 32,
            inputShape: [5],  // Match main app input size after AddTimeInfo
            activation: 'linear',
            name: 'fc1',
            trainable: true
        }));
        this.model.add(tf.layers.layerNormalization({
            axis: [1],
            epsilon: 1e-5,
            center: true,
            scale: true,
            beta_initializer: 'zeros',
            gamma_initializer: 'ones',
            trainable: true,
            name: 'layer_norm1'
        }));
        this.model.add(tf.layers.leakyReLU({
            alpha: 0.01,
            name: 'leaky_relu1'
        }));

        // Hidden layer with layer normalization (match main app exactly)
        this.model.add(tf.layers.dense({
            units: 32,
            activation: 'linear',
            name: 'hidden',
            trainable: true
        }));
        this.model.add(tf.layers.layerNormalization({
            axis: [1],
            epsilon: 1e-5,
            center: true,
            scale: true,
            beta_initializer: 'zeros',
            gamma_initializer: 'ones',
            trainable: true,
            name: 'layer_norm2'
        }));
        this.model.add(tf.layers.leakyReLU({
            alpha: 0.01,
            name: 'leaky_relu2'
        }));

        // Output layer
        this.model.add(tf.layers.dense({
            units: 2,  // Match main app output size
            activation: 'linear',
            name: 'output',
            trainable: true
        }));
    }

    async initialize(sparsity) {
        // Apply sparse initialization with 90% sparsity (match original)
        await this.sparseInit(sparsity);
    }

    async sparseInit(sparsity) {
        // Implement sparse initialization for each dense layer
        const layers = this.model.layers.filter(layer => layer.getClassName() === 'Dense');
        
        for (const layer of layers) {
            const weights = layer.getWeights();
            const w = weights[0];
            const shape = w.shape;
            const [fanOut, fanIn] = shape;
            
            // Create new weights with LeCun initialization
            const newWeights = tf.tidy(() => {
                // LeCun initialization: U[-1/√fan_in, 1/√fan_in]
                const weights = tf.randomUniform(shape, -1.0/Math.sqrt(fanIn), 1.0/Math.sqrt(fanIn));
                
                // Create sparse mask per input neuron (not per output)
                const numZerosPerInput = Math.ceil(sparsity * fanOut);
                const mask = tf.buffer(shape);
                
                // Fill mask with ones initially
                for (let i = 0; i < fanOut; i++) {
                    for (let j = 0; j < fanIn; j++) {
                        mask.set(1, i, j);
                    }
                }
                
                // Zero out random outputs for each input independently
                for (let inIdx = 0; inIdx < fanIn; inIdx++) {
                    const indices = Array.from({length: fanOut}, (_, i) => i);
                    for (let i = indices.length - 1; i > 0; i--) {
                        const j = Math.floor(Math.random() * (i + 1));
                        [indices[i], indices[j]] = [indices[j], indices[i]];
                    }
                    const zeroIndices = indices.slice(0, numZerosPerInput);
                    for (const idx of zeroIndices) {
                        mask.set(0, idx, inIdx);
                    }
                }
                
                // Apply mask to weights
                return tf.mul(weights, mask.toTensor());
            });

            // Set the new weights
            await layer.setWeights([newWeights, weights[1]]);
            newWeights.dispose();
        }
    }

    getTrainableVariables() {
        return this.model.trainableWeights;
    }

    dispose() {
        if (this.model) {
            this.model.dispose();
            this.model = null;
        }
    }
}

// Simple eligibility trace test (match main app parameters)
class TestOptimizer {
    constructor(params, gamma = 0.99, lambda = 0.8, learningRate = 0.01, kappa = 1.0) {
        this.params = params.map(param => {
            if (param instanceof tf.Variable) {
                return param;
            }
            return tf.variable(param.val || param.value || param);
        });
        
        this.paramNames = params.map(p => p.name);
        this.gamma = gamma;
        this.lambda = lambda;
        this.learningRate = learningRate;
        this.kappa = kappa;
        
        this.traces = this.params.map(param => 
            tf.variable(tf.zeros(param.shape))
        );
        
        // Initialize deltaBar to 1.0 to match main app
        this.deltaBar = 1.0;
    }
    
    step(delta, grads, reset = false) {
        const gammaLambda = this.gamma * this.lambda;
        let zSum = 0;

        // Log gradient statistics
        console.log('\nGradient Stats:');
        for (const [name, grad] of Object.entries(grads)) {
            if (!grad) continue;
            
            const data = grad.dataSync();
            const nonZeros = data.filter(x => x !== 0);
            const mean = nonZeros.length > 0 ? 
                nonZeros.reduce((a, b) => a + b, 0) / nonZeros.length : 0;
            const max = Math.max(...data);
            const min = Math.min(...data);
            const zeroCount = data.length - nonZeros.length;
            
            console.log(`Layer: ${name}
  Mean (non-zero): ${mean.toFixed(6)}
  Max: ${max.toFixed(6)}
  Min: ${min.toFixed(6)}
  Zero grads: ${zeroCount}/${data.length} (${(zeroCount/data.length*100).toFixed(2)}%)`);
        }

        // First pass: update traces and compute zSum
        for (let i = 0; i < this.traces.length; i++) {
            const trace = this.traces[i];
            const paramName = this.paramNames[i];
            const grad = grads[paramName];
            
            if (!grad || !trace) continue;
            
            // Update trace: e = γλe + grad
            const newTrace = trace.mul(tf.scalar(gammaLambda)).add(grad);
            trace.assign(newTrace);
            
            // Compute zSum
            const traceData = newTrace.dataSync();
            zSum += traceData.reduce((sum, val) => sum + Math.abs(val), 0);
            
            newTrace.dispose();
        }

        // Compute step size
        const deltaBar = Math.max(Math.abs(delta), 1.0);
        const dotProduct = deltaBar * zSum * this.learningRate * this.kappa;
        const stepSize = dotProduct > 1 ? this.learningRate / dotProduct : this.learningRate;

        // Log trace statistics
        console.log('\nTrace Stats:');
        for (let i = 0; i < this.traces.length; i++) {
            const trace = this.traces[i];
            const paramName = this.paramNames[i];
            
            if (!trace) continue;
            
            const traceData = trace.dataSync();
            const nonZeros = traceData.filter(x => x !== 0);
            const mean = nonZeros.length > 0 ? 
                nonZeros.reduce((a, b) => a + b, 0) / nonZeros.length : 0;
            
            console.log(`Trace for ${paramName}:
  Mean (non-zero): ${mean.toFixed(6)}
  Non-zero elements: ${nonZeros.length}/${traceData.length}
  Step size: ${stepSize.toFixed(6)}`);
        }

        // Log ObGD stats
        console.log('\nObGD Debug:',
            '\nDelta:', delta,
            '\nDelta Bar:', deltaBar,
            '\nZ Sum:', zSum,
            '\nDot Product:', dotProduct,
            '\nStep Size:', stepSize);

        // Second pass: update parameters
        for (let i = 0; i < this.traces.length; i++) {
            const trace = this.traces[i];
            const param = this.params[i];
            
            if (!trace || !param) continue;
            
            // Update parameter: w = w - stepSize * delta * e
            const update = trace.mul(tf.scalar(stepSize * delta));
            const newParam = param.sub(update);
            param.assign(newParam);
            
            update.dispose();
            newParam.dispose();
        }

        if (reset) {
            this.traces.forEach(trace => {
                trace.assign(tf.zeros(trace.shape));
            });
        }
    }
    
    dispose() {
        this.traces.forEach(trace => trace.dispose());
        this.params.forEach(param => {
            if (param instanceof tf.Variable) {
                param.dispose();
            }
        });
    }
}

// Log gradient statistics
async function logGradStats(grads) {
    const results = [];
    
    for (const [name, grad] of Object.entries(grads)) {
        if (!grad) continue;
        
        const data = await grad.data();
        const nonZeros = data.filter(x => x !== 0);
        const mean = nonZeros.length > 0 ? 
            nonZeros.reduce((a, b) => a + b, 0) / nonZeros.length : 0;
        const max = Math.max(...data);
        const min = Math.min(...data);
        const zeroCount = data.length - nonZeros.length;
        
        results.push(`Layer: ${name}
  Mean (non-zero): ${mean.toFixed(6)}
  Max: ${max.toFixed(6)}
  Min: ${min.toFixed(6)}
  Zero grads: ${zeroCount}/${data.length} (${(zeroCount/data.length*100).toFixed(2)}%)\n`);
    }
    
    return results.join('\n');
}

// Test function to simulate main app environment and update loop
async function runTest() {
    // Try higher sparsity levels up to 90%
    const sparsityLevels = [0.75, 0.8, 0.85, 0.9];
    
    for (const sparsity of sparsityLevels) {
        console.log(`\n=== Testing with ${(sparsity*100).toFixed(1)}% sparsity ===`);
        
        const network = new TestNetwork();
        await network.initialize(sparsity);  // Pass sparsity level
        const optimizer = new TestOptimizer(network.getTrainableVariables());
        
        try {
            // Simulate normalized state (after NormalizeObservation wrapper)
            const rawState = [0.5, 0.1, -0.2, 0.3];  // Cart position, velocity, pole angle, pole velocity
            const normalizedState = [-1.2, 0.8, 1.5, -0.9];  // After normalization
            const timeInfo = -0.4;  // Early in episode
            
            const state = tf.tensor2d([[...normalizedState, timeInfo]]);
            const nextState = tf.tensor2d([[...normalizedState.map(x => x * 0.9), timeInfo + 0.1]]);
            
            // Forward pass to get Q-values (match main app exactly)
            const nextQValues = network.model.predict(nextState);
            const maxNextQ = nextQValues.max(1);
            const reward = 0.5;  // Scaled reward
            const gamma = 0.99;
            const done = false;
            const doneMask = done ? 0 : 1;
            const tdTarget = tf.scalar(reward).add(maxNextQ.mul(tf.scalar(gamma * doneMask)));
            
            // Compute gradients (match main app exactly)
            const action = 1;  // Choose action 1 for test
            const {value: qsa, grads} = tf.variableGrads(() => {
                const qValues = network.model.predict(state);
                const actionMask = tf.oneHot([action], 2);  // 2 actions
                return tf.sum(tf.mul(qValues, actionMask));
            });
            
            // Compute TD error (match main app exactly)
            const tdError = tdTarget.sub(qsa);
            const tdErrorValue = tdError.dataSync()[0];
            
            // Update with optimizer
            optimizer.step(tdErrorValue, grads, false);  // Not resetting traces
            
            // Cleanup tensors
            tf.dispose([
                state,
                nextState,
                nextQValues,
                maxNextQ,
                tdTarget,
                qsa,
                tdError,
                ...Object.values(grads).filter(g => g)
            ]);
        } finally {
            optimizer.dispose();
            network.dispose();
        }
    }
}

// Run the test
runTest(); 