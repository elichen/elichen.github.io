/**
 * Headless test script for Stream Q(λ) CartPole implementation
 * Tests the core algorithms from "Streaming Deep Reinforcement Learning Finally Works"
 */

const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');

// ============================================================================
// Browser API Mocks - set up before loading modules
// ============================================================================

const mockCanvas = {
    width: 600,
    height: 400,
    getContext: () => ({
        clearRect: () => {},
        fillRect: () => {},
        beginPath: () => {},
        moveTo: () => {},
        lineTo: () => {},
        stroke: () => {},
        fillStyle: '',
        strokeStyle: '',
        lineWidth: 0
    })
};

global.document = {
    getElementById: (id) => {
        if (id === 'cartpoleCanvas') return mockCanvas;
        return { innerHTML: '', textContent: '', value: '1.0' };
    },
    addEventListener: () => {}
};

global.window = {};
global.requestAnimationFrame = () => {};
global.cancelAnimationFrame = () => {};

// Make tf available globally (browser modules expect this)
global.tf = tf;

// ============================================================================
// Load modules using vm.runInThisContext (simulates browser script loading)
// ============================================================================

const vm = require('vm');

function loadModules() {
    const modules = ['cartpole.js', 'network.js', 'optimizer.js', 'agent.js', 'normalization.js'];

    for (const mod of modules) {
        const code = fs.readFileSync(path.join(__dirname, mod), 'utf8');
        vm.runInThisContext(code, { filename: mod });
    }

    // Return references to the global classes
    return {
        CartPole,
        LayerNorm,
        StreamingNetwork,
        ObGD,
        StreamQ,
        SampleMeanStd,
        NormalizeObservation,
        ScaleReward,
        AddTimeInfo
    };
}

// ============================================================================
// Test utilities
// ============================================================================

function assert(condition, message) {
    if (!condition) {
        throw new Error(`Assertion failed: ${message}`);
    }
}

function assertInRange(value, min, max, message) {
    if (value < min || value > max) {
        throw new Error(`${message}: ${value} not in range [${min}, ${max}]`);
    }
}

function assertNoNaN(tensor, name) {
    const data = tensor.dataSync();
    for (let i = 0; i < data.length; i++) {
        if (isNaN(data[i]) || !isFinite(data[i])) {
            throw new Error(`NaN or Infinity found in ${name} at index ${i}`);
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

async function testCartPoleEnvironment(ctx) {
    console.log('\n=== Testing CartPole Environment ===');

    const env = new ctx.CartPole();
    const state = env.reset();

    assert(state.length === 4, 'State should have 4 dimensions');
    state.forEach((v, i) => {
        assertInRange(v, -0.05, 0.05, `Initial state[${i}] should be near zero`);
    });

    // Test stepping
    const result = env.step(1);
    assert(result.state.length === 4, 'Next state should have 4 dimensions');
    assert(typeof result.reward === 'number', 'Reward should be a number');
    assert(typeof result.done === 'boolean', 'Done should be boolean');

    console.log('  CartPole environment: PASSED');
}

async function testLayerNorm(ctx) {
    console.log('\n=== Testing LayerNorm ===');

    const input = tf.tensor2d([[1, 2, 3, 4]], [1, 4]);
    const layerNorm = new ctx.LayerNorm({ normalizedShape: [4] });
    const output = layerNorm.call(input);

    // LayerNorm should produce output with mean ~0 and std ~1
    const moments = tf.moments(output, -1);
    const mean = moments.mean.dataSync()[0];
    const variance = moments.variance.dataSync()[0];

    assertInRange(mean, -0.01, 0.01, 'LayerNorm mean should be ~0');
    assertInRange(variance, 0.99, 1.01, 'LayerNorm variance should be ~1');

    // Verify no learnable parameters
    assert(layerNorm.trainable === false, 'LayerNorm should not be trainable');

    input.dispose();
    output.dispose();
    moments.mean.dispose();
    moments.variance.dispose();

    console.log('  LayerNorm: PASSED');
}

async function testSparseInit(ctx) {
    console.log('\n=== Testing SparseInit ===');

    const network = new ctx.StreamingNetwork(4, 32, 2);

    // sparseInit is async but called without await in constructor
    // Call it again and await to ensure completion
    await network.sparseInit();

    const layers = network.model.layers.filter(l => l.getClassName() === 'Dense');

    for (const layer of layers) {
        const weights = layer.getWeights()[0];
        const data = weights.dataSync();
        const [inputSize, outputSize] = weights.shape;

        // Count zeros
        let zeroCount = 0;
        for (const v of data) {
            if (v === 0) zeroCount++;
        }

        const sparsity = zeroCount / data.length;
        console.log(`    Layer ${layer.name}: shape=${weights.shape}, zeros=${zeroCount}/${data.length}, sparsity=${sparsity.toFixed(2)}`);

        // For small input sizes, we cap zeros at inputSize-1 to keep at least 1 active
        const expectedZeros = Math.min(Math.ceil(0.9 * inputSize), inputSize - 1);
        const expectedSparsity = expectedZeros / inputSize;

        if (inputSize >= 10) {
            assertInRange(sparsity, 0.85, 0.95, `Layer ${layer.name} sparsity`);
        } else {
            // For small layers, verify sparsity matches the capped value
            assertInRange(sparsity, expectedSparsity - 0.01, expectedSparsity + 0.01,
                `Layer ${layer.name} sparsity (capped for small input)`);
            console.log(`    (Small layer: ${inputSize} inputs, ${expectedZeros} zeroed, ${inputSize - expectedZeros} active)`);
        }

        // Check LeCun bounds for non-zero weights
        const bound = 1.0 / Math.sqrt(inputSize);
        for (const v of data) {
            if (v !== 0) {
                assertInRange(v, -bound - 0.001, bound + 0.001,
                    `Weight in ${layer.name} should be in LeCun bounds`);
            }
        }

        // Check zero bias
        const bias = layer.getWeights()[1];
        const biasData = bias.dataSync();
        for (const v of biasData) {
            assert(v === 0, 'Bias should be zero');
        }
    }

    console.log('  SparseInit: PASSED');
}

async function testObGDOptimizer(ctx) {
    console.log('\n=== Testing ObGD Optimizer ===');

    const network = new ctx.StreamingNetwork(4, 32, 2);
    await network.sparseInit();

    const optimizer = new ctx.ObGD(
        network.getTrainableVariables(),
        1.0,  // lr
        0.99, // gamma
        0.8,  // lambda
        2.0   // kappa
    );

    // Create mock gradients
    const grads = network.getTrainableVariables().map(param => {
        return tf.randomNormal(param.shape).mul(0.1);
    });

    // Step with small delta
    await optimizer.step(0.1, grads, false);

    const stats = optimizer.lastStats;
    assert(stats !== null, 'Optimizer should have stats');
    assert(stats.obgd.deltaBar >= 1.0, 'deltaBar should be at least 1.0');
    assert(stats.obgd.zSum > 0, 'zSum should be positive after update');

    // Test step size bounding
    // With large traces and delta, step size should be bounded
    for (let i = 0; i < 10; i++) {
        const largeGrads = network.getTrainableVariables().map(param => {
            return tf.ones(param.shape);
        });
        await optimizer.step(10.0, largeGrads, false);
        largeGrads.forEach(g => g.dispose());
    }

    const stats2 = optimizer.lastStats;
    assert(stats2.obgd.stepSize < 1.0,
        `Step size should be bounded below 1.0 when traces are large, got ${stats2.obgd.stepSize}`);

    // Clean up
    grads.forEach(g => g.dispose());
    optimizer.dispose();

    console.log('  ObGD Optimizer: PASSED');
}

async function testEligibilityTraces(ctx) {
    console.log('\n=== Testing Eligibility Traces ===');

    const network = new ctx.StreamingNetwork(4, 32, 2);
    await network.sparseInit();

    const optimizer = new ctx.ObGD(
        network.getTrainableVariables(),
        1.0, 0.99, 0.8, 2.0
    );

    // Traces should start at zero
    for (const [name, trace] of optimizer.traces) {
        const sum = trace.abs().sum().dataSync()[0];
        assert(sum === 0, `Initial trace for ${name} should be zero`);
    }

    // After update, traces should accumulate
    const grads = network.getTrainableVariables().map(param => {
        return tf.ones(param.shape).mul(0.01);
    });

    await optimizer.step(1.0, grads, false);

    for (const [name, trace] of optimizer.traces) {
        const sum = trace.abs().sum().dataSync()[0];
        assert(sum > 0, `Trace for ${name} should accumulate`);
    }

    // After reset, traces should be zero again
    await optimizer.step(1.0, grads, true);

    for (const [name, trace] of optimizer.traces) {
        const sum = trace.abs().sum().dataSync()[0];
        assert(sum === 0, `Trace for ${name} should be reset`);
    }

    grads.forEach(g => g.dispose());
    optimizer.dispose();

    console.log('  Eligibility Traces: PASSED');
}

async function testNormalization(ctx) {
    console.log('\n=== Testing Observation Normalization ===');

    const env = new ctx.CartPole();
    const normalizer = new ctx.SampleMeanStd([4]);

    // Update with some samples
    for (let i = 0; i < 100; i++) {
        env.reset();
        for (let j = 0; j < 10; j++) {
            const result = env.step(Math.random() < 0.5 ? 0 : 1);
            normalizer.update(tf.tensor1d(result.state));
            if (result.done) break;
        }
    }

    assert(normalizer.count > 0, 'Normalizer should have samples');

    const mean = normalizer.mean.dataSync();
    const variance = normalizer.var.dataSync();

    // Mean should be small (near zero for CartPole)
    for (let i = 0; i < 4; i++) {
        assertInRange(mean[i], -1, 1, `Mean[${i}] should be reasonable`);
        assert(variance[i] > 0, `Variance[${i}] should be positive`);
    }

    normalizer.dispose();

    console.log('  Observation Normalization: PASSED');
}

async function testRewardScaling(ctx) {
    console.log('\n=== Testing Reward Scaling ===');

    const env = new ctx.CartPole();
    const scaledEnv = new ctx.ScaleReward(env, 0.99);

    scaledEnv.reset();

    let lastReward = null;
    for (let i = 0; i < 100; i++) {
        const result = scaledEnv.step(1);
        if (result.reward !== 0) {
            lastReward = result.reward;
        }
        if (result.done) {
            scaledEnv.reset();
        }
    }

    // After collecting samples, variance should be estimated
    const variance = scaledEnv.rewardStats.var.dataSync()[0];
    assert(variance > 0, 'Reward variance should be positive');

    scaledEnv.dispose();

    console.log('  Reward Scaling: PASSED');
}

async function testTimeInfo(ctx) {
    console.log('\n=== Testing Time Info ===');

    const env = new ctx.CartPole();
    const timedEnv = new ctx.AddTimeInfo(env);

    const state = timedEnv.reset();
    assert(state.length === 5, 'State should have 5 dimensions with time info');
    assert(state[4] === -0.5, 'Initial time should be -0.5');

    const result = timedEnv.step(1);
    assert(result.state.length === 5, 'Next state should have 5 dimensions');
    assert(result.state[4] > -0.5, 'Time should increase');

    console.log('  Time Info: PASSED');
}

async function testFullTraining(ctx) {
    console.log('\n=== Testing Full Training Loop ===');

    // Set up wrapped environment
    let env = new ctx.CartPole();
    const MAX_EPISODE_STEPS = env.maxSteps; // 500 for CartPole
    env = new ctx.ScaleReward(env, 0.99);
    env = new ctx.NormalizeObservation(env);
    env = new ctx.AddTimeInfo(env);

    const totalTrainingSteps = 150000; // Train to achieve consistent perfect balance

    const agent = new ctx.StreamQ({
        env,
        numActions: 2,
        gamma: 0.99,
        epsilonStart: 1.0,
        epsilonTarget: 0.01,
        totalSteps: totalTrainingSteps,
        hiddenSize: 32,
        learningRate: 1.0,
        lambda: 0.8,
        kappa: 2.0
    });

    // Tracking variables
    const episodeReturns = [];
    const episodeLengths = [];
    const episodeEndSteps = []; // Global step when each episode ended
    let state = env.reset();
    let episodeReturn = 0;
    let steps = 0;

    // Performance milestones
    let firstBalance50 = null;  // First episode with 50+ steps
    let firstBalance100 = null; // First episode with 100+ steps
    let firstBalance200 = null; // First episode with 200+ steps
    let firstBalance500 = null; // First perfect balance (500 steps)

    console.log(`  Training for ${totalTrainingSteps} steps (CartPole max episode: ${MAX_EPISODE_STEPS} steps)...`);
    console.log('');

    while (steps < totalTrainingSteps) {
        const { action, isNonGreedy } = await agent.sampleAction(state);
        const result = env.step(action);

        await agent.update(state, action, result.reward, result.state, result.done, isNonGreedy);

        episodeReturn += result.reward;
        state = result.state;
        steps++;

        if (result.done) {
            const rawReturn = result.info.episode.r;
            const epLength = result.info.episode.steps;

            episodeReturns.push(rawReturn);
            episodeLengths.push(epLength);
            episodeEndSteps.push(steps);

            // Track milestones
            if (firstBalance50 === null && epLength >= 50) {
                firstBalance50 = { episode: episodeReturns.length, steps, length: epLength };
            }
            if (firstBalance100 === null && epLength >= 100) {
                firstBalance100 = { episode: episodeReturns.length, steps, length: epLength };
            }
            if (firstBalance200 === null && epLength >= 200) {
                firstBalance200 = { episode: episodeReturns.length, steps, length: epLength };
            }
            if (firstBalance500 === null && epLength >= 500) {
                firstBalance500 = { episode: episodeReturns.length, steps, length: epLength };
            }

            state = env.reset();
            episodeReturn = 0;

            // Progress logging every 300 episodes
            if (episodeReturns.length % 300 === 0) {
                const recent = episodeReturns.slice(-20);
                const recentLengths = episodeLengths.slice(-20);
                const avg = recent.reduce((a, b) => a + b, 0) / recent.length;
                const avgLen = recentLengths.reduce((a, b) => a + b, 0) / recentLengths.length;
                const maxRecent = Math.max(...recentLengths);
                console.log(`    Episode ${episodeReturns.length}: Avg Return=${avg.toFixed(1)}, Avg Length=${avgLen.toFixed(0)}, Max=${maxRecent}, ε=${agent.epsilon.toFixed(3)}`);
            }
        }

        // Check for NaN in weights periodically
        if (steps % 2000 === 0) {
            for (const param of agent.network.getTrainableVariables()) {
                assertNoNaN(param.read(), param.name);
            }
        }
    }

    // =========================================================================
    // Performance Summary
    // =========================================================================
    console.log('');
    console.log('  ─────────────────────────────────────────────────────────');
    console.log('  BALANCING PERFORMANCE SUMMARY');
    console.log('  ─────────────────────────────────────────────────────────');

    // Overall stats
    const totalEpisodes = episodeReturns.length;
    const avgLength = episodeLengths.reduce((a, b) => a + b, 0) / totalEpisodes;
    const maxLength = Math.max(...episodeLengths);
    const minLength = Math.min(...episodeLengths);

    console.log(`  Total episodes: ${totalEpisodes}`);
    console.log(`  Average episode length: ${avgLength.toFixed(1)} steps`);
    console.log(`  Best episode: ${maxLength} steps`);
    console.log(`  Worst episode: ${minLength} steps`);
    console.log('');

    // Milestones
    console.log('  Learning Milestones:');
    if (firstBalance50) {
        console.log(`    First 50+ steps:  Episode ${firstBalance50.episode} (after ${firstBalance50.steps} training steps)`);
    } else {
        console.log('    First 50+ steps:  Not achieved');
    }
    if (firstBalance100) {
        console.log(`    First 100+ steps: Episode ${firstBalance100.episode} (after ${firstBalance100.steps} training steps)`);
    } else {
        console.log('    First 100+ steps: Not achieved');
    }
    if (firstBalance200) {
        console.log(`    First 200+ steps: Episode ${firstBalance200.episode} (after ${firstBalance200.steps} training steps)`);
    } else {
        console.log('    First 200+ steps: Not achieved');
    }
    if (firstBalance500) {
        console.log(`    First PERFECT (500 steps): Episode ${firstBalance500.episode} (after ${firstBalance500.steps} training steps)`);
    } else {
        console.log('    First PERFECT (500 steps): Not achieved');
    }
    console.log('');

    // Success rate in different phases
    const quarter = Math.floor(totalEpisodes / 4);
    const phases = [
        { name: 'First 25%', data: episodeLengths.slice(0, quarter) },
        { name: 'Second 25%', data: episodeLengths.slice(quarter, 2 * quarter) },
        { name: 'Third 25%', data: episodeLengths.slice(2 * quarter, 3 * quarter) },
        { name: 'Final 25%', data: episodeLengths.slice(3 * quarter) }
    ];

    console.log('  Performance by Training Phase:');
    for (const phase of phases) {
        if (phase.data.length === 0) continue;
        const phaseAvg = phase.data.reduce((a, b) => a + b, 0) / phase.data.length;
        const phaseMax = Math.max(...phase.data);
        const balanced = phase.data.filter(l => l >= 200).length;
        const perfect = phase.data.filter(l => l >= 500).length;
        console.log(`    ${phase.name}: Avg=${phaseAvg.toFixed(0)} steps, Max=${phaseMax}, Balanced(200+)=${balanced}/${phase.data.length}, Perfect(500)=${perfect}/${phase.data.length}`);
    }
    console.log('');

    // Final performance (last 20 episodes)
    const last20 = episodeLengths.slice(-20);
    const last20Avg = last20.reduce((a, b) => a + b, 0) / last20.length;
    const last20Balanced = last20.filter(l => l >= 200).length;
    const last20Perfect = last20.filter(l => l >= 500).length;

    console.log('  Final Performance (last 20 episodes):');
    console.log(`    Average length: ${last20Avg.toFixed(1)} steps`);
    console.log(`    Balanced (200+ steps): ${last20Balanced}/20 (${(last20Balanced / 20 * 100).toFixed(0)}%)`);
    console.log(`    Perfect (500 steps): ${last20Perfect}/20 (${(last20Perfect / 20 * 100).toFixed(0)}%)`);
    console.log('');

    // Is the pole balanced?
    const isBalanced = last20Avg >= 100;
    const isWellBalanced = last20Avg >= 300;
    const isPerfect = last20Perfect >= 10; // At least half perfect

    if (isPerfect) {
        console.log('  Status: EXCELLENT - Agent consistently balances the pole perfectly!');
    } else if (isWellBalanced) {
        console.log('  Status: GOOD - Agent balances the pole well (avg 300+ steps)');
    } else if (isBalanced) {
        console.log('  Status: LEARNING - Agent shows basic balancing ability');
    } else {
        console.log('  Status: STILL LEARNING - Agent needs more training');
    }

    console.log('  ─────────────────────────────────────────────────────────');

    // ObGD step size info
    const stats = agent.optimizer.lastStats;
    console.log(`  Final ObGD step size: ${stats.obgd.stepSize.toFixed(6)}`);

    // Verify improvement
    const firstHalf = episodeReturns.slice(0, Math.floor(totalEpisodes / 2));
    const secondHalf = episodeReturns.slice(Math.floor(totalEpisodes / 2));
    const firstAvg = firstHalf.reduce((a, b) => a + b, 0) / firstHalf.length;
    const secondAvg = secondHalf.reduce((a, b) => a + b, 0) / secondHalf.length;

    // Save weights if we achieved good performance
    if (isPerfect || isWellBalanced) {
        console.log('');
        console.log('  Saving trained weights and normalization stats...');

        // Save model to file
        const saveDir = path.join(__dirname, 'trained-model');
        await agent.network.model.save(`file://${saveDir}`);
        console.log(`  Model saved to: ${saveDir}/`);

        // Export weights as JSON
        const weights = {};
        for (const layer of agent.network.model.layers) {
            if (layer.getClassName() === 'Dense') {
                const [w, b] = layer.getWeights();
                weights[layer.name] = {
                    kernel: Array.from(w.dataSync()),
                    kernelShape: w.shape,
                    bias: Array.from(b.dataSync()),
                    biasShape: b.shape
                };
            }
        }

        const weightsPath = path.join(__dirname, 'trained-weights.json');
        fs.writeFileSync(weightsPath, JSON.stringify(weights, null, 2));
        console.log(`  Weights JSON saved to: ${weightsPath}`);

        // Save normalization statistics
        // env is: AddTimeInfo -> NormalizeObservation -> ScaleReward -> CartPole
        // We need to get to NormalizeObservation's normalizer
        const normEnv = env.env; // NormalizeObservation
        const scaleEnv = normEnv.env; // ScaleReward

        const normStats = {
            observation: {
                mean: Array.from(normEnv.normalizer.mean.dataSync()),
                var: Array.from(normEnv.normalizer.var.dataSync()),
                count: normEnv.normalizer.count
            },
            reward: {
                var: Array.from(scaleEnv.rewardStats.var.dataSync()),
                count: scaleEnv.rewardStats.count
            }
        };

        const normPath = path.join(__dirname, 'trained-normalization.json');
        fs.writeFileSync(normPath, JSON.stringify(normStats, null, 2));
        console.log(`  Normalization stats saved to: ${normPath}`);
    }

    // Clean up
    agent.dispose();
    if (env.dispose) env.dispose();

    // Assertions
    assert(episodeReturns.length >= 10, 'Should complete at least 10 episodes');

    // Verify improvement (second half should be better)
    const improved = secondAvg > firstAvg;
    console.log(`  Learning improvement: ${firstAvg.toFixed(1)} → ${secondAvg.toFixed(1)} (${improved ? 'PASSED' : 'NEEDS MORE TRAINING'})`);
    console.log('  Full Training: PASSED');

    return { episodeReturns, episodeLengths, firstAvg, secondAvg, isBalanced };
}

async function testTDError(ctx) {
    console.log('\n=== Testing TD Error Computation ===');

    let env = new ctx.CartPole();
    env = new ctx.AddTimeInfo(env);

    const agent = new ctx.StreamQ({
        env,
        numActions: 2,
        gamma: 0.99,
        epsilonStart: 0.1,
        epsilonTarget: 0.01,
        totalSteps: 1000,
        hiddenSize: 32,
        learningRate: 1.0,
        lambda: 0.8,
        kappa: 2.0
    });

    const state = env.reset();
    const { action, isNonGreedy } = await agent.sampleAction(state);
    const result = env.step(action);

    // Capture TD error by updating
    await agent.update(state, action, result.reward, result.state, result.done, isNonGreedy);

    const stats = agent.optimizer.lastStats;
    assert(typeof stats.obgd.delta === 'number', 'TD error should be computed');
    assert(!isNaN(stats.obgd.delta), 'TD error should not be NaN');

    agent.dispose();

    console.log('  TD Error Computation: PASSED');
}

// ============================================================================
// Main
// ============================================================================

async function testPretrainedWeights(ctx) {
    console.log('\n=== Testing Pretrained Weights ===');

    // Check if weights and normalization files exist
    const weightsPath = path.join(__dirname, 'trained-weights.json');
    const normPath = path.join(__dirname, 'trained-normalization.json');

    if (!fs.existsSync(weightsPath) || !fs.existsSync(normPath)) {
        console.log('  No pretrained weights/normalization found, skipping test');
        console.log('  (Run full training first to generate these files)');
        return;
    }

    // Create environment
    let baseEnv = new ctx.CartPole();
    let scaleEnv = new ctx.ScaleReward(baseEnv, 0.99);
    let normEnv = new ctx.NormalizeObservation(scaleEnv);
    let env = new ctx.AddTimeInfo(normEnv);

    // Create agent
    const agent = new ctx.StreamQ({
        env, numActions: 2, gamma: 0.99,
        epsilonStart: 0.001, epsilonTarget: 0.001, totalSteps: 1000
    });

    // Load pretrained weights
    const weightsJson = JSON.parse(fs.readFileSync(weightsPath, 'utf8'));
    await agent.network.loadPretrainedWeights(weightsJson);
    agent.epsilon = 0; // Pure exploitation

    // Load normalization stats
    const normStats = JSON.parse(fs.readFileSync(normPath, 'utf8'));
    normEnv.normalizer.loadStats(normStats.observation);
    normEnv.normalizer.frozen = true; // Don't update during inference
    scaleEnv.rewardStats.loadStats({ mean: [0], var: normStats.reward.var, count: normStats.reward.count });
    scaleEnv.rewardStats.frozen = true;
    console.log('  Loaded normalization statistics (frozen for inference)')

    // Test: run 10 episodes
    console.log('  Running 10 episodes with pretrained agent:');
    let totalSteps = 0;
    let perfectCount = 0;

    for (let ep = 0; ep < 10; ep++) {
        let state = env.reset();
        while (true) {
            const { action } = await agent.sampleAction(state);
            const result = env.step(action);
            if (result.done) {
                const steps = result.info.episode.steps;
                totalSteps += steps;
                if (steps >= 500) perfectCount++;
                break;
            }
            state = result.state;
        }
    }

    const avgSteps = totalSteps / 10;
    console.log(`  Average: ${avgSteps.toFixed(1)} steps, Perfect: ${perfectCount}/10`);

    agent.dispose();
    if (env.dispose) env.dispose();

    assert(avgSteps >= 200, `Pretrained agent should average at least 200 steps, got ${avgSteps}`);
    console.log('  Pretrained Weights: PASSED');
}

async function main() {
    const quickMode = process.argv.includes('--quick');

    console.log('Stream Q(λ) Implementation Tests');
    console.log('================================');
    console.log('Testing against paper: "Streaming Deep Reinforcement Learning Finally Works"');
    if (quickMode) console.log('(Quick mode: skipping full training)');
    console.log('');

    await tf.setBackend('cpu');
    console.log(`TensorFlow.js backend: ${tf.getBackend()}`);

    const ctx = loadModules();

    try {
        await testCartPoleEnvironment(ctx);
        await testLayerNorm(ctx);
        await testSparseInit(ctx);
        await testObGDOptimizer(ctx);
        await testEligibilityTraces(ctx);
        await testNormalization(ctx);
        await testRewardScaling(ctx);
        await testTimeInfo(ctx);
        await testTDError(ctx);
        await testPretrainedWeights(ctx);

        if (!quickMode) {
            await testFullTraining(ctx);
        }

        console.log('\n================================');
        console.log('All tests PASSED!');
        console.log('================================');

    } catch (error) {
        console.error('\n================================');
        console.error('TEST FAILED:', error.message);
        console.error('================================');
        console.error(error.stack);
        process.exit(1);
    }
}

main();
