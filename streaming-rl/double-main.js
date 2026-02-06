class CircularBuffer {
    constructor(maxSize) {
        this.maxSize = maxSize;
        this.buffer = new Array(maxSize);
        this.currentIndex = 0;
        this.size = 0;
    }

    push(value) {
        this.buffer[this.currentIndex] = value;
        this.currentIndex = (this.currentIndex + 1) % this.maxSize;
        this.size = Math.min(this.size + 1, this.maxSize);
    }

    average() {
        if (this.size === 0) return 0;
        const sum = this.buffer.slice(0, this.size).reduce((a, b) => a + b, 0);
        return sum / this.size;
    }
}

class ContinuousActor {
    constructor(weights, config) {
        this.forceMag = config.forceMag;
        this.xLimit = config.xLimit;
        this.velScale = config.velScale;

        this.fc1Kernel = Float32Array.from(weights.fc1.kernel);
        this.fc1Bias = Float32Array.from(weights.fc1.bias);
        this.fc2Kernel = Float32Array.from(weights.hidden.kernel);
        this.fc2Bias = Float32Array.from(weights.hidden.bias);
        this.fc3Kernel = Float32Array.from(weights.output.kernel);
        this.fc3Bias = Float32Array.from(weights.output.bias);

        this.inputSize = weights.fc1.kernelShape[0];
        this.hiddenSize = weights.fc1.kernelShape[1];

        if (this.inputSize === 9) {
            this.obsMode = 'trig';
            this.useTime = true;
        } else if (this.inputSize === 8) {
            this.obsMode = 'trig';
            this.useTime = false;
        } else if (this.inputSize === 7) {
            this.obsMode = 'raw';
            this.useTime = true;
        } else if (this.inputSize === 6) {
            this.obsMode = 'raw';
            this.useTime = false;
        } else {
            throw new Error(`Unsupported actor input size: ${this.inputSize}`);
        }

        if (this.hiddenSize !== weights.hidden.kernelShape[0] ||
            this.hiddenSize !== weights.hidden.kernelShape[1]) {
            throw new Error('Hidden layer shape mismatch in checkpoint.');
        }
        if (weights.output.kernelShape[0] !== this.hiddenSize || weights.output.kernelShape[1] !== 1) {
            throw new Error('Output layer shape mismatch in checkpoint.');
        }

        this.h1 = new Float32Array(this.hiddenSize);
        this.h2 = new Float32Array(this.hiddenSize);
    }

    buildInput(state, episodeTime) {
        const x = state[0];
        const xDot = state[1];
        const theta1 = state[2];
        const theta1Dot = state[3];
        const theta2 = state[4];
        const theta2Dot = state[5];

        let obs;
        if (this.obsMode === 'trig') {
            obs = [
                x / this.xLimit,
                xDot / this.velScale,
                Math.sin(theta1),
                Math.cos(theta1),
                theta1Dot / this.velScale,
                Math.sin(theta2),
                Math.cos(theta2),
                theta2Dot / this.velScale
            ];
        } else {
            obs = [
                x / this.xLimit,
                xDot / this.velScale,
                theta1 / Math.PI,
                theta1Dot / this.velScale,
                theta2 / Math.PI,
                theta2Dot / this.velScale
            ];
        }

        if (this.useTime) {
            obs.push(episodeTime);
        }

        if (obs.length !== this.inputSize) {
            throw new Error(`Built input size ${obs.length} does not match actor input size ${this.inputSize}`);
        }

        return obs;
    }

    denseRelu(input, kernel, inSize, outSize, bias, out) {
        for (let j = 0; j < outSize; j++) {
            let sum = bias[j];
            for (let i = 0; i < inSize; i++) {
                sum += input[i] * kernel[i * outSize + j];
            }
            out[j] = sum > 0 ? sum : 0;
        }
    }

    denseLinear(input, kernel, inSize, outSize, bias, out) {
        for (let j = 0; j < outSize; j++) {
            let sum = bias[j];
            for (let i = 0; i < inSize; i++) {
                sum += input[i] * kernel[i * outSize + j];
            }
            out[j] = sum;
        }
    }

    act(input) {
        this.denseRelu(input, this.fc1Kernel, this.inputSize, this.hiddenSize, this.fc1Bias, this.h1);
        this.denseRelu(this.h1, this.fc2Kernel, this.hiddenSize, this.hiddenSize, this.fc2Bias, this.h2);
        const out = [0.0];
        this.denseLinear(this.h2, this.fc3Kernel, this.hiddenSize, 1, this.fc3Bias, out);
        return Math.tanh(out[0]) * this.forceMag;
    }
}

class DoubleInferenceRunner {
    constructor() {
        this.animationFrameId = null;
        this.stats = document.getElementById('stats');
        this.policyStats = document.getElementById('policyStats');
        this.episodeReturns = new CircularBuffer(20);
        this.actionAbs = new CircularBuffer(500);
        this.episodeCount = 0;
        this.totalSteps = 0;
        this.episodeSteps = 0;
        this.episodeTime = -0.5;
        this.prevAction = 0.0;
        this.state = null;
    }

    async init() {
        this.env = new CartPoleDouble({
            forceMag: POLICY_CONFIG.forceMag,
            dt: POLICY_CONFIG.dt,
            maxSteps: POLICY_CONFIG.maxSteps
        });

        const response = await fetch(POLICY_CONFIG.weightsPath);
        if (!response.ok) {
            throw new Error(`Failed to load actor weights (${response.status}): ${POLICY_CONFIG.weightsPath}`);
        }
        const weights = await response.json();
        this.actor = new ContinuousActor(weights, {
            forceMag: this.env.forceMag,
            xLimit: this.env.xLimit,
            velScale: POLICY_CONFIG.velScale
        });

        this.stats.innerHTML = 'Continuous TD3 actor loaded. Running inference...';
        this.policyStats.textContent = this.buildPolicyText();

        this.resetEpisode();
        this.run();
    }

    resetEpisode() {
        this.state = this.env.reset();
        this.episodeSteps = 0;
        this.episodeTime = -0.5;
        this.prevAction = 0.0;
    }

    currentGain() {
        if (POLICY_CONFIG.actionGainEarly == null || POLICY_CONFIG.actionGainLate == null) {
            return POLICY_CONFIG.actionGain;
        }
        const frac = this.episodeTime + 0.5;
        return frac < POLICY_CONFIG.actionGainSwitchFrac
            ? POLICY_CONFIG.actionGainEarly
            : POLICY_CONFIG.actionGainLate;
    }

    buildPolicyText() {
        const lines = [
            'Policy:',
            `  checkpoint: ${POLICY_CONFIG.weightsPath}`,
            `  actor input: ${this.actor.obsMode}${this.actor.useTime ? '+time' : ''} (${this.actor.inputSize})`,
            `  force: +/-${this.env.forceMag.toFixed(1)}, dt: ${this.env.dt.toFixed(3)}, maxSteps: ${this.env.maxSteps}`,
            `  gain schedule: early=${POLICY_CONFIG.actionGainEarly}, late=${POLICY_CONFIG.actionGainLate}, switch=${POLICY_CONFIG.actionGainSwitchFrac}`,
            `  smoothing: ${POLICY_CONFIG.actionSmooth}`
        ];
        return lines.join('\n');
    }

    updateStats(lastAction, episodeReturn) {
        const avgReturn = this.episodeReturns.average();
        const maxReturn = this.env.maxSteps * 2;
        const pctMax = (episodeReturn / maxReturn * 100).toFixed(0);
        const avgAction = this.actionAbs.average();

        this.stats.innerHTML = `
            Episode: ${this.episodeCount}<br>
            Return: ${episodeReturn.toFixed(1)} (${pctMax}% of max)<br>
            Steps: ${this.episodeSteps}<br>
            Avg Return (${this.episodeReturns.size}): ${avgReturn.toFixed(1)}<br>
            Last Action: ${lastAction.toFixed(3)}<br>
            Mean |Action| (recent): ${avgAction.toFixed(3)}<br>
            Total Steps: ${this.totalSteps.toLocaleString()}
        `;
    }

    run() {
        const animate = () => {
            const modelInput = this.actor.buildInput(this.state, this.episodeTime);
            let action = this.actor.act(modelInput);

            const gain = this.currentGain();
            action = Math.max(-this.env.forceMag, Math.min(this.env.forceMag, action * gain));

            if (POLICY_CONFIG.actionSmooth > 0) {
                action = (1 - POLICY_CONFIG.actionSmooth) * action + POLICY_CONFIG.actionSmooth * this.prevAction;
            }

            this.prevAction = action;
            this.actionAbs.push(Math.abs(action));

            const result = this.env.step(action);
            this.state = result.state;

            this.episodeSteps += 1;
            this.totalSteps += 1;
            if (this.actor.useTime) {
                this.episodeTime += 1.0 / this.env.maxSteps;
            }

            this.env.render();

            if (result.done) {
                this.episodeCount += 1;
                const episodeReturn = result.info.episode.r;
                this.episodeReturns.push(episodeReturn);
                this.updateStats(action, episodeReturn);
                this.resetEpisode();
            }

            this.animationFrameId = requestAnimationFrame(animate);
        };

        this.animationFrameId = requestAnimationFrame(animate);
    }
}

const POLICY_CONFIG = {
    weightsPath: 'trained-actor-double-sb3-sac-sb3-sac-continue-h512.json',
    forceMag: 10.0,
    dt: 0.02,
    maxSteps: 5000,
    velScale: 5.0,
    actionGain: 1.0,
    actionGainEarly: null,
    actionGainLate: null,
    actionGainSwitchFrac: 0.4,
    actionSmooth: 0.0
};

window.onload = async () => {
    const demo = new DoubleInferenceRunner();
    try {
        await demo.init();
    } catch (error) {
        console.error(error);
        const stats = document.getElementById('stats');
        const policyStats = document.getElementById('policyStats');
        stats.innerHTML = `Failed to initialize inference runner: ${error.message}`;
        if (policyStats) {
            policyStats.textContent = 'Check console and checkpoint path.';
        }
    }
};
