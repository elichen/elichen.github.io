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

class SwingupRunner {
    constructor() {
        this.animationFrameId = null;
        this.stats = document.getElementById('stats');
        this.gradientStats = document.getElementById('gradientStats');
        this.episodeReturns = new CircularBuffer(10);
        this.episodeCount = 0;
        this.episodeSteps = 0;
        this.totalSteps = 0;
        this.learning = true;
        this.speed = 1;  // steps per frame; Infinity runs as many as fit in the frame budget
        this.history = [];
        this.events = [];
        this.chart = new ReturnChart(document.getElementById('returnChart'), document.getElementById('chartTooltip'));
    }

    async init() {
        // Create environment chain with swingup environment
        let baseEnv = new CartPoleSwingup();
        this.baseEnv = baseEnv;
        let scaleEnv = new ScaleReward(baseEnv, 0.99);
        let normEnv = new NormalizeObservation(scaleEnv);
        this.env = new AddTimeInfo(normEnv);

        // Create agent with 64 hidden units (matching swingup training)
        this.agent = new StreamQ({
            env: this.env,
            numActions: 2,
            gamma: 0.99,
            epsilonStart: 0.01,
            epsilonTarget: 0.01,
            totalSteps: 1000,
            hiddenSize: 64,  // Larger network for swingup
            lambda: 0.9,     // Higher lambda for longer credit assignment
            // The 2024 ObGD optimizer often stalled or relapsed after a physics change;
            // the 2026 per-weight bounded optimizer re-adapts in 10-20 episodes
            learningRate: 3e-4
        });

        // Load pretrained swingup weights
        try {
            const weightsResponse = await fetch('trained-weights-swingup.json');
            const weightsJson = await weightsResponse.json();
            await this.agent.network.loadPretrainedWeights(weightsJson);
            this.weightsJson = weightsJson;

            // Load normalization stats
            const normResponse = await fetch('trained-normalization-swingup.json');
            const normStats = await normResponse.json();

            // Load and freeze normalizer stats
            normEnv.normalizer.loadStats(normStats.observation);
            normEnv.normalizer.frozen = true;
            scaleEnv.rewardStats.loadStats({ mean: [0], var: normStats.reward.var, count: normStats.reward.count });
            scaleEnv.rewardStats.frozen = true;

            this.stats.innerHTML = 'Pretrained swingup agent loaded. Running...';
            this.setupControls();
            this.chart.update(this.history, this.events);
            this.run();
        } catch (error) {
            console.error('Error loading pretrained:', error);
            this.stats.innerHTML = `Error loading pretrained agent: ${error.message}`;
        }
    }

    setupControls() {
        const pole = document.getElementById('poleLength');
        const force = document.getElementById('forceMag');
        const showPhysics = () => {
            document.getElementById('poleLengthValue').textContent = `${(+pole.value).toFixed(1)} m`;
            document.getElementById('forceMagValue').textContent = `${force.value} N`;
        };
        const applyPhysics = () => {
            this.baseEnv.setPhysics({ poleLength: +pole.value, forceMag: +force.value });
            showPhysics();
        };
        // Physics changes live while dragging; the chart marks where the drag ended
        pole.addEventListener('input', applyPhysics);
        force.addEventListener('input', applyPhysics);
        pole.addEventListener('change', () => this.markEvent(`pole ${(+pole.value).toFixed(1)} m`));
        force.addEventListener('change', () => this.markEvent(`force ${force.value} N`));
        document.getElementById('resetPhysics').addEventListener('click', () => {
            pole.value = 1.0;
            force.value = 10;
            applyPhysics();
            this.markEvent('default physics');
        });
        // Sliders can keep their values across a reload
        applyPhysics();

        const learning = document.getElementById('learning');
        learning.checked = true;
        learning.addEventListener('change', () => {
            this.learning = learning.checked;
            // Traces from before the pause would credit the wrong steps
            this.agent.optimizer.resetTraces();
            this.markEvent(this.learning ? 'learning on' : 'learning off');
        });

        document.getElementById('resetAgent').addEventListener('click', async () => {
            await this.agent.network.loadPretrainedWeights(this.weightsJson);
            this.agent.optimizer.reset();
            this.markEvent('agent reset');
        });

        const speedButtons = document.querySelectorAll('[data-speed]');
        speedButtons.forEach(button => button.addEventListener('click', () => {
            this.speed = button.dataset.speed === 'max' ? Infinity : +button.dataset.speed;
            speedButtons.forEach(b => b.setAttribute('aria-pressed', String(b === button)));
        }));
    }

    markEvent(label) {
        // Place the hairline partway through the current episode
        this.events.push({ x: this.episodeCount + this.episodeSteps / this.baseEnv.maxSteps, label });
        this.chart.update(this.history, this.events);
    }

    async step(state) {
        const { action, isNonGreedy } = await this.agent.sampleAction(state);
        const result = this.env.step(action);

        // Learn from this transition
        if (this.learning) {
            await this.agent.update(state, action, result.reward, result.state, result.done, isNonGreedy, result.info.truncated);
        }

        this.episodeSteps++;
        this.totalSteps++;

        if (result.done) {
            this.endEpisode(result.info.episode.r);
            return this.env.reset();
        }
        return result.state;
    }

    endEpisode(rawReturn) {
        this.episodeCount++;
        this.episodeReturns.push(rawReturn);
        const avgReturn = this.episodeReturns.average();
        this.history.push({
            episode: this.episodeCount,
            ret: rawReturn,
            avg: avgReturn,
            pole: this.baseEnv.length * 2,
            force: this.baseEnv.forceMag,
            learning: this.learning
        });
        if (this.history.length > 1000) this.history.shift();

        // Calculate percentage of max possible return (1000)
        const pctMax = (rawReturn / 1000 * 100).toFixed(0);

        this.stats.innerHTML = `
            Episode: ${this.episodeCount}<br>
            Return: ${rawReturn.toFixed(1)} (${pctMax}% of max)<br>
            Steps: ${this.episodeSteps}<br>
            Avg Return (${this.episodeReturns.size}): ${avgReturn.toFixed(1)}<br>
            Total Steps: ${this.totalSteps.toLocaleString()}
        `;

        if (this.gradientStats) {
            this.gradientStats.textContent = this.agent.optimizer.getLastStats();
        }

        this.chart.update(this.history, this.events);
        this.episodeSteps = 0;
    }

    async run() {
        let state = this.env.reset();

        const animate = async () => {
            // At max speed, keep stepping for most of a frame, then draw once
            const frameStart = performance.now();
            let steps = 0;
            while (steps < this.speed && (this.speed !== Infinity || performance.now() - frameStart < 25)) {
                state = await this.step(state);
                steps++;
            }

            this.env.render();
            this.animationFrameId = requestAnimationFrame(animate);
        };

        this.animationFrameId = requestAnimationFrame(animate);
    }
}

// Initialize when document is loaded
window.onload = async () => {
    await tf.setBackend('cpu');
    const demo = new SwingupRunner();
    await demo.init();
};
