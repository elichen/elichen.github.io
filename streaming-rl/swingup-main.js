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
        this.totalSteps = 0;
    }

    async init() {
        // Create environment chain with swingup environment
        let baseEnv = new CartPoleSwingup();
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
            lambda: 0.9      // Higher lambda for longer credit assignment
        });

        // Load pretrained swingup weights
        try {
            const weightsResponse = await fetch('trained-weights-swingup.json');
            const weightsJson = await weightsResponse.json();
            await this.agent.network.loadPretrainedWeights(weightsJson);

            // Load normalization stats
            const normResponse = await fetch('trained-normalization-swingup.json');
            const normStats = await normResponse.json();

            // Load and freeze normalizer stats
            normEnv.normalizer.loadStats(normStats.observation);
            normEnv.normalizer.frozen = true;
            scaleEnv.rewardStats.loadStats({ mean: [0], var: normStats.reward.var, count: normStats.reward.count });
            scaleEnv.rewardStats.frozen = true;

            // Set epsilon for minimal exploration
            this.agent.epsilon = 0.001;

            this.stats.innerHTML = 'Pretrained swingup agent loaded. Running...';
            this.run();
        } catch (error) {
            console.error('Error loading pretrained:', error);
            this.stats.innerHTML = `Error loading pretrained agent: ${error.message}`;
        }
    }

    async run() {
        let state = this.env.reset();
        let episodeSteps = 0;

        const animate = async () => {
            // Get action
            const { action, isNonGreedy } = await this.agent.sampleAction(state);
            const result = this.env.step(action);

            // Learn from this transition
            await this.agent.update(state, action, result.reward, result.state, result.done, isNonGreedy);

            episodeSteps++;
            this.totalSteps++;
            state = result.state;

            this.env.render();

            if (result.done) {
                this.episodeCount++;
                const rawReturn = result.info.episode.r;
                this.episodeReturns.push(rawReturn);
                const avgReturn = this.episodeReturns.average();

                // Calculate percentage of max possible return (1000)
                const pctMax = (rawReturn / 1000 * 100).toFixed(0);

                this.stats.innerHTML = `
                    Episode: ${this.episodeCount}<br>
                    Return: ${rawReturn.toFixed(1)} (${pctMax}% of max)<br>
                    Steps: ${episodeSteps}<br>
                    Avg Return (${this.episodeReturns.size}): ${avgReturn.toFixed(1)}<br>
                    Total Steps: ${this.totalSteps.toLocaleString()}
                `;

                if (this.gradientStats) {
                    this.gradientStats.textContent = this.agent.optimizer.getLastStats();
                }

                state = this.env.reset();
                episodeSteps = 0;
            }

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
