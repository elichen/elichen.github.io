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

class DemoRunner {
    constructor() {
        this.animationFrameId = null;
        this.stats = document.getElementById('stats');
        this.gradientStats = document.getElementById('gradientStats');
        this.episodeRewards = new CircularBuffer(10);
        this.episodeCount = 0;
        this.totalSteps = 0;
    }

    async init() {
        // Create environment chain
        let baseEnv = new CartPole();
        let scaleEnv = new ScaleReward(baseEnv, 0.99);
        let normEnv = new NormalizeObservation(scaleEnv);
        this.env = new AddTimeInfo(normEnv);

        // Create agent with small epsilon for continual learning
        this.agent = new StreamQ({
            env: this.env,
            numActions: 2,
            gamma: 0.99,
            epsilonStart: 0.01,
            epsilonTarget: 0.01,
            totalSteps: 1000
        });

        // Load pretrained weights
        try {
            const weightsResponse = await fetch('trained-weights.json');
            const weightsJson = await weightsResponse.json();
            await this.agent.network.loadPretrainedWeights(weightsJson);

            // Load normalization stats
            const normResponse = await fetch('trained-normalization.json');
            const normStats = await normResponse.json();

            // Load and freeze normalizer stats
            normEnv.normalizer.loadStats(normStats.observation);
            normEnv.normalizer.frozen = true;
            scaleEnv.rewardStats.loadStats({ mean: [0], var: normStats.reward.var, count: normStats.reward.count });
            scaleEnv.rewardStats.frozen = true;

            // Set epsilon to 0.01 for minimal exploration
            this.agent.epsilon = 0.01;

            this.stats.innerHTML = 'Pretrained agent loaded. Training...';
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
                this.episodeRewards.push(rawReturn);
                const avgReward = this.episodeRewards.average();

                this.stats.innerHTML = `
                    Episode: ${this.episodeCount}<br>
                    Last Return: ${rawReturn.toFixed(1)}<br>
                    Steps: ${episodeSteps}<br>
                    Avg Return (${this.episodeRewards.size}): ${avgReward.toFixed(1)}<br>
                    Epsilon: ${this.agent.epsilon.toFixed(3)}<br>
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
    const demo = new DemoRunner();
    await demo.init();
};
