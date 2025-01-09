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

class TrainingManager {
    constructor(config = {}) {
        let env = new CartPole();
        env = new ScaleReward(env, config.gamma || 0.99);
        env = new NormalizeObservation(env);
        env = new AddTimeInfo(env);
        this.env = env;
        
        config.env = env;
        this.agent = new StreamQ(config);
        this.episodeRewards = new CircularBuffer(10);
        this.isTraining = true;
        this.stats = document.getElementById('stats');
        this.totalSteps = 0;
        
        this.setupControls();
        this.train();
    }

    setupControls() {
        const modeButton = document.getElementById('toggleTraining');
        modeButton.textContent = 'Switch to Test Mode';
        modeButton.onclick = () => this.toggleMode();
    }

    async toggleMode() {
        this.isTraining = !this.isTraining;
        const modeButton = document.getElementById('toggleTraining');
        
        if (this.isTraining) {
            modeButton.textContent = 'Switch to Test Mode';
            this.train();
        } else {
            modeButton.textContent = 'Switch to Training Mode';
            this.test();
        }
    }

    async train() {
        let episodeReward = 0;
        let state = this.env.reset();
        let episodeCount = 0;

        const animate = async () => {
            if (!this.isTraining) return;

            this.totalSteps++;
            const { action, isNonGreedy } = await this.agent.sampleAction(state);
            const result = this.env.step(action);
            
            episodeReward += result.reward;
            await this.agent.update(state, action, result.reward, result.state, result.done, isNonGreedy);

            state = result.state;

            this.env.render();

            if (result.done) {
                const rawReturn = result.info.episode.r;
                this.episodeRewards.push(rawReturn);
                episodeCount++;
                
                console.log(`Episodic Return: ${rawReturn.toFixed(1)}, Time Step ${this.totalSteps}, Episode Number ${episodeCount}, Epsilon ${this.agent.epsilon.toFixed(3)}`);
                
                const avgReward = this.episodeRewards.average();
                
                this.stats.innerHTML = `
                    Mode: Training<br>
                    Episode: ${episodeCount}<br>
                    Last Reward: ${rawReturn.toFixed(1)}<br>
                    Avg Reward (${this.episodeRewards.size}): ${avgReward.toFixed(1)}<br>
                    Epsilon: ${this.agent.epsilon.toFixed(3)}
                `;

                const gradientStats = document.getElementById('gradientStats');
                if (gradientStats) {
                    gradientStats.textContent = this.agent.optimizer.getLastStats();
                }

                state = this.env.reset();
                episodeReward = 0;
            }

            requestAnimationFrame(animate);
        };

        animate();
    }

    async test() {
        let state = this.env.reset();
        let totalReward = 0;
        
        const testEpisode = async () => {
            if (this.isTraining) return;

            const savedEpsilon = this.agent.epsilon;
            this.agent.epsilon = 0;
            
            const { action } = await this.agent.sampleAction(state);
            const { state: nextState, reward, done } = this.env.step(action);
            
            this.agent.epsilon = savedEpsilon;
            
            totalReward += reward;
            state = nextState;
            
            this.env.render();
            
            if (done) {
                this.stats.innerHTML = `
                    Mode: Testing<br>
                    Test Episode Reward: ${totalReward.toFixed(1)}
                `;
                state = this.env.reset();
                totalReward = 0;
            }
            
            requestAnimationFrame(testEpisode);
        };
        
        testEpisode();
    }

    dispose() {
        if (this.env.dispose) {
            this.env.dispose();
        }
        if (this.agent.dispose) {
            this.agent.dispose();
        }
    }
}

// Initialize when document is loaded
window.onload = async () => {
    await tf.setBackend('cpu');
    const config = {
        learningRate: parseFloat(document.getElementById('learningRate').value),
        gamma: parseFloat(document.getElementById('gamma').value),
        lambda: parseFloat(document.getElementById('lambda').value)
    };
    
    const manager = new TrainingManager(config);
}; 