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
        this.isSwingUpMode = false;  // Start in balance mode
        this.animationFrameId = null;
        this.initializeEnvironment(config);
        this.setupControls();
        this.train();
    }

    initializeEnvironment(config) {
        let env = new CartPole({
            swingUp: this.isSwingUpMode
        });
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
    }

    setupControls() {
        const modeButton = document.getElementById('toggleTraining');
        modeButton.textContent = 'Switch to Test Mode';
        modeButton.onclick = () => this.toggleMode();

        const resetButton = document.getElementById('resetAgent');
        resetButton.onclick = () => this.resetAgent();

        const taskButton = document.getElementById('toggleMode');
        taskButton.onclick = () => {
            this.isSwingUpMode = !this.isSwingUpMode;
            taskButton.textContent = this.isSwingUpMode ? 'Swing Up Mode' : 'Balance Mode';
        };
    }

    resetAgent() {
        // Cancel any existing animation frame
        if (this.animationFrameId !== null) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }

        // Stop current training/testing
        this.isTraining = false;
        
        // Clean up existing agent and environment
        if (this.agent.dispose) {
            this.agent.dispose();
        }
        if (this.env.dispose) {
            this.env.dispose();
        }

        // Get current configuration from UI
        const config = {
            learningRate: 1.0,
            gamma: 0.99,
            lambda: 0.8,
            epsilonStart: parseFloat(document.getElementById('epsilonStart').value),
            epsilonTarget: parseFloat(document.getElementById('epsilonTarget').value),
            totalSteps: parseFloat(document.getElementById('decaySteps').value),
            explorationFraction: 1.0
        };

        // Reinitialize environment and agent
        this.initializeEnvironment(config);
        
        // Restart training
        this.isTraining = true;
        const modeButton = document.getElementById('toggleTraining');
        modeButton.textContent = 'Switch to Test Mode';
        this.train();
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
        // Cancel any existing animation frame
        if (this.animationFrameId !== null) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }

        let episodeReward = 0;
        let state = this.env.reset();
        let episodeCount = 0;

        const animate = async () => {
            if (!this.isTraining) {
                this.animationFrameId = null;
                return;
            }

            this.totalSteps++;
            const { action, isNonGreedy } = await this.agent.sampleAction(state);
            const result = this.env.step(action);
            
            episodeReward += result.reward;
            await this.agent.update(state, action, result.reward, result.state, result.done, isNonGreedy);

            state = result.state;

            this.env.render();

            if (result.done) {
                const rawReturn = result.info.episode.r;
                const steps = result.info.episode.steps;
                const mode = result.info.episode.mode;
                this.episodeRewards.push(rawReturn);
                episodeCount++;
                
                console.log(`Mode: ${mode}, Episodic Return: ${rawReturn.toFixed(1)}, Steps: ${steps}, Episode ${episodeCount}, Epsilon ${this.agent.epsilon.toFixed(3)}`);
                
                const avgReward = this.episodeRewards.average();
                
                this.stats.innerHTML = `
                    Mode: ${mode} Training<br>
                    Episode: ${episodeCount}<br>
                    Last Return: ${rawReturn.toFixed(1)}<br>
                    Steps: ${steps}<br>
                    Avg Return (${this.episodeRewards.size}): ${avgReward.toFixed(1)}<br>
                    Epsilon: ${this.agent.epsilon.toFixed(3)}
                `;

                const gradientStats = document.getElementById('gradientStats');
                if (gradientStats) {
                    gradientStats.textContent = this.agent.optimizer.getLastStats();
                }

                state = this.env.reset();
                episodeReward = 0;
            }

            this.animationFrameId = requestAnimationFrame(animate);
        };

        this.animationFrameId = requestAnimationFrame(animate);
    }

    async test() {
        // Cancel any existing animation frame
        if (this.animationFrameId !== null) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }

        let state = this.env.reset();
        let totalReward = 0;
        
        const testEpisode = async () => {
            if (this.isTraining) {
                this.animationFrameId = null;
                return;
            }

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
            
            this.animationFrameId = requestAnimationFrame(testEpisode);
        };
        
        this.animationFrameId = requestAnimationFrame(testEpisode);
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
        // Hardcoded values
        learningRate: 1.0,
        gamma: 0.99,
        lambda: 0.8,
        
        // Exploration settings from UI
        epsilonStart: parseFloat(document.getElementById('epsilonStart').value),
        epsilonTarget: parseFloat(document.getElementById('epsilonTarget').value),
        totalSteps: parseFloat(document.getElementById('decaySteps').value),
        explorationFraction: 1.0  // Use all decay steps for exploration
    };
    
    const manager = new TrainingManager(config);
}; 