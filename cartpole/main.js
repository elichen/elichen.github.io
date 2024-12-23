class TrainingManager {
    constructor(config = {}) {
        let env = new CartPole();
        env = new NormalizeObservation(env);
        env = new ScaleReward(env, 0.1);
        this.env = env;
        
        this.agent = new StreamQ(config);
        this.episodeRewards = [];
        this.isTraining = true;
        this.stats = document.getElementById('stats');
        
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

            const { action, isNonGreedy } = await this.agent.sampleAction(state);
            const { state: nextState, reward, done } = this.env.step(action);
            
            await this.agent.update(state, action, reward, nextState, done, isNonGreedy);

            episodeReward += reward;
            state = nextState;

            this.env.render();

            if (done) {
                this.episodeRewards.push(episodeReward);
                episodeCount++;
                
                const lastHundred = this.episodeRewards.slice(-100);
                const avgReward = lastHundred.reduce((a, b) => a + b, 0) / lastHundred.length;
                
                this.stats.innerHTML = `
                    Mode: Training<br>
                    Episode: ${episodeCount}<br>
                    Last Reward: ${episodeReward.toFixed(1)}<br>
                    Avg Reward (100): ${avgReward.toFixed(1)}<br>
                    Epsilon: ${this.agent.epsilon.toFixed(3)}
                `;

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
window.onload = () => {
    tf.setBackend('webgl').then(() => {
        const config = {
            learningRate: parseFloat(document.getElementById('learningRate').value),
            gamma: parseFloat(document.getElementById('gamma').value),
            lambda: parseFloat(document.getElementById('lambda').value)
        };
        
        const manager = new TrainingManager(config);
    });
}; 