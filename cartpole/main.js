class TrainingManager {
    constructor(config = {}) {
        this.env = new CartPole();
        this.agent = new StreamQ(config);
        this.episodeRewards = [];
        this.isTraining = false;
        this.stats = document.getElementById('stats');
        
        this.setupControls();
    }

    setupControls() {
        const trainButton = document.getElementById('toggleTraining');
        const testButton = document.getElementById('toggleTesting');

        trainButton.onclick = () => this.toggleTraining();
        testButton.onclick = () => this.testModel();
    }

    async toggleTraining() {
        if (this.isTraining) {
            this.isTraining = false;
            document.getElementById('toggleTraining').textContent = 'Start Training';
        } else {
            this.isTraining = true;
            document.getElementById('toggleTraining').textContent = 'Stop Training';
            await this.train();
        }
    }

    async train() {
        let episodeReward = 0;
        let state = this.env.reset();
        let episodeCount = 0;

        const animate = async () => {
            if (!this.isTraining) return;

            // Sample action and step environment
            const { action, isNonGreedy } = await this.agent.sampleAction(state);
            const { state: nextState, reward, done } = this.env.step(action);
            
            // Update agent
            const delta = await this.agent.update(
                state,
                action,
                reward,
                nextState,
                done,
                isNonGreedy
            );

            episodeReward += reward;
            state = nextState;

            // Render environment
            this.env.render();

            // Handle episode end
            if (done) {
                this.episodeRewards.push(episodeReward);
                episodeCount++;
                
                // Update stats
                const lastHundred = this.episodeRewards.slice(-100);
                const avgReward = lastHundred.reduce((a, b) => a + b, 0) / lastHundred.length;
                
                this.stats.innerHTML = `
                    Episode: ${episodeCount}<br>
                    Last Reward: ${episodeReward.toFixed(1)}<br>
                    Avg Reward (100): ${avgReward.toFixed(1)}<br>
                    Epsilon: ${this.agent.epsilon.toFixed(3)}
                `;

                // Reset for next episode
                state = this.env.reset();
                episodeReward = 0;
            }

            requestAnimationFrame(animate);
        };

        animate();
    }

    async testModel() {
        this.isTraining = false;
        document.getElementById('toggleTraining').textContent = 'Start Training';
        
        let state = this.env.reset();
        let totalReward = 0;
        
        const testEpisode = async () => {
            const { action } = await this.agent.sampleAction(state);
            const { state: nextState, reward, done } = this.env.step(action);
            
            totalReward += reward;
            state = nextState;
            
            this.env.render();
            
            if (done) {
                this.stats.innerHTML = `Test Episode Reward: ${totalReward.toFixed(1)}`;
                return;
            }
            
            requestAnimationFrame(testEpisode);
        };
        
        testEpisode();
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