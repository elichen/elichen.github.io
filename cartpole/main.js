class TrainingManager {
    constructor(config = {}) {
        let env = new CartPole();
        env = new NormalizeObservation(env);
        env = new ScaleReward(env, 0.1);
        this.env = env;
        
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