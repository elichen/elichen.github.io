class PongTrainer {
    constructor() {
        console.log("Initializing PongTrainer");
        this.env = new PongEnvironment();
        this.agent = new DQNAgent();
        this.metrics = new MetricsTracker();
        this.episodeCount = 0;
        this.isTraining = true;
        this.isTesting = false;
    }

    setTestingMode(testing) {
        this.isTesting = testing;
    }

    async test() {
        console.log("Starting testing mode");
        while (this.isTesting) {
            let { state1, state2 } = this.env.reset();
            let done = false;

            while (!done && this.isTesting) {
                await new Promise(resolve => setTimeout(resolve, 16));

                const action1 = this.agent.selectAction(state1);
                const action2 = this.agent.selectAction(state2);

                const result = this.env.step(action1, action2);
                if (result.done) {
                    done = true;
                } else {
                    ({ state1, state2 } = result);
                }
            }
        }
    }

    async train() {
        console.log("Starting training");
        
        while (this.isTraining) {
            await this.trainEpisode();
            this.episodeCount++;
            
            // Update metrics every episode
            if (this.episodeCount % 10 === 0) {
                this.metrics.update();
                await tf.nextFrame(); // Allow UI to update
            }
        }
    }

    async trainEpisode() {
        let { state1, state2 } = this.env.reset();
        let episodeReward1 = 0;
        let episodeReward2 = 0;
        let stepCount = 0;

        while (true) {
            stepCount++;
            
            // Get actions from same agent using different perspectives
            const action1 = this.agent.selectAction(state1);
            const action2 = this.agent.selectAction(state2);

            // Step environment with original actions
            const result = this.env.step(action1, action2);

            // Store experiences from both perspectives
            this.agent.store(state1, action1, result.reward1, result.state1, result.done);
            this.agent.store(state2, action2, result.reward2, result.state2, result.done);

            // Update agent
            await this.agent.update();

            episodeReward1 += result.reward1;
            episodeReward2 += result.reward2;
            
            if (result.done) break;

            ({ state1, state2 } = result);
        }

        const stats = this.env.getStats();
        console.log(`Episode ${this.episodeCount} Summary:
    Stage: ${stats.stage}
    Steps: ${stepCount}
    Rewards: ${episodeReward1.toFixed(2)} | ${episodeReward2.toFixed(2)}
    Epsilon: ${this.agent.epsilon.toFixed(4)}
    Rally: ${stats.currentRally} (Best Ever: ${stats.bestRally})
    Score: ${stats.scores[0]} - ${stats.scores[1]}`);
        
        this.metrics.addEpisodeData({
            reward1: episodeReward1,
            reward2: episodeReward2,
            steps: stats.steps,
            maxRally: stats.bestRally,
            scores: stats.scores
        });
    }

    pause() {
        this.isTraining = false;
    }

    resume() {
        this.isTraining = true;
    }
} 