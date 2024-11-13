class PongTrainer {
    constructor() {
        console.log("Initializing PongTrainer");
        this.env = new PongEnvironment();
        this.agent1 = new PPOAgent();
        this.agent2 = new PPOAgent();
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
            let state = this.env.reset();
            let done = false;

            while (!done && this.isTesting) {
                await new Promise(resolve => setTimeout(resolve, 16)); // ~60fps

                // Get actions from both agents (inference only)
                const { action: action1 } = this.agent1.selectAction(state);
                const { action: action2 } = this.agent2.selectAction(state);

                // Environment step
                const { state: nextState, done: gameDone } = this.env.step(action1, action2);
                state = nextState;
                done = gameDone;
            }
        }
    }

    async train(numEpisodes = 1000) {
        console.log(`Starting training for ${numEpisodes} episodes`);
        for (this.episodeCount = 0; this.episodeCount < numEpisodes; this.episodeCount++) {
            if (!this.isTraining) break;
            
            console.log(`Starting episode ${this.episodeCount}`);
            await this.trainEpisode();
            
            // Update metrics every episode
            if (this.episodeCount % 10 === 0) {
                console.log(`Episode ${this.episodeCount}: Updating metrics`);
                this.metrics.update();
                await tf.nextFrame(); // Allow UI to update
            }
        }
    }

    async trainEpisode() {
        let state = this.env.reset();
        let episodeReward1 = 0;
        let episodeReward2 = 0;
        let stepCount = 0;

        while (true) {
            stepCount++;
            
            // Add delay to make the game visible
            await new Promise(resolve => setTimeout(resolve, 16)); // ~60fps

            // Agent 1 action
            const { action: action1, value: value1, logProb: logProb1 } = this.agent1.selectAction(state);
            
            // Agent 2 action
            const { action: action2, value: value2, logProb: logProb2 } = this.agent2.selectAction(state);

            // Environment step
            const { state: nextState, reward1, reward2, done } = this.env.step(action1, action2);

            // Store experiences
            this.agent1.memory.store(state, action1, reward1, value1, logProb1, done);
            this.agent2.memory.store(state, action2, reward2, value2, logProb2, done);

            episodeReward1 += reward1;
            episodeReward2 += reward2;
            state = nextState;

            if (done) {
                console.log(`Episode finished after ${stepCount} steps`);
                console.log(`Rewards - Agent1: ${episodeReward1.toFixed(2)}, Agent2: ${episodeReward2.toFixed(2)}`);
                
                // Update metrics
                const stats = this.env.getStats();
                this.metrics.addEpisodeData({
                    reward1: episodeReward1,
                    reward2: episodeReward2,
                    steps: stats.steps,
                    maxRally: stats.maxRally,
                    scores: stats.scores
                });

                // Update both agents
                if (this.agent1.memory.states.length >= this.agent1.batchSize) {
                    console.log("Updating agents with batch size:", this.agent1.memory.states.length);
                    await this.agent1.update();
                    await this.agent2.update();
                }
                break;
            }
        }
    }

    pause() {
        this.isTraining = false;
    }

    resume() {
        this.isTraining = true;
    }
} 