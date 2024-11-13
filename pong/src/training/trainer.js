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
            let { state1, state2 } = this.env.reset();
            let done = false;

            while (!done && this.isTesting) {
                await new Promise(resolve => setTimeout(resolve, 16)); // ~60fps

                // Get actions from both agents using their perspectives
                const { action: action1 } = this.agent1.selectAction(state1);
                const { action: action2 } = this.agent2.selectAction(state2);

                // Environment step
                const result = this.env.step(action1, action2);
                if (result.done) {
                    done = true;
                } else {
                    ({ state1, state2 } = result);
                }
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
        try {
            let { state1, state2 } = this.env.reset();
            let episodeReward1 = 0;
            let episodeReward2 = 0;
            let stepCount = 0;

            while (true) {
                stepCount++;
                
                // Add delay to make the game visible
                await new Promise(resolve => setTimeout(resolve, 16)); // ~60fps

                // Agent 1 action using its perspective
                const { action: action1, value: value1, logProb: logProb1 } = 
                    this.agent1.selectAction(state1);
                
                // Agent 2 action using its perspective
                const { action: action2, value: value2, logProb: logProb2 } = 
                    this.agent2.selectAction(state2);

                // Environment step
                const result = this.env.step(action1, action2);

                // Store experiences with respective perspectives
                this.agent1.memory.store(state1, action1, result.reward1, value1, logProb1, result.done);
                this.agent2.memory.store(state2, action2, result.reward2, value2, logProb2, result.done);

                episodeReward1 += result.reward1;
                episodeReward2 += result.reward2;
                
                if (result.done) {
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

                // Update states for next iteration using destructuring
                ({ state1, state2 } = result);
            }
        } catch (error) {
            console.error("Error in training episode:", error);
            throw error;
        }
    }

    pause() {
        this.isTraining = false;
    }

    resume() {
        this.isTraining = true;
    }
} 