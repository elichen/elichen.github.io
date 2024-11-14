class MetricsTracker {
    constructor() {
        this.episodeRewards1 = [];
        this.episodeRewards2 = [];
        this.episodeLengths = [];
        this.rallyLengths = [];
    }

    addEpisodeData(data) {
        this.episodeRewards1.push(data.reward1);
        this.episodeRewards2.push(data.reward2);
        this.episodeLengths.push(data.steps);
        this.rallyLengths.push(data.maxRally);

        // Keep only last 100 episodes of data
        if (this.episodeRewards1.length > 100) {
            this.episodeRewards1.shift();
            this.episodeRewards2.shift();
            this.episodeLengths.shift();
            this.rallyLengths.shift();
        }
    }

    update() {
        // No-op - removed console logging
    }
} 