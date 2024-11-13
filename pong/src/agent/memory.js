class Memory {
    constructor() {
        this.states = [];
        this.actions = [];
        this.rewards = [];
        this.values = [];
        this.logProbs = [];
        this.dones = [];
    }

    store(state, action, reward, value, logProb, done) {
        this.states.push(state);
        this.actions.push(action);
        this.rewards.push(reward);
        this.values.push(value);
        this.logProbs.push(logProb);
        this.dones.push(done);
    }

    clear() {
        this.states = [];
        this.actions = [];
        this.rewards = [];
        this.values = [];
        this.logProbs = [];
        this.dones = [];
    }

    computeGAE(gamma = 0.99, lambda = 0.95) {
        const advantages = new Array(this.rewards.length);
        let lastGAE = 0;

        for (let t = this.rewards.length - 1; t >= 0; t--) {
            const nextValue = t === this.rewards.length - 1 ? 0 : this.values[t + 1];
            const nextNonTerminal = t === this.rewards.length - 1 ? 0 : !this.dones[t];
            
            const delta = this.rewards[t] + gamma * nextValue * nextNonTerminal - this.values[t];
            lastGAE = delta + gamma * lambda * nextNonTerminal * lastGAE;
            advantages[t] = lastGAE;
        }

        const returns = advantages.map((adv, i) => adv + this.values[i]);
        return { advantages, returns };
    }
} 