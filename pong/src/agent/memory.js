class Memory {
    constructor() {
        this.states = [];
        this.actions = [];
        this.rewards = [];
        this.nextStates = [];
        this.dones = [];
        this.maxSize = 10000;  // Limit memory size
    }

    store(state, action, reward, nextState, done) {
        this.states.push(state);
        this.actions.push(action);
        this.rewards.push(reward);
        this.nextStates.push(nextState);
        this.dones.push(done);

        // Remove oldest memories if we exceed maxSize
        if (this.states.length > this.maxSize) {
            this.states.shift();
            this.actions.shift();
            this.rewards.shift();
            this.nextStates.shift();
            this.dones.shift();
        }
    }

    clear() {
        this.states = [];
        this.actions = [];
        this.rewards = [];
        this.nextStates = [];
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