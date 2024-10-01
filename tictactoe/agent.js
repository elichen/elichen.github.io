class DQNAgent {
  constructor(epsilon = 1.0, epsilonDecay = 0.999, epsilonMin = 0.1, gamma = 0.95, batchSize = 150, maxMemorySize = 100000, alpha = 0.6, beta = 0.4, betaIncrement = 0.001) {
    this.model = new TicTacToeModel();
    this.epsilon = epsilon;
    this.epsilonDecay = epsilonDecay;
    this.epsilonMin = epsilonMin;
    this.gamma = gamma;
    this.batchSize = batchSize;
    this.maxMemorySize = maxMemorySize;
    this.memory = [];
    this.priorities = [];
    this.alpha = alpha; // Controls how much prioritization is used
    this.beta = beta; // Controls importance sampling weights
    this.betaIncrement = betaIncrement; // How much to increase beta each time
    this.maxPriority = 1.0; // Max priority for new experiences
  }

  act(state, isTraining = true) {
    if (isTraining && Math.random() < this.epsilon) {
      return Math.floor(Math.random() * 9);
    } else {
      const qValues = this.model.predict(state);
      return tf.argMax(qValues, 1).dataSync()[0];
    }
  }

  remember(state, action, reward, nextState, done) {
    if (this.memory.length >= this.maxMemorySize) {
      this.memory.shift();
      this.priorities.shift();
    }
    this.memory.push([state, action, reward, nextState, done]);
    this.priorities.push(this.maxPriority);
  }

  async replay() {
    if (this.memory.length === 0) return;

    const batchSize = Math.min(this.batchSize, this.memory.length);
    const [batch, indices, importanceWeights] = this.getPrioritizedBatch(batchSize);
    const states = batch.map(experience => experience[0]);
    const nextStates = batch.map(experience => experience[3]);

    const currentQs = this.model.predict(states);
    const nextQs = this.model.predict(nextStates, true);

    const x = [];
    const y = [];
    const newPriorities = [];

    for (let i = 0; i < batchSize; i++) {
      const [state, action, reward, nextState, done] = batch[i];
      let newQ = reward;
      if (!done) {
        const nextQsMain = this.model.predict(nextState).arraySync()[0];
        const bestAction = tf.argMax(nextQsMain).dataSync()[0];
        newQ += this.gamma * nextQs.arraySync()[i][bestAction];
      }
      const targetQ = currentQs.arraySync()[i];
      const oldQ = targetQ[action];
      targetQ[action] = newQ;
      
      x.push(state);
      y.push(targetQ);

      // Calculate new priority
      const error = Math.abs(oldQ - newQ);
      newPriorities.push(error);
    }

    // Update priorities
    for (let i = 0; i < indices.length; i++) {
      this.priorities[indices[i]] = newPriorities[i];
    }
    this.maxPriority = Math.max(this.maxPriority, ...newPriorities);

    // Apply importance weights to the loss
    const weightedLoss = (yTrue, yPred) => {
      const losses = tf.losses.meanSquaredError(yTrue, yPred);
      return tf.mul(losses, tf.tensor(importanceWeights)).mean();
    };

    await this.model.train(x, y, weightedLoss);

    // Increase beta
    this.beta = Math.min(1.0, this.beta + this.betaIncrement);
  }

  getPrioritizedBatch(batchSize) {
    const priorities = this.priorities.map(p => Math.pow(p, this.alpha));
    const totalPriority = priorities.reduce((a, b) => a + b, 0);
    const probabilities = priorities.map(p => p / totalPriority);

    const indices = [];
    for (let i = 0; i < batchSize; i++) {
      const r = Math.random();
      let cumSum = 0;
      for (let j = 0; j < probabilities.length; j++) {
        cumSum += probabilities[j];
        if (r <= cumSum) {
          indices.push(j);
          break;
        }
      }
    }

    const batch = indices.map(index => this.memory[index]);
    const maxWeight = Math.pow(this.memory.length * Math.min(...probabilities), -this.beta);
    const importanceWeights = indices.map(index => 
      Math.pow(this.memory.length * probabilities[index], -this.beta) / maxWeight
    );

    return [batch, indices, importanceWeights];
  }

  getMemorySize() {
    return this.memory.length;
  }

  decayEpsilon() {
    if (this.epsilon > this.epsilonMin) {
      this.epsilon *= this.epsilonDecay;
    }
  }
}