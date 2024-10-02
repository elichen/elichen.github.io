class DQNAgent {
  constructor(epsilon = 1.0, epsilonDecay = 0.999, epsilonMin = 0.1, gamma = 0.95, batchSize = 150, maxMemorySize = 100000) {
    this.model = new TicTacToeModel();
    this.epsilon = epsilon;
    this.epsilonDecay = epsilonDecay;
    this.epsilonMin = epsilonMin;
    this.gamma = gamma;
    this.batchSize = batchSize;
    this.maxMemorySize = maxMemorySize;
    this.memory = [];
  }

  act(state, isTraining = true) {
    if (isTraining && Math.random() < this.epsilon) {
      const validMoves = game.getValidMoves();
      return validMoves[Math.floor(Math.random() * validMoves.length)];
    } else {
      const qValues = this.model.predict(state);
      return tf.argMax(qValues, 1).dataSync()[0];
    }
  }

  remember(state, action, reward, nextState, done) {
    if (this.memory.length >= this.maxMemorySize) {
      this.memory.shift();
    }
    this.memory.push([state, action, reward, nextState, done]);
  }

  async replay() {
    if (this.memory.length < this.batchSize) return;

    const batch = this.getRandomBatch(this.batchSize);
    const states = batch.map(experience => experience[0]);
    const nextStates = batch.map(experience => experience[3]);

    const currentQs = this.model.predict(states);
    const nextQs = this.model.predict(nextStates, true);

    const x = [];
    const y = [];

    for (let i = 0; i < this.batchSize; i++) {
      const [state, action, reward, nextState, done] = batch[i];
      let newQ = reward;
      if (!done) {
        const nextQsMain = this.model.predict(nextState).arraySync()[0];
        const bestAction = tf.argMax(nextQsMain).dataSync()[0];
        newQ += this.gamma * nextQs.arraySync()[i][bestAction];
      }
      const targetQ = currentQs.arraySync()[i];
      targetQ[action] = newQ;
      
      x.push(state);
      y.push(targetQ);
    }

    await this.model.train(x, y);
  }

  getRandomBatch(batchSize) {
    const indices = [];
    while (indices.length < batchSize) {
      const index = Math.floor(Math.random() * this.memory.length);
      if (!indices.includes(index)) {
        indices.push(index);
      }
    }
    return indices.map(index => this.memory[index]);
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