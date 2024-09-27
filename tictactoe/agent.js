class DQNAgent {
  constructor(epsilon = 1.0, epsilonDecay = 0.995, epsilonMin = 0.01, gamma = 0.95, batchSize = 32, maxMemorySize = 10000) {
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
      return Math.floor(Math.random() * 9);
    } else {
      const qValues = this.model.predict(state);
      return tf.argMax(qValues, 1).dataSync()[0];
    }
  }

  remember(state, action, reward, nextState, done) {
    if (this.memory.length >= this.maxMemorySize) {
      this.memory.shift(); // Remove the oldest experience
    }
    this.memory.push([state, action, reward, nextState, done]);
  }

  async replay() {
    if (this.memory.length < this.batchSize) return;

    const batch = this.getRandomBatch();
    const states = batch.map(experience => experience[0]);
    const nextStates = batch.map(experience => experience[3]);

    const currentQs = this.model.predict(states);
    const nextQs = this.model.predict(nextStates);

    const x = [];
    const y = [];

    for (let i = 0; i < this.batchSize; i++) {
      const [state, action, reward, nextState, done] = batch[i];
      let newQ = reward;
      if (!done) {
        newQ += this.gamma * Math.max(...nextQs.arraySync()[i]);
      }
      const targetQ = currentQs.arraySync()[i];
      targetQ[action] = newQ;
      x.push(state);
      y.push(targetQ);
    }

    await this.model.train(x, y);

    if (this.epsilon > this.epsilonMin) {
      this.epsilon *= this.epsilonDecay;
    }
  }

  getRandomBatch() {
    const batchIndices = new Array(this.batchSize).fill(0).map(() => 
      Math.floor(Math.random() * this.memory.length)
    );
    return batchIndices.map(index => this.memory[index]);
  }

  getMemorySize() {
    return this.memory.length;
  }
}