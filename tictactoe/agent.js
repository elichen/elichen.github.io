class DQNAgent {
  constructor(epsilonStart = 0.7, epsilonEnd = 0.1, gamma = 0.999, batchSize = 150, maxMemorySize = 100000, fixedEpsilonSteps = 100, decayEpsilonSteps = 100) {
    this.model = new TicTacToeModel();
    this.epsilon = epsilonStart;
    this.epsilonEnd = epsilonEnd;
    this.gamma = gamma;
    this.batchSize = batchSize;
    this.maxMemorySize = maxMemorySize;
    this.memory = [];
    this.fixedEpsilonSteps = fixedEpsilonSteps;
    this.decayEpsilonRate = (epsilonStart-epsilonEnd) / decayEpsilonSteps;
    this.currentStep = 0;
  }

  act(state, isTraining = true) {
    if (isTraining && Math.random() < this.epsilon) {
      const validMoves = game.getValidMoves();
      return validMoves[Math.floor(Math.random() * validMoves.length)];
    } else {
      return tf.tidy(() => {
        const qValues = this.model.predict(state);
        return tf.argMax(qValues, 1).dataSync()[0];
      });
    }
  }

  remember(state, action, reward, nextState, done) {
    if (this.memory.length >= this.maxMemorySize) {
      this.memory.shift();
    }
    this.memory.push([state, action, reward, nextState, done]);
  }

  async replay() {
    if (this.memory.length < this.batchSize) return null;

    const batch = this.getRandomBatch(this.batchSize);
    const states = batch.map(experience => experience[0]);
    const nextStates = batch.map(experience => experience[3]);

    const x = [];
    const y = [];

    tf.tidy(() => {
      const currentQs = this.model.predict(states);
      const nextQs = this.model.predict(nextStates, true);

      for (let i = 0; i < this.batchSize; i++) {
        const [state, action, reward, nextState, done] = batch[i];
        let newQ = reward;
        if (!done) {
          const nextQsMain = this.model.predict(nextState);
          const bestAction = tf.argMax(nextQsMain).dataSync()[0];
          newQ += this.gamma * nextQs.arraySync()[i][bestAction];
        }
        const targetQ = currentQs.arraySync()[i];
        targetQ[action] = newQ;
        
        x.push(state);
        y.push(targetQ);
      }
    });

    const loss = await this.model.train(x, y);
    return loss;
  }

  async trainStep(state, action, reward, nextState, done) {
    const currentQ = this.model.predict(state);
    const nextQ = this.model.predict(nextState, true);
    
    const currentQArray = await currentQ.array();
    const nextQArray = await nextQ.array();
    
    let newQ = reward;
    if (!done) {
      const nextQsMain = this.model.predict(nextState);
      const nextQsMainArray = await nextQsMain.array();
      const bestAction = tf.argMax(nextQsMainArray).dataSync()[0];
      newQ += this.gamma * nextQArray[0][bestAction];
    }
    
    const targetQ = currentQArray[0];
    targetQ[action] = newQ;
    
    const loss = await this.model.train([state], [targetQ]);
    
    // Clean up tensors
    currentQ.dispose();
    nextQ.dispose();
    
    return loss;
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
    this.currentStep++;
    if (this.currentStep <= this.fixedEpsilonSteps) {
      // Do nothing, keep epsilon fixed
    } else if (this.epsilon > 0) {
      // Linear decay
      this.epsilon = Math.max(this.epsilonEnd, this.epsilon - this.decayEpsilonRate);
    }
  }
}