class DQNAgent {
  constructor(epsilonStart = 0.7, epsilonEnd = 0.1, gamma = 0.999, batchSize = 100, maxMemorySize = 100000, fixedEpsilonSteps = 250, decayEpsilonSteps = 250) {
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

  act(state, isTraining = true, validMoves) {
    if (isTraining && Math.random() < this.epsilon) {
      // Choose a random valid move during exploration
      return validMoves[Math.floor(Math.random() * validMoves.length)];
    } else {
      return tf.tidy(() => {
        const qValues = this.model.predict(state);
        
        // Create a mask for valid moves
        const mask = new Array(9).fill(-Infinity);
        validMoves.forEach(move => mask[move] = 0);
        const maskTensor = tf.tensor1d(mask);
        
        // Add the mask to qValues to make invalid moves have very low values
        const maskedQValues = qValues.add(maskTensor);
        
        return tf.argMax(maskedQValues, 1).dataSync()[0];
      });
    }
  }

  remember(state, action, reward, nextState, done) {
    if (this.memory.length >= this.maxMemorySize) {
      this.memory.shift();
    }
    
    // Check if the experience already exists in memory
    const experienceExists = this.memory.some(experience => 
      this.areExperiencesEqual(experience, [state, action, reward, nextState, done])
    );

    // Only add the experience if it doesn't already exist
    if (!experienceExists) {
      this.memory.push([state, action, reward, nextState, done]);
    }
  }

  areExperiencesEqual(exp1, exp2) {
    return (
      JSON.stringify(exp1[0]) === JSON.stringify(exp2[0]) && // state
      exp1[1] === exp2[1] && // action
      exp1[2] === exp2[2] && // reward
      JSON.stringify(exp1[3]) === JSON.stringify(exp2[3]) && // nextState
      exp1[4] === exp2[4] // done
    );
  }

  async replay() {
    if (this.memory.length < this.batchSize) return null;

    console.log(`Replay history size: ${this.memory.length}`);

    const batch = this.getRandomBatch(this.batchSize);

    const states = batch.map(experience => experience[0]);
    const actions = batch.map(experience => experience[1]);
    const rewards = batch.map(experience => experience[2]);
    const nextStates = batch.map(experience => experience[3]);
    const dones = batch.map(experience => experience[4]);

    const oneHotStates = states.map(state => this.model.convertToOneHot(state));
    const oneHotNextStates = nextStates.map(state => this.model.convertToOneHot(state));

    let loss;

    try {
      const x = tf.tensor2d(oneHotStates);
      let y;

      tf.tidy(() => {
        const currentQs = this.model.predict(x);

        const nextStatesTensor = tf.tensor2d(oneHotNextStates);
        const nextQsMain = this.model.predict(nextStatesTensor);
        const nextQsTarget = this.model.predict(nextStatesTensor, true);

        const actionsTensor = tf.tensor1d(actions, 'int32');
        const rewardsTensor = tf.tensor1d(rewards, 'float32');
        const notDones = tf.tensor1d(dones.map(done => done ? 0 : 1), 'float32');

        const validMovesMasks = nextStates.map(state => {
          const mask = new Array(9).fill(-Infinity);
          const validMoves = this.getValidMoves(state);
          validMoves.forEach(move => mask[move] = 0);
          return mask;
        });
        const validMovesMasksTensor = tf.tensor2d(validMovesMasks);

        const maskedNextQsMain = nextQsMain.add(validMovesMasksTensor);
        const bestActions = maskedNextQsMain.argMax(1).toInt();

        const batchIndices = tf.range(0, this.batchSize, 1, 'int32');
        const indices = tf.stack([batchIndices, bestActions], 1);

        const targetQValues = tf.gatherND(nextQsTarget, indices).mul(notDones);
        const Q_targets = rewardsTensor.add(targetQValues.mul(this.gamma));

        const mask = tf.oneHot(actionsTensor, 9).toFloat();
        const Q_targets_expanded = Q_targets.expandDims(1);
        
        y = tf.keep(currentQs.mul(tf.scalar(1).sub(mask)).add(Q_targets_expanded.mul(mask)));
      });

      // Train the model
      loss = await this.model.train(x, y);
      
      x.dispose();
      y.dispose();

    } catch (error) {
      console.error("Error during replay:", error);
    }

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

  // Add this method to the DQNAgent class
  getValidMoves(state) {
    return state.reduce((validMoves, cell, index) => {
      if (cell === 0) validMoves.push(index);
      return validMoves;
    }, []);
  }
}
