class DQNAgent {
  constructor(epsilonStart = 0.7, epsilonEnd = 0.1, gamma = 0.999, batchSize = 500, maxMemorySize = 100000, fixedEpsilonSteps = 1000, decayEpsilonSteps = 1000) {
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
		    return Math.floor(Math.random() * 9);
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

	  // Extract states, actions, rewards, nextStates, and done flags from the batch
	  const states = batch.map(experience => experience[0]);
	  const actions = batch.map(experience => experience[1]);
	  const rewards = batch.map(experience => experience[2]);
	  const nextStates = batch.map(experience => experience[3]);
	  const dones = batch.map(experience => experience[4]);

	  // Initialize tensors
	  const x = tf.tensor2d(states); // Input states
	  let y; // Targets

	  tf.tidy(() => {
	    const currentQs = this.model.predict(states); // Main network predictions for current states

	    // Get illegal moves mask for current states
	    const illegalMovesMask = this.getIllegalMovesMask(states); // shape [batchSize, 9]
	    const legalMovesMask = tf.scalar(1).sub(illegalMovesMask); // shape [batchSize, 9]

	    // Zero out Q-values for illegal moves in currentQs
	    const maskedCurrentQs = currentQs.mul(legalMovesMask); // shape [batchSize, 9]

	    const nextQsMain = this.model.predict(nextStates); // Main network predictions for next states
	    const nextQsTarget = this.model.predict(nextStates, true); // Target network predictions for next states

	    // Convert actions, rewards, dones to tensors
	    const actionsTensor = tf.tensor1d(actions, 'int32');
	    const rewardsTensor = tf.tensor1d(rewards, 'float32');
	    const notDones = tf.tensor1d(dones.map(done => done ? 0 : 1), 'float32');

	    // Get the best actions from the main network's predictions
	    const bestActions = nextQsMain.argMax(1).toInt();

	    // Prepare indices to gather Q-values from the target network
	    const batchIndices = tf.range(0, this.batchSize, 1, 'int32');
	    const indices = tf.stack([batchIndices, bestActions], 1);

	    // Gather the Q-values of the best actions from the target network
	    const targetQValues = tf.gatherND(nextQsTarget, indices).mul(notDones);

	    // Compute the target Q-values
	    const Q_targets = rewardsTensor.add(targetQValues.mul(this.gamma));

	    // Update current Q-values with target Q-values for the taken actions
	    const mask = tf.oneHot(actionsTensor, 9).toFloat();

	    const Q_targets_expanded = Q_targets.expandDims(1); // shape [batchSize, 1]
	    const targetQsUpdate = mask.mul(Q_targets_expanded); // shape [batchSize, 9]
	    const updatedQs = maskedCurrentQs.add(targetQsUpdate);

	    y = tf.keep(updatedQs);
	  });

	  // Train the model
	  const loss = await this.model.train(x, y);
	  x.dispose();
	  y.dispose();
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

  getIllegalMovesMask(states) {
    // states is an array of shape [batchSize, 9]
    return tf.tidy(() => {
      const statesTensor = tf.tensor2d(states); // shape [batchSize, 9]
      // Cells are empty if value is 0
      const illegalMovesMask = statesTensor.notEqual(tf.scalar(0)).toFloat(); // shape [batchSize, 9]
      return illegalMovesMask; // 1 where move is illegal, 0 where move is legal
    });
  }
}