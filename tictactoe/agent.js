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

	  // Initialize tensors
	  const x = tf.tensor2d(states); // Input states
	  let y; // Targets

	  // Use tf.tidy to manage memory
	  tf.tidy(() => {
	    const currentQs = this.model.predict(states); // Main network predictions for current states
	    const nextQsMain = this.model.predict(nextStates); // Main network predictions for next states
	    const nextQsTarget = this.model.predict(nextStates, true); // Target network predictions for next states

	    // Extract actions, rewards, and done flags from the batch
	    const actions = tf.tensor1d(batch.map(experience => experience[1]), 'int32');
	    const rewards = tf.tensor1d(batch.map(experience => experience[2]), 'float32');
	    const dones = batch.map(experience => experience[4]);
	    const notDones = tf.tensor1d(dones.map(done => done ? 0 : 1), 'float32');

	    // Get the best actions from the main network's predictions
	    const bestActions = nextQsMain.argMax(1).toInt();

	    // Prepare indices to gather Q-values from the target network
	    const batchIndices = tf.range(0, this.batchSize, 1, 'int32');
	    const indices = tf.stack([batchIndices, bestActions], 1);

	    // Gather the Q-values of the best actions from the target network
	    const targetQValues = tf.gatherND(nextQsTarget, indices).mul(notDones);

	    // Compute the target Q-values
	    const Q_targets = rewards.add(targetQValues.mul(this.gamma));

	    // Create a mask for updating the Q-values
	    const mask = tf.oneHot(actions, 9).toFloat();

	    // Update the Q-values
	    const Q_targets_expanded = Q_targets.expandDims(1);
	    const targetQsUpdate = mask.mul(Q_targets_expanded);
	    const updatedCurrentQs = currentQs.mul(tf.scalar(1).sub(mask));
	    const updatedQs = updatedCurrentQs.add(targetQsUpdate);

	    // Keep the updated Q-values tensor
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
}