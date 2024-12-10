class DQNAgent {
  constructor(
    epsilonStart = 0.3,
    epsilonEnd = 0.1,
    gamma = 0.995,
    batchSize = 64,
    maxMemorySize = 1000,
    fixedEpsilonSteps = 100,
    decayEpsilonSteps = 1000
  ) {
    this.model = new TicTacToeModel();
    this.epsilon = epsilonStart;
    this.epsilonEnd = epsilonEnd;
    this.gamma = gamma;
    this.batchSize = batchSize;
    this.maxMemorySize = maxMemorySize;
    this.memory = []; // Using an array instead of a Map
    this.memoryIndex = 0; // Pointer for circular buffer

    this.fixedEpsilonSteps = fixedEpsilonSteps;
    this.decayEpsilonRate = (epsilonStart - epsilonEnd) / decayEpsilonSteps;
    this.currentStep = 0;
    this.isTraining = false; // We will simply check conditions and train immediately.
    this.frameCount = 0;
    this.trainFrequency = 4;
    this.visualization = new Visualization();
    this.memorySize = 0; // Track how many elements are actually in memory
  }

  act(state, isTraining = true, validMoves) {
    if (isTraining && Math.random() < this.epsilon) {
      return validMoves[Math.floor(Math.random() * validMoves.length)];
    } else {
      return tf.tidy(() => {
        const qValues = this.model.predict(state);

        const mask = new Array(9).fill(-Infinity);
        validMoves.forEach(move => mask[move] = 0);
        const maskTensor = tf.tensor1d(mask);

        const maskedQValues = qValues.add(maskTensor);
        return tf.argMax(maskedQValues, 1).dataSync()[0];
      });
    }
  }

  async remember(state, action, reward, nextState, done) {
    // Store experience in replay buffer
    const experience = [state, action, reward, nextState, done];

    if (this.memorySize < this.maxMemorySize) {
      this.memory.push(experience);
      this.memorySize++;
    } else {
      // Overwrite oldest experience
      this.memory[this.memoryIndex] = experience;
    }

    this.memoryIndex = (this.memoryIndex + 1) % this.maxMemorySize;

    this.frameCount++;
    if (this.frameCount % this.trainFrequency === 0 && this.memorySize >= this.batchSize) {
      return await this.replay();
    }
    return null;
  }

  async replay() {
    if (this.memorySize < this.batchSize) return null;

    // No need for a training queue now, as we won't be calling replay concurrently.
    this.isTraining = true;

    try {
      const batch = this.getRandomBatch(this.batchSize);

      const states = batch.map(ex => ex[0]);
      const actions = batch.map(ex => ex[1]);
      const rewards = batch.map(ex => ex[2]);
      const nextStates = batch.map(ex => ex[3]);
      const dones = batch.map(ex => ex[4]);

      const oneHotStates = states.map(state => this.model.convertToOneHot(state));
      const oneHotNextStates = nextStates.map(state => this.model.convertToOneHot(state));

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

        y = tf.keep(
          currentQs.mul(tf.scalar(1).sub(mask)).add(Q_targets_expanded.mul(mask))
        );
      });

      const loss = await this.model.train(x, y);
      x.dispose();
      y.dispose();

      return loss;
    } catch (error) {
      console.error("Error during replay:", error);
      return null;
    } finally {
      this.isTraining = false;
    }
  }

  getRandomBatch(batchSize) {
    const batch = [];
    for (let i = 0; i < batchSize; i++) {
      const randomIndex = Math.floor(Math.random() * this.memorySize);
      batch.push(this.memory[randomIndex]);
    }
    return batch;
  }

  getMemorySize() {
    return this.memorySize;
  }

  decayEpsilon() {
    this.currentStep++;
    if (this.currentStep <= this.fixedEpsilonSteps) {
      // Keep epsilon fixed initially
    } else if (this.epsilon > 0) {
      this.epsilon = Math.max(this.epsilonEnd, this.epsilon - this.decayEpsilonRate);
    }
  }

  getValidMoves(state) {
    return state.reduce((validMoves, cell, index) => {
      if (cell === 0) validMoves.push(index);
      return validMoves;
    }, []);
  }
}