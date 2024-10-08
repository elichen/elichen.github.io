class TicTacToeModel {
  constructor() {
    this.mainModel = this.createModel();
    this.targetModel = this.createModel();
    this.updateTargetModel(); // Initialize target model with main model weights
    this.episodeCount = 0;
  }

  createModel() {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 100, activation: 'relu', inputShape: [9] }));
    model.add(tf.layers.dense({ units: 160, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 160, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 100, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 9, activation: 'linear' }));
    model.compile({ optimizer: tf.train.adam(0.001), loss: 'meanSquaredError' });
    return model;
  }

  predict(state, useTargetNetwork = false) {
    const model = useTargetNetwork ? this.targetModel : this.mainModel;
    return tf.tidy(() => {
      const stateTensor = Array.isArray(state[0]) ? tf.tensor2d(state) : tf.tensor2d([state], [1, 9]);
      return model.predict(stateTensor);
    });
  }

  async train(states, targets) {
    const fitConfig = {
      epochs: 1,
      verbose: 0  // Set to 0 to suppress console output
    };

    const stateTensor = tf.tensor2d(states);
    const targetTensor = tf.tensor2d(targets);

    try {
      const result = await this.mainModel.fit(stateTensor, targetTensor, fitConfig);
      const loss = result.history.loss[0];

      this.episodeCount++;
      if (this.episodeCount % 100 === 0) {
        this.updateTargetModel();
      }

      return loss;
    } finally {
      stateTensor.dispose();
      targetTensor.dispose();
    }
  }

  updateTargetModel() {
    this.targetModel.setWeights(this.mainModel.getWeights());
  }
}