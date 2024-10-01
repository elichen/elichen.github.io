class TicTacToeModel {
  constructor() {
    this.mainModel = this.createModel();
    this.targetModel = this.createModel();
    this.updateTargetModel(); // Initialize target model with main model weights
    this.episodeCount = 0;
  }

  createModel() {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 128, activation: 'relu', inputShape: [9] }));
    model.add(tf.layers.dense({ units: 256, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 256, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 9, activation: 'linear' }));
    model.compile({ optimizer: tf.train.adam(0.0001), loss: 'meanSquaredError' });
    return model;
  }

  predict(state, useTargetNetwork = false) {
    const model = useTargetNetwork ? this.targetModel : this.mainModel;
    // Check if it's a single state or a batch of states
    if (Array.isArray(state[0])) {
      // It's a batch of states
      return model.predict(tf.tensor2d(state));
    } else {
      // It's a single state
      return model.predict(tf.tensor2d([state], [1, 9]));
    }
  }

  async train(states, targets) {
    const result = await this.mainModel.fit(tf.tensor2d(states), tf.tensor2d(targets), {
      epochs: 1
    });

    this.episodeCount++;
    if (this.episodeCount % 10 === 0) {
      this.updateTargetModel();
    }

    return result;
  }

  updateTargetModel() {
    this.targetModel.setWeights(this.mainModel.getWeights());
  }
}