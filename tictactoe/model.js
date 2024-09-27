class TicTacToeModel {
  constructor() {
    this.mainModel = this.createModel();
    this.targetModel = this.createModel();
    this.updateTargetModel(); // Initialize target model with main model weights
    this.episodeCount = 0;
  }

  createModel() {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 256, activation: 'relu', inputShape: [9] }));
    model.add(tf.layers.dense({ units: 256, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 9, activation: 'linear' }));
    model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });
    return model;
  }

  predict(state, useTargetNetwork = false) {
    const model = useTargetNetwork ? this.targetModel : this.mainModel;
    let tensor;
    if (Array.isArray(state[0])) {
      tensor = tf.tensor2d(state);
    } else {
      tensor = tf.tensor2d([state], [1, 9]);
    }
    const prediction = model.predict(tensor);
    tensor.dispose();
    return prediction;
  }

  async train(states, targets) {
    const stateTensor = tf.tensor2d(states);
    const targetTensor = tf.tensor2d(targets);

    const result = await this.mainModel.fit(stateTensor, targetTensor, {
      epochs: 1,
      shuffle: true,
    });

    stateTensor.dispose();
    targetTensor.dispose();

    this.episodeCount++;
    if (this.episodeCount % 5 === 0) { // Updated frequency
      this.updateTargetModel();
    }

    return result;
  }

  updateTargetModel() {
    this.targetModel.setWeights(this.mainModel.getWeights());
  }
}