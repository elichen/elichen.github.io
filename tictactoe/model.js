class TicTacToeModel {
  constructor() {
    this.model = this.createModel();
  }

  createModel() {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 256, activation: 'relu', inputShape: [9] }));
    model.add(tf.layers.dense({ units: 256, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 9, activation: 'linear' }));
    model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });
    return model;
  }

  predict(state) {
    // Check if it's a single state or a batch of states
    if (Array.isArray(state[0])) {
      // It's a batch of states
      return this.model.predict(tf.tensor2d(state));
    } else {
      // It's a single state
      return this.model.predict(tf.tensor2d([state], [1, 9]));
    }
  }

  async train(states, targets) {
    // Ensure states and targets are 2D tensors
    return this.model.fit(tf.tensor2d(states), tf.tensor2d(targets), {
      epochs: 1,
      shuffle: true,
    });
  }
}