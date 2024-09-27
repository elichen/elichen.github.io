class TicTacToeModel {
  constructor() {
    this.model = this.createModel();
  }

  createModel() {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 64, activation: 'relu', inputShape: [9] }));
    model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 9, activation: 'linear' }));
    model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });
    return model;
  }

  predict(state) {
    console.log('State in predict:', state);
    console.log('State shape in predict:', state.length);
    
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
    console.log('States in train:', states);
    console.log('States shape in train:', states.length, states[0].length);
    console.log('Targets in train:', targets);
    console.log('Targets shape in train:', targets.length, targets[0].length);
    // Ensure states and targets are 2D tensors
    return this.model.fit(tf.tensor2d(states), tf.tensor2d(targets), {
      epochs: 1,
      shuffle: true,
    });
  }
}