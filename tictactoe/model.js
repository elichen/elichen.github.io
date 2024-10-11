class TicTacToeModel {
  constructor() {
    this.mainModel = this.createModel();
    this.targetModel = this.createModel();
    this.updateTargetModel(); // Initialize target model with main model weights
    this.episodeCount = 0;
  }

  createModel() {
    const model = tf.sequential();
    // Change input shape from 9 to 27
    model.add(tf.layers.dense({ units: 100, activation: 'relu', inputShape: [27] }));
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
      // Check if state is already a tensor
      const stateTensor = state instanceof tf.Tensor ? state : tf.tensor2d([this.convertToOneHot(state)]);
      return model.predict(stateTensor);
    });
  }

	async train(stateTensor, targetTensor) {
	  const fitConfig = {
	    epochs: 1,
	    verbose: 0  // Set to 0 to suppress console output
	  };

	  const result = await this.mainModel.fit(stateTensor, targetTensor, fitConfig);
	  const loss = result.history.loss[0];

	  this.episodeCount++;
	  if (this.episodeCount % 100 === 0) {
	    this.updateTargetModel();
	  }

	  return loss;
	}

  updateTargetModel() {
    this.targetModel.setWeights(this.mainModel.getWeights());
  }

  // Add a new method to convert the state to 1-hot encoding
  convertToOneHot(state) {
    return state.flatMap(value => {
      if (value === 1) return [1, 0, 0];
      if (value === -1) return [0, 1, 0];
      return [0, 0, 1]; // Empty cell
    });
  }
}