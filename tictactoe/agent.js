class PPOAgent {
  constructor() {
    this.model = null;
    this.isLoaded = false;
  }

  async loadModel() {
    try {
      console.log('Loading PPO model...');
      this.model = await tf.loadLayersModel('./tfjs_model/model.json');
      this.isLoaded = true;
      console.log('✅ PPO model loaded successfully!');
      return true;
    } catch (error) {
      console.error('Failed to load PPO model:', error);
      this.isLoaded = false;
      return false;
    }
  }

  act(state, validMoves) {
    if (!this.isLoaded || !this.model) {
      throw new Error('Model not loaded. Please train and export a model first.');
    }

    return tf.tidy(() => {
      // Convert state to tensor
      const stateTensor = tf.tensor2d([state], [1, 9]);

      // Get action logits from the model
      const actionLogits = this.model.predict(stateTensor);

      // Apply action masking to prevent invalid moves
      const mask = new Array(9).fill(-Infinity);
      validMoves.forEach(move => mask[move] = 0);
      const maskTensor = tf.tensor1d(mask);

      // Add mask to logits (invalid positions become -Infinity)
      const maskedLogits = actionLogits.add(maskTensor);

      // Select action with highest logit (greedy for deployed model)
      const action = tf.argMax(maskedLogits, 1).dataSync()[0];
      return action;
    });
  }
}