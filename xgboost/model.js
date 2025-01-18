class XGBoostModel {
    constructor(learningRate = 0.1, maxDepth = 3, nEstimators = 100) {
        this.learningRate = learningRate;
        this.maxDepth = maxDepth;
        this.nEstimators = nEstimators;
        this.models = [];
    }

    async train(X, y, progressCallback) {
        let predictions = tf.zeros([X.shape[0]]);
        
        for (let i = 0; i < this.nEstimators; i++) {
            const residuals = y.sub(predictions);
            
            const model = tf.sequential({
                layers: [
                    tf.layers.dense({units: 16, activation: 'relu', inputShape: [X.shape[1]]}),
                    tf.layers.dense({units: 1})
                ]
            });
            
            model.compile({
                optimizer: tf.train.adam(this.learningRate),
                loss: 'meanSquaredError'
            });
            
            await model.fit(X, residuals, {
                epochs: 1,
                verbose: 0
            });
            
            const weakPredictions = model.predict(X);
            predictions = predictions.add(tf.mul(weakPredictions.squeeze(), this.learningRate));
            
            this.models.push(model);
            progressCallback((i + 1) / this.nEstimators * 100);
        }
    }

    predict(X) {
        let predictions = tf.zeros([X.shape[0]]);
        
        for (const model of this.models) {
            const weakPredictions = model.predict(X);
            predictions = predictions.add(tf.mul(weakPredictions.squeeze(), this.learningRate));
        }
        
        return predictions;
    }
} 