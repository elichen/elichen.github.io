class DecisionTree {
    constructor(maxDepth = 3) {
        this.maxDepth = maxDepth;
        this.root = null;
        this.minSamplesSplit = 5;  // Add minimum samples to split
        this.minImpurityDecrease = 0.001;  // Add minimum impurity decrease
    }

    async train(X, y) {
        const features = await X.array();
        const targets = await y.array();
        this.root = this.buildTree(features, targets, 0);
    }

    buildTree(features, targets, depth) {
        if (depth >= this.maxDepth || 
            this.isLeaf(targets) || 
            features.length < this.minSamplesSplit) {
            return {
                isLeaf: true,
                value: this.calculatePrediction(targets)
            };
        }

        const bestSplit = this.findBestSplit(features, targets);
        if (!bestSplit || bestSplit.gain < this.minImpurityDecrease) {
            return {
                isLeaf: true,
                value: this.calculatePrediction(targets)
            };
        }

        const { feature, threshold, leftIndices, rightIndices, gain } = bestSplit;

        // Don't split if either child would be too small
        if (leftIndices.length < this.minSamplesSplit || 
            rightIndices.length < this.minSamplesSplit) {
            return {
                isLeaf: true,
                value: this.calculatePrediction(targets)
            };
        }

        const leftFeatures = leftIndices.map(i => features[i]);
        const leftTargets = leftIndices.map(i => targets[i]);
        const rightFeatures = rightIndices.map(i => features[i]);
        const rightTargets = rightIndices.map(i => targets[i]);

        return {
            isLeaf: false,
            feature: feature,
            threshold: threshold,
            gain: gain,
            left: this.buildTree(leftFeatures, leftTargets, depth + 1),
            right: this.buildTree(rightFeatures, rightTargets, depth + 1)
        };
    }

    findBestSplit(features, targets) {
        let bestGain = -Infinity;
        let bestSplit = null;

        // Calculate base impurity
        const baseImpurity = this.calculateImpurity(targets);
        if (baseImpurity === 0) return null;

        // Try each feature
        for (let feature = 0; feature < features[0].length; feature++) {
            const values = features.map(f => f[feature]);
            
            // Get unique values excluding NaN
            const uniqueValues = [...new Set(values.filter(v => !isNaN(v)))].sort((a, b) => a - b);
            
            if (uniqueValues.length < 2) continue;

            // Try more split points for better granularity
            for (let i = 1; i < uniqueValues.length; i++) {
                const threshold = (uniqueValues[i-1] + uniqueValues[i]) / 2;
                
                const leftIndices = [];
                const rightIndices = [];

                for (let j = 0; j < features.length; j++) {
                    if (features[j][feature] <= threshold) {
                        leftIndices.push(j);
                    } else {
                        rightIndices.push(j);
                    }
                }

                // Skip if split is too unbalanced
                const minSize = Math.max(5, features.length * 0.05);
                if (leftIndices.length < minSize || rightIndices.length < minSize) continue;

                const gain = this.calculateGain(targets, leftIndices, rightIndices);
                if (gain > bestGain) {
                    bestGain = gain;
                    bestSplit = { feature, threshold, leftIndices, rightIndices, gain };
                }
            }
        }

        return bestSplit;
    }

    calculateImpurity(targets) {
        const mean = targets.reduce((a, b) => a + b, 0) / targets.length;
        if (mean === 0 || mean === 1) return 0;
        return mean * (1 - mean);  // Gini impurity for binary classification
    }

    calculateGain(targets, leftIndices, rightIndices) {
        const parentImpurity = this.calculateImpurity(targets);
        const leftImpurity = this.calculateImpurity(leftIndices.map(i => targets[i]));
        const rightImpurity = this.calculateImpurity(rightIndices.map(i => targets[i]));
        
        const leftWeight = leftIndices.length / targets.length;
        const rightWeight = rightIndices.length / targets.length;
        
        return parentImpurity - (leftWeight * leftImpurity + rightWeight * rightImpurity);
    }

    isLeaf(targets) {
        return new Set(targets).size === 1;
    }

    calculatePrediction(targets) {
        return targets.reduce((a, b) => a + b, 0) / targets.length;
    }

    predict(features) {
        return features.map(feature => this.predictSingle(feature, this.root));
    }

    predictSingle(feature, node) {
        if (node.isLeaf) {
            return node.value;
        }
        if (feature[node.feature] <= node.threshold) {
            return this.predictSingle(feature, node.left);
        }
        return this.predictSingle(feature, node.right);
    }
}

class XGBoostModel {
    constructor(learningRate = 0.05, maxDepth = 6, nEstimators = 500) {
        this.learningRate = learningRate;
        this.maxDepth = maxDepth;
        this.nEstimators = nEstimators;
        this.trees = [];
        this.initialPrediction = 0;
        this.patience = 20;  // Early stopping patience
        this.bestLoss = Infinity;
        this.treesWithoutImprovement = 0;
    }

    async train(X, y, progressCallback) {
        const features = await X.array();
        const targets = await y.array();
        
        // Split data for validation
        const [trainFeatures, trainTargets, valFeatures, valTargets] = this.splitValidation(features, targets);
        
        // Initialize with base prediction
        const meanTarget = trainTargets.reduce((a, b) => a + b, 0) / trainTargets.length;
        this.initialPrediction = Math.log(meanTarget / (1 - meanTarget));
        
        let trainPredictions = Array(trainFeatures.length).fill(this.initialPrediction);
        let valPredictions = Array(valFeatures.length).fill(this.initialPrediction);
        let bestAccuracy = 0;
        let completedTrees = 0;
        
        for (let i = 0; i < this.nEstimators; i++) {
            // Calculate gradients (residuals)
            const probabilities = trainPredictions.map(p => 1 / (1 + Math.exp(-p)));
            const gradients = trainTargets.map((t, j) => t - probabilities[j]);
            
            // Train tree on gradients with subsample
            const tree = new DecisionTree(this.maxDepth);
            const sampleIndices = this.subsample(trainFeatures.length, 0.8);
            const sampledFeatures = sampleIndices.map(idx => trainFeatures[idx]);
            const sampledGradients = sampleIndices.map(idx => gradients[idx]);
            
            await tree.train(
                tf.tensor2d(sampledFeatures), 
                tf.tensor1d(sampledGradients)
            );
            
            // Update predictions with line search
            const trainTreePredict = tree.predict(trainFeatures);
            const valTreePredict = tree.predict(valFeatures);
            
            const bestLR = this.lineSearch(trainPredictions, trainTreePredict, trainTargets);
            
            trainPredictions = trainPredictions.map((pred, j) => 
                pred + bestLR * trainTreePredict[j]);
            valPredictions = valPredictions.map((pred, j) => 
                pred + bestLR * valTreePredict[j]);
            
            // Calculate validation loss
            const valProbs = valPredictions.map(p => 1 / (1 + Math.exp(-p)));
            const valLoss = this.calculateLoss(valProbs, valTargets);
            
            // Early stopping check
            if (valLoss < this.bestLoss) {
                this.bestLoss = valLoss;
                this.treesWithoutImprovement = 0;
            } else {
                this.treesWithoutImprovement++;
                if (this.treesWithoutImprovement >= this.patience) {
                    // Make sure progress shows 100% when early stopping
                    progressCallback(100);
                    break;
                }
            }
            
            this.trees.push(tree);
            completedTrees++;
            
            // Update progress based on completed trees
            progressCallback((completedTrees / this.nEstimators) * 100);
        }
    }

    splitValidation(features, targets, valSize = 0.2) {
        const numVal = Math.floor(features.length * valSize);
        const indices = Array.from({length: features.length}, (_, i) => i);
        
        // Shuffle indices
        for (let i = indices.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [indices[i], indices[j]] = [indices[j], indices[i]];
        }
        
        const trainIndices = indices.slice(numVal);
        const valIndices = indices.slice(0, numVal);
        
        return [
            trainIndices.map(i => features[i]),
            trainIndices.map(i => targets[i]),
            valIndices.map(i => features[i]),
            valIndices.map(i => targets[i])
        ];
    }

    calculateAccuracy(predictions, actual) {
        let correct = 0;
        let total = 0;
        
        for (let i = 0; i < predictions.length; i++) {
            const predicted = predictions[i] >= 0.5 ? 1 : 0;
            if (predicted === actual[i]) {
                correct++;
            }
            total++;
        }
        
        return (correct / total) * 100;
    }

    getPredictionDistribution(predictions) {
        const bins = {
            '0.0-0.2': 0,
            '0.2-0.4': 0,
            '0.4-0.6': 0,
            '0.6-0.8': 0,
            '0.8-1.0': 0
        };
        predictions.forEach(p => {
            if (p < 0.2) bins['0.0-0.2']++;
            else if (p < 0.4) bins['0.2-0.4']++;
            else if (p < 0.6) bins['0.4-0.6']++;
            else if (p < 0.8) bins['0.6-0.8']++;
            else bins['0.8-1.0']++;
        });
        return Object.fromEntries(
            Object.entries(bins).map(([k, v]) => [k, (v / predictions.length * 100).toFixed(1) + '%'])
        );
    }

    async predict(X) {
        const features = await X.array();
        let predictions = Array(features.length).fill(this.initialPrediction);
        
        for (const tree of this.trees) {
            const treePredict = tree.predict(features);
            predictions = predictions.map((pred, i) => 
                pred + this.learningRate * treePredict[i]);
        }
        
        // Return a proper TensorFlow tensor
        const probabilities = predictions.map(p => 1 / (1 + Math.exp(-p)));
        return tf.tensor1d(probabilities);
    }

    subsample(length, ratio) {
        const indices = Array.from({length}, (_, i) => i);
        const n = Math.floor(length * ratio);
        for (let i = length - 1; i > length - n - 1; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [indices[i], indices[j]] = [indices[j], indices[i]];
        }
        return indices.slice(length - n);
    }

    // Add line search for optimal learning rate
    lineSearch(predictions, treePredict, targets) {
        const learningRates = [0.01, 0.05, 0.1, 0.2, 0.5];
        let bestLR = this.learningRate;
        let bestLoss = Infinity;
        
        for (const lr of learningRates) {
            const newPreds = predictions.map((p, i) => p + lr * treePredict[i]);
            const probs = newPreds.map(p => 1 / (1 + Math.exp(-p)));
            const loss = this.calculateLoss(probs, targets);
            
            if (loss < bestLoss) {
                bestLoss = loss;
                bestLR = lr;
            }
        }
        
        return bestLR;
    }

    calculateLoss(predictions, targets) {
        return predictions.reduce((sum, pred, i) => {
            const p = Math.max(Math.min(pred, 1 - 1e-15), 1e-15);
            return sum - (targets[i] * Math.log(p) + (1 - targets[i]) * Math.log(1 - p));
        }, 0) / predictions.length;
    }
} 