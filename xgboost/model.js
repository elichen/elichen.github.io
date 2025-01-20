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
    constructor(learningRate = 0.03, maxDepth = 5, nEstimators = 1000) {
        this.learningRate = learningRate;
        this.maxDepth = maxDepth;
        this.nEstimators = nEstimators;
        this.trees = [];
        this.initialPrediction = 0;
        this.minSamplesSplit = 10;
        this.subsampleRatio = 0.8;
    }

    async train(X, y, progressCallback) {
        const features = await X.array();
        const targets = await y.array();
        
        // Initialize with base prediction
        const meanTarget = targets.reduce((a, b) => a + b, 0) / targets.length;
        const p = Math.min(Math.max(meanTarget, 1e-15), 1 - 1e-15);
        this.initialPrediction = Math.log(p / (1 - p));
        
        let predictions = Array(features.length).fill(this.initialPrediction);
        let completedTrees = 0;
        
        for (let i = 0; i < this.nEstimators; i++) {
            const probabilities = predictions.map(p => 1 / (1 + Math.exp(-p)));
            const gradients = targets.map((t, j) => t - probabilities[j]);
            
            const tree = new DecisionTree(this.maxDepth);
            const sampleIndices = this.subsample(features.length, this.subsampleRatio);
            const sampledFeatures = sampleIndices.map(idx => features[idx]);
            const sampledGradients = sampleIndices.map(idx => gradients[idx]);
            
            await tree.train(
                tf.tensor2d(sampledFeatures), 
                tf.tensor1d(sampledGradients)
            );
            
            const treePredict = tree.predict(features);
            const bestLR = this.lineSearch(predictions, treePredict, targets);
            
            predictions = predictions.map((pred, j) => 
                pred + bestLR * treePredict[j]);
            
            this.trees.push(tree);
            completedTrees++;
            progressCallback((completedTrees / this.nEstimators) * 100);
        }
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
        
        console.log('Initial prediction:', this.initialPrediction);
        console.log('Sample feature vector:', features[0]);
        
        for (const tree of this.trees) {
            const treePredict = tree.predict(features);
            predictions = predictions.map((pred, i) => 
                pred + this.learningRate * treePredict[i]);
        }
        
        // Log intermediate values
        console.log('Raw predictions:', predictions.slice(0, 5));
        
        // Return probabilities with protection against extreme values
        const probabilities = predictions.map(p => {
            const logit = Math.min(Math.max(p, -100), 100); // Prevent overflow
            const prob = 1 / (1 + Math.exp(-logit));
            console.log('Converting prediction:', p, 'to probability:', prob);
            return prob;
        });
        
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