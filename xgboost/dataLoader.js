async function loadData(filePath, progressCallback) {
    const response = await fetch(filePath);
    const text = await response.text();
    const rows = text.split('\n');
    const headers = rows[0].split(',');
    const data = [];
    
    // Define features and their preprocessing rules
    const features = {
        'Pclass': {type: 'numeric', default: 3},
        'Age': {type: 'numeric', default: 30},
        'SibSp': {type: 'numeric', default: 0},
        'Parch': {type: 'numeric', default: 0},
        'Fare': {type: 'numeric', default: 32.2},
        'Sex': {type: 'categorical', values: ['male', 'female']},
        'Embarked': {type: 'categorical', values: ['S', 'C', 'Q']}
    };
    
    for (let i = 1; i < rows.length; i++) {
        if (rows[i].trim() === '') continue;
        
        const values = rows[i].split(',');
        const row = {};
        
        headers.forEach((header, index) => {
            const headerTrim = header.trim();
            const value = values[index]?.trim();
            
            if (features[headerTrim]) {
                if (features[headerTrim].type === 'numeric') {
                    row[headerTrim] = value && !isNaN(value) ? 
                        parseFloat(value) : features[headerTrim].default;
                } else if (features[headerTrim].type === 'categorical') {
                    // One-hot encode categorical variables
                    features[headerTrim].values.forEach(val => {
                        row[`${headerTrim}_${val}`] = value === val ? 1 : 0;
                    });
                }
            } else if (headerTrim === 'Survived') {
                row['target'] = parseInt(value);
            }
        });
        
        data.push(row);
        progressCallback((i / rows.length) * 100);
    }
    
    // Get flattened feature list including one-hot encoded features
    const flattenedFeatures = Object.entries(features).flatMap(([key, value]) => {
        if (value.type === 'categorical') {
            return value.values.map(val => `${key}_${val}`);
        }
        return key;
    });
    
    return { data, headers, features: flattenedFeatures };
}

function preprocessData(data, features, target) {
    // Normalize numeric features
    const numericStats = {};
    features.forEach(feature => {
        if (!feature.includes('_')) {  // Numeric features don't have underscore
            const values = data.map(row => row[feature]);
            const mean = values.reduce((a, b) => a + b, 0) / values.length;
            const std = Math.sqrt(values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length);
            numericStats[feature] = { mean, std };
        }
    });

    const X = data.map(row => features.map(feature => {
        if (!feature.includes('_')) {
            // Normalize numeric features
            return (row[feature] - numericStats[feature].mean) / 
                   (numericStats[feature].std || 1);
        }
        // Binary features (one-hot encoded) don't need normalization
        return row[feature];
    }));
    
    const y = data.map(row => row[target]);
    
    return {
        X: tf.tensor2d(X),
        y: tf.tensor1d(y),
        numericStats  // Return stats for use with test data
    };
}

function calculateMetrics(actual, predicted) {
    const n = actual.length;
    
    // Round predictions for binary classification metrics
    const roundedPredictions = predicted.map(p => p >= 0.5 ? 1 : 0);
    
    // Calculate Accuracy
    const accuracy = actual.reduce((sum, val, i) => 
        sum + (roundedPredictions[i] === val ? 1 : 0), 0) / n;
    
    // Calculate MSE using raw probabilities
    const mse = actual.reduce((sum, val, i) => 
        sum + Math.pow(val - predicted[i], 2), 0) / n;
    
    // Calculate RMSE
    const rmse = Math.sqrt(mse);
    
    // Calculate confusion matrix
    const tp = actual.reduce((sum, val, i) => 
        sum + (val === 1 && roundedPredictions[i] === 1 ? 1 : 0), 0);
    const tn = actual.reduce((sum, val, i) => 
        sum + (val === 0 && roundedPredictions[i] === 0 ? 1 : 0), 0);
    const fp = actual.reduce((sum, val, i) => 
        sum + (val === 0 && roundedPredictions[i] === 1 ? 1 : 0), 0);
    const fn = actual.reduce((sum, val, i) => 
        sum + (val === 1 && roundedPredictions[i] === 0 ? 1 : 0), 0);
    
    // Calculate precision and recall
    const precision = tp / (tp + fp) || 0;
    const recall = tp / (tp + fn) || 0;
    const f1 = 2 * (precision * recall) / (precision + recall) || 0;
    
    return {
        accuracy: (accuracy * 100).toFixed(2) + '%',
        precision: (precision * 100).toFixed(2) + '%',
        recall: (recall * 100).toFixed(2) + '%',
        f1: f1.toFixed(4),
        rmse: rmse.toFixed(4),
        mse: mse.toFixed(4)
    };
} 