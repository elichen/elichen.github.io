async function loadData(filename, progressCallback) {
    const response = await fetch(filename);
    const text = await response.text();
    const lines = text.split('\n');
    const headers = lines[0].split(',');
    
    // Detect feature types from first data row
    const firstRow = lines[1].split(',');
    const featureTypes = headers.map((_, i) => {
        const value = firstRow[i]?.trim();
        return !isNaN(value) ? 'numeric' : 'categorical';
    });

    const data = [];
    const features = [];
    
    // Identify features (excluding target and ID columns)
    headers.forEach((header, i) => {
        const name = header.trim();
        if (name !== 'Survived' && name !== 'PassengerId') {
            features.push(name);
        }
    });

    // Parse data rows
    for (let i = 1; i < lines.length; i++) {
        if (!lines[i].trim()) continue;
        
        const values = lines[i].split(',');
        const row = {};
        
        headers.forEach((header, j) => {
            const name = header.trim();
            let value = values[j]?.trim();
            
            // Convert numeric values
            if (featureTypes[j] === 'numeric' && value) {
                value = parseFloat(value);
            }
            
            row[name] = value || null;
        });
        
        // Convert target variable
        if ('Survived' in row) {
            row.target = row.Survived;
            delete row.Survived;
        }
        
        data.push(row);
        
        if (progressCallback) {
            progressCallback((i / lines.length) * 100);
        }
    }
    
    return { data, features };
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