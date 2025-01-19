async function loadData(filename, progressCallback) {
    const response = await fetch(filename);
    const text = await response.text();
    const lines = text.split('\n');
    const headers = lines[0].split(',').map(h => h.trim());
    
    // Define important features we want to keep
    const keepFeatures = [
        'Pclass',
        'Sex',
        'Age',
        'SibSp',
        'Parch',
        'Fare',
        'Embarked'
    ];
    
    // Detect feature types from first data row
    const firstRow = lines[1].split(',');
    const featureTypes = headers.map((_, i) => {
        const value = firstRow[i]?.trim();
        return !isNaN(value) ? 'numeric' : 'categorical';
    });

    const data = [];
    const features = [];
    
    // Helper function to parse CSV line with quotes
    function parseCSVLine(line) {
        const values = [];
        let currentValue = '';
        let insideQuotes = false;
        
        for (let i = 0; i < line.length; i++) {
            const char = line[i];
            
            if (char === '"') {
                insideQuotes = !insideQuotes;
            } else if (char === ',' && !insideQuotes) {
                values.push(currentValue.trim());
                currentValue = '';
            } else {
                currentValue += char;
            }
        }
        values.push(currentValue.trim());
        
        // Remove quotes from values
        return values.map(v => v.replace(/^"|"$/g, ''));
    }
    
    // Identify features we want to keep
    headers.forEach((header, i) => {
        const name = header.trim();
        if (keepFeatures.includes(name)) {
            features.push(name);
        }
    });

    // Parse data rows
    for (let i = 1; i < lines.length; i++) {
        if (!lines[i].trim()) continue;
        
        const values = parseCSVLine(lines[i]);
        const row = {};
        
        headers.forEach((header, j) => {
            const name = header.trim();
            let value = values[j];
            
            // Only process features we want to keep
            if (keepFeatures.includes(name)) {
                // Convert numeric values
                if (name !== 'Sex' && name !== 'Embarked' && value) {
                    value = parseFloat(value);
                }
                row[name] = value || null;
            }
        });
        
        // Convert target variable (Survived column)
        if (values[1]) {  // Survived is always in column 1
            row.target = parseInt(values[1]);
        }
        
        data.push(row);
        
        if (progressCallback) {
            progressCallback((i / lines.length) * 100);
        }
    }

    // Debug log for target values
    console.log('Data sample with target:', data.slice(0, 5));
    
    return { data, features };
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