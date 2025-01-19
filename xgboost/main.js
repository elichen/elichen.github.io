const target = 'target';

const statusElement = document.getElementById('status');
const dataProgressElement = document.getElementById('dataProgress');
const trainingProgressElement = document.getElementById('trainingProgress');

async function run() {
    try {
        // Load and preprocess training data
        statusElement.textContent = 'Loading training data...';
        const trainData = await loadData('train.csv', 
            progress => dataProgressElement.value = progress);
        
        console.log('Train data sample:', trainData.data.slice(0, 3));
        console.log('Features:', trainData.features);
        
        // Visualize raw data using detected features
        visualizeData(trainData.data, trainData.features, target, 'dataViz');
        
        // Preprocess data
        statusElement.textContent = 'Preprocessing data...';
        const { X: XTrain, y: yTrain, features: processedFeatures, stats: trainStats } = 
            await preprocessData(trainData.data, trainData.features, target);
        
        console.log('Training tensor shapes:', {
            X: XTrain.shape,
            y: yTrain?.shape
        });
        
        // Train model
        statusElement.textContent = 'Training model...';
        const model = new XGBoostModel(
            0.05,   // Lower learning rate for stability
            6,      // Deeper trees
            500     // More trees
        );
        await model.train(XTrain, yTrain, 
            progress => trainingProgressElement.value = progress);
        
        // Load and preprocess test data
        statusElement.textContent = 'Loading test data...';
        const testData = await loadData('test.csv', 
            progress => dataProgressElement.value = progress);
        const { X: XTest } = await preprocessData(testData.data, processedFeatures, undefined, trainStats);
        
        // Make predictions
        statusElement.textContent = 'Making predictions...';
        const predictions = await model.predict(XTest);
        const predictedValues = await predictions.array();
        
        console.log('Prediction samples:', predictedValues.slice(0, 5));
        
        // Clean up tensors
        tf.dispose([predictions, XTrain, yTrain, XTest]);
        
        // Update status with prediction summary
        const survivedCount = predictedValues.filter(p => p >= 0.45).length;
        const totalCount = predictedValues.length;
        const survivedPercent = ((survivedCount / totalCount) * 100).toFixed(1);
        
        statusElement.textContent = `Predictions complete! Predicted ${survivedCount} survivors (${survivedPercent}%) out of ${totalCount} passengers`;
        
        // Visualize test results
        visualizeTestResults(testData.data, predictedValues, 'performanceViz');
        
    } catch (error) {
        statusElement.textContent = `Error: ${error.message}`;
        console.error('Full error:', error);
    }
}

async function preprocessData(data, features, target, trainStats = null) {
    // Add debug logging
    console.log('Preprocessing data:', {
        dataLength: data.length,
        features,
        target,
        trainStats: trainStats ? 'present' : 'absent',
        sampleRow: data[0],
        sampleTarget: target ? data[0][target] : 'no target'  // Debug target value
    });

    // Deep copy the data
    const processedData = data.map(row => ({...row}));
    
    // Debug target values
    if (target) {
        const targetValues = processedData.map(row => row[target]);
        console.log('Target values sample:', targetValues.slice(0, 5));
        console.log('Target statistics:', {
            count: targetValues.length,
            nonNull: targetValues.filter(v => v != null).length,
            sample: targetValues.slice(0, 10)
        });
    }

    // Calculate statistics for numeric features (only during training)
    const numericFeatures = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare'];
    const stats = trainStats || numericFeatures.reduce((acc, feature) => {
        const values = data.map(row => row[feature]).filter(v => v != null);
        if (values.length) {
            const mean = values.reduce((a, b) => a + b, 0) / values.length;
            const std = Math.sqrt(values.reduce((a, b) => 
                a + Math.pow(b - mean, 2), 0) / values.length) || 1;
            acc[feature] = { mean, std };
        }
        return acc;
    }, {});

    console.log('Feature statistics:', stats);
    
    // Process each row
    processedData.forEach(row => {
        // Handle numeric features first
        numericFeatures.forEach(feature => {
            if (row[feature] == null) {
                row[feature] = stats[feature].mean;
            }
            row[feature] = (row[feature] - stats[feature].mean) / stats[feature].std;
        });

        // Handle categorical variables
        if (row.Sex) {
            row.Sex_male = row.Sex.toLowerCase() === 'male' ? 1 : 0;
            row.Sex_female = row.Sex.toLowerCase() === 'female' ? 1 : 0;
        }
        
        if (row.Embarked) {
            row.Embarked_S = row.Embarked === 'S' ? 1 : 0;
            row.Embarked_C = row.Embarked === 'C' ? 1 : 0;
            row.Embarked_Q = row.Embarked === 'Q' ? 1 : 0;
        }
        
        // Feature engineering
        row.FamilySize = row.SibSp + row.Parch + 1;
        row.FarePerPerson = row.Fare / row.FamilySize;
        row.IsChild = row.Age < -1 ? 1 : 0;  // Normalized age
        row.IsElderly = row.Age > 1 ? 1 : 0;  // Normalized age
    });

    // Log sample processed row
    console.log('Sample processed row:', processedData[0]);
    
    // Define final feature list - keep order consistent
    const finalFeatures = [
        ...numericFeatures,
        'Sex_male',
        'Sex_female',
        'Embarked_S',
        'Embarked_C',
        'Embarked_Q',
        'FamilySize',
        'FarePerPerson',
        'IsChild',
        'IsElderly'
    ];

    // Create feature matrix
    const X = tf.tensor2d(processedData.map(row => 
        finalFeatures.map(feature => row[feature] ?? 0)
    ));

    // Log sample feature vector
    const sampleVector = await X.slice([0, 0], [1, finalFeatures.length]).array();
    console.log('Sample feature vector:', sampleVector);

    // Create target vector if target exists
    const y = target ? tf.tensor1d(processedData.map(row => row[target])) : undefined;

    // Add debug logging
    console.log('Preprocessing complete:', {
        finalFeatures,
        XShape: X.shape,
        yShape: y?.shape,
        sampleProcessedRow: processedData[0],
        sampleFeatureVector: finalFeatures.map(f => processedData[0][f] ?? 0).slice(0, 5)
    });

    return { X, y, features: finalFeatures, stats };
}

window.onload = run; 