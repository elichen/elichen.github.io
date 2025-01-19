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
        const { X: XTrain, y: yTrain } = preprocessData(trainData.data, trainData.features, target);
        
        console.log('Training tensor shapes:', {
            X: XTrain.shape,
            y: yTrain.shape
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
        const { X: XTest } = preprocessData(testData.data, trainData.features, target);
        
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

function preprocessData(data, features, target) {
    // Deep copy the data
    const processedData = data.map(row => ({...row}));
    
    // Calculate statistics for imputation
    const stats = features.reduce((acc, feature) => {
        const values = data.map(row => row[feature]).filter(v => v != null);
        if (values.length && typeof values[0] === 'number') {
            acc[feature] = {
                mean: values.reduce((a, b) => a + b, 0) / values.length
            };
        }
        return acc;
    }, {});
    
    // Process each row
    processedData.forEach(row => {
        // Handle categorical variables
        if (row.Sex) {
            row.Sex_male = row.Sex.toLowerCase() === 'male' ? 1 : 0;
            row.Sex_female = row.Sex.toLowerCase() === 'female' ? 1 : 0;
        }
        
        // Impute missing numeric values
        features.forEach(feature => {
            if (stats[feature] && row[feature] == null) {
                row[feature] = stats[feature].mean;
            }
        });
        
        // Feature engineering
        row.FamilySize = (row.SibSp || 0) + (row.Parch || 0) + 1;
        row.FarePerPerson = (row.Fare || stats.Fare.mean) / row.FamilySize;
        row.IsChild = (row.Age || stats.Age.mean) < 12 ? 1 : 0;
        row.IsElderly = (row.Age || stats.Age.mean) > 60 ? 1 : 0;
        
        // Extract title from name
        const match = row.Name?.match(/([A-Za-z]+)\./);
        const title = match ? match[1] : 'Other';
        row.Title_Mr = title === 'Mr' ? 1 : 0;
        row.Title_Mrs = title === 'Mrs' ? 1 : 0;
        row.Title_Miss = title === 'Miss' ? 1 : 0;
        row.Title_Master = title === 'Master' ? 1 : 0;
        
        // Encode cabin deck
        const deck = row.Cabin ? row.Cabin[0] : 'Unknown';
        ['A', 'B', 'C', 'D', 'E', 'F', 'G'].forEach(d => {
            row[`Deck_${d}`] = deck === d ? 1 : 0;
        });
    });
    
    // Update feature list
    const engineeredFeatures = [
        'FamilySize',
        'FarePerPerson',
        'IsChild',
        'IsElderly',
        'Title_Mr',
        'Title_Mrs',
        'Title_Miss',
        'Title_Master',
        ...['A', 'B', 'C', 'D', 'E', 'F', 'G'].map(d => `Deck_${d}`)
    ];
    
    const allFeatures = [...features, ...engineeredFeatures];

    // Create tensors
    const X = tf.tensor2d(processedData.map(row => 
        allFeatures.map(feature => row[feature] || 0)
    ));

    const y = target && processedData[0].hasOwnProperty(target) ? 
        tf.tensor1d(processedData.map(row => row[target])) : 
        undefined;

    return { X, y, features: allFeatures };
}

window.onload = run; 