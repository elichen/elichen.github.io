const features = ['feature1', 'feature2', 'feature3']; // Update with your actual feature names
const target = 'target'; // Update with your actual target column name

const statusElement = document.getElementById('status');
const dataProgressElement = document.getElementById('dataProgress');
const trainingProgressElement = document.getElementById('trainingProgress');

async function run() {
    try {
        // Load and preprocess training data
        statusElement.textContent = 'Loading training data...';
        const trainData = await loadData('train.csv', 
            progress => dataProgressElement.value = progress);
        
        // Visualize raw data
        visualizeData(trainData.data, features, target, 'dataViz');
        
        // Preprocess data
        statusElement.textContent = 'Preprocessing data...';
        const { X: XTrain, y: yTrain } = preprocessData(trainData.data, features, target);
        
        // Train model
        statusElement.textContent = 'Training model...';
        const model = new XGBoostModel();
        await model.train(XTrain, yTrain, 
            progress => trainingProgressElement.value = progress);
        
        // Load and preprocess test data
        statusElement.textContent = 'Loading test data...';
        const testData = await loadData('test.csv', 
            progress => dataProgressElement.value = progress);
        const { X: XTest, y: yTest } = preprocessData(testData.data, features, target);
        
        // Make predictions
        statusElement.textContent = 'Making predictions...';
        const predictions = model.predict(XTest);
        
        // Visualize performance
        const actualValues = await yTest.array();
        const predictedValues = await predictions.array();
        visualizePerformance(actualValues, predictedValues, 'performanceViz');
        
        statusElement.textContent = 'Complete! Check the visualizations below.';
        
    } catch (error) {
        statusElement.textContent = `Error: ${error.message}`;
        console.error(error);
    }
}

// Run when page loads
window.onload = run; 