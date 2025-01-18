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
            0.1,    // learning rate
            5,      // max depth
            100     // number of trees
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
        
        // Create Kaggle submission format
        const submission = testData.data.map((row, i) => ({
            PassengerId: row.PassengerId,
            Survived: predictedValues[i] >= 0.45 ? 1 : 0
        }));
        
        console.log('Submission samples:', submission.slice(0, 5));
        
        // Update status with prediction summary
        const survivedCount = submission.filter(row => row.Survived === 1).length;
        const totalCount = submission.length;
        const survivedPercent = ((survivedCount / totalCount) * 100).toFixed(1);
        
        statusElement.textContent = `Predictions complete! Predicted ${survivedCount} survivors (${survivedPercent}%) out of ${totalCount} passengers`;
        
        // Add download link for submission
        const csvContent = 'PassengerId,Survived\n' + 
            submission.map(row => `${row.PassengerId},${row.Survived}`).join('\n');
        const blob = new Blob([csvContent], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'submission.csv';
        a.textContent = 'Download Predictions';
        a.className = 'download-button';
        document.querySelector('.container').appendChild(a);
        
    } catch (error) {
        statusElement.textContent = `Error: ${error.message}`;
        console.error('Full error:', error);
    }
}

window.onload = run; 