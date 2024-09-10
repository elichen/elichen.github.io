// TensorFlow.js Bigram Language Model with Embeddings

// Load the dataset (tiny shakespeare from input.txt) as a string
async function loadTextDataset(url) {
    const response = await fetch(url);
    return await response.text();
}

// DataLoader: Converts text into integer indices and sets up input/output for bigrams
class DataLoader {
    constructor(text) {
        this.text = text;
        this.chars = Array.from(new Set(text.split(''))); // Unique characters
        this.vocabSize = this.chars.length;

        // Create character to index mapping
        this.stoi = {};
        this.itos = {};
        this.chars.forEach((char, index) => {
            this.stoi[char] = index;
            this.itos[index] = char;
        });

        this.data = text.split('').map(char => this.stoi[char]);
    }

    getBatch(batchSize) {
        // Create input/output bigram pairs for the batch
        let inputs = [];
        let targets = [];

        for (let i = 0; i < batchSize; i++) {
            const idx = Math.floor(Math.random() * (this.data.length - 1));
            inputs.push(this.data[idx]);      // Input character
            targets.push(this.data[idx + 1]); // Next character (bigram)
        }

        return { inputs: tf.tensor(inputs, [batchSize]), targets: tf.tensor(targets, [batchSize]) };
    }
}

// Define the Bigram Model
class BigramLanguageModel {
    constructor(vocabSize) {
        this.vocabSize = vocabSize;

        // Create a simple embedding model using TensorFlow.js
        this.model = tf.sequential();
        this.model.add(tf.layers.embedding({ inputDim: vocabSize, outputDim: 32, inputLength: 1 }));
        this.model.add(tf.layers.flatten()); // Flatten embedding output
        this.model.add(tf.layers.dense({ units: vocabSize, activation: 'softmax' }));

        // Compile the model with Adam optimizer and cross-entropy loss
        this.model.compile({
            optimizer: tf.train.adam(),
            loss: 'sparseCategoricalCrossentropy',
            metrics: ['accuracy'],
        });
    }

    async train(dataLoader, epochs, batchSize) {
        const statusElement = document.getElementById('status');
        const progressElement = document.getElementById('trainingProgress');
        progressElement.style.display = 'block';
        progressElement.max = epochs;

        for (let epoch = 0; epoch < epochs; epoch++) {
            const { inputs, targets } = dataLoader.getBatch(batchSize);

            const history = await this.model.fit(inputs, targets, {
                batchSize,
                epochs: 1,
                shuffle: true
            });

            statusElement.textContent = `Epoch ${epoch + 1}/${epochs}, Loss: ${history.history.loss[0].toFixed(4)}, Accuracy: ${history.history.acc[0].toFixed(4)}`;
            progressElement.value = epoch + 1;
        }

        statusElement.textContent = 'Training complete!';
    }

    // Generate text from a starting character
    async generateText(startChar, numChars, dataLoader) {
        let result = [startChar];
        let currentCharIdx = dataLoader.stoi[startChar];

        for (let i = 0; i < numChars - 1; i++) {
            const input = tf.tensor([[currentCharIdx]]);

            // Predict next character
            const predictions = await this.model.predict(input);
            const probabilities = predictions.dataSync();
            
            // Sample from the probability distribution
            const predictedIdx = this.sampleFromDistribution(probabilities);
            const predictedChar = dataLoader.itos[predictedIdx];

            result.push(predictedChar);
            currentCharIdx = predictedIdx;
        }

        return result.join('');
    }

    sampleFromDistribution(probabilities) {
        const sum = probabilities.reduce((a, b) => a + b, 0);
        const normalized = probabilities.map(p => p / sum);
        const random = Math.random();
        let cumulative = 0;
        
        for (let i = 0; i < normalized.length; i++) {
            cumulative += normalized[i];
            if (random < cumulative) {
                return i;
            }
        }
        
        return normalized.length - 1; // Fallback to last index
    }
}

// Load dataset, train model, and generate text
document.getElementById('trainButton').addEventListener('click', async () => {
    const datasetURL = 'input.txt'; // The URL/path of the tiny shakespeare dataset
    const statusElement = document.getElementById('status');
    const progressElement = document.getElementById('trainingProgress');
    const generateButton = document.getElementById('generateButton');
    const outputElement = document.getElementById('output');

    statusElement.textContent = 'Status: Loading dataset...';
    progressElement.style.display = 'block';
    progressElement.value = 0;

    try {
        const text = await loadTextDataset(datasetURL);
        statusElement.textContent = 'Status: Preparing data...';

        // Initialize DataLoader and Bigram Model
        const dataLoader = new DataLoader(text);
        const model = new BigramLanguageModel(dataLoader.vocabSize);

        statusElement.textContent = 'Status: Training model...';
        
        // Train the model with the dataset
        const epochs = 1000;
        const batchSize = 64;
        await model.train(dataLoader, epochs, batchSize);

        // Enable the "Generate Text" button after training
        generateButton.disabled = false;

        // Generate text on button click
        generateButton.addEventListener('click', async () => {
            const startChar = 'H'; // Starting character for text generation
            const numChars = 100; // Number of characters to generate
            const generatedText = await model.generateText(startChar, numChars, dataLoader);
            outputElement.textContent = generatedText;
        });

    } catch (error) {
        statusElement.textContent = 'Error loading dataset!';
        console.error(error);
    }
});
