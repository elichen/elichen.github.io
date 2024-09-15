// TensorFlow.js Bigram Language Model with Embeddings

// Load the dataset (tiny shakespeare from input.txt) as a string
async function loadTextDataset(url) {
    const response = await fetch(url);
    return await response.text();
}

// DataLoader: Converts text into integer indices and sets up input/output for bigrams
class DataLoader {
    constructor(text, seqLength) {
        this.text = text;
        this.seqLength = seqLength;

        // Build vocabulary
        this.chars = Array.from(new Set(text));
        this.char2idx = {};
        this.idx2char = {};
        this.chars.forEach((c, i) => {
            this.char2idx[c] = i;
            this.idx2char[i] = c;
        });
        this.vocabSize = this.chars.length;

        // Convert text to indices
        this.textIndices = Array.from(text).map(c => this.char2idx[c]);
    }

    getBatch(batchSize) {
        const inputs = [];
        const targets = [];
        for (let i = 0; i < batchSize; i++) {
            const startIdx = Math.floor(Math.random() * (this.textIndices.length - this.seqLength - 1));
            const inputSeq = this.textIndices.slice(startIdx, startIdx + this.seqLength);
            const targetSeq = this.textIndices.slice(startIdx + 1, startIdx + this.seqLength + 1);
            inputs.push(inputSeq);
            targets.push(targetSeq);
        }

        const inputsTensor = tf.tensor2d(inputs, [batchSize, this.seqLength], 'int32');

        // Convert targets to one-hot encoding
        const targetsTensor = tf.oneHot(tf.tensor2d(targets, [batchSize, this.seqLength], 'int32'), this.vocabSize).toFloat();

        return { inputs: inputsTensor, targets: targetsTensor };
    }
}

// Define the Bigram Model
class BigramLanguageModel {
    constructor(vocabSize, seqLength) {
        this.vocabSize = vocabSize;

        // Model parameters
        const embedDim = 128;    // Embedding size for each token
        const numHeads = 8;      // Number of attention heads
        const numLayers = 4;     // Number of transformer blocks

        // Input layers
        const tokenInputs = tf.input({ shape: [seqLength], dtype: 'int32' });
        const positionInputs = tf.input({ shape: [seqLength], dtype: 'int32' });

        // Token embeddings
        const tokenEmbeddingLayer = tf.layers.embedding({ inputDim: this.vocabSize, outputDim: embedDim });
        const tokenEmbeddings = tokenEmbeddingLayer.apply(tokenInputs);

        // Positional embeddings
        const positionEmbeddingLayer = tf.layers.embedding({ inputDim: seqLength, outputDim: embedDim });
        const positionEmbeddings = positionEmbeddingLayer.apply(positionInputs);

        // Combine embeddings
        let x = tf.layers.add().apply([tokenEmbeddings, positionEmbeddings]);

        // Transformer blocks
        for (let i = 0; i < numLayers; i++) {
            // Layer normalization
            let attnInput = tf.layers.layerNormalization({ epsilon: 1e-6 }).apply(x);

            // Multi-head self-attention
            const attentionLayer = new MultiHeadSelfAttention({ numHeads, embedDim });
            let attnOutput = attentionLayer.apply(attnInput);

            // Add & Norm
            x = tf.layers.add().apply([x, attnOutput]);

            // Feed-forward network
            let ffnInput = tf.layers.layerNormalization({ epsilon: 1e-6 }).apply(x);
            let ffnOutput = tf.layers.dense({ units: embedDim * 4, activation: 'relu' }).apply(ffnInput);
            ffnOutput = tf.layers.dense({ units: embedDim }).apply(ffnOutput);

            // Add & Norm
            x = tf.layers.add().apply([x, ffnOutput]);
        }

        // Final layer normalization
        x = tf.layers.layerNormalization({ epsilon: 1e-6 }).apply(x);

        // Output layer
        const logits = tf.layers.dense({ units: this.vocabSize }).apply(x);

        // Define the model
        this.model = tf.model({ inputs: [tokenInputs, positionInputs], outputs: logits });

        // Compile the model with 'categoricalCrossentropy' loss
        this.model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'categoricalCrossentropy',
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

            // Check the shapes of inputs and targets
            console.log('Epoch', epoch);
            console.log('inputs.shape:', inputs.shape);
            console.log('targets.shape:', targets.shape);

            // Generate position indices for the input sequences
            const seqLength = inputs.shape[1];
            const positionIndices = tf.tensor2d(
                Array.from({ length: batchSize }, () =>
                    Array.from({ length: seqLength }, (_, i) => i)
                ),
                [batchSize, seqLength],
                'int32'
            );

            const history = await this.model.fit([inputs, positionIndices], targets, {
                epochs: 1,
                verbose: 0,
            });

            statusElement.innerText = `Epoch ${epoch + 1}/${epochs} - Loss: ${history.history.loss[0].toFixed(4)}`;
            progressElement.value = epoch + 1;
            await tf.nextFrame();
        }

        progressElement.style.display = 'none';
    }

    // Generate text from a starting character
    async generateText(startChar, numChars, dataLoader) {
        let result = [startChar];
        let currentCharIdx = dataLoader.char2idx[startChar];

        for (let i = 0; i < numChars - 1; i++) {
            const input = tf.tensor([[currentCharIdx]]);

            // Predict next character
            const predictions = this.model.predict([input, this.getPositionIndices(1, 1)]);
            const probabilities = predictions.dataSync();
            
            // Sample from the probability distribution
            const predictedIdx = this.sampleFromDistribution(probabilities);
            const predictedChar = dataLoader.idx2char[predictedIdx];

            result.push(predictedChar);
            currentCharIdx = predictedIdx;
        }

        return result.join('');
    }

    // Helper method to generate position indices for single input
    getPositionIndices(batchSize, seqLength) {
        return tf.tensor2d(
            Array.from({ length: batchSize }, () =>
                Array.from({ length: seqLength }, (_, i) => i)
            ),
            [batchSize, seqLength],
            'int32'
        );
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

        // Define sequence length
        const seqLength = 100;

        // Initialize DataLoader and BigramLanguageModel
        const dataLoader = new DataLoader(text, seqLength);
        const model = new BigramLanguageModel(dataLoader.vocabSize, seqLength);

        statusElement.textContent = 'Status: Training model...';
        
        // Train the model with the dataset
        const epochs = 100;
        const batchSize = 64;
        await model.train(dataLoader, epochs, batchSize);

        // Enable the "Generate Text" button after training
        generateButton.disabled = false;

        // Generate text on button click
        generateButton.addEventListener('click', async () => {
            const startChar = 'T'; // Starting character for text generation
            const numChars = 100; // Number of characters to generate
            const generatedText = await model.generateText(startChar, numChars, dataLoader);
            outputElement.textContent = generatedText;
        });

    } catch (error) {
        statusElement.textContent = 'Error loading dataset!';
        console.error(error);
    }
});

// Define a custom MultiHeadSelfAttention layer
class MultiHeadSelfAttention extends tf.layers.Layer {
    constructor(config) {
        super(config);
        this.numHeads = config.numHeads;
        this.embedDim = config.embedDim;

        // Ensure embedDim is divisible by numHeads
        if (this.embedDim % this.numHeads !== 0) {
            throw new Error('embedDim must be divisible by numHeads');
        }

        this.projectionDim = this.embedDim / this.numHeads;

        // Define dense layers for linear projections
        this.queryDense = tf.layers.dense({ units: this.embedDim });
        this.keyDense = tf.layers.dense({ units: this.embedDim });
        this.valueDense = tf.layers.dense({ units: this.embedDim });
        this.combineHeadsDense = tf.layers.dense({ units: this.embedDim });
    }

    call(inputs, kwargs) {
        const x = inputs;

        // Linear projections
        let query = this.queryDense.apply(x);
        let key = this.keyDense.apply(x);
        let value = this.valueDense.apply(x);

        // Split heads
        query = this.splitHeads(query);
        key = this.splitHeads(key);
        value = this.splitHeads(value);

        // Scaled dot-product attention
        let attentionScores = tf.matMul(query, key, false, true);
        attentionScores = tf.mul(attentionScores, 1 / Math.sqrt(this.projectionDim));

        // Apply softmax
        let attentionWeights = tf.softmax(attentionScores);

        // Attention output
        let attentionOutput = tf.matMul(attentionWeights, value);

        // Combine heads
        let combinedHeads = this.combineHeads(attentionOutput);

        // Final linear layer
        let output = this.combineHeadsDense.apply(combinedHeads);

        return output;
    }

    splitHeads(x) {
        const batchSize = x.shape[0];
        const seqLength = x.shape[1];

        // [batch_size, seq_length, num_heads, projection_dim]
        let xReshaped = tf.reshape(x, [batchSize, seqLength, this.numHeads, this.projectionDim]);

        // [batch_size, num_heads, seq_length, projection_dim]
        return tf.transpose(xReshaped, [0, 2, 1, 3]);
    }

    combineHeads(x) {
        // [batch_size, num_heads, seq_length, projection_dim]
        let xTransposed = tf.transpose(x, [0, 2, 1, 3]);

        const batchSize = x.shape[0];
        const seqLength = x.shape[2]; // Adjusted index for seq_length

        // Reshape to [batch_size, seq_length, embed_dim]
        return tf.reshape(xTransposed, [batchSize, seqLength, this.embedDim]);
    }

    static get className() {
        return 'MultiHeadSelfAttention';
    }
}

// Register the custom layer
tf.serialization.registerClass(MultiHeadSelfAttention);
