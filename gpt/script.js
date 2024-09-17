// TensorFlow.js Transformer Language Model

// Load the dataset (tiny shakespeare from input.txt) as a string
async function loadTextDataset(url) {
    const response = await fetch(url);
    return await response.text();
}

// DataLoader: Converts text into integer indices and sets up input/output for batches
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
            let startIdx;
            do {
                startIdx = Math.floor(Math.random() * (this.textIndices.length - this.seqLength));
            } while (startIdx + this.seqLength >= this.textIndices.length);
            
            const inputSeq = this.textIndices.slice(startIdx, startIdx + this.seqLength);
            const targetSeq = this.textIndices.slice(startIdx + 1, startIdx + this.seqLength + 1);
            inputs.push(inputSeq);
            targets.push(targetSeq);
        }

        const inputsTensor = tf.tensor2d(inputs, [batchSize, this.seqLength], 'int32');
        const targetsTensor = tf.oneHot(tf.tensor2d(targets, [batchSize, this.seqLength], 'int32'), this.vocabSize).toFloat();
        
        return { inputs: inputsTensor, targets: targetsTensor };
    }
}

// SharedDataLoader: Extends DataLoader to use shared vocabulary
class SharedDataLoader extends DataLoader {
    constructor(text, seqLength, char2idx, idx2char, vocabSize) {
        super(text, seqLength);
        this.char2idx = char2idx;
        this.idx2char = idx2char;
        this.vocabSize = vocabSize;
        this.textIndices = Array.from(text).map(c => this.char2idx[c] || 0);
    }
}

// Define the GPT Model
class GPT {
    constructor(vocabSize, seqLength) {
        this.vocabSize = vocabSize;
        this.seqLength = seqLength;

        // Model parameters
        const embedDim = 256;    // Embedding size for each token
        const numHeads = 4;      // Number of attention heads
        const numLayers = 2;     // Number of transformer blocks

        // Input layers
        const tokenInputs = tf.input({ shape: [seqLength], dtype: 'int32' });
        const positionInputs = tf.input({ shape: [seqLength], dtype: 'int32' });
        const attentionMask = tf.input({ shape: [1, seqLength, seqLength], dtype: 'float32' });

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

            // Multi-head self-attention with attention mask
            const attentionLayer = new MultiHeadSelfAttention({ numHeads, embedDim });
            let attnOutput = attentionLayer.apply([attnInput, attentionMask]);

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
        this.model = tf.model({ inputs: [tokenInputs, positionInputs, attentionMask], outputs: logits });

        this.model.compile({
            optimizer: tf.train.adam(0.001), // Learning rate: 0.001
            loss: (labels, logits) => tf.losses.softmaxCrossEntropy(labels, logits).mean(),
            metrics: ['accuracy'],
        });

        // Log the model summary (if possible)
        try {
            this.model.summary();
        } catch (e) {
            console.log('Model summary unavailable in TensorFlow.js:', e);
        }
    }

    async train(dataLoader, epochs, batchSize) {
        const statusElement = document.getElementById('status');
        const progressElement = document.getElementById('trainingProgress');
        progressElement.style.display = 'block';
        progressElement.max = epochs;

        for (let epoch = 0; epoch < epochs; epoch++) {
            const { inputs, targets } = dataLoader.getBatch(batchSize);
            const seqLength = inputs.shape[1];
            const positionIndices = tf.tensor2d(
                Array.from({ length: batchSize }, () =>
                    Array.from({ length: seqLength }, (_, i) => i)
                ),
                [batchSize, seqLength],
                'int32'
            );

            // Create attention mask for full sequence length
            const attentionMask = tf.ones([batchSize, 1, seqLength, seqLength]);

            const history = await this.model.fit([inputs, positionIndices, attentionMask], targets, {
                epochs: 1,
                verbose: 0,
            });

            // Log training history
            console.log('Training History:', history);
            if (history.history) {
                // Depending on TensorFlow.js version, accuracy might be under 'accuracy' or 'acc'
                const accuracy = history.history.accuracy || history.history.acc;
                console.log(`Epoch ${epoch + 1} - Loss: ${history.history.loss[0].toFixed(4)}, Accuracy: ${accuracy ? accuracy[0].toFixed(4) : 'N/A'}`);
            }

            statusElement.innerText = `Epoch ${epoch + 1}/${epochs} - Loss: ${history.history.loss[0].toFixed(4)}`;
            progressElement.value = epoch + 1;
            await tf.nextFrame();
        }

        progressElement.style.display = 'none';
    }

    // Generate text from a starting sequence
    async generateText(startSequence, numChars, dataLoader) {
        let result = Array.from(startSequence);
        let currentSequence = result.map(c => dataLoader.char2idx[c] || 0);
        
        for (let i = 0; i < numChars; i++) {
            // Pad or truncate the sequence to match the expected input length
            let paddedSequence = [...currentSequence];
            if (paddedSequence.length < this.seqLength) {
                paddedSequence = Array(this.seqLength - paddedSequence.length).fill(0).concat(paddedSequence);
            } else if (paddedSequence.length > this.seqLength) {
                paddedSequence = paddedSequence.slice(-this.seqLength);
            }

            const input = tf.tensor([paddedSequence], [1, this.seqLength], 'int32');
            const positionIndices = this.getPositionIndices(1, this.seqLength);
            const attentionMask = tf.ones([1, 1, this.seqLength, this.seqLength]);

            const logits = this.model.predict([input, positionIndices, attentionMask]);
            const logitsLast = logits.slice([0, this.seqLength - 1, 0], [1, 1, this.vocabSize]);
            const probabilities = Array.from(tf.softmax(logitsLast).dataSync());

            const predictedIdx = this.sampleFromDistribution(probabilities);
            const predictedChar = dataLoader.idx2char[predictedIdx];

            result.push(predictedChar);
            currentSequence.push(predictedIdx);
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

// Register the custom MultiHeadSelfAttention layer
class MultiHeadSelfAttention extends tf.layers.Layer {
    constructor(config) {
        super(config);
        this.numHeads = config.numHeads;
        this.embedDim = config.embedDim;

        if (this.embedDim % this.numHeads !== 0) {
            throw new Error('embedDim must be divisible by numHeads');
        }

        this.projectionDim = this.embedDim / this.numHeads;

        this.queryDense = tf.layers.dense({ units: this.embedDim });
        this.keyDense = tf.layers.dense({ units: this.embedDim });
        this.valueDense = tf.layers.dense({ units: this.embedDim });
        this.combineHeadsDense = tf.layers.dense({ units: this.embedDim });
    }

    call(inputs, kwargs) {
        const [x, mask] = inputs;

        const batchSize = x.shape[0];
        const seqLength = x.shape[1];

        let query = this.queryDense.apply(x);
        let key = this.keyDense.apply(x);
        let value = this.valueDense.apply(x);

        query = this.splitHeads(query, batchSize);
        key = this.splitHeads(key, batchSize);
        value = this.splitHeads(value, batchSize);

        const scaledAttention = this.scaledDotProductAttention(query, key, value, mask);
        const concatenatedHeads = tf.reshape(scaledAttention, [batchSize, seqLength, this.embedDim]);
        const output = this.combineHeadsDense.apply(concatenatedHeads);

        return output;
    }

    splitHeads(x, batchSize) {
        const seqLength = x.shape[1];
        const xReshaped = tf.reshape(x, [batchSize, seqLength, this.numHeads, this.projectionDim]);
        return tf.transpose(xReshaped, [0, 2, 1, 3]);
    }

    scaledDotProductAttention(query, key, value, mask) {
        const matmulQK = tf.matMul(query, key, false, true);
        const scaledMatmulQK = tf.mul(matmulQK, 1 / Math.sqrt(this.projectionDim));
        
        // Reshape mask to match the shape of scaledMatmulQK
        const reshapedMask = tf.reshape(mask, [mask.shape[0], 1, mask.shape[2], mask.shape[3]]);
        
        const maskedScaledMatmulQK = tf.mul(scaledMatmulQK, reshapedMask);
        const maskedScaledMatmulQK_sub = tf.sub(maskedScaledMatmulQK, tf.mul(tf.sub(1, reshapedMask), 1e9));
        
        const attentionWeights = tf.softmax(maskedScaledMatmulQK_sub, -1);
        return tf.matMul(attentionWeights, value);
    }

    computeOutputShape(inputShape) {
        return inputShape[0];
    }

    static get className() {
        return 'MultiHeadSelfAttention';
    }
}

// Register the custom layer
tf.serialization.registerClass(MultiHeadSelfAttention);

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

        const seqLength = 10;

        // Split the dataset into training and validation sets (90% training, 10% validation)
        const splitText = (text, validationSplit) => {
            const splitIndex = Math.floor(text.length * (1 - validationSplit));
            const trainText = text.slice(0, splitIndex);
            const valText = text.slice(splitIndex);
            return [trainText, valText];
        };

        const [trainText, valText] = splitText(text, 0.01);

        // Build vocabulary from the full text
        const chars = Array.from(new Set(text));
        const char2idx = {};
        const idx2char = {};
        chars.forEach((c, i) => {
            char2idx[c] = i;
            idx2char[i] = c;
        });
        const vocabSize = chars.length;

        // Initialize DataLoader instances for training and validation with shared vocab
        const trainDataLoader = new SharedDataLoader(trainText, seqLength, char2idx, idx2char, vocabSize);
        const valDataLoader = new SharedDataLoader(valText, seqLength, char2idx, idx2char, vocabSize);

        // Initialize GPT model with shared vocabSize
        const model = new GPT(vocabSize, seqLength);

        statusElement.textContent = 'Status: Training model...';
        
        // Train the model with the dataset
        const epochs = 50;
        const batchSize = 64;
        await model.train(trainDataLoader, epochs, batchSize);

        // Enable the "Generate Text" button after training
        generateButton.disabled = false;

        // Generate text on button click
        generateButton.addEventListener('click', async () => {
            const startSequence = 'The '; // Starting sequence with length <= seqLength
            const numChars = 100; // Number of characters to generate
            const generatedText = await model.generateText(startSequence, numChars, trainDataLoader);
            outputElement.textContent = generatedText;
        });

        // After training, evaluate on validation data
        console.log('Evaluating on validation data...');
        const validationBatches = Math.floor(valDataLoader.textIndices.length / batchSize);
        let totalValLoss = 0;
        let totalValAccuracy = 0;

        for (let i = 0; i < validationBatches; i++) {
            const { inputs, targets } = valDataLoader.getBatch(batchSize);
            const seqLength = inputs.shape[1];
            const positionIndices = tf.tensor2d(
                Array.from({ length: batchSize }, () =>
                    Array.from({ length: seqLength }, (_, idx) => idx)
                ),
                [batchSize, seqLength],
                'int32'
            );
            const attentionMask = tf.ones([batchSize, 1, seqLength, seqLength]);

            // Evaluate the model on the validation batch
            const evaluation = model.model.evaluate([inputs, positionIndices, attentionMask], targets, { verbose: 0 });

            // Accumulate loss and accuracy
            totalValLoss += evaluation[0].dataSync()[0];
            totalValAccuracy += evaluation[1].dataSync()[0];
        }

        // Calculate average validation metrics
        const avgValLoss = totalValLoss / validationBatches;
        const avgValAccuracy = totalValAccuracy / validationBatches;

        console.log(`Validation Loss: ${avgValLoss.toFixed(4)}, Validation Accuracy: ${avgValAccuracy.toFixed(4)}`);

    } catch (error) {
        statusElement.textContent = 'Error loading dataset!';
        console.error(error);
    }
});
