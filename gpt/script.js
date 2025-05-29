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
            this.char2idx[c] = i; // Corrected with 'this.'
            this.idx2char[i] = c; // Corrected with 'this.'
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
        const targetsTensor = tf.oneHot(tf.tensor2d(targets, [batchSize, this.seqLength], 'int32'), this.vocabSize);
        
        return { inputs: inputsTensor, targets: targetsTensor };
    }
}

// SharedDataLoader: Extends DataLoader to use shared vocabulary
class SharedDataLoader extends DataLoader {
    constructor(text, seqLength, char2idx, idx2char, vocabSize) {
        super(text, seqLength);
        this.char2idx = char2idx; // Correctly using 'this.'
        this.idx2char = idx2char; // Correctly using 'this.'
        this.vocabSize = vocabSize;
        this.textIndices = Array.from(text).map(c => this.char2idx[c] || 0);
    }
}

// Define the GPT Model
class GPT {
    constructor(vocabSize, seqLength) {
        this.vocabSize = vocabSize;
        this.seqLength = seqLength;

        // Model parameters - reduced for better training
        const embedDim = 128;    // Embedding size for each token
        const numHeads = 4;      // Number of attention heads
        const numLayers = 2;     // Number of transformer blocks

        // Input layers
        const tokenInputs = tf.input({ shape: [null], dtype: 'int32' });
        const positionInputs = tf.input({ shape: [null], dtype: 'int32' });
        const attentionMask = tf.input({ shape: [1, null, null], dtype: 'float32' });

        // Token embeddings
        const tokenEmbeddingLayer = tf.layers.embedding({ inputDim: this.vocabSize, outputDim: embedDim });
        const tokenEmbeddings = tokenEmbeddingLayer.apply(tokenInputs);

        // Positional embeddings
        const positionEmbeddingLayer = tf.layers.embedding({ inputDim: seqLength, outputDim: embedDim });
        const positionEmbeddings = positionEmbeddingLayer.apply(positionInputs);

        // Combine embeddings
        let x = tf.layers.add().apply([tokenEmbeddings, positionEmbeddings]);
        x = tf.layers.dropout({ rate: 0.05 }).apply(x); // Reduced dropout

        // Transformer blocks
        for (let i = 0; i < numLayers; i++) {
            // Layer normalization
            let attnInput = tf.layers.layerNormalization({ epsilon: 1e-6 }).apply(x);

            // Multi-head self-attention with attention mask
            const attentionLayer = new MultiHeadSelfAttention({ numHeads, embedDim });
            let attnOutput = attentionLayer.apply([attnInput, attentionMask]);
            attnOutput = tf.layers.dropout({ rate: 0.05 }).apply(attnOutput); // Reduced dropout

            // Add & Norm
            x = tf.layers.add().apply([x, attnOutput]);

            // Feed-forward network
            let ffnInput = tf.layers.layerNormalization({ epsilon: 1e-6 }).apply(x);
            let ffnOutput = tf.layers.dense({ units: embedDim * 4, activation: 'elu' }).apply(ffnInput); // ELU is similar to GELU
            ffnOutput = tf.layers.dense({ units: embedDim }).apply(ffnOutput);
            ffnOutput = tf.layers.dropout({ rate: 0.05 }).apply(ffnOutput); // Reduced dropout

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
            optimizer: tf.train.adam(0.001), // Higher learning rate for smaller model
            loss: (yTrue, yPred) => {
                // Reshape for loss calculation like minGPT
                const yTrueFlat = tf.reshape(yTrue, [-1, this.vocabSize]);
                const yPredFlat = tf.reshape(yPred, [-1, this.vocabSize]);
                return tf.losses.softmaxCrossEntropy(yTrueFlat, yPredFlat);
            },
            metrics: ['accuracy'],
        });

        // Model summary commented out to reduce console output
        // this.model.summary();
        
        // Debug: Check trainable parameters count
        const totalParams = this.model.countParams();
        console.log(`Total trainable parameters: ${totalParams}`);
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

                // Create causal attention mask
                const attentionMask = createCausalMask(batchSize, seqLength);
                
                const history = await this.model.fit([inputs, positionIndices, attentionMask], targets, {
                    epochs: 1,
                    verbose: 0,
                });

                // Dispose tensors to free memory
                inputs.dispose();
                targets.dispose();
                positionIndices.dispose();
                attentionMask.dispose();

                // Log training history every 20 epochs
                if ((epoch + 1) % 20 === 0 && history.history) {
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
            if (currentSequence.length > this.seqLength) {
                currentSequence = currentSequence.slice(-this.seqLength);
            }
        
            const input = tf.tensor([currentSequence], [1, currentSequence.length], 'int32');
            const positionIndices = this.getPositionIndices(1, currentSequence.length);
            const attentionMask = createCausalMask(1, currentSequence.length);
        
            const logits = this.model.predict([input, positionIndices, attentionMask], { training: false });
            const logitsLast = logits.slice([0, currentSequence.length - 1, 0], [1, 1, this.vocabSize]);
            
            // Reshape logitsLast from [1, 1, vocabSize] to [1, vocabSize]
            const logits2D = logitsLast.squeeze([1]); // {{ edit_squeeze }}
            
            // Use tf.multinomial to sample from the logits
            const sampled = tf.multinomial(logits2D, 1); // {{ edit_multinomial }}
            const sampledArray = await sampled.array();
            const predictedIdx = sampledArray[0][0];
            const predictedChar = dataLoader.idx2char[predictedIdx];
        
            result.push(predictedChar);
            currentSequence.push(predictedIdx);
        
            // Dispose tensors to free memory
            input.dispose();
            positionIndices.dispose();
            attentionMask.dispose();
            logits.dispose();
            logitsLast.dispose();
            logits2D.dispose(); // {{ edit_dispose_logits2D }}
            sampled.dispose();
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

        // Initialize with Xavier/Glorot initialization
        const initConfig = {
            kernelInitializer: 'glorotUniform',
            biasInitializer: 'zeros'
        };
        
        this.queryDense = tf.layers.dense({ units: this.embedDim, ...initConfig });
        this.keyDense = tf.layers.dense({ units: this.embedDim, ...initConfig });
        this.valueDense = tf.layers.dense({ units: this.embedDim, ...initConfig });
        this.combineHeadsDense = tf.layers.dense({ units: this.embedDim, ...initConfig });
    }

    build(inputShape) {
        super.build(inputShape);
        const embeddingShape = inputShape[0];
        
        this.queryDense.build(embeddingShape);
        this.keyDense.build(embeddingShape);
        this.valueDense.build(embeddingShape);
        this.combineHeadsDense.build(embeddingShape);
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
        
        // Apply mask: add large negative value to positions where mask is 0
        const maskedScaledMatmulQK = tf.where(
            tf.equal(reshapedMask, 0),
            tf.mul(tf.onesLike(scaledMatmulQK), -1e10),
            scaledMatmulQK
        );
        
        const attentionWeights = tf.softmax(maskedScaledMatmulQK, -1);
        return tf.matMul(attentionWeights, value);
    }

    computeOutputShape(inputShape) {
        return inputShape[0];
    }

    getConfig() {
        const config = super.getConfig();
        Object.assign(config, {
            numHeads: this.numHeads,
            embedDim: this.embedDim
        });
        return config;
    }

    static get className() {
        return 'MultiHeadSelfAttention';
    }
}

// Register the custom layer
tf.serialization.registerClass(MultiHeadSelfAttention);

// Helper function to create a causal mask
function createCausalMask(batchSize, seqLength) {
    return tf.tidy(() => {
        // Create a lower triangular matrix [seqLength, seqLength]
        const mask = tf.linalg.bandPart(tf.ones([seqLength, seqLength]), -1, 0);
        
        // Reshape the mask to [1, 1, seqLength, seqLength]
        const maskReshaped = mask.reshape([1, 1, seqLength, seqLength]);
        
        // Tile to [batchSize, 1, seqLength, seqLength]
        return maskReshaped.tile([batchSize, 1, 1, 1]);
    });
}

// Load dataset, train model, and generate text
let trainedModel = null;
let dataLoader = null;

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

        const seqLength = 64;

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
        trainedModel = new GPT(vocabSize, seqLength);
        dataLoader = trainDataLoader;

        statusElement.textContent = 'Status: Training model...';
        
        // Train the model with the dataset
        const epochs = 300; // More epochs since model is still learning
        const batchSize = 16; // Smaller batch for more frequent updates
        await trainedModel.train(trainDataLoader, epochs, batchSize);

        // Enable the "Generate Text" button after training
        generateButton.disabled = false;
        statusElement.textContent = 'Status: Training complete!';

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
            const attentionMask = createCausalMask(batchSize, seqLength);

            // Evaluate the model on the validation batch
            const evaluation = trainedModel.model.evaluate([inputs, positionIndices, attentionMask], targets, { verbose: 0 });

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

// Generate text button handler
document.getElementById('generateButton').addEventListener('click', async () => {
    if (!trainedModel || !dataLoader) {
        console.error('Model not trained yet!');
        return;
    }
    
    const outputElement = document.getElementById('output');
    const statusElement = document.getElementById('status');
    
    statusElement.textContent = 'Status: Generating text...';
    
    const startSequence = 'The '; // Starting sequence
    const numChars = 200; // Number of characters to generate
    const generatedText = await trainedModel.generateText(startSequence, numChars, dataLoader);
    
    outputElement.textContent = generatedText;
    statusElement.textContent = 'Status: Text generated!';
});
