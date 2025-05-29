#!/usr/bin/env node

// CLI version of the GPT model for Node.js
// Usage: node train_cli.js [--epochs 100] [--generate]

const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');
const readline = require('readline');

// Parse command line arguments
const args = process.argv.slice(2);
const epochs = args.includes('--epochs') ? 
    parseInt(args[args.indexOf('--epochs') + 1]) : 100;
const shouldGenerate = args.includes('--generate');
const modelPath = './saved_model';

// Load the dataset
function loadTextDataset(filename) {
    return fs.readFileSync(filename, 'utf8');
}

// DataLoader class
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
        const targetsTensor = tf.oneHot(tf.tensor2d(targets, [batchSize, this.seqLength], 'int32'), this.vocabSize);
        
        return { inputs: inputsTensor, targets: targetsTensor };
    }
}

// Multi-Head Self-Attention Layer
class MultiHeadSelfAttention extends tf.layers.Layer {
    static get className() {
        return 'MultiHeadSelfAttention';
    }
    
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

// GPT Model
class GPT {
    constructor(vocabSize, seqLength) {
        this.vocabSize = vocabSize;
        this.seqLength = seqLength;

        // Model parameters - reduced for better training
        const embedDim = 128;
        const numHeads = 4;
        const numLayers = 2;

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

        // Debug: Check trainable parameters count
        const totalParams = this.model.countParams();
        console.log(`Total trainable parameters: ${totalParams}`);
    }

    async train(dataLoader, epochs, batchSize, progressCallback) {
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
                const loss = history.history.loss[0].toFixed(4);
                const acc = accuracy ? accuracy[0].toFixed(4) : 'N/A';
                console.log(`Epoch ${epoch + 1} - Loss: ${loss}, Accuracy: ${acc}`);
                if (progressCallback) {
                    progressCallback(epoch + 1, epochs, loss, acc);
                }
            }
            
            await tf.nextFrame();
        }
    }

    async generateText(startSequence, numChars, dataLoader, temperature = 0.8) {
        let result = Array.from(startSequence);
        let currentSequence = result.map(c => dataLoader.char2idx[c] || 0);
        
        for (let i = 0; i < numChars; i++) {
            if (currentSequence.length > this.seqLength) {
                currentSequence = currentSequence.slice(-this.seqLength);
            }
        
            const input = tf.tensor([currentSequence], [1, currentSequence.length], 'int32');
            const positionIndices = tf.tensor2d(
                Array.from({ length: 1 }, () =>
                    Array.from({ length: currentSequence.length }, (_, i) => i)
                ),
                [1, currentSequence.length],
                'int32'
            );
            const attentionMask = createCausalMask(1, currentSequence.length);
        
            const logits = this.model.predict([input, positionIndices, attentionMask], { training: false });
            const logitsLast = logits.slice([0, currentSequence.length - 1, 0], [1, 1, this.vocabSize]);
            
            // Reshape logitsLast from [1, 1, vocabSize] to [1, vocabSize]
            const logits2D = logitsLast.squeeze([1]);
            
            // Apply temperature scaling for better generation
            const scaledLogits = tf.div(logits2D, temperature);
            
            // Use tf.multinomial to sample from the scaled logits
            const sampled = tf.multinomial(scaledLogits, 1);
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
            logits2D.dispose();
            scaledLogits.dispose();
            sampled.dispose();
        }
    
        return result.join('');
    }

    async saveModel(path) {
        await this.model.save(`file://${path}`);
        console.log(`Model saved to ${path}`);
    }

    async loadModel(path) {
        this.model = await tf.loadLayersModel(`file://${path}/model.json`);
        console.log(`Model loaded from ${path}`);
    }
}

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

// Main CLI function
async function main() {
    console.log('GPT Training CLI');
    console.log('================');
    
    try {
        // Load dataset
        console.log('Loading dataset...');
        const text = loadTextDataset('input.txt');
        console.log(`Loaded ${text.length} characters`);
        
        const seqLength = 64;
        const batchSize = 16;
        
        // Split data
        const splitIndex = Math.floor(text.length * 0.99);
        const trainText = text.slice(0, splitIndex);
        
        // Build vocabulary from full text
        const chars = Array.from(new Set(text));
        const vocabSize = chars.length;
        console.log(`Vocabulary size: ${vocabSize}`);
        
        // Initialize DataLoader
        const dataLoader = new DataLoader(trainText, seqLength);
        
        // Initialize or load model
        const model = new GPT(vocabSize, seqLength);
        
        // Check if we should load existing model
        if (fs.existsSync(`${modelPath}/model.json`) && !args.includes('--fresh')) {
            console.log('Loading existing model...');
            await model.loadModel(modelPath);
        }
        
        // Train if not just generating
        if (!shouldGenerate || args.includes('--train')) {
            console.log(`\nTraining for ${epochs} epochs...`);
            console.log('Batch size:', batchSize);
            console.log('Sequence length:', seqLength);
            console.log('');
            
            const startTime = Date.now();
            
            await model.train(dataLoader, epochs, batchSize, (epoch, total, loss, acc) => {
                const elapsed = (Date.now() - startTime) / 1000;
                const avgTime = elapsed / epoch;
                const remaining = avgTime * (total - epoch);
                console.log(`Progress: ${epoch}/${total} - ETA: ${Math.round(remaining)}s`);
            });
            
            const totalTime = (Date.now() - startTime) / 1000;
            console.log(`\nTraining completed in ${totalTime.toFixed(1)}s`);
            
            // Save model
            await model.saveModel(modelPath);
        }
        
        // Generate text
        if (shouldGenerate || args.includes('--train')) {
            console.log('\nGenerating text...\n');
            
            const prompts = ['The ', 'What ', 'To be', 'O '];
            const temperatures = [0.5, 0.8, 1.0];
            
            for (const prompt of prompts) {
                console.log(`Prompt: "${prompt}"`);
                for (const temp of temperatures) {
                    const generated = await model.generateText(prompt, 100, dataLoader, temp);
                    console.log(`  Temp ${temp}: ${generated}`);
                }
                console.log('');
            }
        }
        
        // Interactive mode
        if (args.includes('--interactive')) {
            console.log('\nInteractive mode (type "quit" to exit)');
            const rl = readline.createInterface({
                input: process.stdin,
                output: process.stdout
            });
            
            const askPrompt = () => {
                rl.question('\nEnter prompt: ', async (prompt) => {
                    if (prompt.toLowerCase() === 'quit') {
                        rl.close();
                        process.exit(0);
                    }
                    
                    const generated = await model.generateText(prompt, 200, dataLoader, 0.8);
                    console.log(`Generated: ${generated}`);
                    askPrompt();
                });
            };
            
            askPrompt();
        }
        
    } catch (error) {
        console.error('Error:', error);
        process.exit(1);
    }
}

// Run if called directly
if (require.main === module) {
    main();
}

module.exports = { GPT, DataLoader, MultiHeadSelfAttention };