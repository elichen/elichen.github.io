#!/usr/bin/env node

// Optimal training script for coherent text generation
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');

// Import components from train_cli.js
const { MultiHeadSelfAttention } = require('./train_cli.js');

// Register the custom layer
tf.serialization.registerClass(MultiHeadSelfAttention);

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

// Optimized GPT Model
class OptimalGPT {
    constructor(vocabSize, seqLength) {
        this.vocabSize = vocabSize;
        this.seqLength = seqLength;

        // Optimized hyperparameters based on research
        const embedDim = 256;    // Larger embedding dimension
        const numHeads = 8;      // More attention heads
        const numLayers = 4;     // Deeper network
        const dropout = 0.1;     // Standard dropout

        // Input layers
        const tokenInputs = tf.input({ shape: [null], dtype: 'int32' });
        const positionInputs = tf.input({ shape: [null], dtype: 'int32' });
        const attentionMask = tf.input({ shape: [1, null, null], dtype: 'float32' });

        // Token embeddings with better initialization
        const tokenEmbeddingLayer = tf.layers.embedding({ 
            inputDim: this.vocabSize, 
            outputDim: embedDim,
            embeddings_initializer: tf.initializers.randomNormal({ stddev: 0.02 })
        });
        const tokenEmbeddings = tokenEmbeddingLayer.apply(tokenInputs);

        // Positional embeddings
        const positionEmbeddingLayer = tf.layers.embedding({ 
            inputDim: seqLength, 
            outputDim: embedDim,
            embeddings_initializer: tf.initializers.randomNormal({ stddev: 0.02 })
        });
        const positionEmbeddings = positionEmbeddingLayer.apply(positionInputs);

        // Combine embeddings
        let x = tf.layers.add().apply([tokenEmbeddings, positionEmbeddings]);
        x = tf.layers.dropout({ rate: dropout }).apply(x);

        // Transformer blocks
        for (let i = 0; i < numLayers; i++) {
            // Pre-norm architecture
            let attnInput = tf.layers.layerNormalization({ epsilon: 1e-5 }).apply(x);

            // Multi-head self-attention
            const attentionLayer = new MultiHeadSelfAttention({ numHeads, embedDim });
            let attnOutput = attentionLayer.apply([attnInput, attentionMask]);
            attnOutput = tf.layers.dropout({ rate: dropout }).apply(attnOutput);

            // Residual connection
            x = tf.layers.add().apply([x, attnOutput]);

            // Feed-forward network
            let ffnInput = tf.layers.layerNormalization({ epsilon: 1e-5 }).apply(x);
            let ffnOutput = tf.layers.dense({ 
                units: embedDim * 4, 
                activation: 'relu',
                kernelInitializer: tf.initializers.heNormal()
            }).apply(ffnInput);
            ffnOutput = tf.layers.dense({ 
                units: embedDim,
                kernelInitializer: tf.initializers.randomNormal({ stddev: 0.02 })
            }).apply(ffnOutput);
            ffnOutput = tf.layers.dropout({ rate: dropout }).apply(ffnOutput);

            // Residual connection
            x = tf.layers.add().apply([x, ffnOutput]);
        }

        // Final layer normalization
        x = tf.layers.layerNormalization({ epsilon: 1e-5 }).apply(x);

        // Output layer with zero initialization for stability
        const logits = tf.layers.dense({ 
            units: this.vocabSize,
            kernelInitializer: tf.initializers.zeros()
        }).apply(x);

        // Define the model
        this.model = tf.model({ inputs: [tokenInputs, positionInputs, attentionMask], outputs: logits });

        // Compile with optimized settings
        this.model.compile({
            optimizer: tf.train.adam(0.0003, 0.9, 0.98, 1e-9), // Transformer-specific betas
            loss: (yTrue, yPred) => {
                const yTrueFlat = tf.reshape(yTrue, [-1, this.vocabSize]);
                const yPredFlat = tf.reshape(yPred, [-1, this.vocabSize]);
                return tf.losses.softmaxCrossEntropy(yTrueFlat, yPredFlat);
            },
            metrics: ['accuracy'],
        });

        const totalParams = this.model.countParams();
        console.log(`Model initialized with ${totalParams} parameters`);
    }

    async train(dataLoader, epochs, batchSize, checkpointInterval = 100) {
        const losses = [];
        let bestLoss = Infinity;
        let bestEpoch = 0;
        
        console.log(`Starting training for ${epochs} epochs...`);
        const startTime = Date.now();

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

            // Dispose tensors
            inputs.dispose();
            targets.dispose();
            positionIndices.dispose();
            attentionMask.dispose();

            const loss = history.history.loss[0];
            const accuracy = history.history.accuracy || history.history.acc;
            const acc = accuracy ? accuracy[0] : 0;
            
            losses.push(loss);

            // Track best loss
            if (loss < bestLoss) {
                bestLoss = loss;
                bestEpoch = epoch + 1;
            }

            // Log progress
            if ((epoch + 1) % 10 === 0) {
                const elapsed = (Date.now() - startTime) / 1000;
                const eta = (elapsed / (epoch + 1)) * (epochs - epoch - 1);
                console.log(`Epoch ${epoch + 1}/${epochs} - Loss: ${loss.toFixed(4)}, Acc: ${acc.toFixed(4)}, Best: ${bestLoss.toFixed(4)} @ ${bestEpoch}, ETA: ${Math.round(eta)}s`);
            }

            // Save checkpoint
            if ((epoch + 1) % checkpointInterval === 0) {
                await this.saveCheckpoint(`./checkpoints/epoch_${epoch + 1}`, dataLoader);
            }

            // Generate sample for quality check
            if ((epoch + 1) % 50 === 0) {
                const sample = await this.generateText('The ', 80, dataLoader, 0.8);
                console.log(`  Sample: "${sample}"`);
            }
            
            await tf.nextFrame();
        }

        const totalTime = (Date.now() - startTime) / 1000;
        console.log(`\nTraining completed in ${totalTime.toFixed(1)}s`);
        console.log(`Best loss: ${bestLoss.toFixed(4)} at epoch ${bestEpoch}`);

        return { finalLoss: losses[losses.length - 1], bestLoss, bestEpoch };
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
            
            const logits2D = logitsLast.squeeze([1]);
            const scaledLogits = tf.div(logits2D, temperature);
            const sampled = tf.multinomial(scaledLogits, 1);
            const sampledArray = await sampled.array();
            const predictedIdx = sampledArray[0][0];
            const predictedChar = dataLoader.idx2char[predictedIdx];
        
            result.push(predictedChar);
            currentSequence.push(predictedIdx);
        
            // Dispose tensors
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

    async saveCheckpoint(path, dataLoader) {
        if (!fs.existsSync(path)) {
            fs.mkdirSync(path, { recursive: true });
        }
        
        await this.model.save(`file://${path}`);
        
        // Save vocabulary info
        fs.writeFileSync(`${path}/vocab.json`, JSON.stringify({
            char2idx: dataLoader.char2idx,
            idx2char: dataLoader.idx2char,
            vocabSize: dataLoader.vocabSize,
            seqLength: this.seqLength
        }));
        
        console.log(`  Checkpoint saved to ${path}`);
    }

    async saveForWeb(path, dataLoader) {
        if (!fs.existsSync(path)) {
            fs.mkdirSync(path, { recursive: true });
        }
        
        await this.model.save(`file://${path}`);
        
        // Save complete info for web loading
        const info = {
            char2idx: dataLoader.char2idx,
            idx2char: dataLoader.idx2char,
            vocabSize: dataLoader.vocabSize,
            seqLength: this.seqLength,
            modelType: 'gpt-coherent',
            version: '1.0'
        };
        
        fs.writeFileSync(`${path}/model_info.json`, JSON.stringify(info, null, 2));
        
        console.log(`\nModel saved for web deployment at ${path}`);
        console.log('Files created:');
        console.log('  - model.json (model architecture)');
        console.log('  - model_info.json (vocabulary and config)');
        console.log('  - group1-shard1of1.bin (model weights)');
    }
}

// Helper function to create a causal mask
function createCausalMask(batchSize, seqLength) {
    return tf.tidy(() => {
        const mask = tf.linalg.bandPart(tf.ones([seqLength, seqLength]), -1, 0);
        const maskReshaped = mask.reshape([1, 1, seqLength, seqLength]);
        return maskReshaped.tile([batchSize, 1, 1, 1]);
    });
}

// Main training function
async function trainOptimalModel() {
    console.log('Training Optimal GPT Model for Coherent Text Generation');
    console.log('======================================================\n');
    
    // Load dataset
    const text = loadTextDataset('input.txt');
    console.log(`Dataset loaded: ${text.length} characters`);
    
    // Use substantial training data
    const trainSize = Math.min(text.length * 0.9, 800000);
    const trainText = text.slice(0, trainSize);
    
    // Hyperparameters
    const seqLength = 64;
    const batchSize = 24;  // Balanced for memory and gradient quality
    const epochs = 1500;   // Enough for convergence
    
    // Create data loader
    const dataLoader = new DataLoader(trainText, seqLength);
    console.log(`Vocabulary size: ${dataLoader.vocabSize}`);
    console.log(`Training on ${trainText.length} characters\n`);
    
    // Create checkpoint directory
    if (!fs.existsSync('./checkpoints')) {
        fs.mkdirSync('./checkpoints');
    }
    
    // Create and train model
    const model = new OptimalGPT(dataLoader.vocabSize, seqLength);
    
    // Train the model
    const result = await model.train(dataLoader, epochs, batchSize, 250);
    
    // Generate final samples
    console.log('\n\nFinal Text Generation Samples:');
    console.log('==============================\n');
    
    const testPrompts = [
        { prompt: 'The ', temp: 0.6, desc: 'Low temperature (more conservative)' },
        { prompt: 'The ', temp: 0.8, desc: 'Medium temperature (balanced)' },
        { prompt: 'The ', temp: 1.0, desc: 'High temperature (more creative)' },
        { prompt: 'What ', temp: 0.8, desc: 'Question prompt' },
        { prompt: 'In the ', temp: 0.8, desc: 'Narrative prompt' },
        { prompt: 'To be or ', temp: 0.7, desc: 'Famous quote completion' },
        { prompt: 'O ', temp: 0.8, desc: 'Shakespearean exclamation' }
    ];
    
    for (const { prompt, temp, desc } of testPrompts) {
        const generated = await model.generateText(prompt, 150, dataLoader, temp);
        console.log(`${desc}:`);
        console.log(`Prompt: "${prompt}" (temp=${temp})`);
        console.log(`Generated: "${generated}"\n`);
    }
    
    // Save final model for web
    await model.saveForWeb('./web_model', dataLoader);
    
    // Create a README for the model
    const readme = `# GPT Model for Web Deployment

## Model Info
- Type: Character-level GPT
- Parameters: ${model.model.countParams()}
- Architecture: 4-layer transformer, 256 embedding dim, 8 attention heads
- Training: ${epochs} epochs on Shakespeare text
- Final Loss: ${result.finalLoss.toFixed(4)}
- Best Loss: ${result.bestLoss.toFixed(4)} (epoch ${result.bestEpoch})

## Usage
1. Copy the web_model directory to your web server
2. Load using TensorFlow.js:
   \`\`\`javascript
   const model = await tf.loadLayersModel('./web_model/model.json');
   const modelInfo = await fetch('./web_model/model_info.json').then(r => r.json());
   \`\`\`

## Files
- model.json: Model architecture
- group1-shard1of1.bin: Model weights
- model_info.json: Vocabulary and configuration
`;
    
    fs.writeFileSync('./web_model/README.md', readme);
    
    console.log('\nTraining complete! Model ready for deployment.');
    console.log('Copy the ./web_model directory to your web application.');
}

// Run if called directly
if (require.main === module) {
    trainOptimalModel().catch(console.error);
}

module.exports = { OptimalGPT, createCausalMask };