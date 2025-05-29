#!/usr/bin/env node

// Resumable training script that works within 10-minute windows
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');

// Import components
const { MultiHeadSelfAttention } = require('./train_cli.js');
tf.serialization.registerClass(MultiHeadSelfAttention);

// Load dataset and create data loader (same as train_optimal.js)
function loadTextDataset(filename) {
    return fs.readFileSync(filename, 'utf8');
}

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

// Helper function to create causal mask
function createCausalMask(batchSize, seqLength) {
    return tf.tidy(() => {
        const mask = tf.linalg.bandPart(tf.ones([seqLength, seqLength]), -1, 0);
        const maskReshaped = mask.reshape([1, 1, seqLength, seqLength]);
        return maskReshaped.tile([batchSize, 1, 1, 1]);
    });
}

// Load or create training state
function loadTrainingState() {
    const statePath = './training_state.json';
    if (fs.existsSync(statePath)) {
        return JSON.parse(fs.readFileSync(statePath, 'utf8'));
    }
    return {
        startEpoch: 0,
        bestLoss: Infinity,
        bestEpoch: 0,
        losses: []
    };
}

// Save training state
function saveTrainingState(state) {
    fs.writeFileSync('./training_state.json', JSON.stringify(state, null, 2));
}

// Main training function
async function trainWithResume() {
    console.log('Resumable GPT Training (10-minute sessions)');
    console.log('==========================================\n');
    
    // Load dataset
    const text = loadTextDataset('input.txt');
    const trainSize = Math.min(text.length * 0.9, 800000);
    const trainText = text.slice(0, trainSize);
    
    // Hyperparameters
    const seqLength = 64;
    const batchSize = 24;
    const totalEpochs = 5000;
    const epochsPerSession = 400; // Target for 10-minute window
    
    // Create data loader
    const dataLoader = new DataLoader(trainText, seqLength);
    console.log(`Vocabulary size: ${dataLoader.vocabSize}`);
    console.log(`Training on ${trainText.length} characters\n`);
    
    // Load training state
    const state = loadTrainingState();
    const startEpoch = state.startEpoch;
    const endEpoch = Math.min(startEpoch + epochsPerSession, totalEpochs);
    
    console.log(`Resuming from epoch ${startEpoch + 1}`);
    console.log(`Training epochs ${startEpoch + 1} to ${endEpoch}\n`);
    
    // Load or create model
    let model;
    const lastCheckpoint = `./checkpoints/epoch_${Math.floor(startEpoch / 50) * 50}`;
    
    if (startEpoch > 0 && fs.existsSync(lastCheckpoint)) {
        console.log(`Loading checkpoint from ${lastCheckpoint}`);
        model = await tf.loadLayersModel(`file://${lastCheckpoint}/model.json`);
        
        // Recompile model
        model.compile({
            optimizer: tf.train.adam(0.0003, 0.9, 0.98, 1e-9),
            loss: (yTrue, yPred) => {
                const yTrueFlat = tf.reshape(yTrue, [-1, dataLoader.vocabSize]);
                const yPredFlat = tf.reshape(yPred, [-1, dataLoader.vocabSize]);
                return tf.losses.softmaxCrossEntropy(yTrueFlat, yPredFlat);
            },
            metrics: ['accuracy'],
        });
    } else {
        // Create new model (using OptimalGPT architecture from train_optimal.js)
        console.log('Creating new model...');
        
        const embedDim = 256;
        const numHeads = 8;
        const numLayers = 4;
        const dropout = 0.1;

        // Model building code (same as OptimalGPT)
        const tokenInputs = tf.input({ shape: [null], dtype: 'int32' });
        const positionInputs = tf.input({ shape: [null], dtype: 'int32' });
        const attentionMask = tf.input({ shape: [1, null, null], dtype: 'float32' });

        const tokenEmbeddingLayer = tf.layers.embedding({ 
            inputDim: dataLoader.vocabSize, 
            outputDim: embedDim,
            embeddings_initializer: tf.initializers.randomNormal({ stddev: 0.02 })
        });
        const tokenEmbeddings = tokenEmbeddingLayer.apply(tokenInputs);

        const positionEmbeddingLayer = tf.layers.embedding({ 
            inputDim: seqLength, 
            outputDim: embedDim,
            embeddings_initializer: tf.initializers.randomNormal({ stddev: 0.02 })
        });
        const positionEmbeddings = positionEmbeddingLayer.apply(positionInputs);

        let x = tf.layers.add().apply([tokenEmbeddings, positionEmbeddings]);
        x = tf.layers.dropout({ rate: dropout }).apply(x);

        // Transformer blocks
        for (let i = 0; i < numLayers; i++) {
            let attnInput = tf.layers.layerNormalization({ epsilon: 1e-5 }).apply(x);

            const attentionLayer = new MultiHeadSelfAttention({ numHeads, embedDim });
            let attnOutput = attentionLayer.apply([attnInput, attentionMask]);
            attnOutput = tf.layers.dropout({ rate: dropout }).apply(attnOutput);

            x = tf.layers.add().apply([x, attnOutput]);

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

            x = tf.layers.add().apply([x, ffnOutput]);
        }

        x = tf.layers.layerNormalization({ epsilon: 1e-5 }).apply(x);

        const logits = tf.layers.dense({ 
            units: dataLoader.vocabSize,
            kernelInitializer: tf.initializers.zeros()
        }).apply(x);

        model = tf.model({ inputs: [tokenInputs, positionInputs, attentionMask], outputs: logits });

        model.compile({
            optimizer: tf.train.adam(0.0003, 0.9, 0.98, 1e-9),
            loss: (yTrue, yPred) => {
                const yTrueFlat = tf.reshape(yTrue, [-1, dataLoader.vocabSize]);
                const yPredFlat = tf.reshape(yPred, [-1, dataLoader.vocabSize]);
                return tf.losses.softmaxCrossEntropy(yTrueFlat, yPredFlat);
            },
            metrics: ['accuracy'],
        });

        console.log(`Model initialized with ${model.countParams()} parameters\n`);
    }
    
    // Training loop
    const sessionStartTime = Date.now();
    let bestLoss = state.bestLoss;
    let bestEpoch = state.bestEpoch;
    const losses = [...state.losses];
    
    for (let epoch = startEpoch; epoch < endEpoch; epoch++) {
        const { inputs, targets } = dataLoader.getBatch(batchSize);
        const seqLen = inputs.shape[1];
        const positionIndices = tf.tensor2d(
            Array.from({ length: batchSize }, () =>
                Array.from({ length: seqLen }, (_, i) => i)
            ),
            [batchSize, seqLen],
            'int32'
        );

        const attentionMask = createCausalMask(batchSize, seqLen);
        
        const history = await model.fit([inputs, positionIndices, attentionMask], targets, {
            epochs: 1,
            verbose: 0,
        });

        inputs.dispose();
        targets.dispose();
        positionIndices.dispose();
        attentionMask.dispose();

        const loss = history.history.loss[0];
        const accuracy = history.history.accuracy || history.history.acc;
        const acc = accuracy ? accuracy[0] : 0;
        
        losses.push(loss);

        if (loss < bestLoss) {
            bestLoss = loss;
            bestEpoch = epoch + 1;
        }

        // Log progress
        if ((epoch + 1) % 10 === 0) {
            const elapsed = (Date.now() - sessionStartTime) / 1000;
            const eta = (elapsed / (epoch - startEpoch + 1)) * (endEpoch - epoch - 1);
            console.log(`Epoch ${epoch + 1}/${totalEpochs} - Loss: ${loss.toFixed(4)}, Acc: ${acc.toFixed(4)}, Best: ${bestLoss.toFixed(4)} @ ${bestEpoch}, ETA: ${Math.round(eta)}s`);
        }

        // Save checkpoint every 50 epochs
        if ((epoch + 1) % 50 === 0) {
            const checkpointPath = `./checkpoints/epoch_${epoch + 1}`;
            if (!fs.existsSync('./checkpoints')) {
                fs.mkdirSync('./checkpoints');
            }
            await model.save(`file://${checkpointPath}`);
            
            // Save vocabulary info
            fs.writeFileSync(`${checkpointPath}/vocab.json`, JSON.stringify({
                char2idx: dataLoader.char2idx,
                idx2char: dataLoader.idx2char,
                vocabSize: dataLoader.vocabSize,
                seqLength: seqLength
            }));
            
            console.log(`  Checkpoint saved to ${checkpointPath}`);
        }

        // Generate sample every 50 epochs
        if ((epoch + 1) % 50 === 0) {
            const sample = await generateText(model, 'The ', 80, dataLoader, 0.8);
            console.log(`  Sample: "${sample}"`);
        }
        
        await tf.nextFrame();
    }
    
    // Update and save state
    const newState = {
        startEpoch: endEpoch,
        bestLoss: bestLoss,
        bestEpoch: bestEpoch,
        losses: losses
    };
    saveTrainingState(newState);
    
    const sessionTime = (Date.now() - sessionStartTime) / 1000;
    console.log(`\nSession completed in ${sessionTime.toFixed(1)}s`);
    console.log(`Trained epochs ${startEpoch + 1} to ${endEpoch}`);
    console.log(`Current best loss: ${bestLoss.toFixed(4)} at epoch ${bestEpoch}`);
    
    if (endEpoch >= totalEpochs) {
        console.log('\nTraining complete!');
        
        // Save final model for web
        await saveForWeb(model, './web_model', dataLoader);
        
        // Generate final samples
        console.log('\n\nFinal Text Generation Samples:');
        console.log('==============================\n');
        
        const testPrompts = [
            { prompt: 'The ', temp: 0.8 },
            { prompt: 'What ', temp: 0.8 },
            { prompt: 'To be or ', temp: 0.7 }
        ];
        
        for (const { prompt, temp } of testPrompts) {
            const generated = await generateText(model, prompt, 150, dataLoader, temp);
            console.log(`Prompt: "${prompt}" (temp=${temp})`);
            console.log(`Generated: "${generated}"\n`);
        }
    } else {
        console.log(`\nRun again to continue training from epoch ${endEpoch + 1}`);
    }
}

// Text generation function
async function generateText(model, startSequence, numChars, dataLoader, temperature = 0.8) {
    let result = Array.from(startSequence);
    let currentSequence = result.map(c => dataLoader.char2idx[c] || 0);
    
    for (let i = 0; i < numChars; i++) {
        if (currentSequence.length > 64) {
            currentSequence = currentSequence.slice(-64);
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
    
        const logits = model.predict([input, positionIndices, attentionMask], { training: false });
        const logitsLast = logits.slice([0, currentSequence.length - 1, 0], [1, 1, dataLoader.vocabSize]);
        
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

// Save for web function
async function saveForWeb(model, path, dataLoader) {
    if (!fs.existsSync(path)) {
        fs.mkdirSync(path, { recursive: true });
    }
    
    await model.save(`file://${path}`);
    
    const info = {
        char2idx: dataLoader.char2idx,
        idx2char: dataLoader.idx2char,
        vocabSize: dataLoader.vocabSize,
        seqLength: 64,
        modelType: 'gpt-coherent',
        version: '1.0'
    };
    
    fs.writeFileSync(`${path}/model_info.json`, JSON.stringify(info, null, 2));
    
    console.log(`\nModel saved for web deployment at ${path}`);
}

// Run if called directly
if (require.main === module) {
    trainWithResume().catch(console.error);
}

module.exports = { trainWithResume };