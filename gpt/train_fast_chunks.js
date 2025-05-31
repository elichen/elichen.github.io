#!/usr/bin/env node

// Fast chunk-based training for quick iterations
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const { MultiHeadSelfAttention } = require('./train_cli.js');
tf.serialization.registerClass(MultiHeadSelfAttention);

// Simple state management
function loadState() {
    const statePath = './fast_training_state.json';
    if (fs.existsSync(statePath)) {
        return JSON.parse(fs.readFileSync(statePath, 'utf8'));
    }
    return {
        epoch: 0,
        bestLoss: Infinity,
        modelPath: null
    };
}

function saveState(state) {
    fs.writeFileSync('./fast_training_state.json', JSON.stringify(state, null, 2));
}

// Create model
function createModel(vocabSize, seqLength) {
    const embedDim = 128; // Smaller for faster training
    const numHeads = 4;
    const numLayers = 3;
    const dropout = 0.1;

    const tokenInputs = tf.input({ shape: [null], dtype: 'int32' });
    const positionInputs = tf.input({ shape: [null], dtype: 'int32' });
    const attentionMask = tf.input({ shape: [1, null, null], dtype: 'float32' });

    const tokenEmbedding = tf.layers.embedding({ 
        inputDim: vocabSize, 
        outputDim: embedDim,
        embeddingsInitializer: tf.initializers.randomNormal({ stddev: 0.02 })
    });
    const positionEmbedding = tf.layers.embedding({ 
        inputDim: seqLength, 
        outputDim: embedDim,
        embeddingsInitializer: tf.initializers.randomNormal({ stddev: 0.02 })
    });

    const tokenEmbeddings = tokenEmbedding.apply(tokenInputs);
    const positionEmbeddings = positionEmbedding.apply(positionInputs);
    
    let x = tf.layers.add().apply([tokenEmbeddings, positionEmbeddings]);
    x = tf.layers.dropout({ rate: dropout }).apply(x);

    for (let i = 0; i < numLayers; i++) {
        let attnInput = tf.layers.layerNormalization({ epsilon: 1e-5 }).apply(x);
        const attention = new MultiHeadSelfAttention({ numHeads, embedDim });
        let attnOutput = attention.apply([attnInput, attentionMask]);
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
        units: vocabSize,
        kernelInitializer: tf.initializers.zeros()
    }).apply(x);

    return tf.model({ 
        inputs: [tokenInputs, positionInputs, attentionMask], 
        outputs: logits 
    });
}

function createCausalMask(batchSize, seqLength) {
    return tf.tidy(() => {
        const mask = tf.linalg.bandPart(tf.ones([seqLength, seqLength]), -1, 0);
        return mask.reshape([1, 1, seqLength, seqLength]).tile([batchSize, 1, 1, 1]);
    });
}

// Fast training function
async function trainChunk() {
    console.log('Fast Chunk Training');
    console.log('==================\n');
    
    // Load state
    const state = loadState();
    console.log('Current state:', state);
    
    // Load data
    const text = fs.readFileSync('input.txt', 'utf8').slice(0, 500000);
    const chars = Array.from(new Set(text));
    const char2idx = {};
    const idx2char = {};
    chars.forEach((c, i) => {
        char2idx[c] = i;
        idx2char[i] = c;
    });
    const vocabSize = chars.length;
    const textIndices = Array.from(text).map(c => char2idx[c]);
    
    console.log(`Vocabulary: ${vocabSize} chars`);
    console.log(`Text: ${text.length} chars\n`);
    
    // Parameters
    const seqLength = 64;
    const batchSize = 32;
    const stepsPerChunk = 50; // Quick chunks
    const chunksPerRun = 10;
    
    // Load or create model
    let model;
    if (state.modelPath && fs.existsSync(state.modelPath)) {
        console.log('Loading model from:', state.modelPath);
        model = await tf.loadLayersModel(`file://${state.modelPath}/model.json`);
    } else {
        console.log('Creating new model...');
        model = createModel(vocabSize, seqLength);
    }
    
    console.log(`Model params: ${model.countParams()}\n`);
    
    // Compile with appropriate learning rate
    const lr = state.epoch < 100 ? 0.001 : 0.0001;
    model.compile({
        optimizer: tf.train.adam(lr),
        loss: tf.losses.softmaxCrossEntropy,
        metrics: ['accuracy']
    });
    
    // Training chunks
    for (let chunk = 0; chunk < chunksPerRun; chunk++) {
        let chunkLoss = 0;
        let chunkAcc = 0;
        
        for (let step = 0; step < stepsPerChunk; step++) {
            // Create batch
            const inputs = [];
            const targets = [];
            
            for (let i = 0; i < batchSize; i++) {
                const idx = Math.floor(Math.random() * (textIndices.length - seqLength - 1));
                inputs.push(textIndices.slice(idx, idx + seqLength));
                targets.push(textIndices.slice(idx + 1, idx + seqLength + 1));
            }
            
            const inputTensor = tf.tensor2d(inputs, [batchSize, seqLength], 'int32');
            const targetTensor = tf.oneHot(tf.tensor2d(targets, [batchSize, seqLength], 'int32'), vocabSize);
            const positions = tf.tensor2d(
                Array(batchSize).fill(Array.from({length: seqLength}, (_, i) => i)),
                [batchSize, seqLength],
                'int32'
            );
            const mask = createCausalMask(batchSize, seqLength);
            
            const history = await model.fit(
                [inputTensor, positions, mask],
                targetTensor,
                { epochs: 1, verbose: 0 }
            );
            
            chunkLoss += history.history.loss[0];
            const acc = history.history.acc || history.history.accuracy;
            chunkAcc += acc ? acc[0] : 0;
            
            // Cleanup
            inputTensor.dispose();
            targetTensor.dispose();
            positions.dispose();
            mask.dispose();
            
            if (step % 10 === 0) {
                await tf.nextFrame();
            }
        }
        
        chunkLoss /= stepsPerChunk;
        chunkAcc /= stepsPerChunk;
        state.epoch++;
        
        console.log(`Chunk ${chunk + 1}/${chunksPerRun} (Epoch ${state.epoch}) - Loss: ${chunkLoss.toFixed(4)}, Acc: ${chunkAcc.toFixed(4)}`);
        
        if (chunkLoss < state.bestLoss) {
            state.bestLoss = chunkLoss;
            console.log('  New best loss!');
        }
        
        // Save checkpoint every 5 chunks
        if ((chunk + 1) % 5 === 0) {
            const checkpointPath = `./checkpoints/fast_epoch_${state.epoch}`;
            if (!fs.existsSync(checkpointPath)) {
                fs.mkdirSync(checkpointPath, { recursive: true });
            }
            await model.save(`file://${checkpointPath}`);
            state.modelPath = checkpointPath;
            saveState(state);
            console.log(`  Checkpoint saved: ${checkpointPath}`);
        }
        
        // Generate sample
        if ((chunk + 1) % 5 === 0) {
            const sample = await generateSample(model, 'The ', 100, char2idx, idx2char, vocabSize, seqLength);
            console.log(`  Sample: "${sample}"`);
        }
    }
    
    // Save final model for this run
    const finalPath = `./checkpoints/fast_final_${state.epoch}`;
    if (!fs.existsSync(finalPath)) {
        fs.mkdirSync(finalPath, { recursive: true });
    }
    await model.save(`file://${finalPath}`);
    state.modelPath = finalPath;
    saveState(state);
    
    // Save to web_model if loss is good
    if (state.bestLoss < 2.0) {
        console.log('\nSaving to web_model...');
        await model.save('file://./web_model');
        
        const modelInfo = {
            char2idx: char2idx,
            idx2char: idx2char,
            vocabSize: vocabSize,
            seqLength: seqLength,
            epochsTrained: state.epoch,
            currentLoss: state.bestLoss
        };
        fs.writeFileSync('./web_model/model_info.json', JSON.stringify(modelInfo, null, 2));
    }
    
    console.log('\nChunk training complete!');
    console.log(`Total epochs: ${state.epoch}`);
    console.log(`Best loss: ${state.bestLoss.toFixed(4)}`);
    
    if (state.bestLoss > 1.2) {
        console.log('Run again to continue training!');
    } else {
        console.log('Target loss achieved!');
    }
}

async function generateSample(model, prompt, length, char2idx, idx2char, vocabSize, seqLength) {
    let result = Array.from(prompt);
    let currentSeq = result.map(c => char2idx[c] || 0);
    
    for (let i = 0; i < length; i++) {
        if (currentSeq.length > seqLength) {
            currentSeq = currentSeq.slice(-seqLength);
        }
        
        const input = tf.tensor2d([currentSeq], [1, currentSeq.length], 'int32');
        const positions = tf.tensor2d([Array.from({length: currentSeq.length}, (_, i) => i)], [1, currentSeq.length], 'int32');
        const mask = createCausalMask(1, currentSeq.length);
        
        const predictions = model.predict([input, positions, mask]);
        const lastToken = predictions.slice([0, currentSeq.length - 1, 0], [1, 1, vocabSize]);
        const scaledLogits = tf.div(lastToken.squeeze(), 0.8);
        const sampled = tf.multinomial(scaledLogits, 1);
        const sampledIdx = await sampled.array();
        
        const nextChar = idx2char[sampledIdx[0]];
        result.push(nextChar);
        currentSeq.push(sampledIdx[0]);
        
        // Cleanup
        input.dispose();
        positions.dispose();
        mask.dispose();
        predictions.dispose();
        lastToken.dispose();
        scaledLogits.dispose();
        sampled.dispose();
    }
    
    return result.join('');
}

// Run
trainChunk().catch(console.error);