#!/usr/bin/env node

const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const { MultiHeadSelfAttention } = require('./train_cli.js');

// Register the custom layer
tf.serialization.registerClass(MultiHeadSelfAttention);

// Helper function to create causal mask
function createCausalMask(batchSize, seqLength) {
    return tf.tidy(() => {
        const mask = tf.linalg.bandPart(tf.ones([seqLength, seqLength]), -1, 0);
        const maskReshaped = mask.reshape([1, 1, seqLength, seqLength]);
        return maskReshaped.tile([batchSize, 1, 1, 1]);
    });
}

// Text generation function
async function generateText(model, startSequence, numChars, vocab, temperature = 0.8) {
    let result = Array.from(startSequence);
    let currentSequence = result.map(c => vocab.char2idx[c] || 0);
    
    for (let i = 0; i < numChars; i++) {
        if (currentSequence.length > vocab.seqLength) {
            currentSequence = currentSequence.slice(-vocab.seqLength);
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
        const logitsLast = logits.slice([0, currentSequence.length - 1, 0], [1, 1, vocab.vocabSize]);
        
        const logits2D = logitsLast.squeeze([1]);
        const scaledLogits = tf.div(logits2D, temperature);
        const sampled = tf.multinomial(scaledLogits, 1);
        const sampledArray = await sampled.array();
        const predictedIdx = sampledArray[0][0];
        const predictedChar = vocab.idx2char[predictedIdx];
    
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

async function main() {
    console.log('Checking model quality from best checkpoint...\n');
    
    // Load model from epoch 1200 (closest to best epoch 1181)
    const checkpointPath = './checkpoints/epoch_1200';
    console.log(`Loading model from ${checkpointPath}`);
    
    const model = await tf.loadLayersModel(`file://${checkpointPath}/model.json`);
    const vocab = JSON.parse(fs.readFileSync(`${checkpointPath}/vocab.json`, 'utf8'));
    
    console.log(`Model loaded. Best loss was 1.9861 at epoch 1181\n`);
    
    // Generate samples with different temperatures
    const prompts = [
        { prompt: 'The ', temp: 0.5, desc: 'Low temperature (0.5)' },
        { prompt: 'The ', temp: 0.8, desc: 'Medium temperature (0.8)' },
        { prompt: 'The ', temp: 1.0, desc: 'High temperature (1.0)' },
        { prompt: 'What ', temp: 0.8 },
        { prompt: 'To be or ', temp: 0.8 },
        { prompt: 'Lord ', temp: 0.8 },
        { prompt: 'I am ', temp: 0.8 }
    ];
    
    console.log('Text Generation Samples from Best Model:');
    console.log('========================================\n');
    
    for (const { prompt, temp, desc } of prompts) {
        const generated = await generateText(model, prompt, 200, vocab, temp);
        console.log(`${desc || `Prompt: "${prompt}"`} (temp=${temp}):`);
        console.log(`"${generated}"\n`);
    }
}

main().catch(console.error);