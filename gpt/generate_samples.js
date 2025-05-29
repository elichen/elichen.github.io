#!/usr/bin/env node

// Script to generate text samples from the latest checkpoint
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');

// Import components
const { MultiHeadSelfAttention } = require('./train_cli.js');
tf.serialization.registerClass(MultiHeadSelfAttention);

// Load vocabulary from checkpoint
function loadVocab(checkpointPath) {
    const vocabPath = path.join(checkpointPath, 'vocab.json');
    return JSON.parse(fs.readFileSync(vocabPath, 'utf8'));
}

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
    console.log('GPT Text Generation Samples');
    console.log('===========================\n');
    
    // Find latest checkpoint
    const checkpoints = fs.readdirSync('./checkpoints')
        .filter(f => f.startsWith('epoch_'))
        .map(f => ({ name: f, epoch: parseInt(f.split('_')[1]) }))
        .sort((a, b) => b.epoch - a.epoch);
    
    if (checkpoints.length === 0) {
        console.log('No checkpoints found!');
        return;
    }
    
    const latestCheckpoint = `./checkpoints/${checkpoints[0].name}`;
    console.log(`Loading model from ${latestCheckpoint}\n`);
    
    // Load model and vocabulary
    const model = await tf.loadLayersModel(`file://${latestCheckpoint}/model.json`);
    const vocab = loadVocab(latestCheckpoint);
    
    // Get training state
    const state = JSON.parse(fs.readFileSync('./training_state.json', 'utf8'));
    console.log(`Model trained for ${checkpoints[0].epoch} epochs`);
    console.log(`Best loss: ${state.bestLoss.toFixed(4)} at epoch ${state.bestEpoch}\n`);
    
    // Generate samples with different temperatures and prompts
    const samples = [
        { prompt: 'The ', temp: 0.5, desc: 'Conservative (temp=0.5)' },
        { prompt: 'The ', temp: 0.8, desc: 'Balanced (temp=0.8)' },
        { prompt: 'The ', temp: 1.0, desc: 'Creative (temp=1.0)' },
        { prompt: 'What ', temp: 0.8, desc: 'Question prompt' },
        { prompt: 'In the ', temp: 0.8, desc: 'Narrative prompt' },
        { prompt: 'To be or ', temp: 0.7, desc: 'Famous quote' },
        { prompt: 'O ', temp: 0.8, desc: 'Shakespearean style' },
        { prompt: 'Lord ', temp: 0.8, desc: 'Character prompt' },
        { prompt: 'I am ', temp: 0.8, desc: 'First person' },
        { prompt: 'When ', temp: 0.8, desc: 'Temporal prompt' }
    ];
    
    console.log('Text Generation Examples:');
    console.log('------------------------\n');
    
    for (const { prompt, temp, desc } of samples) {
        const generated = await generateText(model, prompt, 200, vocab, temp);
        console.log(`${desc}:`);
        console.log(`Prompt: "${prompt}"`);
        console.log(`Generated: "${generated}"`);
        console.log('---\n');
    }
    
    // Longer sample
    console.log('Longer sample (500 characters, temp=0.8):');
    const longSample = await generateText(model, 'The king ', 500, vocab, 0.8);
    console.log(`"${longSample}"`);
}

// Run
main().catch(console.error);