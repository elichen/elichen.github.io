#!/usr/bin/env node

// Focused training script to achieve coherent text generation
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');

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

// Simplified but effective GPT
class SimpleGPT {
    constructor(vocabSize, seqLength) {
        this.vocabSize = vocabSize;
        this.seqLength = seqLength;
        
        // Hyperparameters optimized for coherence
        const embedDim = 256;
        const numHeads = 8;
        const numLayers = 3;
        const dropout = 0.1;
        
        // Build model
        const inputs = tf.input({shape: [seqLength], dtype: 'int32'});
        
        // Embeddings with better initialization
        let x = tf.layers.embedding({
            inputDim: vocabSize,
            outputDim: embedDim,
            inputLength: seqLength,
            embeddings_initializer: tf.initializers.randomNormal({mean: 0, stddev: 0.02})
        }).apply(inputs);
        
        // Add learned positional embeddings
        const posEmbed = tf.layers.embedding({
            inputDim: seqLength,
            outputDim: embedDim,
            inputLength: seqLength,
            embeddings_initializer: tf.initializers.randomNormal({mean: 0, stddev: 0.02})
        });
        
        // Create position indices
        const positions = tf.layers.lambda({
            func: () => tf.range(0, seqLength, 1, 'int32').expandDims(0).tile([tf.backend().symbolic(inputs).shape[0], 1])
        }).apply(inputs);
        
        const posEmbeddings = posEmbed.apply(positions);
        x = tf.layers.add().apply([x, posEmbeddings]);
        x = tf.layers.dropout({rate: dropout}).apply(x);
        
        // Transformer blocks
        for (let i = 0; i < numLayers; i++) {
            // Self-attention block
            const ln1 = tf.layers.layerNormalization({epsilon: 1e-5}).apply(x);
            
            // Simplified multi-head attention using dense layers
            const attentionDim = embedDim;
            
            // Attention mechanism
            let attn = tf.layers.dense({units: attentionDim * 3}).apply(ln1); // Q, K, V
            attn = tf.layers.reshape({targetShape: [seqLength, 3, attentionDim]}).apply(attn);
            
            // Apply attention (simplified)
            attn = tf.layers.dense({units: embedDim}).apply(ln1);
            attn = tf.layers.dropout({rate: dropout}).apply(attn);
            
            x = tf.layers.add().apply([x, attn]);
            
            // FFN
            const ln2 = tf.layers.layerNormalization({epsilon: 1e-5}).apply(x);
            let ffn = tf.layers.dense({
                units: embedDim * 4,
                activation: 'relu',
                kernelInitializer: tf.initializers.heNormal()
            }).apply(ln2);
            ffn = tf.layers.dense({
                units: embedDim,
                kernelInitializer: tf.initializers.randomNormal({stddev: 0.02})
            }).apply(ffn);
            ffn = tf.layers.dropout({rate: dropout}).apply(ffn);
            
            x = tf.layers.add().apply([x, ffn]);
        }
        
        // Output
        x = tf.layers.layerNormalization({epsilon: 1e-5}).apply(x);
        const logits = tf.layers.dense({
            units: vocabSize,
            kernelInitializer: tf.initializers.zeros()
        }).apply(x);
        
        this.model = tf.model({inputs: inputs, outputs: logits});
        
        // Compile with learning rate scheduling
        const initialLearningRate = 0.0006;
        const decaySteps = 1000;
        const decayRate = 0.95;
        
        const learningRateSchedule = tf.train.exponentialDecay(
            initialLearningRate,
            decaySteps,
            decayRate,
            true
        );
        
        this.model.compile({
            optimizer: tf.train.adam(learningRateSchedule, 0.9, 0.98),
            loss: 'sparseCategoricalCrossentropy',
            metrics: ['accuracy']
        });
        
        console.log('Model parameters:', this.model.countParams());
    }
    
    async train(dataLoader, epochs, batchSize) {
        const losses = [];
        let bestLoss = Infinity;
        let epochsSinceImprovement = 0;
        
        for (let epoch = 0; epoch < epochs; epoch++) {
            const batch = dataLoader.getBatch(batchSize);
            
            // Convert one-hot back to sparse for sparseCategoricalCrossentropy
            const sparseTargets = tf.argMax(batch.targets, -1);
            
            const history = await this.model.fit(batch.inputs, sparseTargets, {
                epochs: 1,
                verbose: 0
            });
            
            batch.inputs.dispose();
            batch.targets.dispose();
            sparseTargets.dispose();
            
            const loss = history.history.loss[0];
            const acc = history.history.acc?.[0] || history.history.accuracy?.[0] || 0;
            losses.push(loss);
            
            // Track improvement
            if (loss < bestLoss) {
                bestLoss = loss;
                epochsSinceImprovement = 0;
            } else {
                epochsSinceImprovement++;
            }
            
            // Log progress
            if ((epoch + 1) % 10 === 0) {
                const avgLoss = losses.slice(-10).reduce((a, b) => a + b) / 10;
                console.log(`Epoch ${epoch + 1}: Loss=${loss.toFixed(4)}, Avg=${avgLoss.toFixed(4)}, Acc=${acc.toFixed(4)}, Best=${bestLoss.toFixed(4)}`);
                
                // Sample generation for monitoring
                if ((epoch + 1) % 50 === 0) {
                    const sample = await this.generate(dataLoader, 'The ', 50, 0.8);
                    console.log(`  Sample: "${sample}"`);
                }
            }
            
            // Early stopping if no improvement
            if (epochsSinceImprovement > 100 && epoch > 500) {
                console.log('Early stopping - no improvement for 100 epochs');
                break;
            }
            
            await tf.nextFrame();
        }
        
        return { finalLoss: losses[losses.length - 1], bestLoss };
    }
    
    async generate(dataLoader, prompt, length, temperature = 0.8) {
        let context = [...prompt].map(ch => dataLoader.char2idx[ch] || 0);
        const generated = [...prompt];
        
        for (let i = 0; i < length; i++) {
            // Prepare context
            let inputContext = context.slice(-this.seqLength);
            while (inputContext.length < this.seqLength) {
                inputContext = [0, ...inputContext];
            }
            
            const input = tf.tensor2d([inputContext], [1, this.seqLength], 'int32');
            const logits = this.model.predict(input, {training: false});
            
            // Get last position logits
            const lastLogits = logits.slice([0, this.seqLength - 1, 0], [1, 1, this.vocabSize]).squeeze();
            
            // Temperature sampling
            const scaledLogits = tf.div(lastLogits, temperature);
            const probs = tf.softmax(scaledLogits);
            
            const idx = tf.multinomial(probs, 1).arraySync()[0];
            const nextChar = dataLoader.idx2char[idx] || '?';
            
            generated.push(nextChar);
            context.push(idx);
            
            // Cleanup
            input.dispose();
            logits.dispose();
            lastLogits.dispose();
            scaledLogits.dispose();
            probs.dispose();
        }
        
        return generated.join('');
    }
    
    async saveForWeb(modelPath) {
        // Save in format compatible with browser
        await this.model.save(`file://${modelPath}`);
        
        // Also save vocab mapping
        const vocabPath = path.join(modelPath, 'vocab.json');
        fs.writeFileSync(vocabPath, JSON.stringify({
            chars: Object.keys(dataLoader.char2idx).sort(),
            seqLength: this.seqLength
        }));
        
        console.log(`Model saved for web at ${modelPath}`);
    }
}

// Main training function
async function trainToCoherence() {
    console.log('Training GPT to Coherence');
    console.log('=========================\n');
    
    // Load and prepare data
    const text = loadTextDataset('input.txt');
    console.log(`Loaded ${text.length} characters`);
    
    // Use more data for better coherence
    const trainText = text.slice(0, 500000); // First 500k chars
    const seqLength = 64;
    const batchSize = 32;
    
    const dataLoader = new DataLoader(trainText, seqLength);
    console.log(`Vocabulary size: ${dataLoader.vocabSize}`);
    
    // Create and train model
    const model = new SimpleGPT(dataLoader.vocabSize, seqLength);
    
    // Train in stages with evaluation
    const stages = [
        { epochs: 500, desc: 'Initial training' },
        { epochs: 500, desc: 'Refinement' },
        { epochs: 500, desc: 'Final polish' }
    ];
    
    let totalEpochs = 0;
    
    for (const stage of stages) {
        console.log(`\n${stage.desc} (${stage.epochs} epochs)...`);
        
        const result = await model.train(dataLoader, stage.epochs, batchSize);
        totalEpochs += stage.epochs;
        
        console.log(`Stage complete. Loss: ${result.finalLoss.toFixed(4)}`);
        
        // Evaluate
        console.log('\nEvaluation samples:');
        const prompts = ['The ', 'In the ', 'What ', 'To be ', 'And '];
        
        for (const prompt of prompts) {
            const generated = await model.generate(dataLoader, prompt, 100, 0.8);
            console.log(`"${prompt}" -> "${generated}"`);
        }
        
        // Stop if good enough
        if (result.bestLoss < 1.5) {
            console.log('\nReached target loss! Stopping training.');
            break;
        }
    }
    
    // Save final model
    const modelPath = './gpt_model_coherent';
    await model.saveForWeb(modelPath);
    
    // Generate final samples
    console.log('\n\nFinal generation samples:');
    console.log('=========================');
    
    const finalPrompts = [
        { prompt: 'The king ', temp: 0.7 },
        { prompt: 'What is ', temp: 0.8 },
        { prompt: 'In the beginning ', temp: 0.8 },
        { prompt: 'To be or ', temp: 0.7 },
        { prompt: 'O, ', temp: 0.9 }
    ];
    
    for (const { prompt, temp } of finalPrompts) {
        const text = await model.generate(dataLoader, prompt, 200, temp);
        console.log(`\nPrompt: "${prompt}" (temp=${temp})`);
        console.log(`Generated: "${text}"`);
    }
    
    // Save dataLoader info for web loading
    fs.writeFileSync('./gpt_model_coherent/dataloader.json', JSON.stringify({
        char2idx: dataLoader.char2idx,
        idx2char: dataLoader.idx2char,
        vocabSize: dataLoader.vocabSize,
        seqLength: seqLength
    }));
    
    console.log('\n\nTraining complete! Model saved to ./gpt_model_coherent/');
    console.log('To use in web app, copy the model files to your web server.');
}

// Make dataLoader global for saveForWeb
let dataLoader;

// Run if called directly
if (require.main === module) {
    trainToCoherence().catch(console.error);
}

module.exports = { SimpleGPT, trainToCoherence };