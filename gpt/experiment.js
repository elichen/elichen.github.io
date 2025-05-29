#!/usr/bin/env node

// Hyperparameter experimentation script
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');

// Import GPT and DataLoader from train_cli.js
const { GPT, DataLoader } = require('./train_cli.js');

// Experiment configurations
const experiments = [
    {
        name: 'baseline',
        epochs: 500,
        batchSize: 16,
        seqLength: 64,
        embedDim: 128,
        numHeads: 4,
        numLayers: 2,
        learningRate: 0.001,
        dropout: 0.05
    },
    {
        name: 'larger_model',
        epochs: 500,
        batchSize: 16,
        seqLength: 64,
        embedDim: 256,
        numHeads: 8,
        numLayers: 4,
        learningRate: 0.0005,
        dropout: 0.1
    },
    {
        name: 'longer_context',
        epochs: 500,
        batchSize: 16,
        seqLength: 128,
        embedDim: 128,
        numHeads: 4,
        numLayers: 2,
        learningRate: 0.001,
        dropout: 0.05
    },
    {
        name: 'smaller_batch',
        epochs: 500,
        batchSize: 8,
        seqLength: 64,
        embedDim: 128,
        numHeads: 4,
        numLayers: 2,
        learningRate: 0.002,
        dropout: 0.05
    },
    {
        name: 'mini_fast',
        epochs: 1000,
        batchSize: 32,
        seqLength: 32,
        embedDim: 64,
        numHeads: 4,
        numLayers: 2,
        learningRate: 0.003,
        dropout: 0.1
    }
];

// Modified GPT class for experiments
class ExperimentalGPT extends GPT {
    constructor(vocabSize, config) {
        // Override the hardcoded parameters
        const originalSeqLength = config.seqLength;
        const originalEmbedDim = config.embedDim;
        const originalNumHeads = config.numHeads;
        const originalNumLayers = config.numLayers;
        const originalDropout = config.dropout;
        
        // Temporarily modify the constructor
        super(vocabSize, originalSeqLength);
        
        // Need to rebuild model with custom parameters
        this.buildCustomModel(vocabSize, config);
    }
    
    buildCustomModel(vocabSize, config) {
        // Clear existing model
        if (this.model) {
            this.model.dispose();
        }
        
        const { seqLength, embedDim, numHeads, numLayers, learningRate, dropout } = config;
        
        // Input layers
        const tokenInputs = tf.input({ shape: [null], dtype: 'int32' });
        const positionInputs = tf.input({ shape: [null], dtype: 'int32' });
        const attentionMask = tf.input({ shape: [1, null, null], dtype: 'float32' });

        // Token embeddings
        const tokenEmbeddingLayer = tf.layers.embedding({ inputDim: vocabSize, outputDim: embedDim });
        const tokenEmbeddings = tokenEmbeddingLayer.apply(tokenInputs);

        // Positional embeddings
        const positionEmbeddingLayer = tf.layers.embedding({ inputDim: seqLength, outputDim: embedDim });
        const positionEmbeddings = positionEmbeddingLayer.apply(positionInputs);

        // Combine embeddings
        let x = tf.layers.add().apply([tokenEmbeddings, positionEmbeddings]);
        x = tf.layers.dropout({ rate: dropout }).apply(x);

        // Import MultiHeadSelfAttention from train_cli
        const MultiHeadSelfAttention = require('./train_cli.js').MultiHeadSelfAttention || this.MultiHeadSelfAttention;

        // Transformer blocks
        for (let i = 0; i < numLayers; i++) {
            // Layer normalization
            let attnInput = tf.layers.layerNormalization({ epsilon: 1e-6 }).apply(x);

            // Multi-head self-attention with attention mask
            const attentionLayer = new MultiHeadSelfAttention({ numHeads, embedDim });
            let attnOutput = attentionLayer.apply([attnInput, attentionMask]);
            attnOutput = tf.layers.dropout({ rate: dropout }).apply(attnOutput);

            // Add & Norm
            x = tf.layers.add().apply([x, attnOutput]);

            // Feed-forward network
            let ffnInput = tf.layers.layerNormalization({ epsilon: 1e-6 }).apply(x);
            let ffnOutput = tf.layers.dense({ units: embedDim * 4, activation: 'elu' }).apply(ffnInput);
            ffnOutput = tf.layers.dense({ units: embedDim }).apply(ffnOutput);
            ffnOutput = tf.layers.dropout({ rate: dropout }).apply(ffnOutput);

            // Add & Norm
            x = tf.layers.add().apply([x, ffnOutput]);
        }

        // Final layer normalization
        x = tf.layers.layerNormalization({ epsilon: 1e-6 }).apply(x);

        // Output layer
        const logits = tf.layers.dense({ units: vocabSize }).apply(x);

        // Define the model
        this.model = tf.model({ inputs: [tokenInputs, positionInputs, attentionMask], outputs: logits });

        this.model.compile({
            optimizer: tf.train.adam(learningRate),
            loss: (yTrue, yPred) => {
                const yTrueFlat = tf.reshape(yTrue, [-1, vocabSize]);
                const yPredFlat = tf.reshape(yPred, [-1, vocabSize]);
                return tf.losses.softmaxCrossEntropy(yTrueFlat, yPredFlat);
            },
            metrics: ['accuracy'],
        });

        const totalParams = this.model.countParams();
        console.log(`Model parameters: ${totalParams}`);
    }
}

// Evaluation function
async function evaluateModel(model, dataLoader, numSamples = 5) {
    const results = {
        samples: [],
        avgLength: 0,
        uniqueChars: new Set(),
        wordLikeRatio: 0
    };
    
    const prompts = ['The ', 'What ', 'To be', 'O ', 'In '];
    
    for (const prompt of prompts.slice(0, numSamples)) {
        const generated = await model.generateText(prompt, 100, dataLoader, 0.8);
        results.samples.push({ prompt, text: generated });
        
        // Analyze generated text
        for (const char of generated) {
            results.uniqueChars.add(char);
        }
    }
    
    // Calculate word-like ratio (sequences of letters followed by space)
    const allGenerated = results.samples.map(s => s.text).join(' ');
    const wordLikePattern = /[a-zA-Z]+\s/g;
    const matches = allGenerated.match(wordLikePattern) || [];
    results.wordLikeRatio = matches.length / (allGenerated.length / 5); // Approximate words per length
    
    return results;
}

// Run experiment
async function runExperiment(config) {
    console.log(`\n${'='.repeat(60)}`);
    console.log(`Running experiment: ${config.name}`);
    console.log(`Config: ${JSON.stringify(config, null, 2)}`);
    console.log(`${'='.repeat(60)}\n`);
    
    const startTime = Date.now();
    const logFile = `./experiments/log_${config.name}_${Date.now()}.txt`;
    
    // Create experiments directory
    if (!fs.existsSync('./experiments')) {
        fs.mkdirSync('./experiments');
    }
    
    // Load data
    const text = fs.readFileSync('input.txt', 'utf8');
    const trainText = text.slice(0, Math.floor(text.length * 0.9));
    const dataLoader = new DataLoader(trainText, config.seqLength);
    
    // Create model
    const model = config.name === 'baseline' ? 
        new GPT(dataLoader.vocabSize, config.seqLength) :
        new ExperimentalGPT(dataLoader.vocabSize, config);
    
    // Training with logging
    const losses = [];
    const accuracies = [];
    
    for (let epoch = 0; epoch < config.epochs; epoch++) {
        const { inputs, targets } = dataLoader.getBatch(config.batchSize);
        const seqLength = inputs.shape[1];
        const positionIndices = tf.tensor2d(
            Array.from({ length: config.batchSize }, () =>
                Array.from({ length: seqLength }, (_, i) => i)
            ),
            [config.batchSize, seqLength],
            'int32'
        );

        const attentionMask = createCausalMask(config.batchSize, seqLength);
        
        const history = await model.model.fit([inputs, positionIndices, attentionMask], targets, {
            epochs: 1,
            verbose: 0,
        });

        inputs.dispose();
        targets.dispose();
        positionIndices.dispose();
        attentionMask.dispose();

        const loss = history.history.loss[0];
        const acc = history.history.acc?.[0] || history.history.accuracy?.[0] || 0;
        losses.push(loss);
        accuracies.push(acc);

        // Log progress
        if ((epoch + 1) % 20 === 0) {
            const evalResults = await evaluateModel(model, dataLoader, 2);
            const logEntry = {
                epoch: epoch + 1,
                loss: loss.toFixed(4),
                accuracy: acc.toFixed(4),
                wordLikeRatio: evalResults.wordLikeRatio.toFixed(2),
                sample: evalResults.samples[0].text.substring(0, 50)
            };
            
            console.log(`Epoch ${epoch + 1}: Loss=${loss.toFixed(4)}, Acc=${acc.toFixed(4)}, WordRatio=${evalResults.wordLikeRatio.toFixed(2)}`);
            console.log(`  Sample: "${evalResults.samples[0].text.substring(0, 50)}..."`);
            
            fs.appendFileSync(logFile, JSON.stringify(logEntry) + '\n');
        }
        
        await tf.nextFrame();
    }
    
    // Final evaluation
    console.log('\nFinal Evaluation:');
    const finalEval = await evaluateModel(model, dataLoader);
    
    // Save results
    const results = {
        config,
        trainTime: (Date.now() - startTime) / 1000,
        finalLoss: losses[losses.length - 1],
        finalAccuracy: accuracies[accuracies.length - 1],
        avgLoss: losses.slice(-50).reduce((a, b) => a + b) / 50,
        evaluation: finalEval
    };
    
    fs.writeFileSync(`./experiments/results_${config.name}_${Date.now()}.json`, JSON.stringify(results, null, 2));
    
    // Save model if it's good
    if (results.finalLoss < 2.0 && results.evaluation.wordLikeRatio > 0.5) {
        const modelPath = `./experiments/model_${config.name}_loss${results.finalLoss.toFixed(2)}`;
        await model.saveModel(modelPath);
        console.log(`Model saved to ${modelPath}`);
    }
    
    return results;
}

// Helper function to create a causal mask
function createCausalMask(batchSize, seqLength) {
    return tf.tidy(() => {
        const mask = tf.linalg.bandPart(tf.ones([seqLength, seqLength]), -1, 0);
        const maskReshaped = mask.reshape([1, 1, seqLength, seqLength]);
        return maskReshaped.tile([batchSize, 1, 1, 1]);
    });
}

// Main execution
async function main() {
    console.log('GPT Hyperparameter Optimization Study');
    console.log('=====================================\n');
    
    const allResults = [];
    
    // Run baseline first
    const baselineResult = await runExperiment(experiments[0]);
    allResults.push(baselineResult);
    
    // Quick test with mini model
    const miniResult = await runExperiment(experiments[4]);
    allResults.push(miniResult);
    
    // Compare results
    console.log('\n\nExperiment Summary:');
    console.log('===================');
    
    allResults.sort((a, b) => a.finalLoss - b.finalLoss);
    
    for (const result of allResults) {
        console.log(`\n${result.config.name}:`);
        console.log(`  Final Loss: ${result.finalLoss.toFixed(4)}`);
        console.log(`  Final Accuracy: ${result.finalAccuracy.toFixed(4)}`);
        console.log(`  Word-like Ratio: ${result.evaluation.wordLikeRatio.toFixed(2)}`);
        console.log(`  Train Time: ${result.trainTime.toFixed(1)}s`);
        console.log(`  Sample: "${result.evaluation.samples[0].text.substring(0, 80)}..."`);
    }
    
    // Save summary
    fs.writeFileSync('./experiments/summary.json', JSON.stringify({
        timestamp: new Date().toISOString(),
        experiments: allResults
    }, null, 2));
}

// Run if called directly
if (require.main === module) {
    main().catch(console.error);
}

module.exports = { ExperimentalGPT, runExperiment };