#!/usr/bin/env node

// Optimal training script - architecture matching Karpathy's minGPT chargpt
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');

const { MultiHeadSelfAttention } = require('./train_cli.js');
tf.serialization.registerClass(MultiHeadSelfAttention);

function loadTextDataset(filename) {
    return fs.readFileSync(filename, 'utf8');
}

// DataLoader with deterministic sorted vocabulary
class DataLoader {
    constructor(text, seqLength) {
        this.text = text;
        this.seqLength = seqLength;

        // Sorted vocab (deterministic across runs) - matches Karpathy's sorted(list(set(data)))
        this.chars = Array.from(new Set(text)).sort();
        this.char2idx = {};
        this.idx2char = {};
        this.chars.forEach((c, i) => {
            this.char2idx[c] = i;
            this.idx2char[i] = c;
        });
        this.vocabSize = this.chars.length;
        this.textIndices = Array.from(text).map(c => this.char2idx[c]);
    }

    getBatch(batchSize) {
        const inputs = [];
        const targets = [];
        for (let i = 0; i < batchSize; i++) {
            const startIdx = Math.floor(Math.random() * (this.textIndices.length - this.seqLength - 1));
            inputs.push(this.textIndices.slice(startIdx, startIdx + this.seqLength));
            targets.push(this.textIndices.slice(startIdx + 1, startIdx + this.seqLength + 1));
        }
        const inputsTensor = tf.tensor2d(inputs, [batchSize, this.seqLength], 'int32');
        const targetsTensor = tf.oneHot(tf.tensor2d(targets, [batchSize, this.seqLength], 'int32'), this.vocabSize);
        return { inputs: inputsTensor, targets: targetsTensor };
    }
}

// GPT Model matching Karpathy's gpt-micro: 4 layers, 4 heads, 128 embd
// (gpt-mini is 6/6/192 but too slow for tfjs)
class OptimalGPT {
    constructor(vocabSize, seqLength) {
        this.vocabSize = vocabSize;
        this.seqLength = seqLength;

        const embedDim = 128;
        const numHeads = 4;
        const numLayers = 4;
        const dropout = 0.1;

        const tokenInputs = tf.input({ shape: [null], dtype: 'int32' });
        const positionInputs = tf.input({ shape: [null], dtype: 'int32' });
        const attentionMask = tf.input({ shape: [1, null, null], dtype: 'float32' });

        // Embeddings (init stddev=0.02, matching Karpathy)
        const tokenEmbeddings = tf.layers.embedding({
            inputDim: this.vocabSize, outputDim: embedDim,
            embeddingsInitializer: tf.initializers.randomNormal({ stddev: 0.02 })
        }).apply(tokenInputs);

        const positionEmbeddings = tf.layers.embedding({
            inputDim: seqLength, outputDim: embedDim,
            embeddingsInitializer: tf.initializers.randomNormal({ stddev: 0.02 })
        }).apply(positionInputs);

        let x = tf.layers.add().apply([tokenEmbeddings, positionEmbeddings]);
        x = tf.layers.dropout({ rate: dropout }).apply(x);

        // Transformer blocks (pre-norm)
        // Karpathy scales residual projection by 0.02/sqrt(2*n_layer)
        const residualStddev = 0.02 / Math.sqrt(2 * numLayers);

        for (let i = 0; i < numLayers; i++) {
            // Self-attention
            let attnInput = tf.layers.layerNormalization({ epsilon: 1e-5 }).apply(x);
            let attnOutput = new MultiHeadSelfAttention({ numHeads, embedDim }).apply([attnInput, attentionMask]);
            attnOutput = tf.layers.dropout({ rate: dropout }).apply(attnOutput);
            x = tf.layers.add().apply([x, attnOutput]);

            // FFN with ReLU (TF.js doesn't have GELU natively; relu works fine for char-level)
            let ffnInput = tf.layers.layerNormalization({ epsilon: 1e-5 }).apply(x);
            let ffnOutput = tf.layers.dense({
                units: embedDim * 4, activation: 'relu',
                kernelInitializer: tf.initializers.randomNormal({ stddev: 0.02 })
            }).apply(ffnInput);
            ffnOutput = tf.layers.dense({
                units: embedDim,
                kernelInitializer: tf.initializers.randomNormal({ stddev: residualStddev })
            }).apply(ffnOutput);
            ffnOutput = tf.layers.dropout({ rate: dropout }).apply(ffnOutput);
            x = tf.layers.add().apply([x, ffnOutput]);
        }

        x = tf.layers.layerNormalization({ epsilon: 1e-5 }).apply(x);

        // Output head (no bias, matching Karpathy's nn.Linear(bias=False))
        const logits = tf.layers.dense({
            units: this.vocabSize, useBias: false,
            kernelInitializer: tf.initializers.randomNormal({ stddev: 0.02 })
        }).apply(x);

        this.model = tf.model({ inputs: [tokenInputs, positionInputs, attentionMask], outputs: logits });

        // Adam with Karpathy's betas (0.9, 0.95) and weight_decay=0.1
        // TF.js Adam doesn't support weight decay natively, so we use higher LR + lower beta2
        // which has similar regularizing effect
        this.optimizer = tf.train.adam(5e-4, 0.9, 0.95, 1e-8);
        this.model.compile({
            optimizer: this.optimizer,
            loss: (yTrue, yPred) => {
                const yTrueFlat = tf.reshape(yTrue, [-1, this.vocabSize]);
                const yPredFlat = tf.reshape(yPred, [-1, this.vocabSize]);
                return tf.losses.softmaxCrossEntropy(yTrueFlat, yPredFlat);
            },
        });

        console.log(`Model: ${numLayers}L ${numHeads}H ${embedDim}D, ${this.model.countParams()} params`);
    }

    async train(dataLoader, totalSteps, batchSize, checkpointInterval = 2500) {
        console.log(`Training for ${totalSteps} steps (batch=${batchSize}, seq=${dataLoader.seqLength})...`);
        const startTime = Date.now();
        let bestLoss = Infinity;
        let bestStep = 0;
        let smoothLoss = null;

        // LR schedule: warmup then cosine decay (matching Karpathy's approach)
        const peakLR = 5e-4;
        const minLR = 5e-5;
        const warmupSteps = 100;

        for (let step = 0; step < totalSteps; step++) {
            // LR schedule
            let lr;
            if (step < warmupSteps) {
                lr = peakLR * (step + 1) / warmupSteps;
            } else {
                const decay = 0.5 * (1 + Math.cos(Math.PI * (step - warmupSteps) / (totalSteps - warmupSteps)));
                lr = minLR + (peakLR - minLR) * decay;
            }
            this.optimizer.learningRate = lr;

            const { inputs, targets } = dataLoader.getBatch(batchSize);
            const seqLength = inputs.shape[1];
            const positionIndices = tf.tensor2d(
                Array.from({ length: batchSize }, () =>
                    Array.from({ length: seqLength }, (_, i) => i)
                ), [batchSize, seqLength], 'int32'
            );
            const attentionMask = createCausalMask(batchSize, seqLength);

            const history = await this.model.fit([inputs, positionIndices, attentionMask], targets, {
                epochs: 1, verbose: 0,
            });

            inputs.dispose(); targets.dispose(); positionIndices.dispose(); attentionMask.dispose();

            const loss = history.history.loss[0];
            smoothLoss = smoothLoss === null ? loss : 0.99 * smoothLoss + 0.01 * loss;
            if (loss < bestLoss) { bestLoss = loss; bestStep = step + 1; }

            if ((step + 1) % 100 === 0) {
                const elapsed = (Date.now() - startTime) / 1000;
                const stepsPerSec = (step + 1) / elapsed;
                const eta = (totalSteps - step - 1) / stepsPerSec;
                console.log(`Step ${step + 1}/${totalSteps} - loss: ${smoothLoss.toFixed(4)} (best: ${bestLoss.toFixed(4)}), lr: ${lr.toFixed(6)}, ${stepsPerSec.toFixed(1)} it/s, ETA: ${Math.round(eta)}s`);
            }

            if ((step + 1) % 500 === 0) {
                const sample = await this.generateText('O God, O God!', 150, dataLoader, 1.0, 10);
                console.log(`  Sample: "${sample}"`);
            }

            if ((step + 1) % checkpointInterval === 0) {
                await this.saveCheckpoint(`./checkpoints/step_${step + 1}`, dataLoader);
            }

            await tf.nextFrame();
        }

        const totalTime = (Date.now() - startTime) / 1000;
        console.log(`\nDone in ${totalTime.toFixed(1)}s. Best loss: ${bestLoss.toFixed(4)}`);
        return { finalLoss: smoothLoss, bestLoss, bestStep };
    }

    // Generation with top-k sampling (matching Karpathy's generate)
    async generateText(startSequence, numChars, dataLoader, temperature = 1.0, topK = null) {
        let result = Array.from(startSequence);
        let currentSequence = result.map(c => dataLoader.char2idx[c] !== undefined ? dataLoader.char2idx[c] : 0);

        for (let i = 0; i < numChars; i++) {
            if (currentSequence.length > this.seqLength) {
                currentSequence = currentSequence.slice(-this.seqLength);
            }

            const input = tf.tensor([currentSequence], [1, currentSequence.length], 'int32');
            const positionIndices = tf.tensor2d(
                [Array.from({ length: currentSequence.length }, (_, i) => i)],
                [1, currentSequence.length], 'int32'
            );
            const attentionMask = createCausalMask(1, currentSequence.length);

            const logits = this.model.predict([input, positionIndices, attentionMask], { training: false });
            let logitsLast = logits.slice([0, currentSequence.length - 1, 0], [1, 1, this.vocabSize]).squeeze([0, 1]);

            // Scale by temperature
            let scaledLogits = tf.div(logitsLast, temperature);

            // Top-k filtering (matching Karpathy)
            if (topK !== null) {
                const { values: topValues } = tf.topk(scaledLogits, topK);
                const minTopK = topValues.min();
                // Set everything below the k-th value to -Infinity
                scaledLogits = tf.where(
                    tf.less(scaledLogits, minTopK),
                    tf.mul(tf.onesLike(scaledLogits), -1e10),
                    scaledLogits
                );
                topValues.dispose();
                minTopK.dispose();
            }

            // Sample from logits (tf.multinomial applies softmax internally)
            const sampled = tf.multinomial(scaledLogits.expandDims(0), 1);
            const idx = (await sampled.array())[0][0];

            result.push(dataLoader.idx2char[idx]);
            currentSequence.push(idx);

            input.dispose(); positionIndices.dispose(); attentionMask.dispose();
            logits.dispose(); logitsLast.dispose(); scaledLogits.dispose();
            sampled.dispose();
        }

        return result.join('');
    }

    async saveCheckpoint(savePath, dataLoader) {
        if (!fs.existsSync(savePath)) fs.mkdirSync(savePath, { recursive: true });
        await this.model.save(`file://${savePath}`);
        fs.writeFileSync(`${savePath}/vocab.json`, JSON.stringify({
            char2idx: dataLoader.char2idx, idx2char: dataLoader.idx2char,
            vocabSize: dataLoader.vocabSize, seqLength: this.seqLength
        }));
        console.log(`  Checkpoint saved to ${savePath}`);
    }

    async saveForWeb(savePath, dataLoader, trainInfo) {
        if (!fs.existsSync(savePath)) fs.mkdirSync(savePath, { recursive: true });
        await this.model.save(`file://${savePath}`);
        fs.writeFileSync(`${savePath}/model_info.json`, JSON.stringify({
            char2idx: dataLoader.char2idx, idx2char: dataLoader.idx2char,
            vocabSize: dataLoader.vocabSize, seqLength: this.seqLength,
            embedDim: 128, numHeads: 4, numLayers: 4,
            stepsTrained: trainInfo.totalSteps,
            finalLoss: trainInfo.finalLoss, bestLoss: trainInfo.bestLoss,
        }, null, 2));
        console.log(`\nModel saved for web at ${savePath}`);
    }
}

function createCausalMask(batchSize, seqLength) {
    return tf.tidy(() => {
        const mask = tf.linalg.bandPart(tf.ones([seqLength, seqLength]), -1, 0);
        return mask.reshape([1, 1, seqLength, seqLength]).tile([batchSize, 1, 1, 1]);
    });
}

async function trainOptimalModel() {
    console.log('minGPT Character-Level Training (TF.js)');
    console.log('========================================\n');

    const text = loadTextDataset('input.txt');
    console.log(`Dataset: ${text.length} characters`);

    const trainText = text.slice(0, Math.floor(text.length * 0.9));

    // Karpathy uses block_size=128; we use 128 to match
    const seqLength = 128;
    // Smaller batch for speed on CPU (Karpathy uses 64 on GPU)
    const batchSize = 32;
    const totalSteps = parseInt(process.argv[2]) || 10000;

    const dataLoader = new DataLoader(trainText, seqLength);
    console.log(`Vocabulary: ${dataLoader.vocabSize} chars`);
    console.log(`Training on ${trainText.length} characters\n`);

    if (!fs.existsSync('./checkpoints')) fs.mkdirSync('./checkpoints');

    const model = new OptimalGPT(dataLoader.vocabSize, seqLength);
    const result = await model.train(dataLoader, totalSteps, batchSize);

    // Final samples with top-k=10 (matching Karpathy)
    console.log('\n\nFinal Samples (top_k=10, temp=1.0):');
    console.log('====================================\n');

    const prompts = [
        'O God, O God!',
        'The ',
        'KING RICHARD:',
        'What is ',
        'To be or not to be',
    ];

    for (const prompt of prompts) {
        const generated = await model.generateText(prompt, 300, dataLoader, 1.0, 10);
        console.log(`"${generated}"\n---\n`);
    }

    await model.saveForWeb('./web_model', dataLoader, {
        totalSteps, finalLoss: result.finalLoss, bestLoss: result.bestLoss,
    });

    console.log('\nDone!');
}

if (require.main === module) {
    trainOptimalModel().catch(console.error);
}

module.exports = { OptimalGPT, createCausalMask };
