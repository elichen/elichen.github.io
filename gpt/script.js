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

        this.queryDense = tf.layers.dense({ units: this.embedDim });
        this.keyDense = tf.layers.dense({ units: this.embedDim });
        this.valueDense = tf.layers.dense({ units: this.embedDim });
        this.combineHeadsDense = tf.layers.dense({ units: this.embedDim });
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
        return Object.assign({}, config, {
            numHeads: this.numHeads,
            embedDim: this.embedDim
        });
    }
}

// Register the custom layer
tf.serialization.registerClass(MultiHeadSelfAttention);

class GPT {
    constructor(vocabSize, seqLength, modelInfo = null) {
        this.vocabSize = vocabSize;
        this.seqLength = seqLength;
        this.model = null;
        this.trainData = null;
        this.modelInfo = modelInfo;
        this.totalEpochsTrained = modelInfo ? modelInfo.epochsTrained || 1500 : 0;
        this.currentLoss = modelInfo ? modelInfo.currentLoss || 1.99 : null;
    }

    async buildModel() {
        // Model architecture parameters (matching train_optimal.js)
        const embedDim = 256;
        const numHeads = 8;
        const numLayers = 4;
        const dropout = 0.1;

        // Input layers
        const tokenInputs = tf.input({ shape: [null], dtype: 'int32' });
        const positionInputs = tf.input({ shape: [null], dtype: 'int32' });
        const attentionMask = tf.input({ shape: [1, null, null], dtype: 'float32' });

        // Token embeddings
        const tokenEmbeddingLayer = tf.layers.embedding({ 
            inputDim: this.vocabSize, 
            outputDim: embedDim,
            embeddingsInitializer: tf.initializers.randomNormal({ stddev: 0.02 })
        });
        const tokenEmbeddings = tokenEmbeddingLayer.apply(tokenInputs);

        // Positional embeddings
        const positionEmbeddingLayer = tf.layers.embedding({ 
            inputDim: this.seqLength, 
            outputDim: embedDim,
            embeddingsInitializer: tf.initializers.randomNormal({ stddev: 0.02 })
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

        // Output layer
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

        console.log('Model built with', this.model.countParams(), 'parameters');
    }

    createCausalMask(batchSize, seqLength) {
        return tf.tidy(() => {
            const mask = tf.linalg.bandPart(tf.ones([seqLength, seqLength]), -1, 0);
            const maskReshaped = mask.reshape([1, 1, seqLength, seqLength]);
            return maskReshaped.tile([batchSize, 1, 1, 1]);
        });
    }

    async train(epochs = 100, batchSize = 24, onProgress = null) {
        if (!this.trainData) {
            throw new Error('No training data loaded');
        }

        const { inputs, targets, vocabulary } = this.trainData;
        const numBatches = Math.floor(inputs.length / batchSize);
        let totalLoss = 0;
        let totalAcc = 0;
        let batchCount = 0;

        for (let epoch = 0; epoch < epochs; epoch++) {
            let epochLoss = 0;
            let epochAcc = 0;
            
            // Shuffle indices for each epoch
            const indices = tf.util.createShuffledIndices(inputs.length);
            
            for (let i = 0; i < numBatches; i++) {
                const batchIndices = indices.slice(i * batchSize, (i + 1) * batchSize);
                const batchInputs = batchIndices.map(idx => inputs[idx]);
                const batchTargets = batchIndices.map(idx => targets[idx]);

                const inputTensor = tf.tensor2d(batchInputs, [batchSize, this.seqLength], 'int32');
                const targetTensor = tf.oneHot(tf.tensor2d(batchTargets, [batchSize, this.seqLength], 'int32'), this.vocabSize);
                
                const positionIndices = tf.tensor2d(
                    Array.from({ length: batchSize }, () =>
                        Array.from({ length: this.seqLength }, (_, i) => i)
                    ),
                    [batchSize, this.seqLength],
                    'int32'
                );

                const attentionMask = this.createCausalMask(batchSize, this.seqLength);

                const history = await this.model.fit(
                    [inputTensor, positionIndices, attentionMask], 
                    targetTensor, 
                    {
                        epochs: 1,
                        verbose: 0
                    }
                );

                epochLoss += history.history.loss[0];
                epochAcc += history.history.acc[0] || history.history.accuracy[0] || 0;

                // Clean up tensors
                inputTensor.dispose();
                targetTensor.dispose();
                positionIndices.dispose();
                attentionMask.dispose();

                await tf.nextFrame();
            }

            epochLoss /= numBatches;
            epochAcc /= numBatches;
            totalLoss += epochLoss;
            totalAcc += epochAcc;
            batchCount++;
            
            this.totalEpochsTrained++;
            this.currentLoss = epochLoss;

            if (onProgress) {
                onProgress({
                    epoch: epoch + 1,
                    totalEpochs: epochs,
                    loss: epochLoss,
                    accuracy: epochAcc,
                    totalEpochsTrained: this.totalEpochsTrained
                });
            }

            // Log every 10 epochs
            if ((epoch + 1) % 10 === 0) {
                console.log(`Epoch ${epoch + 1}/${epochs} - Loss: ${epochLoss.toFixed(4)}, Acc: ${epochAcc.toFixed(4)}`);
            }
        }

        const avgLoss = totalLoss / batchCount;
        const avgAcc = totalAcc / batchCount;
        return { loss: avgLoss, accuracy: avgAcc };
    }

    async generateText(startSequence, length = 100, temperature = 0.8) {
        if (!this.model || !this.trainData) {
            throw new Error('Model not ready for generation');
        }

        const { vocabulary } = this.trainData;
        let result = Array.from(startSequence);
        let currentSequence = result.map(c => vocabulary.char2idx[c] || 0);

        for (let i = 0; i < length; i++) {
            // Trim sequence if it's too long
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
            const attentionMask = this.createCausalMask(1, currentSequence.length);

            const logits = this.model.predict([input, positionIndices, attentionMask], { training: false });
            const logitsLast = logits.slice([0, currentSequence.length - 1, 0], [1, 1, this.vocabSize]);
            
            const logits2D = logitsLast.squeeze([1]);
            const scaledLogits = tf.div(logits2D, temperature);
            const sampled = tf.multinomial(scaledLogits, 1);
            const sampledArray = await sampled.array();
            const predictedIdx = sampledArray[0][0];
            const predictedChar = vocabulary.idx2char[predictedIdx];

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

    async loadPretrainedModel() {
        try {
            console.log('Loading pre-trained model...');
            this.model = await tf.loadLayersModel('./web_model/model.json');
            
            // Recompile the model
            this.model.compile({
                optimizer: tf.train.adam(0.0003, 0.9, 0.98, 1e-9),
                loss: (yTrue, yPred) => {
                    const yTrueFlat = tf.reshape(yTrue, [-1, this.vocabSize]);
                    const yPredFlat = tf.reshape(yPred, [-1, this.vocabSize]);
                    return tf.losses.softmaxCrossEntropy(yTrueFlat, yPredFlat);
                },
                metrics: ['accuracy'],
            });
            
            console.log('Pre-trained model loaded successfully');
            return true;
        } catch (error) {
            console.error('Failed to load pre-trained model:', error);
            return false;
        }
    }
}

// Data loading and preprocessing
async function loadData() {
    try {
        // First try to load model info
        const modelInfoResponse = await fetch('./web_model/model_info.json');
        if (modelInfoResponse.ok) {
            const modelInfo = await modelInfoResponse.json();
            console.log('Loaded model info with vocabulary');
            
            // Load training data
            const response = await fetch('input.txt');
            const text = await response.text();
            const trainSize = Math.min(text.length * 0.9, 800000);
            const trainText = text.slice(0, trainSize);
            
            // Use the vocabulary from the model
            const vocabulary = {
                chars: Object.keys(modelInfo.char2idx).sort((a, b) => modelInfo.char2idx[a] - modelInfo.char2idx[b]),
                char2idx: modelInfo.char2idx,
                idx2char: modelInfo.idx2char,
                size: modelInfo.vocabSize
            };
            
            // Convert text to sequences
            const sequences = [];
            for (let i = 0; i < trainText.length - modelInfo.seqLength; i++) {
                const seq = trainText.slice(i, i + modelInfo.seqLength);
                const target = trainText.slice(i + 1, i + modelInfo.seqLength + 1);
                
                const seqIndices = Array.from(seq).map(c => vocabulary.char2idx[c] || 0);
                const targetIndices = Array.from(target).map(c => vocabulary.char2idx[c] || 0);
                
                sequences.push({ input: seqIndices, target: targetIndices });
            }
            
            // Shuffle sequences
            for (let i = sequences.length - 1; i > 0; i--) {
                const j = Math.floor(Math.random() * (i + 1));
                [sequences[i], sequences[j]] = [sequences[j], sequences[i]];
            }
            
            const inputs = sequences.map(s => s.input);
            const targets = sequences.map(s => s.target);
            
            return { 
                inputs, 
                targets, 
                vocabulary, 
                modelInfo 
            };
        }
    } catch (error) {
        console.log('No pre-trained model found, creating new vocabulary');
    }
    
    // Fallback: create new vocabulary
    const response = await fetch('input.txt');
    const text = await response.text();
    const trainSize = Math.min(text.length * 0.9, 800000);
    const trainText = text.slice(0, trainSize);
    
    // Build vocabulary
    const chars = Array.from(new Set(trainText));
    const char2idx = {};
    const idx2char = {};
    chars.forEach((char, idx) => {
        char2idx[char] = idx;
        idx2char[idx] = char;
    });
    
    const vocabulary = {
        chars,
        char2idx,
        idx2char,
        size: chars.length
    };
    
    // Convert text to sequences
    const sequences = [];
    const seqLength = 64;
    for (let i = 0; i < trainText.length - seqLength; i++) {
        const seq = trainText.slice(i, i + seqLength);
        const target = trainText.slice(i + 1, i + seqLength + 1);
        
        const seqIndices = Array.from(seq).map(c => char2idx[c]);
        const targetIndices = Array.from(target).map(c => char2idx[c]);
        
        sequences.push({ input: seqIndices, target: targetIndices });
    }
    
    // Shuffle sequences
    for (let i = sequences.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [sequences[i], sequences[j]] = [sequences[j], sequences[i]];
    }
    
    const inputs = sequences.map(s => s.input);
    const targets = sequences.map(s => s.target);
    
    return { inputs, targets, vocabulary, modelInfo: null };
}

// UI Management
let gptModel = null;
let isTraining = false;

document.addEventListener('DOMContentLoaded', async () => {
    const trainButton = document.getElementById('trainButton');
    const generateButton = document.getElementById('generateButton');
    const status = document.getElementById('status');
    const output = document.getElementById('output');
    const progress = document.getElementById('trainingProgress');
    const modelInfo = document.getElementById('modelInfo');

    // Load data
    status.textContent = 'Status: Loading data...';
    const data = await loadData();
    
    // Initialize model
    const seqLength = data.modelInfo ? data.modelInfo.seqLength : 64;
    gptModel = new GPT(data.vocabulary.size, seqLength, data.modelInfo);
    gptModel.trainData = data;
    
    // Try to load pre-trained model
    const modelLoaded = await gptModel.loadPretrainedModel();
    
    if (modelLoaded) {
        status.textContent = 'Status: Pre-trained model loaded successfully!';
        modelInfo.innerHTML = `
            <div style="background: #f0f0f0; padding: 10px; margin: 10px 0; border-radius: 5px;">
                <strong>Model Status:</strong><br>
                Total epochs trained: ${gptModel.totalEpochsTrained}<br>
                Current loss: ${gptModel.currentLoss ? gptModel.currentLoss.toFixed(4) : 'Unknown'}<br>
                Parameters: ${(gptModel.model.countParams() / 1000000).toFixed(2)}M
            </div>
        `;
        generateButton.disabled = false;
    } else {
        status.textContent = 'Status: Building new model...';
        await gptModel.buildModel();
        status.textContent = 'Status: Ready to train new model';
    }
    
    trainButton.disabled = false;

    trainButton.addEventListener('click', async () => {
        if (isTraining) return;
        
        isTraining = true;
        trainButton.disabled = true;
        generateButton.disabled = true;
        
        status.textContent = 'Status: Training...';
        
        try {
            const result = await gptModel.train(100, 24, (info) => {
                const progressPercent = (info.epoch / info.totalEpochs) * 100;
                progress.value = progressPercent;
                status.textContent = `Status: Training... Epoch ${info.epoch}/${info.totalEpochs} - Loss: ${info.loss.toFixed(4)}, Acc: ${info.accuracy.toFixed(4)}`;
                
                // Update model info
                modelInfo.innerHTML = `
                    <div style="background: #f0f0f0; padding: 10px; margin: 10px 0; border-radius: 5px;">
                        <strong>Model Status:</strong><br>
                        Total epochs trained: ${info.totalEpochsTrained}<br>
                        Current loss: ${info.loss.toFixed(4)}<br>
                        Current accuracy: ${(info.accuracy * 100).toFixed(1)}%<br>
                        Parameters: ${(gptModel.model.countParams() / 1000000).toFixed(2)}M
                    </div>
                `;
            });
            
            status.textContent = `Status: Training complete! Final loss: ${result.loss.toFixed(4)}, Accuracy: ${result.accuracy.toFixed(4)}`;
            generateButton.disabled = false;
        } catch (error) {
            console.error('Training error:', error);
            status.textContent = 'Status: Training failed - ' + error.message;
        } finally {
            isTraining = false;
            trainButton.disabled = false;
            progress.value = 0;
        }
    });

    generateButton.addEventListener('click', async () => {
        generateButton.disabled = true;
        status.textContent = 'Status: Generating text...';
        
        try {
            const prompt = 'The ';
            const temperature = 0.8;
            const generated = await gptModel.generateText(prompt, 200, temperature);
            
            output.innerHTML = `<div class="output-box">${generated}</div>`;
            
            status.textContent = 'Status: Text generation complete!';
        } catch (error) {
            console.error('Generation error:', error);
            status.textContent = 'Status: Generation failed - ' + error.message;
            output.innerHTML = '<p style="color: red;">Error generating text. Please train the model first.</p>';
        } finally {
            generateButton.disabled = false;
        }
    });
});