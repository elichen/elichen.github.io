// Multi-Head Self-Attention Layer
// Uses addWeight() for proper save/load serialization.
// Flattens [B,S,E] to [B*S,E] before matmul with 2D weights to avoid
// TF.js BatchMatMul gradient shape mismatch bug.
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
    }

    build(inputShape) {
        const d = inputShape[0][inputShape[0].length - 1];

        this.queryKernel = this.addWeight('queryKernel', [d, this.embedDim], 'float32', tf.initializers.glorotUniform());
        this.queryBias = this.addWeight('queryBias', [this.embedDim], 'float32', tf.initializers.zeros());
        this.keyKernel = this.addWeight('keyKernel', [d, this.embedDim], 'float32', tf.initializers.glorotUniform());
        this.keyBias = this.addWeight('keyBias', [this.embedDim], 'float32', tf.initializers.zeros());
        this.valueKernel = this.addWeight('valueKernel', [d, this.embedDim], 'float32', tf.initializers.glorotUniform());
        this.valueBias = this.addWeight('valueBias', [this.embedDim], 'float32', tf.initializers.zeros());
        this.combineKernel = this.addWeight('combineKernel', [this.embedDim, this.embedDim], 'float32', tf.initializers.glorotUniform());
        this.combineBias = this.addWeight('combineBias', [this.embedDim], 'float32', tf.initializers.zeros());

        super.build(inputShape);
    }

    call(inputs, kwargs) {
        const [x, mask] = inputs;
        const batchSize = x.shape[0];
        const seqLength = x.shape[1];

        const xFlat = tf.reshape(x, [-1, this.embedDim]);

        let query = tf.reshape(tf.add(tf.matMul(xFlat, this.queryKernel.read()), this.queryBias.read()), [batchSize, seqLength, this.embedDim]);
        let key = tf.reshape(tf.add(tf.matMul(xFlat, this.keyKernel.read()), this.keyBias.read()), [batchSize, seqLength, this.embedDim]);
        let value = tf.reshape(tf.add(tf.matMul(xFlat, this.valueKernel.read()), this.valueBias.read()), [batchSize, seqLength, this.embedDim]);

        query = tf.transpose(tf.reshape(query, [batchSize, seqLength, this.numHeads, this.projectionDim]), [0, 2, 1, 3]);
        key = tf.transpose(tf.reshape(key, [batchSize, seqLength, this.numHeads, this.projectionDim]), [0, 2, 1, 3]);
        value = tf.transpose(tf.reshape(value, [batchSize, seqLength, this.numHeads, this.projectionDim]), [0, 2, 1, 3]);

        const matmulQK = tf.matMul(query, key, false, true);
        const scaledScores = tf.mul(matmulQK, 1 / Math.sqrt(this.projectionDim));

        const maskedScores = tf.where(
            tf.equal(mask, 0),
            tf.mul(tf.onesLike(scaledScores), -1e10),
            scaledScores
        );

        const attnWeights = tf.softmax(maskedScores, -1);
        const attnOutput = tf.matMul(attnWeights, value);

        const transposed = tf.transpose(attnOutput, [0, 2, 1, 3]);
        const concatenated = tf.reshape(transposed, [batchSize, seqLength, this.embedDim]);

        const concatFlat = tf.reshape(concatenated, [-1, this.embedDim]);
        return tf.reshape(tf.add(tf.matMul(concatFlat, this.combineKernel.read()), this.combineBias.read()), [batchSize, seqLength, this.embedDim]);
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

tf.serialization.registerClass(MultiHeadSelfAttention);

// Inference-only GPT wrapper
class GPT {
    constructor(vocabSize, seqLength, vocabulary) {
        this.vocabSize = vocabSize;
        this.seqLength = seqLength;
        this.model = null;
        this.vocabulary = vocabulary;
    }

    createCausalMask(batchSize, seqLength) {
        return tf.tidy(() => {
            const mask = tf.linalg.bandPart(tf.ones([seqLength, seqLength]), -1, 0);
            return mask.reshape([1, 1, seqLength, seqLength]).tile([batchSize, 1, 1, 1]);
        });
    }

    async loadModel() {
        this.model = await tf.loadLayersModel('./web_model/model.json');
        console.log('Model loaded:', this.model.countParams(), 'parameters');
    }

    async generateText(startSequence, length = 200, temperature = 1.0, topK = 10, onToken = null) {
        let result = Array.from(startSequence);
        let currentSequence = result.map(c => this.vocabulary.char2idx[c] !== undefined ? this.vocabulary.char2idx[c] : 0);

        if (onToken) onToken(startSequence);
        console.log(`Generating ${length} chars from "${startSequence}" (temp=${temperature}, topK=${topK})`);
        const t0 = performance.now();

        for (let i = 0; i < length; i++) {
            if (currentSequence.length > this.seqLength) {
                currentSequence = currentSequence.slice(-this.seqLength);
            }

            const sampled = tf.tidy(() => {
                const input = tf.tensor([currentSequence], [1, currentSequence.length], 'int32');
                const positionIndices = tf.tensor2d(
                    [Array.from({ length: currentSequence.length }, (_, i) => i)],
                    [1, currentSequence.length], 'int32'
                );
                const attentionMask = this.createCausalMask(1, currentSequence.length);

                const logits = this.model.predict([input, positionIndices, attentionMask], { training: false });
                const logitsLast = logits.slice([0, currentSequence.length - 1, 0], [1, 1, this.vocabSize]).squeeze([0, 1]);

                let scaledLogits = tf.div(logitsLast, temperature);

                if (topK !== null) {
                    const { values: topValues } = tf.topk(scaledLogits, topK);
                    const minTopK = topValues.min();
                    scaledLogits = tf.where(
                        tf.less(scaledLogits, minTopK),
                        tf.mul(tf.onesLike(scaledLogits), -1e10),
                        scaledLogits
                    );
                }

                return tf.multinomial(scaledLogits.expandDims(0), 1);
            });

            const predictedIdx = (await sampled.array())[0][0];
            sampled.dispose();

            const predictedChar = this.vocabulary.idx2char[predictedIdx];
            result.push(predictedChar);
            currentSequence.push(predictedIdx);

            if (onToken) onToken(result.join(''));

            if (i === 0) console.log(`First token: "${predictedChar}" (${(performance.now() - t0).toFixed(0)}ms)`);
            if ((i + 1) % 50 === 0) console.log(`Token ${i + 1}/${length} (${(performance.now() - t0).toFixed(0)}ms)`);

            await tf.nextFrame();
        }

        console.log(`Done: ${length} tokens in ${(performance.now() - t0).toFixed(0)}ms`);
        return result.join('');
    }
}

// App
let gptModel = null;

document.addEventListener('DOMContentLoaded', async () => {
    const generateButton = document.getElementById('generateButton');
    const promptInput = document.getElementById('promptInput');
    const tokensSlider = document.getElementById('numTokens');
    const tokensValue = document.getElementById('tokensValue');
    const tempSlider = document.getElementById('temperature');
    const tempValue = document.getElementById('tempValue');
    const status = document.getElementById('status');
    const output = document.getElementById('output');

    tokensSlider.addEventListener('input', () => {
        tokensValue.textContent = tokensSlider.value;
    });
    tempSlider.addEventListener('input', () => {
        tempValue.textContent = tempSlider.value;
    });

    status.textContent = 'Loading model...';
    console.log('Fetching model_info.json...');

    try {
        const modelInfoResponse = await fetch('./web_model/model_info.json');
        const modelInfo = await modelInfoResponse.json();
        console.log('Model info loaded:', modelInfo.vocabSize, 'vocab,', modelInfo.seqLength, 'seq');

        const vocabulary = {
            char2idx: modelInfo.char2idx,
            idx2char: modelInfo.idx2char,
        };

        gptModel = new GPT(modelInfo.vocabSize, modelInfo.seqLength, vocabulary);
        console.log('Loading model weights...');
        await gptModel.loadModel();

        status.textContent = `Model loaded (${(gptModel.model.countParams() / 1000).toFixed(0)}K params, loss ${modelInfo.bestLoss.toFixed(2)})`;
        generateButton.disabled = false;
        console.log('Ready for generation');
    } catch (error) {
        console.error('Model load failed:', error);
        status.textContent = 'Failed to load model: ' + error.message;
        return;
    }

    generateButton.addEventListener('click', async () => {
        generateButton.disabled = true;
        output.textContent = '';
        status.textContent = 'Generating...';

        try {
            const prompt = promptInput.value || 'The ';
            const numTokens = parseInt(tokensSlider.value);
            const temperature = parseFloat(tempSlider.value);
            await gptModel.generateText(prompt, numTokens, temperature, 10, (text) => {
                output.textContent = text;
            });
            status.textContent = 'Done.';
        } catch (error) {
            console.error(error);
            status.textContent = 'Generation failed: ' + error.message;
        } finally {
            generateButton.disabled = false;
        }
    });
});
