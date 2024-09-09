// Configurable parameters
const NUM_ITERATIONS = 1;
const OUTPUT_LENGTH = 100;
const TRAIN_SPLIT = 0.9;

let vocab;
let trainData;
let valData;
let model;

async function loadData() {
    const response = await fetch('input.txt');
    const text = await response.text();
    return text;
}

function createVocab(text) {
    return Array.from(new Set(text)).sort();
}

function encode(text, vocab) {
    return text.split('').map(char => vocab.indexOf(char));
}

function decode(encoded, vocab) {
    return encoded.map(index => index < vocab.length ? vocab[index] : '?').join('');
}

function createDataset(encoded, windowSize, vocabSize) {
    const x = [];
    const y = [];
    for (let i = 0; i < encoded.length - windowSize; i++) {
        x.push(encoded[i]);
        y.push(encoded[i + 1]);
    }
    return [
        tf.tensor2d(x, [x.length, 1], 'int32'),
        tf.oneHot(tf.tensor1d(y, 'int32'), vocabSize)
    ];
}

function createModel(vocabSize) {
    const model = tf.sequential();
    model.add(tf.layers.embedding({inputDim: vocabSize, outputDim: 16, inputLength: 1}));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({units: vocabSize}));

    const learningRate = 0.001;
    const decay = learningRate / NUM_ITERATIONS;
    const optimizer = tf.train.adam(learningRate, 0.9, 0.999, 1e-7, decay);

    model.compile({
        optimizer: optimizer,
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });
    return model;
}

async function trainModel() {
    const status = document.getElementById('status');
    const progressBar = document.getElementById('trainingProgress');
    status.textContent = 'Status: Preparing for training...';
    progressBar.style.display = 'block';
    progressBar.value = 0;

    console.log('Training data shape:', trainData[0].shape, trainData[1].shape);
    console.log('Validation data shape:', valData[0].shape, valData[1].shape);

    const batchSize = 256;
    const totalBatches = Math.ceil(trainData[0].shape[0] / batchSize);
    console.log('Total batches per epoch:', totalBatches);

    try {
        console.log('Starting model.fit()');
        let currentEpoch = 0;
        const history = await model.fit(trainData[0], trainData[1], {
            epochs: NUM_ITERATIONS,
            validationData: valData,
            batchSize: batchSize,
            shuffle: true,
            callbacks: {
                onEpochBegin: (epoch) => {
                    currentEpoch = epoch;
                    console.log(`Starting epoch ${epoch + 1}`);
                    status.textContent = `Status: Training... Epoch ${epoch + 1}/${NUM_ITERATIONS}`;
                    progressBar.value = (epoch / NUM_ITERATIONS) * 100;
                },
                onBatchEnd: (batch, logs) => {
                    if (batch % 100 === 0) {
                        console.log('Raw batch logs:', logs);
                        const batchWithinEpoch = batch % totalBatches;
                        const statusText = `Status: Training... Epoch ${currentEpoch + 1}/${NUM_ITERATIONS}, Batch ${batchWithinEpoch}/${totalBatches}`;
                        status.textContent = statusText;
                        console.log(statusText + `, Loss: ${logs.loss.toFixed(4)}`);
                        
                        const progress = ((currentEpoch + batchWithinEpoch / totalBatches) / NUM_ITERATIONS) * 100;
                        console.log('Calculated progress:', progress);
                        progressBar.value = progress;
                    }
                },
                onEpochEnd: (epoch, logs) => {
                    console.log(`Epoch ${epoch + 1} completed: loss = ${logs.loss.toFixed(4)}, val_loss = ${logs.val_loss.toFixed(4)}`);
                }
            }
        });

        console.log('Training completed');
        console.log('Final training history:', history.history);
        const finalLoss = history.history.loss[history.history.loss.length - 1];
        const finalValLoss = history.history.val_loss[history.history.val_loss.length - 1];
        console.log(`Final training perplexity: ${calculatePerplexity(finalLoss).toFixed(2)}`);
        console.log(`Final validation perplexity: ${calculatePerplexity(finalValLoss).toFixed(2)}`);

        status.textContent = 'Status: Training complete';
        progressBar.style.display = 'none';
        document.getElementById('generateButton').disabled = false;

    } catch (error) {
        console.error('Error during training:', error);
        status.textContent = 'Status: Training failed';
        progressBar.style.display = 'none';
    }
}

function generateText() {
    let input = tf.tensor2d([[Math.floor(Math.random() * vocab.length)]], [1, 1], 'int32');
    let output = [];

    for (let i = 0; i < OUTPUT_LENGTH; i++) {
        const prediction = model.predict(input);
        const probabilities = tf.softmax(prediction.squeeze());
        const nextCharIndex = tf.multinomial(probabilities, 1).dataSync()[0];
        output.push(nextCharIndex);
        input = tf.tensor2d([[nextCharIndex]], [1, 1], 'int32');
        
        // Clean up tensors to prevent memory leaks
        prediction.dispose();
        probabilities.dispose();
    }

    const generatedText = decode(output, vocab);
    document.getElementById('output').textContent = generatedText;
}

function calculatePerplexity(loss) {
    return Math.exp(loss);
}

async function init() {
    console.log('Initializing...');
    const text = await loadData();
    console.log('Data loaded, length:', text.length);
    
    vocab = createVocab(text);
    console.log('Vocabulary created, size:', vocab.length);
    
    const encoded = encode(text, vocab);
    console.log('Text encoded, length:', encoded.length);

    const splitIndex = Math.floor(encoded.length * TRAIN_SPLIT);
    console.log('Split index:', splitIndex);
    
    trainData = createDataset(encoded.slice(0, splitIndex), 1, vocab.length);
    valData = createDataset(encoded.slice(splitIndex), 1, vocab.length);
    console.log('Datasets created');
    console.log('Train data shapes:', trainData[0].shape, trainData[1].shape);
    console.log('Val data shapes:', valData[0].shape, valData[1].shape);

    model = createModel(vocab.length);
    console.log('Model created');

    document.getElementById('trainButton').addEventListener('click', trainModel);
    document.getElementById('generateButton').addEventListener('click', generateText);
    console.log('Initialization complete');
}

init();