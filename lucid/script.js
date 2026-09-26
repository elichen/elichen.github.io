// Lucid Feature Visualization for InceptionV3
// Inspired by the original Lucid library (https://github.com/tensorflow/lucid)

// Global state
let inceptionModel = null;
let isOptimizing = false;
let visualizationHistory = [];
const MODEL_INPUT_RESOLUTION = 299;
const DISPLAY_RESOLUTION = 512;
const IMAGENET_LABELS_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt';
const IMAGENET_OUTPUT_LABEL = 'ImageNet Classification';
const DEFAULT_IMAGENET_CLASS_COUNT = 1001;
const fourierBasisCache = new Map();
let imagenetLabels = [];
let imagenetClassCount = DEFAULT_IMAGENET_CLASS_COUNT;

// Layer configuration for InceptionV3
const INCEPTION_PREFIX = 'module_apply_default/InceptionV3/InceptionV3/';
let INCEPTION_LAYERS = {
    'Mixed_6a': {
        name: `${INCEPTION_PREFIX}Mixed_6a/concat`,
        channels: 768,
        description: 'Early mixed layer - basic patterns and textures'
    },
    'Mixed_6b': {
        name: `${INCEPTION_PREFIX}Mixed_6b/concat`,
        channels: 768,
        description: 'Mid-level features - parts and components'
    },
    'Mixed_6c': {
        name: `${INCEPTION_PREFIX}Mixed_6c/concat`,
        channels: 768,
        description: 'Complex patterns - recurring motifs'
    },
    'Mixed_6d': {
        name: `${INCEPTION_PREFIX}Mixed_6d/concat`,
        channels: 768,
        description: 'Higher abstractions - object parts'
    },
    'Mixed_6e': {
        name: `${INCEPTION_PREFIX}Mixed_6e/concat`,
        channels: 768,
        description: 'Late layer - abstract object features'
    }
};

// DOM Elements
const elements = {
    // Status
    statusIndicator: document.getElementById('statusIndicator'),
    statusText: document.getElementById('statusText'),
    statusDot: document.querySelector('.status-dot'),

    // Controls
    layerSelect: document.getElementById('layerSelect'),
    targetHeading: document.getElementById('targetHeading'),
    objectiveMode: document.getElementById('objectiveMode'),
    objectiveDescription: document.getElementById('objectiveDescription'),
    targetDescription: document.getElementById('targetDescription'),
    targetDetail: document.getElementById('targetDetail'),
    targetDetailValue: document.getElementById('targetDetailValue'),
    channelIndex: document.getElementById('channelIndex'),
    channelSlider: document.getElementById('channelSlider'),
    channelMax: document.getElementById('channelMax'),
    visualizeBtn: document.getElementById('visualizeBtn'),
    btnText: document.getElementById('btnText'),

    // Settings
    steps: document.getElementById('steps'),
    learningRate: document.getElementById('learningRate'),
    l2Weight: document.getElementById('l2Weight'),
    tvWeight: document.getElementById('tvWeight'),
    transformStrength: document.getElementById('transformStrength'),
    showProgress: document.getElementById('showProgress'),

    // Visualization
    canvas: document.getElementById('visualizationCanvas'),
    progressOverlay: document.getElementById('progressOverlay'),
    progressFill: document.getElementById('progressFill'),
    progressText: document.getElementById('progressText'),

    // Results
    resultControls: document.getElementById('resultControls'),
    downloadBtn: document.getElementById('downloadBtn'),
    shareBtn: document.getElementById('shareBtn'),
    visualizationInfo: document.getElementById('visualizationInfo'),
    infoLayer: document.getElementById('infoLayer'),
    infoMode: document.getElementById('infoMode'),
    infoChannel: document.getElementById('infoChannel'),
    infoLoss: document.getElementById('infoLoss'),
    infoTime: document.getElementById('infoTime'),

    // Gallery
    gallery: document.getElementById('gallery'),
    galleryGrid: document.getElementById('galleryGrid')
};

// ========== Initialization ==========

async function init() {
    console.log('Initializing Lucid Feature Visualization...');

    // Set up TensorFlow.js
    await selectBackend();
    console.log('TensorFlow.js backend:', tf.getBackend());

    // Load model
    await loadModel();

    // Set up event listeners
    setupEventListeners();
    updateObjectiveModeUI();
}

// WebGPU is fastest where available; fall back to WebGL, then CPU.
async function selectBackend() {
    for (const backendName of ['webgpu', 'webgl', 'cpu']) {
        try {
            if (await tf.setBackend(backendName)) {
                await tf.ready();
                return;
            }
        } catch (error) {
            console.warn(`Unable to initialize TensorFlow.js backend "${backendName}".`, error);
        }
    }
    throw new Error('Unable to initialize any TensorFlow.js backend.');
}

async function loadModel() {
    try {
        updateStatus('Loading InceptionV3 model...', 'loading');

        // Load InceptionV3 from TensorFlow Hub
        inceptionModel = await tf.loadGraphModel(
            'https://tfhub.dev/google/tfjs-model/imagenet/inception_v3/classification/3/default/1',
            { fromTFHub: true }
        );

        configureImageNetOutput();

        // Discover and populate all available layers
        discoverAllLayers();
        await loadImageNetLabels();

        updateStatus('Model ready', 'ready');
        enableControls(true);
        updateObjectiveModeUI();

        console.log('InceptionV3 model loaded successfully');
    } catch (error) {
        console.error('Failed to load model:', error);
        updateStatus('Failed to load model', 'error');
    }
}

function discoverAllLayers() {
    // Clear existing layer options
    const layerSelect = elements.layerSelect;
    layerSelect.innerHTML = '';

    // InceptionV3 common layer names and their typical channel counts
    // We'll try to detect and use all available layers
    const allLayers = {};

    // Standard InceptionV3 layers with known channel counts
    const knownLayers = [
        // Early layers
        { pattern: 'Conv2d_1a_3x3', channels: 32, description: 'First convolution' },
        { pattern: 'Conv2d_2a_3x3', channels: 32, description: 'Early features' },
        { pattern: 'Conv2d_2b_3x3', channels: 64, description: 'Edge detection' },
        { pattern: 'Conv2d_3b_1x1', channels: 80, description: 'Basic patterns' },
        { pattern: 'Conv2d_4a_3x3', channels: 192, description: 'Texture features' },

        // Mixed layers (Inception modules)
        { pattern: 'Mixed_5b', channels: 256, description: 'Low-level combinations' },
        { pattern: 'Mixed_5c', channels: 288, description: 'Pattern compositions' },
        { pattern: 'Mixed_5d', channels: 288, description: 'Complex textures' },
        { pattern: 'Mixed_6a', channels: 768, description: 'Mid-level features' },
        { pattern: 'Mixed_6b', channels: 768, description: 'Object parts' },
        { pattern: 'Mixed_6c', channels: 768, description: 'Complex patterns' },
        { pattern: 'Mixed_6d', channels: 768, description: 'Higher abstractions' },
        { pattern: 'Mixed_6e', channels: 768, description: 'Abstract features' },
        { pattern: 'Mixed_7a', channels: 1280, description: 'High-level features' },
        { pattern: 'Mixed_7b', channels: 2048, description: 'Complex objects' },
        { pattern: 'Mixed_7c', channels: 2048, description: 'Final abstractions' },
    ];

    // Try to find these layers in the model - use a single test input
    const testInput = tf.zeros([1, 299, 299, 3]);

    knownLayers.forEach(layerInfo => {
        // Common suffixes for layer outputs
        const suffixes = ['/concat', '/Relu', '/add', ''];

        for (const suffix of suffixes) {
            const layerName = `${INCEPTION_PREFIX}${layerInfo.pattern}${suffix}`;

            // Try to execute with this layer name to see if it exists
            try {
                const output = inceptionModel.execute(testInput, layerName);

                if (output) {
                    // Layer exists! Get its shape
                    const shape = output.shape;
                    const channels = shape[shape.length - 1]; // Last dimension is channels

                    allLayers[layerInfo.pattern] = {
                        name: layerName,
                        channels: channels || layerInfo.channels,
                        description: layerInfo.description
                    };

                    output.dispose();
                    break; // Found this layer, move to next
                }
            } catch (e) {
                // Layer doesn't exist with this suffix, try next
            }
        }
    });

    // Dispose of test input
    testInput.dispose();

    // Update the global INCEPTION_LAYERS
    Object.assign(INCEPTION_LAYERS, allLayers);

    // Populate the select dropdown
    const groups = {
        'Early Layers': ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3', 'Conv2d_3b_1x1', 'Conv2d_4a_3x3'],
        'Mid-Level Mixed': ['Mixed_5b', 'Mixed_5c', 'Mixed_5d', 'Mixed_6a', 'Mixed_6b', 'Mixed_6c', 'Mixed_6d', 'Mixed_6e'],
        'High-Level Mixed': ['Mixed_7a', 'Mixed_7b', 'Mixed_7c']
    };

    // Add optgroups for better organization
    Object.entries(groups).forEach(([groupName, layerNames]) => {
        const optgroup = document.createElement('optgroup');
        optgroup.label = groupName;

        layerNames.forEach(layerName => {
            if (allLayers[layerName]) {
                const option = document.createElement('option');
                option.value = layerName;
                option.textContent = `${layerName} (${allLayers[layerName].channels} channels)`;
                optgroup.appendChild(option);
            }
        });

        if (optgroup.children.length > 0) {
            layerSelect.appendChild(optgroup);
        }
    });

    // Set default selection to Mixed_6b if available
    if (allLayers['Mixed_6b']) {
        layerSelect.value = 'Mixed_6b';
    } else if (layerSelect.options.length > 0) {
        layerSelect.selectedIndex = Math.floor(layerSelect.options.length / 2);
    }

    // Update channel slider based on selection
    if (layerSelect.options.length > 0) {
        updateChannelRange();
    }

    console.log(`Discovered ${Object.keys(allLayers).length} layers in InceptionV3`);
}

function configureImageNetOutput() {
    const outputShape = inceptionModel?.outputs?.[0]?.shape;
    const outputUnits = Array.isArray(outputShape) ? outputShape[outputShape.length - 1] : null;

    if (Number.isInteger(outputUnits) && outputUnits > 0) {
        imagenetClassCount = outputUnits;
    }
}

async function loadImageNetLabels() {
    try {
        const response = await fetch(IMAGENET_LABELS_URL);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const text = await response.text();
        const labels = text
            .split(/\r?\n/)
            .map(label => label.trim())
            .filter(Boolean);

        imagenetLabels = labels.slice(0, imagenetClassCount);
        console.log(`Loaded ${imagenetLabels.length} ImageNet labels`);
    } catch (error) {
        console.warn('Failed to load ImageNet labels:', error);
        imagenetLabels = [];
    }
}

function getImageNetLabel(classIndex) {
    if (classIndex === 0) {
        return imagenetLabels[0] || 'background';
    }

    return imagenetLabels[classIndex] || `ImageNet class ${classIndex}`;
}

function getVisualizationLayerLabel(objectiveMode, layerKey) {
    return objectiveMode === 'class' ? IMAGENET_OUTPUT_LABEL : layerKey;
}

function formatTargetValue(objectiveMode, targetIndex) {
    if (objectiveMode === 'class') {
        return `${targetIndex} (${getImageNetLabel(targetIndex)})`;
    }

    return `${targetIndex}`;
}

function sanitizeForFilename(value) {
    return value
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, '-')
        .replace(/^-+|-+$/g, '')
        .slice(0, 40) || 'target';
}

// ========== Image Parameterization ==========
// Lucid's param.image(fft=True, decorrelate=True): the image is a 1/f-scaled Fourier spectrum in
// a decorrelated color space, squashed into [0, 1] with a sigmoid.

// Square root of ImageNet's color covariance, from lucid.optvis.param.color.
const COLOR_CORRELATION_SVD_SQRT = [
    [0.26, 0.09, 0.02],
    [0.27, 0.00, -0.05],
    [0.27, -0.09, 0.03]
];

// 1x1 conv filter computing rgb = decorrelated @ C^T, with C normalized by its largest column norm as in Lucid.
function colorFilterValues() {
    const C = COLOR_CORRELATION_SVD_SQRT;
    const maxNorm = Math.max(...[0, 1, 2].map(j => Math.hypot(C[0][j], C[1][j], C[2][j])));
    return [0, 1, 2].flatMap(i => [0, 1, 2].map(o => C[o][i] / maxNorm));
}

// DFT matrices for a real 2D FFT done as matrix products (tfjs has no gradient for its FFT ops).
function getFourierBasis(size) {
    if (fourierBasisCache.has(size)) {
        return fourierBasisCache.get(size);
    }

    const freqWidth = Math.floor(size / 2) + 1;

    const cosY = new Float32Array(size * size);
    const sinY = new Float32Array(size * size);
    for (let k = 0; k < size; k++) {
        for (let y = 0; y < size; y++) {
            const angle = 2 * Math.PI * k * y / size;
            cosY[k * size + y] = Math.cos(angle);
            sinY[k * size + y] = Math.sin(angle);
        }
    }

    const cosX = new Float32Array(freqWidth * size);
    const sinX = new Float32Array(freqWidth * size);
    for (let k = 0; k < freqWidth; k++) {
        for (let x = 0; x < size; x++) {
            const angle = 2 * Math.PI * k * x / size;
            cosX[k * size + x] = Math.cos(angle);
            sinX[k * size + x] = Math.sin(angle);
        }
    }

    // Only non-negative x frequencies are stored, so every column but DC (and Nyquist,
    // for even sizes) also stands in for its mirror image when inverting.
    const columnWeights = new Float32Array(freqWidth).map((_, k) =>
        (k === 0 || 2 * k === size ? 1 : 2) / size
    );

    // Lucid's 1/f scaling: low frequencies take bigger steps than high ones, so the optimizer
    // builds coherent structure instead of pixel noise. It also folds in Lucid's divide-by-4
    // and the 1/size of the inverse DFT.
    const spectrumScale = new Float32Array(size * freqWidth);
    for (let y = 0; y < size; y++) {
        const fy = (y <= (size - 1) / 2 ? y : y - size) / size;
        for (let x = 0; x < freqWidth; x++) {
            const freq = Math.max(Math.hypot(x / size, fy), 1 / size);
            spectrumScale[y * freqWidth + x] = 1 / freq / 4;
        }
    }

    const basis = {
        size,
        freqWidth,
        // One copy per color channel for batched matmuls.
        cosY: tf.tidy(() => tf.tensor2d(cosY, [size, size]).expandDims(0).tile([3, 1, 1])),
        sinY: tf.tidy(() => tf.tensor2d(sinY, [size, size]).expandDims(0).tile([3, 1, 1])),
        cosX: tf.tensor2d(cosX, [freqWidth, size]),
        sinX: tf.tensor2d(sinX, [freqWidth, size]),
        columnWeights: tf.tensor1d(columnWeights),
        spectrumScale: tf.tensor2d(spectrumScale, [size, freqWidth])
    };

    fourierBasisCache.set(size, basis);
    return basis;
}

// tf.variable keeps its initial tensor registered, so dispose that tensor once the variable exists.
function randomVariable(shape, stddev) {
    const initial = tf.randomNormal(shape, 0, stddev);
    const variable = tf.variable(initial);
    initial.dispose();
    return variable;
}

function createFourierParameter(size) {
    const basis = getFourierBasis(size);
    const shape = [3, size, basis.freqWidth];

    return {
        basis,
        colorFilter: tf.tensor4d(colorFilterValues(), [1, 1, 3, 3]),
        // Lucid initializes the spectrum with small noise (sd 0.01), which renders as near-uniform gray.
        realVar: randomVariable(shape, 0.01),
        imagVar: randomVariable(shape, 0.01)
    };
}

function disposeFourierParameter(parameterization) {
    parameterization.realVar.dispose();
    parameterization.imagVar.dispose();
    parameterization.colorFilter.dispose();
}

// Spectrum -> [size, size, 3] image in [0, 1]: inverse real FFT, recorrelate colors, sigmoid.
function renderFourierImage(parameterization) {
    return tf.tidy(() => {
        const { basis, colorFilter, realVar, imagVar } = parameterization;
        const { size, freqWidth } = basis;
        const real = realVar.mul(basis.spectrumScale);
        const imag = imagVar.mul(basis.spectrumScale);
        const colReal = tf.matMul(basis.cosY, real).sub(tf.matMul(basis.sinY, imag)).mul(basis.columnWeights);
        const colImag = tf.matMul(basis.cosY, imag).add(tf.matMul(basis.sinY, real)).mul(basis.columnWeights);
        const decorrelated = tf.matMul(colReal.reshape([3 * size, freqWidth]), basis.cosX)
            .sub(tf.matMul(colImag.reshape([3 * size, freqWidth]), basis.sinX))
            .reshape([3, size, size]);
        const rgb = tf.conv2d(decorrelated.transpose([1, 2, 0]).expandDims(0), colorFilter, 1, 'valid');
        return tf.sigmoid(rgb).squeeze([0]);
    });
}

// ========== Transformation Robustness ==========

function padImage(image, amount, fillValue = 0.5) {
    if (amount <= 0) {
        return tf.clone(image);
    }

    return tf.pad(image, [[amount, amount], [amount, amount], [0, 0]], fillValue);
}

function jitterCrop(image, amount) {
    if (amount <= 0) {
        return tf.clone(image);
    }

    const [height, width, channels] = image.shape;
    if (height <= amount || width <= amount) {
        return tf.clone(image);
    }

    const yOffset = Math.floor(Math.random() * (amount + 1));
    const xOffset = Math.floor(Math.random() * (amount + 1));

    return image.slice([yOffset, xOffset, 0], [height - amount, width - amount, channels]);
}

function randomScaleImage(image, strength) {
    if (strength <= 0) {
        return tf.clone(image);
    }

    const lucidScaleChoices = Array.from({ length: 11 }, (_, index) => 1 + (index - 5) / 50);
    const selectedScale = lucidScaleChoices[Math.floor(Math.random() * lucidScaleChoices.length)];
    const scale = 1 + (selectedScale - 1) * strength;

    if (Math.abs(scale - 1) < 1e-3) {
        return tf.clone(image);
    }

    const [height, width] = image.shape;
    const scaledHeight = Math.max(32, Math.round(height * scale));
    const scaledWidth = Math.max(32, Math.round(width * scale));

    return tf.tidy(() => {
        const batched = image.expandDims(0);
        const resized = tf.image.resizeBilinear(batched, [scaledHeight, scaledWidth], false, true);
        return resized.squeeze([0]);
    });
}

// Bilinear rotation about the center, built from gathers so gradients reach the image
// (tfjs has no gradient for tf.image.rotateWithOffset). Samples past the edge clamp to it,
// which is the constant padding added earlier in the pipeline.
function rotateImage(image, degrees) {
    const [height, width, channels] = image.shape;
    const count = height * width;
    const cos = Math.cos(degrees * Math.PI / 180);
    const sin = Math.sin(degrees * Math.PI / 180);
    const centerY = (height - 1) / 2;
    const centerX = (width - 1) / 2;
    const indices = [0, 1, 2, 3].map(() => new Int32Array(count));
    const weights = [0, 1, 2, 3].map(() => new Float32Array(count));

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const dy = y - centerY;
            const dx = x - centerX;
            const sourceY = Math.min(height - 1, Math.max(0, centerY + cos * dy - sin * dx));
            const sourceX = Math.min(width - 1, Math.max(0, centerX + sin * dy + cos * dx));
            const y0 = Math.floor(sourceY);
            const x0 = Math.floor(sourceX);
            const y1 = Math.min(y0 + 1, height - 1);
            const x1 = Math.min(x0 + 1, width - 1);
            const fy = sourceY - y0;
            const fx = sourceX - x0;
            const p = y * width + x;
            indices[0][p] = y0 * width + x0; weights[0][p] = (1 - fy) * (1 - fx);
            indices[1][p] = y0 * width + x1; weights[1][p] = (1 - fy) * fx;
            indices[2][p] = y1 * width + x0; weights[2][p] = fy * (1 - fx);
            indices[3][p] = y1 * width + x1; weights[3][p] = fy * fx;
        }
    }

    return tf.tidy(() => {
        const pixels = image.reshape([count, channels]);
        const corners = indices.map((corner, i) =>
            tf.gather(pixels, tf.tensor1d(corner, 'int32')).mul(tf.tensor2d(weights[i], [count, 1]))
        );
        return tf.addN(corners).reshape([height, width, channels]);
    });
}

function randomRotateImage(image, strength) {
    // Lucid's angles: -10..10 degrees, with extra weight on 0.
    const angles = [...Array.from({ length: 21 }, (_, i) => i - 10), 0, 0, 0, 0, 0];
    const degrees = angles[Math.floor(Math.random() * angles.length)] * strength;
    return degrees === 0 ? tf.clone(image) : rotateImage(image, degrees);
}

function applyStandardTransforms(image, strength) {
    return tf.tidy(() => {
        if (strength <= 0) {
            return tf.clone(image);
        }

        const padAmount = Math.max(1, Math.round(12 * strength));
        const jitterLarge = Math.max(1, Math.round(8 * strength));
        const jitterSmall = Math.max(1, Math.round(4 * strength));

        let transformed = padImage(image, padAmount);
        transformed = jitterCrop(transformed, jitterLarge);
        transformed = randomScaleImage(transformed, strength);
        transformed = randomRotateImage(transformed, strength);
        transformed = jitterCrop(transformed, jitterSmall);

        return transformed;
    });
}

// ========== Regularization ==========

function totalVariation(image) {
    return tf.tidy(() => {
        const [height, width, channels] = image.shape;

        // Calculate vertical differences (between rows)
        // Compare row i with row i+1 for i in [0, height-2]
        const yTop = image.slice([0, 0, 0], [height - 1, width, channels]);
        const yBottom = image.slice([1, 0, 0], [height - 1, width, channels]);
        const yDiff = tf.sub(yBottom, yTop);
        const yVar = tf.mean(tf.abs(yDiff));

        // Calculate horizontal differences (between columns)
        // Compare column j with column j+1 for j in [0, width-2]
        const xLeft = image.slice([0, 0, 0], [height, width - 1, channels]);
        const xRight = image.slice([0, 1, 0], [height, width - 1, channels]);
        const xDiff = tf.sub(xRight, xLeft);
        const xVar = tf.mean(tf.abs(xDiff));

        // Return sum of the two scalar values
        return tf.add(yVar, xVar);
    });
}

// Distance from mid-gray, so the penalty reins in saturated pixels instead of darkening the image.
function l2Penalty(image) {
    return tf.tidy(() => {
        return tf.mean(tf.square(image.sub(0.5)));
    });
}

// ========== Objective Functions ==========

function computeChannelObjective(batchedImage, layerName, channelIndex) {
    return tf.tidy(() => {
        const activations = inceptionModel.execute(batchedImage, layerName);
        const channelActivations = activations.slice(
            [0, 0, 0, channelIndex],
            [1, -1, -1, 1]
        );

        return tf.mean(channelActivations);
    });
}

function computeNeuronObjective(batchedImage, layerName, channelIndex) {
    return tf.tidy(() => {
        const activations = inceptionModel.execute(batchedImage, layerName);
        const [, height, width] = activations.shape;
        const centerY = Math.floor(height / 2);
        const centerX = Math.floor(width / 2);
        const neuronActivation = activations.slice(
            [0, centerY, centerX, channelIndex],
            [1, 1, 1, 1]
        );

        return tf.mean(neuronActivation);
    });
}

function computeClassObjective(batchedImage, classIndex) {
    return tf.tidy(() => {
        const output = inceptionModel.execute(batchedImage);
        const scores = Array.isArray(output) ? output[0] : output;
        const flattenedScores = scores.reshape([scores.shape[0], -1]);
        const classActivation = flattenedScores.slice([0, classIndex], [1, 1]);

        return tf.mean(classActivation);
    });
}

function getObjectiveMeta(mode) {
    if (mode === 'class') {
        return {
            label: 'Class',
            buttonText: 'Visualize Class',
            description: 'Target the model output directly by maximizing a final ImageNet class activation'
        };
    }

    if (mode === 'channel') {
        return {
            label: 'Channel',
            buttonText: 'Visualize Channel',
            description: 'Target the full activation map of the selected channel for broader, faster-emerging features'
        };
    }

    return {
        label: 'Neuron',
        buttonText: 'Visualize Neuron',
        description: 'Target the center neuron of the selected channel for Lucid-style localized features'
    };
}

function updateObjectiveModeUI() {
    const meta = getObjectiveMeta(elements.objectiveMode.value);
    elements.objectiveDescription.textContent = meta.description;

    if (!isOptimizing) {
        elements.btnText.textContent = meta.buttonText;
    }

    updateChannelRange();
}

// ========== Main Optimization Loop ==========

async function optimizeVisualization(layerKey, channelIndex, config) {
    const startTime = Date.now();
    const layerInfo = config.objectiveMode === 'class' ? null : INCEPTION_LAYERS[layerKey];
    const resultLayerLabel = getVisualizationLayerLabel(config.objectiveMode, layerKey);
    const totalSteps = config.steps;
    let objectiveFn;

    if (config.objectiveMode === 'class') {
        objectiveFn = batchedImage => computeClassObjective(batchedImage, channelIndex);
    } else if (config.objectiveMode === 'channel') {
        objectiveFn = batchedImage => computeChannelObjective(batchedImage, layerInfo.name, channelIndex);
    } else {
        objectiveFn = batchedImage => computeNeuronObjective(batchedImage, layerInfo.name, channelIndex);
    }

    let finalObjective = 0;

    updateProgress(0, 'Initializing Fourier basis...');

    let finalImage = null;
    let displayTensor = null;
    const parameterization = createFourierParameter(MODEL_INPUT_RESOLUTION);
    const optimizer = tf.train.adam(config.learningRate);

    try {
        for (let step = 1; step <= totalSteps; step++) {
            const loss = optimizer.minimize(() => tf.tidy(() => {
                const image = renderFourierImage(parameterization);
                const transformedImage = applyStandardTransforms(image, config.transformStrength);
                const activation = objectiveFn(transformedImage.expandDims(0));
                const l2 = l2Penalty(image).mul(config.l2Weight);
                const tv = totalVariation(image).mul(config.tvWeight);

                return activation.sub(l2).sub(tv).neg();
            }), true, [parameterization.realVar, parameterization.imagVar]);

            // Reading the loss stalls the GPU pipeline, so only do it when reporting progress.
            if (step === 1 || step % 5 === 0 || step === totalSteps) {
                finalObjective = -((await loss.data())[0]);
                updateProgress(
                    (step / totalSteps) * 100,
                    `Step ${step}/${totalSteps} (objective: ${finalObjective.toFixed(3)})`
                );

                if (config.showProgress && (step === 1 || step % 10 === 0 || step === totalSteps)) {
                    const previewImage = renderFourierImage(parameterization);
                    await displayImage(previewImage);
                    previewImage.dispose();
                }

                await tf.nextFrame();
            }
            loss.dispose();
        }

        finalImage = renderFourierImage(parameterization);
        displayTensor = tf.tidy(() => {
            const batched = finalImage.expandDims(0);
            const resized = tf.image.resizeBilinear(batched, [DISPLAY_RESOLUTION, DISPLAY_RESOLUTION]);
            return resized.squeeze([0]);
        });

        await displayImage(displayTensor);

        const endTime = Date.now();
        const elapsedTime = ((endTime - startTime) / 1000).toFixed(1);

        updateVisualizationInfo(config.objectiveMode, resultLayerLabel, channelIndex, finalObjective, elapsedTime);

        await addToHistory(config.objectiveMode, layerKey, channelIndex, displayTensor);
    } finally {
        disposeFourierParameter(parameterization);
        optimizer.dispose();
        if (finalImage) finalImage.dispose();
        if (displayTensor) displayTensor.dispose();
    }
}

// ========== Display Functions ==========

async function displayImage(imageTensor) {
    const canvas = elements.canvas;
    const processedImage = tf.tidy(() => {
        let displayImageTensor = tf.clipByValue(imageTensor, 0, 1);

        if (displayImageTensor.shape[0] !== DISPLAY_RESOLUTION || displayImageTensor.shape[1] !== DISPLAY_RESOLUTION) {
            const batched = displayImageTensor.expandDims(0);
            const resized = tf.image.resizeBilinear(batched, [DISPLAY_RESOLUTION, DISPLAY_RESOLUTION]);
            displayImageTensor = resized.squeeze([0]);
        }

        return displayImageTensor;
    });

    await tf.browser.toPixels(processedImage, canvas);
    processedImage.dispose();
}

function updateProgress(percent, message) {
    elements.progressFill.style.width = `${percent}%`;
    elements.progressText.textContent = message;
}

function updateStatus(message, status) {
    elements.statusText.textContent = message;

    if (status === 'ready') {
        elements.statusDot.classList.add('ready');
    } else {
        elements.statusDot.classList.remove('ready');
    }
}

function updateVisualizationInfo(objectiveMode, layerKey, channelIndex, loss, time) {
    elements.infoLayer.textContent = layerKey;
    elements.infoMode.textContent = getObjectiveMeta(objectiveMode).label;
    elements.infoChannel.textContent = formatTargetValue(objectiveMode, channelIndex);
    elements.infoLoss.textContent = loss.toFixed(4);
    elements.infoTime.textContent = `${time}s`;

    elements.visualizationInfo.classList.remove('hidden');
}

function enableControls(enabled) {
    elements.layerSelect.disabled = !enabled || elements.objectiveMode.value === 'class';
    elements.objectiveMode.disabled = !enabled;
    elements.channelIndex.disabled = !enabled;
    elements.channelSlider.disabled = !enabled;
    elements.visualizeBtn.disabled = !enabled;
}

// ========== Gallery Functions ==========

async function addToHistory(objectiveMode, layerKey, channelIndex, imageTensor) {
    // Create thumbnail
    const thumbnailCanvas = document.createElement('canvas');
    thumbnailCanvas.width = 150;
    thumbnailCanvas.height = 150;

    const thumbnailImage = tf.tidy(() => {
        const clipped = tf.clipByValue(imageTensor, 0, 1);
        const batched = clipped.expandDims(0);
        const resized = tf.image.resizeBilinear(batched, [150, 150]);
        return resized.squeeze([0]);
    });

    // Draw to canvas (async operation)
    await tf.browser.toPixels(thumbnailImage, thumbnailCanvas);

    // Clean up
    thumbnailImage.dispose();

    // Add to history
    visualizationHistory.unshift({
        mode: objectiveMode,
        layer: getVisualizationLayerLabel(objectiveMode, layerKey),
        sourceLayer: layerKey,
        channel: channelIndex,
        targetLabel: formatTargetValue(objectiveMode, channelIndex),
        image: thumbnailCanvas.toDataURL(),
        timestamp: Date.now()
    });

    // Keep only last 12 visualizations
    if (visualizationHistory.length > 12) {
        visualizationHistory = visualizationHistory.slice(0, 12);
    }

    updateGallery();
}

function updateGallery() {
    if (visualizationHistory.length === 0) {
        elements.gallery.classList.add('hidden');
        return;
    }

    elements.gallery.classList.remove('hidden');
    elements.galleryGrid.innerHTML = '';

    visualizationHistory.forEach(item => {
        const galleryItem = document.createElement('div');
        galleryItem.className = 'gallery-item';
        galleryItem.innerHTML = `
            <img src="${item.image}" alt="Feature visualization">
            <div class="gallery-info">
                ${item.mode} • ${item.layer}:${item.targetLabel || item.channel}
            </div>
        `;

        galleryItem.addEventListener('click', () => {
            // Load this configuration
            elements.objectiveMode.value = item.mode || 'neuron';
            updateObjectiveModeUI();

            if (item.mode !== 'class' && item.sourceLayer) {
                elements.layerSelect.value = item.sourceLayer;
            }

            updateChannelRange();
            elements.channelIndex.value = item.channel;
            elements.channelSlider.value = item.channel;
            updateTargetDetail();
        });

        elements.galleryGrid.appendChild(galleryItem);
    });
}

// ========== Event Listeners ==========

function setupEventListeners() {
    // Layer selection change
    elements.layerSelect.addEventListener('change', updateChannelRange);
    elements.objectiveMode.addEventListener('change', updateObjectiveModeUI);

    // Channel input sync
    elements.channelIndex.addEventListener('input', (e) => {
        elements.channelSlider.value = e.target.value;
        updateTargetDetail();
    });

    elements.channelSlider.addEventListener('input', (e) => {
        elements.channelIndex.value = e.target.value;
        updateTargetDetail();
    });

    // Visualize button
    elements.visualizeBtn.addEventListener('click', startVisualization);

    // Download button
    elements.downloadBtn.addEventListener('click', downloadVisualization);

    // Share button
    elements.shareBtn.addEventListener('click', shareSettings);
}

function updateChannelRange() {
    const objectiveMode = elements.objectiveMode.value;
    const isClassMode = objectiveMode === 'class';
    const maxTarget = isClassMode
        ? imagenetClassCount - 1
        : (INCEPTION_LAYERS[elements.layerSelect.value]?.channels || 0) - 1;

    elements.layerSelect.disabled = !inceptionModel || isClassMode;
    elements.targetHeading.textContent = isClassMode ? 'ImageNet Class' : 'Feature Channel';
    elements.targetDescription.textContent = isClassMode
        ? 'Choose a final ImageNet class index to maximize at the model output'
        : 'Choose a channel; the objective setting decides whether to target its center neuron or full activation map';

    if (maxTarget < 0) {
        elements.channelIndex.max = 0;
        elements.channelSlider.max = 0;
        elements.channelMax.textContent = '/ -';
        updateTargetDetail();
        return;
    }

    elements.channelIndex.max = maxTarget;
    elements.channelSlider.max = maxTarget;
    elements.channelMax.textContent = isClassMode
        ? `/ ${imagenetClassCount}`
        : `/ ${maxTarget + 1}`;

    // Clamp current value
    if (Number.parseInt(elements.channelIndex.value, 10) > maxTarget) {
        elements.channelIndex.value = 0;
        elements.channelSlider.value = 0;
    }

    updateTargetDetail();
}

function updateTargetDetail() {
    if (elements.objectiveMode.value === 'class') {
        const classIndex = Number.parseInt(elements.channelIndex.value, 10) || 0;
        elements.targetDetailValue.textContent = getImageNetLabel(classIndex);
        elements.targetDetail.hidden = false;
        return;
    }

    elements.targetDetailValue.textContent = '';
    elements.targetDetail.hidden = true;
}

async function startVisualization() {
    if (isOptimizing) return;

    isOptimizing = true;
    elements.visualizeBtn.classList.add('running');
    elements.btnText.textContent = 'Optimizing...';
    elements.progressOverlay.classList.remove('hidden');
    elements.resultControls.classList.add('hidden');

    const config = {
        steps: parseInt(elements.steps.value),
        learningRate: parseFloat(elements.learningRate.value),
        l2Weight: parseFloat(elements.l2Weight.value),
        tvWeight: parseFloat(elements.tvWeight.value),
        transformStrength: parseFloat(elements.transformStrength.value),
        showProgress: elements.showProgress.checked,
        objectiveMode: elements.objectiveMode.value
    };

    const layerKey = elements.layerSelect.value;
    const channelIndex = parseInt(elements.channelIndex.value);

    try {
        await optimizeVisualization(layerKey, channelIndex, config);
        elements.resultControls.classList.remove('hidden');
    } catch (error) {
        console.error('Optimization failed:', error);
        alert('Optimization failed: ' + error.message);
    } finally {
        isOptimizing = false;
        elements.visualizeBtn.classList.remove('running');
        elements.btnText.textContent = getObjectiveMeta(elements.objectiveMode.value).buttonText;
        elements.progressOverlay.classList.add('hidden');
    }
}

function downloadVisualization() {
    const link = document.createElement('a');
    const layerKey = elements.layerSelect.value;
    const channelIndex = elements.channelIndex.value;
    const objectiveMode = elements.objectiveMode.value;
    if (objectiveMode === 'class') {
        const classSlug = sanitizeForFilename(getImageNetLabel(Number.parseInt(channelIndex, 10) || 0));
        link.download = `lucid_class_imagenet_${channelIndex}_${classSlug}.png`;
    } else {
        link.download = `lucid_${objectiveMode}_${layerKey}_channel${channelIndex}.png`;
    }
    link.href = elements.canvas.toDataURL();
    link.click();
}

function shareSettings() {
    const settings = {
        objectiveMode: elements.objectiveMode.value,
        layer: elements.layerSelect.value,
        channel: elements.channelIndex.value,
        steps: elements.steps.value,
        learningRate: elements.learningRate.value,
        l2Weight: elements.l2Weight.value,
        tvWeight: elements.tvWeight.value,
        transformStrength: elements.transformStrength.value
    };

    if (settings.objectiveMode === 'class') {
        settings.classLabel = getImageNetLabel(Number.parseInt(settings.channel, 10) || 0);
    }

    const settingsText = JSON.stringify(settings, null, 2);
    navigator.clipboard.writeText(settingsText).then(() => {
        alert('Settings copied to clipboard!');
    });
}

// Initialize on page load
window.addEventListener('load', init);
