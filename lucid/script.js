// Lucid Feature Visualization for InceptionV3
// Inspired by the original Lucid library (https://github.com/tensorflow/lucid)

// Global state
let inceptionModel = null;
let isOptimizing = false;
let visualizationHistory = [];
const MODEL_INPUT_RESOLUTION = 299;
const DISPLAY_RESOLUTION = 512;
const DEFAULT_FOURIER_FREQUENCIES = 48;
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
    await tf.ready();
    console.log('TensorFlow.js backend:', tf.getBackend());

    // Load model
    await loadModel();

    // Set up event listeners
    setupEventListeners();
    updateObjectiveModeUI();
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

// ========== Color Decorrelation ==========

// Color correlation matrix from Lucid library (based on ImageNet statistics)
// This transforms from decorrelated space to RGB
const COLOR_CORRELATION_SVD_SQRT = [
    [0.26, 0.09, 0.02],
    [0.27, 0.00, -0.05],
    [0.27, -0.09, 0.03]
];

// Normalize the color correlation matrix
const COLOR_CORRELATION_NORMALIZED = (() => {
    const maxNorm = Math.max(
        ...COLOR_CORRELATION_SVD_SQRT.map(row =>
            Math.sqrt(row.reduce((sum, v) => sum + v * v, 0))
        )
    );
    return COLOR_CORRELATION_SVD_SQRT.map(row => row.map(v => v / maxNorm));
})();

// Apply color decorrelation: transform from decorrelated space to RGB
function toRGB(decorrelatedImage) {
    return tf.tidy(() => {
        const colorMatrix = tf.tensor2d(COLOR_CORRELATION_NORMALIZED);
        const [h, w, c] = decorrelatedImage.shape;
        const reshaped = decorrelatedImage.reshape([h * w, c]);
        const transformed = tf.matMul(reshaped, colorMatrix, false, true);
        return transformed.reshape([h, w, c]);
    });
}

// ========== Fourier Parameterization ==========

function buildSignedFrequencies(count) {
    const frequencies = [0];
    let value = 1;

    while (frequencies.length < count) {
        frequencies.push(value);
        if (frequencies.length < count) {
            frequencies.push(-value);
        }
        value++;
    }

    return frequencies;
}

function createBasisY(size, frequencies) {
    const cosValues = new Float32Array(size * frequencies.length);
    const sinValues = new Float32Array(size * frequencies.length);

    for (let y = 0; y < size; y++) {
        for (let index = 0; index < frequencies.length; index++) {
            const angle = 2 * Math.PI * frequencies[index] * y / size;
            const offset = y * frequencies.length + index;
            cosValues[offset] = Math.cos(angle);
            sinValues[offset] = Math.sin(angle);
        }
    }

    return {
        cos: tf.tensor2d(cosValues, [size, frequencies.length]),
        sin: tf.tensor2d(sinValues, [size, frequencies.length])
    };
}

function createBasisX(size, frequencies) {
    const cosValues = new Float32Array(frequencies.length * size);
    const sinValues = new Float32Array(frequencies.length * size);

    for (let index = 0; index < frequencies.length; index++) {
        for (let x = 0; x < size; x++) {
            const angle = 2 * Math.PI * frequencies[index] * x / size;
            const offset = index * size + x;
            cosValues[offset] = Math.cos(angle);
            sinValues[offset] = Math.sin(angle);
        }
    }

    return {
        cos: tf.tensor2d(cosValues, [frequencies.length, size]),
        sin: tf.tensor2d(sinValues, [frequencies.length, size])
    };
}

function getFourierBasis(size, frequencyCount = DEFAULT_FOURIER_FREQUENCIES, decayPower = 1) {
    const basisKey = `${size}:${frequencyCount}:${decayPower}`;
    if (fourierBasisCache.has(basisKey)) {
        return fourierBasisCache.get(basisKey);
    }

    const yFrequencies = buildSignedFrequencies(Math.min(frequencyCount, size));
    const xFrequencies = buildSignedFrequencies(Math.min(frequencyCount, size));
    const minFrequency = 1 / size;
    const scaleValues = new Float32Array(yFrequencies.length * xFrequencies.length);

    for (let yIndex = 0; yIndex < yFrequencies.length; yIndex++) {
        for (let xIndex = 0; xIndex < xFrequencies.length; xIndex++) {
            const radialFrequency = Math.max(
                Math.hypot(yFrequencies[yIndex] / size, xFrequencies[xIndex] / size),
                minFrequency
            );
            scaleValues[yIndex * xFrequencies.length + xIndex] =
                (1 / Math.pow(radialFrequency, decayPower)) * size;
        }
    }

    const scaleTensor = tf.tensor2d(scaleValues, [yFrequencies.length, xFrequencies.length]);
    const basisY = createBasisY(size, yFrequencies);
    const basisX = createBasisX(size, xFrequencies);
    const basis = {
        basisYCos: basisY.cos,
        basisYSin: basisY.sin,
        basisXCos: basisX.cos,
        basisXSin: basisX.sin,
        scale: scaleTensor,
        normalization: size * size * 4
    };

    fourierBasisCache.set(basisKey, basis);
    return basis;
}

function createFourierParameter(size) {
    const basis = getFourierBasis(size);
    const coefficientShape = [3, basis.scale.shape[0], basis.scale.shape[1]];

    return {
        size,
        basis,
        realVar: tf.variable(tf.randomNormal(coefficientShape, 0, 0.01)),
        imagVar: tf.variable(tf.randomNormal(coefficientShape, 0, 0.01))
    };
}

function disposeFourierParameter(parameterization) {
    parameterization.realVar.dispose();
    parameterization.imagVar.dispose();
}

function renderFourierImage(parameterization) {
    return tf.tidy(() => {
        // WebGL in tfjs does not support complex64 kernels for these matmuls, so
        // express the same Fourier basis using real-valued cosine/sine products.
        const scaledReal = parameterization.realVar.mul(parameterization.basis.scale.expandDims(0));
        const scaledImag = parameterization.imagVar.mul(parameterization.basis.scale.expandDims(0));
        const channels = [];

        for (let channel = 0; channel < 3; channel++) {
            const realCoefficients = scaledReal.slice([channel, 0, 0], [1, -1, -1]).squeeze([0]);
            const imagCoefficients = scaledImag.slice([channel, 0, 0], [1, -1, -1]).squeeze([0]);

            const rowReal = tf.matMul(parameterization.basis.basisYCos, realCoefficients)
                .sub(tf.matMul(parameterization.basis.basisYSin, imagCoefficients));
            const rowImag = tf.matMul(parameterization.basis.basisYCos, imagCoefficients)
                .add(tf.matMul(parameterization.basis.basisYSin, realCoefficients));

            const spatialReal = tf.matMul(rowReal, parameterization.basis.basisXCos)
                .sub(tf.matMul(rowImag, parameterization.basis.basisXSin));

            channels.push(spatialReal);
        }

        const decorrelatedImage = tf
            .stack(channels, -1)
            .div(parameterization.basis.normalization);

        return tf.sigmoid(toRGB(decorrelatedImage));
    });
}

function resizeImage(image, resolution) {
    if (image.shape[0] === resolution && image.shape[1] === resolution) {
        return tf.clone(image);
    }

    return tf.tidy(() => {
        const batched = image.expandDims(0);
        const resized = tf.image.resizeBilinear(batched, [resolution, resolution]);
        return resized.squeeze([0]);
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
        const resized = tf.image.resizeBilinear(batched, [scaledHeight, scaledWidth]);
        return resized.squeeze([0]);
    });
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
        transformed = jitterCrop(transformed, jitterSmall);

        return transformed;
    });
}

function buildOptimizationStages(totalSteps) {
    let resolutions;
    let weights;

    if (totalSteps < 24) {
        resolutions = [128, MODEL_INPUT_RESOLUTION];
        weights = [0.4, 0.6];
    } else if (totalSteps < 48) {
        resolutions = [128, 224, MODEL_INPUT_RESOLUTION];
        weights = [0.25, 0.35, 0.4];
    } else {
        resolutions = [128, 192, 256, MODEL_INPUT_RESOLUTION];
        weights = [0.15, 0.2, 0.25, 0.4];
    }

    const stages = resolutions.map((resolution, index) => ({
        resolution,
        steps: 1,
        weight: weights[index]
    }));

    let remainingSteps = Math.max(0, totalSteps - stages.length);
    const totalWeight = weights.reduce((sum, value) => sum + value, 0);

    stages.forEach((stage, index) => {
        if (index === stages.length - 1 || remainingSteps === 0) {
            return;
        }

        const weightedSteps = Math.floor((remainingSteps * stage.weight) / totalWeight);
        stage.steps += weightedSteps;
        remainingSteps -= weightedSteps;
    });

    stages[stages.length - 1].steps += remainingSteps;
    return stages;
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

function l2Penalty(image) {
    return tf.tidy(() => {
        return tf.mean(tf.square(image));
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
    const stages = buildOptimizationStages(totalSteps);
    let objectiveFn;

    if (config.objectiveMode === 'class') {
        objectiveFn = batchedImage => computeClassObjective(batchedImage, channelIndex);
    } else if (config.objectiveMode === 'channel') {
        objectiveFn = batchedImage => computeChannelObjective(batchedImage, layerInfo.name, channelIndex);
    } else {
        objectiveFn = batchedImage => computeNeuronObjective(batchedImage, layerInfo.name, channelIndex);
    }

    let completedSteps = 0;
    let finalObjective = 0;

    updateProgress(0, 'Initializing Fourier basis...');

    let finalImage = null;
    let displayTensor = null;
    const parameterization = createFourierParameter(MODEL_INPUT_RESOLUTION);
    const optimizer = tf.train.adam(config.learningRate);

    try {
        for (const stage of stages) {
            for (let stageStep = 0; stageStep < stage.steps; stageStep++) {
                const loss = optimizer.minimize(() => tf.tidy(() => {
                    const baseImage = renderFourierImage(parameterization);
                    const stagedImage = resizeImage(baseImage, stage.resolution);
                    const transformedImage = applyStandardTransforms(stagedImage, config.transformStrength);
                    const batchedImage = transformedImage.expandDims(0);
                    const activation = objectiveFn(batchedImage);
                    const l2 = l2Penalty(baseImage).mul(config.l2Weight);
                    const tv = totalVariation(baseImage).mul(config.tvWeight);

                    return activation.sub(l2).sub(tv).neg();
                }), true, [parameterization.realVar, parameterization.imagVar]);

                finalObjective = -((await loss.data())[0]);
                loss.dispose();
                completedSteps++;

                if (completedSteps === 1 || completedSteps % 5 === 0 || completedSteps === totalSteps) {
                    const progress = (completedSteps / totalSteps) * 100;
                    updateProgress(
                        progress,
                        `Stage ${stage.resolution}px • Step ${completedSteps}/${totalSteps} (objective: ${finalObjective.toFixed(3)})`
                    );

                    if (config.showProgress && (completedSteps === 1 || completedSteps % 10 === 0 || completedSteps === totalSteps)) {
                        const previewImage = renderFourierImage(parameterization);
                        await displayImage(previewImage);
                        previewImage.dispose();
                    }

                    await tf.nextFrame();
                }
            }
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
