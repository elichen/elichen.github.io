// Lucid Feature Visualization for InceptionV3
// Inspired by the original Lucid library (https://github.com/tensorflow/lucid)

// Global state
let inceptionModel = null;
let isOptimizing = false;
let visualizationHistory = [];

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
        channels: 1280,
        description: 'Late layer - complex object representations'
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
}

async function loadModel() {
    try {
        updateStatus('Loading InceptionV3 model...', 'loading');

        // Load InceptionV3 from TensorFlow Hub
        inceptionModel = await tf.loadGraphModel(
            'https://tfhub.dev/google/tfjs-model/imagenet/inception_v3/classification/3/default/1',
            { fromTFHub: true }
        );

        // Discover and populate all available layers
        discoverAllLayers();

        updateStatus('Model ready', 'ready');
        enableControls(true);

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

// ========== Image Initialization ==========

// Initialize image with low-frequency biased noise (approximates Fourier initialization)
function initializeImage(size) {
    return tf.tidy(() => {
        // Start with small random noise in decorrelated color space
        const decorrelated = tf.randomNormal([size, size, 3], 0, 0.01);

        // Apply color correlation to get RGB
        const rgb = toRGB(decorrelated);

        // Apply sigmoid to get bounded values, then scale to [-1, 1]
        return tf.sigmoid(rgb).mul(2).sub(1);
    });
}

// Apply Gaussian blur to reduce high frequencies (approximates frequency penalization)
function gaussianBlur(image) {
    return tf.tidy(() => {
        // Simple box blur using average pooling and upsampling
        const [h, w, c] = image.shape;
        const batched = image.expandDims(0);

        // Downsample then upsample for blur effect
        const pooled = tf.avgPool(batched, 2, 1, 'same');

        return pooled.squeeze(0);
    });
}

// ========== Transformation Robustness ==========

function applyRandomTransform(image, strength = 0.5) {
    // Simplified transform - just jitter for now to avoid disposal issues
    const maxJitter = Math.floor(8 * strength);
    const jitterY = Math.floor(Math.random() * (maxJitter * 2 + 1)) - maxJitter;
    const jitterX = Math.floor(Math.random() * (maxJitter * 2 + 1)) - maxJitter;

    const transformed = rollImage(image, jitterY, jitterX);

    return {
        transformed,
        params: { jitterY, jitterX }
    };
}

function reverseTransform(gradient, params) {
    return tf.tidy(() => {
        // Reverse jitter
        return rollImage(gradient, -params.jitterY, -params.jitterX);
    });
}

function rollImage(image, shiftY, shiftX) {
    // For zero shift, just return a clone
    if (shiftY === 0 && shiftX === 0) {
        return tf.clone(image);
    }

    return tf.tidy(() => {
        const [height, width, channels] = image.shape;

        // Ensure shifts are integers
        shiftY = Math.floor(shiftY);
        shiftX = Math.floor(shiftX);

        // Normalize shifts to be within bounds
        const yShift = ((shiftY % height) + height) % height;
        const xShift = ((shiftX % width) + width) % width;

        if (yShift === 0 && xShift === 0) {
            return tf.clone(image);
        }

        // Use slice and concat for rolling instead of gather
        let result = image;

        if (yShift !== 0) {
            const top = result.slice([0, 0, 0], [yShift, width, channels]);
            const bottom = result.slice([yShift, 0, 0], [height - yShift, width, channels]);
            result = tf.concat([bottom, top], 0);
        }

        if (xShift !== 0) {
            const left = result.slice([0, 0, 0], [height, xShift, channels]);
            const right = result.slice([0, xShift, 0], [height, width - xShift, channels]);
            result = tf.concat([right, left], 1);
        }

        // Final dimension check
        if (result.shape[0] !== height || result.shape[1] !== width || result.shape[2] !== channels) {
            throw new Error(`rollImage: output shape [${result.shape}] doesn't match input shape [${height},${width},${channels}]`);
        }

        return result;
    });
}

function rotateImage(image, angleDegrees) {
    return tf.tidy(() => {
        // Simple rotation using affine transform
        // For small angles, we can approximate
        const radians = angleDegrees * Math.PI / 180;
        const cos = Math.cos(radians);
        const sin = Math.sin(radians);

        // This is a simplified rotation - for production, use tf.image.transform
        return image; // Placeholder - implement proper rotation if needed
    });
}

function scaleImage(image, scale) {
    const [height, width, channels] = image.shape;

    // For small scale changes, just return a clone to avoid numerical issues
    if (Math.abs(scale - 1.0) < 0.01) {
        return tf.clone(image);
    }

    const newHeight = Math.round(height * scale);
    const newWidth = Math.round(width * scale);

    // Resize to new dimensions
    const batched = image.expandDims(0);
    const scaled = tf.image.resizeBilinear(batched, [newHeight, newWidth]);
    const squeezed = scaled.squeeze();

    // Clean up intermediate tensors
    batched.dispose();
    scaled.dispose();

    // Ensure we return exact original dimensions
    if (newHeight === height && newWidth === width) {
        return squeezed;
    }

    // Now process squeezed to get back to original size
    let result;

    if (scale > 1) {
        // Crop center to get back to original size
        const yStart = Math.floor((newHeight - height) / 2);
        const xStart = Math.floor((newWidth - width) / 2);

        result = squeezed.slice([yStart, xStart, 0], [height, width, channels]);
        squeezed.dispose();
    } else {
        // Pad to get back to original size
        const yPadTotal = height - newHeight;
        const xPadTotal = width - newWidth;

        const yPadTop = Math.floor(yPadTotal / 2);
        const yPadBottom = yPadTotal - yPadTop;
        const xPadLeft = Math.floor(xPadTotal / 2);
        const xPadRight = xPadTotal - xPadLeft;

        const paddings = [
            [yPadTop, yPadBottom],
            [xPadLeft, xPadRight],
            [0, 0]
        ];

        result = tf.pad(squeezed, paddings);
        squeezed.dispose();
    }

    // Final dimension check and correction
    if (result.shape[0] !== height || result.shape[1] !== width || result.shape[2] !== channels) {
        console.warn(`scaleImage: correcting shape from [${result.shape}] to [${height},${width},${channels}]`);
        const b = result.expandDims(0);
        const corrected = tf.image.resizeBilinear(b, [height, width]).squeeze();
        b.dispose();
        result.dispose();
        return corrected;
    }

    return result;
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

function frequencyPenalty(image, alpha = 1.5) {
    return tf.tidy(() => {
        // Approximate frequency penalty using gradient magnitude
        // High frequencies correspond to rapid changes (high gradients)
        const [height, width, channels] = image.shape;

        // Compute gradients (approximates high-frequency content)
        // We need to compute on the overlapping region only
        const dx = image.slice([0, 1, 0], [height - 1, width - 1, channels])
            .sub(image.slice([0, 0, 0], [height - 1, width - 1, channels]));
        const dy = image.slice([1, 0, 0], [height - 1, width - 1, channels])
            .sub(image.slice([0, 0, 0], [height - 1, width - 1, channels]));

        // Now both dx and dy have shape [height-1, width-1, channels]
        // Gradient magnitude
        const gradMagnitude = tf.sqrt(dx.square().add(dy.square()));

        // Apply higher penalty to larger gradients (high frequencies)
        // This approximates the 1/f^alpha weighting
        const penalty = tf.mean(tf.pow(gradMagnitude, alpha));

        return penalty;
    });
}

// ========== Neuron Objective ==========

function computeNeuronObjective(batchedImage, layerName, channelIndex) {
    return tf.tidy(() => {
        // Get layer activations
        const activations = inceptionModel.execute(batchedImage, layerName);

        // Extract specific channel and compute mean activation
        // activations shape: [batch, height, width, channels]
        const channelActivations = activations.slice(
            [0, 0, 0, channelIndex],
            [1, -1, -1, 1]
        );

        return tf.mean(channelActivations);
    });
}

// ========== Main Optimization Loop ==========

async function optimizeNeuron(layerKey, channelIndex, config) {
    const startTime = Date.now();
    const layerInfo = INCEPTION_LAYERS[layerKey];

    // InceptionV3 expects 299x299 input
    const resolution = 299;
    const totalSteps = config.steps;

    let stepCount = 0;
    let finalLoss = 0;

    updateProgress(0, 'Initializing image...');

    // Initialize image with color-decorrelated noise
    let image = initializeImage(resolution);

    // Create variable for optimization
    const imageVar = tf.variable(image);
    image.dispose();

    updateProgress(0, 'Starting optimization...');

    // Main optimization loop
    for (let step = 0; step < totalSteps; step++) {
        // Apply random jitter for robustness
        const jitterAmount = Math.floor(8 * config.transformStrength);
        const jitterY = Math.floor(Math.random() * (jitterAmount * 2 + 1)) - jitterAmount;
        const jitterX = Math.floor(Math.random() * (jitterAmount * 2 + 1)) - jitterAmount;

        // Compute gradient
        const { value: loss, grads } = tf.variableGrads(() => {
            return tf.tidy(() => {
                // Apply jitter
                const jittered = rollImage(imageVar, jitterY, jitterX);

                // Prepare for model (add batch dimension)
                const batched = jittered.expandDims(0);

                // Compute neuron activation (objective to maximize)
                const activation = computeNeuronObjective(batched, layerInfo.name, channelIndex);

                // Regularization
                const l2 = l2Penalty(imageVar).mul(config.l2Weight);
                const tv = totalVariation(imageVar).mul(config.tvWeight);

                // Return negative because we want to maximize activation
                // (variableGrads minimizes by default)
                return activation.sub(l2).sub(tv).neg();
            });
        });

        // Get the gradient for our image variable
        const grad = grads[imageVar.name];

        if (grad) {
            // Normalize gradient and apply update (gradient ASCENT)
            const updateResult = tf.tidy(() => {
                const gradStd = tf.moments(grad).variance.sqrt().add(1e-8);
                const normalizedGrad = grad.div(gradStd);
                // Subtract because we negated the loss above
                const updated = imageVar.sub(normalizedGrad.mul(config.learningRate));
                return tf.clipByValue(updated, -1, 1);
            });

            imageVar.assign(updateResult);
            updateResult.dispose();
            grad.dispose();
        }

        // Apply periodic blur to suppress high frequencies (every 4 steps)
        if (step % 4 === 0 && step > 0) {
            const blurred = gaussianBlur(imageVar);
            imageVar.assign(blurred);
            blurred.dispose();
        }

        finalLoss = (await loss.data())[0];
        loss.dispose();

        stepCount++;

        // Update progress and show preview
        if (step % 5 === 0 || step === totalSteps - 1) {
            const progress = ((step + 1) / totalSteps) * 100;
            updateProgress(progress, `Step ${stepCount}/${totalSteps} (activation: ${(-finalLoss).toFixed(3)})`);

            if (config.showProgress && step % 10 === 0) {
                await displayImage(imageVar);
            }

            await tf.nextFrame();
        }
    }

    // Final display
    const finalImage = imageVar.clone();

    // Resize to 512x512 for display
    const displaySize = 512;
    const resizedImage = tf.tidy(() => {
        const batched = finalImage.expandDims(0);
        const resized = tf.image.resizeBilinear(batched, [displaySize, displaySize]);
        return resized.squeeze();
    });

    await displayImage(resizedImage);

    const endTime = Date.now();
    const elapsedTime = ((endTime - startTime) / 1000).toFixed(1);

    // Update info panel (negate loss since we negated it for optimization)
    updateVisualizationInfo(layerKey, channelIndex, -finalLoss, elapsedTime);

    // Add to history
    await addToHistory(layerKey, channelIndex, resizedImage);

    // Cleanup
    imageVar.dispose();
    finalImage.dispose();
    resizedImage.dispose();
}

// ========== Display Functions ==========

async function displayImage(imageTensor) {
    const canvas = elements.canvas;
    const ctx = canvas.getContext('2d');

    // Process the image outside of async tidy
    const processedImage = tf.tidy(() => {
        // Convert from [-1, 1] to [0, 1]
        const normalized = imageTensor.add(1).div(2);
        const clipped = tf.clipByValue(normalized, 0, 1);

        // Resize to canvas size if needed
        let displayImage = clipped;
        if (clipped.shape[0] !== 512 || clipped.shape[1] !== 512) {
            const batched = clipped.expandDims(0);
            const resized = tf.image.resizeBilinear(batched, [512, 512]);
            displayImage = resized.squeeze();
        }

        return displayImage;
    });

    // Draw to canvas (async operation outside of tidy)
    await tf.browser.toPixels(processedImage, canvas);

    // Clean up
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

function updateVisualizationInfo(layerKey, channelIndex, loss, time) {
    elements.infoLayer.textContent = layerKey;
    elements.infoChannel.textContent = channelIndex;
    elements.infoLoss.textContent = loss.toFixed(4);
    elements.infoTime.textContent = `${time}s`;

    elements.visualizationInfo.classList.remove('hidden');
}

function enableControls(enabled) {
    elements.layerSelect.disabled = !enabled;
    elements.channelIndex.disabled = !enabled;
    elements.channelSlider.disabled = !enabled;
    elements.visualizeBtn.disabled = !enabled;
}

// ========== Gallery Functions ==========

async function addToHistory(layerKey, channelIndex, imageTensor) {
    // Create thumbnail
    const thumbnailCanvas = document.createElement('canvas');
    thumbnailCanvas.width = 150;
    thumbnailCanvas.height = 150;

    // Process the image for thumbnail
    const thumbnailImage = tf.tidy(() => {
        const normalized = imageTensor.add(1).div(2);
        const clipped = tf.clipByValue(normalized, 0, 1);
        const batched = clipped.expandDims(0);
        const resized = tf.image.resizeBilinear(batched, [150, 150]);
        return resized.squeeze();
    });

    // Draw to canvas (async operation)
    await tf.browser.toPixels(thumbnailImage, thumbnailCanvas);

    // Clean up
    thumbnailImage.dispose();

    // Add to history
    visualizationHistory.unshift({
        layer: layerKey,
        channel: channelIndex,
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
            <img src="${item.image}" alt="Neuron visualization">
            <div class="gallery-info">
                ${item.layer}:${item.channel}
            </div>
        `;

        galleryItem.addEventListener('click', () => {
            // Load this configuration
            elements.layerSelect.value = item.layer;
            updateChannelRange();
            elements.channelIndex.value = item.channel;
            elements.channelSlider.value = item.channel;
        });

        elements.galleryGrid.appendChild(galleryItem);
    });
}

// ========== Event Listeners ==========

function setupEventListeners() {
    // Layer selection change
    elements.layerSelect.addEventListener('change', updateChannelRange);

    // Channel input sync
    elements.channelIndex.addEventListener('input', (e) => {
        elements.channelSlider.value = e.target.value;
    });

    elements.channelSlider.addEventListener('input', (e) => {
        elements.channelIndex.value = e.target.value;
    });

    // Visualize button
    elements.visualizeBtn.addEventListener('click', startVisualization);

    // Download button
    elements.downloadBtn.addEventListener('click', downloadVisualization);

    // Share button
    elements.shareBtn.addEventListener('click', shareSettings);
}

function updateChannelRange() {
    const layerKey = elements.layerSelect.value;
    if (!layerKey) return; // Exit if no layer selected

    const layerInfo = INCEPTION_LAYERS[layerKey];
    if (!layerInfo) return; // Exit if layer info not found

    const maxChannel = layerInfo.channels - 1;

    elements.channelIndex.max = maxChannel;
    elements.channelSlider.max = maxChannel;
    elements.channelMax.textContent = `/ ${layerInfo.channels}`;

    // Clamp current value
    if (elements.channelIndex.value > maxChannel) {
        elements.channelIndex.value = 0;
        elements.channelSlider.value = 0;
    }
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
        showProgress: elements.showProgress.checked
    };

    const layerKey = elements.layerSelect.value;
    const channelIndex = parseInt(elements.channelIndex.value);

    try {
        await optimizeNeuron(layerKey, channelIndex, config);
        elements.resultControls.classList.remove('hidden');
    } catch (error) {
        console.error('Optimization failed:', error);
        alert('Optimization failed: ' + error.message);
    } finally {
        isOptimizing = false;
        elements.visualizeBtn.classList.remove('running');
        elements.btnText.textContent = 'Visualize Neuron';
        elements.progressOverlay.classList.add('hidden');
    }
}

function downloadVisualization() {
    const link = document.createElement('a');
    const layerKey = elements.layerSelect.value;
    const channelIndex = elements.channelIndex.value;
    link.download = `lucid_${layerKey}_channel${channelIndex}.png`;
    link.href = elements.canvas.toDataURL();
    link.click();
}

function shareSettings() {
    const settings = {
        layer: elements.layerSelect.value,
        channel: elements.channelIndex.value,
        steps: elements.steps.value,
        learningRate: elements.learningRate.value,
        l2Weight: elements.l2Weight.value,
        tvWeight: elements.tvWeight.value,
        transformStrength: elements.transformStrength.value
    };

    const settingsText = JSON.stringify(settings, null, 2);
    navigator.clipboard.writeText(settingsText).then(() => {
        alert('Settings copied to clipboard!');
    });
}

// Initialize on page load
window.addEventListener('load', init);