// Global state
let inceptionModel = null;
let inputImage = null;
let stream = null;

// DOM Elements
const cameraBtn = document.getElementById('cameraBtn');
const uploadBtn = document.getElementById('uploadBtn');
const fileInput = document.getElementById('fileInput');
const cameraModal = document.getElementById('cameraModal');
const video = document.getElementById('video');
const captureBtn = document.getElementById('captureBtn');
const closeCameraBtn = document.getElementById('closeCameraBtn');
const imagePreview = document.getElementById('imagePreview');
const inputCanvas = document.getElementById('inputCanvas');
const iterationsSlider = document.getElementById('iterationsSlider');
const iterationsValue = document.getElementById('iterationsValue');
const dreamBtn = document.getElementById('dreamBtn');
const progressSection = document.getElementById('progressSection');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const resultsSection = document.getElementById('resultsSection');
const originalCanvas = document.getElementById('originalCanvas');
const outputCanvas = document.getElementById('outputCanvas');
const downloadBtn = document.getElementById('downloadBtn');
const resetBtn = document.getElementById('resetBtn');
const layerSelect = document.getElementById('layerSelect');

// Deep Dream configuration inspired by Lucid feature visualization tricks
const INCEPTION_INPUT_SIZE = 299;
const LAYER_PRESETS = {
    multi: [
        { name: 'module_apply_default/InceptionV3/InceptionV3/Mixed_5b/concat', weight: 1.0 },
        { name: 'module_apply_default/InceptionV3/InceptionV3/Mixed_5d/concat', weight: 1.0 }
    ],
    mixed3: [{ name: 'module_apply_default/InceptionV3/InceptionV3/Mixed_5b/concat', weight: 1 }],
    mixed4: [{ name: 'module_apply_default/InceptionV3/InceptionV3/Mixed_5d/concat', weight: 1 }],
    mixed5: [{ name: 'module_apply_default/InceptionV3/InceptionV3/Mixed_6b/concat', weight: 1 }],
    mixed6: [{ name: 'module_apply_default/InceptionV3/InceptionV3/Mixed_6d/concat', weight: 1 }],
    mixed7: [{ name: 'module_apply_default/InceptionV3/InceptionV3/Mixed_7b/concat', weight: 1 }]
};

const DREAM_OPTIONS = {
    stepSize: 0.01,
    jitter: 16,
    tvStrength: 0,
    contentStrength: 0,
    contentBlend: 0,
    smoothing: 0
};

let activeLayerKey = 'multi';
let activeLayers = LAYER_PRESETS[activeLayerKey];

// Event Listeners
cameraBtn.addEventListener('click', openCamera);
uploadBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFileUpload);
captureBtn.addEventListener('click', capturePhoto);
closeCameraBtn.addEventListener('click', closeCamera);
dreamBtn.addEventListener('click', generateDream);
downloadBtn.addEventListener('click', downloadResult);
resetBtn.addEventListener('click', reset);
iterationsSlider.addEventListener('input', (e) => {
    iterationsValue.textContent = e.target.value;
});
if (layerSelect) {
    layerSelect.addEventListener('change', (e) => {
        const key = e.target.value;
        activeLayerKey = key;
        activeLayers = LAYER_PRESETS[key] || LAYER_PRESETS.multi;
    });
}

// Initialize
async function init() {
    console.log('Initializing Deep Dream...');
    await tf.ready();
    console.log('TensorFlow.js backend:', tf.getBackend());
}

// Camera Functions
async function openCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480 }
        });
        video.srcObject = stream;
        cameraModal.classList.remove('hidden');
    } catch (error) {
        alert('Could not access camera: ' + error.message);
    }
}

function closeCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    cameraModal.classList.add('hidden');
}

function capturePhoto() {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);

    canvas.toBlob(blob => {
        displayImage(blob);
        closeCamera();
    });
}

// File Upload
function handleFileUpload(e) {
    const file = e.target.files[0];
    if (file) {
        displayImage(file);
    }
}

// Display Image
function displayImage(blob) {
    const img = new Image();
    img.onload = () => {
        // Use a larger size for better quality (512x512)
        const maxSize = 512;
        const ctx = inputCanvas.getContext('2d');

        // Calculate dimensions maintaining aspect ratio
        let width = img.width;
        let height = img.height;

        if (width > maxSize || height > maxSize) {
            const scale = Math.min(maxSize / width, maxSize / height);
            width = Math.round(width * scale);
            height = Math.round(height * scale);
        }

        inputCanvas.width = width;
        inputCanvas.height = height;
        ctx.drawImage(img, 0, 0, width, height);

        imagePreview.classList.remove('hidden');
        dreamBtn.disabled = false;

        // Store the tensor
        if (inputImage) inputImage.dispose();
        inputImage = tf.browser.fromPixels(inputCanvas);
    };
    img.src = URL.createObjectURL(blob);
}

// Load InceptionV3 Model
async function loadModel() {
    if (inceptionModel) return inceptionModel;

    updateProgress(0, 'Loading InceptionV3 model...');
    // Load InceptionV3 from TensorFlow Hub
    inceptionModel = await tf.loadGraphModel(
        'https://tfhub.dev/google/tfjs-model/imagenet/inception_v3/classification/3/default/1',
        { fromTFHub: true }
    );
    updateProgress(10, 'Model loaded!');
    return inceptionModel;
}

// InceptionV3 helper utilities inspired by Lucid's feature visualization stack
function preprocessForInception(image) {
    return tf.tidy(() => {
        const rank = image.shape.length;
        const batched = rank === 4 ? image : tf.expandDims(image, 0);
        const resized = tf.image.resizeBilinear(batched, [INCEPTION_INPUT_SIZE, INCEPTION_INPUT_SIZE], true);
        const normalized = resized.mul(2).sub(1); // scale [0,1] -> [-1,1]
        return normalized;
    });
}

function computeLayerObjective(batchedImage, layers = activeLayers) {
    return tf.tidy(() => {
        const scores = layers.map(({ name, weight }) =>
            tf.tidy(() => {
                const activation = inferLayer(batchedImage, name);
                // Simple mean activation (matching TensorFlow tutorial)
                const energy = tf.mean(activation);
                return energy.mul(weight);
            })
        );
        return tf.addN(scores);
    });
}

function inferLayer(batchedImage, layerName) {
    // InceptionV3 is a GraphModel, so we need to execute it and get intermediate outputs
    return tf.tidy(() => {
        const result = inceptionModel.execute(batchedImage, layerName);
        return result;
    });
}

function totalVariation(image) {
    return tf.tidy(() => {
        const [height, width, channels] = image.shape;
        if (height < 2 || width < 2) {
            return tf.scalar(0);
        }

        const yDiff = image
            .slice([1, 0, 0], [height - 1, width, channels])
            .sub(image.slice([0, 0, 0], [height - 1, width, channels]));
        const xDiff = image
            .slice([0, 1, 0], [height, width - 1, channels])
            .sub(image.slice([0, 0, 0], [height, width - 1, channels]));

        const yTerm = tf.mean(tf.abs(yDiff));
        const xTerm = tf.mean(tf.abs(xDiff));

        yDiff.dispose();
        xDiff.dispose();

        const result = yTerm.add(xTerm);
        yTerm.dispose();
        xTerm.dispose();
        return result;
    });
}

function rollImage(image, shiftY, shiftX) {
    return tf.tidy(() => {
        const [height, width, channels] = image.shape;
        if (height === undefined || width === undefined || channels === undefined) {
            throw new Error('rollImage expects a rank-3 tensor with known spatial dimensions.');
        }

        const yShift = ((shiftY % height) + height) % height;
        const xShift = ((shiftX % width) + width) % width;

        if (yShift === 0 && xShift === 0) {
            return tf.clone(image);
        }

        let shifted = image;
        if (yShift !== 0) {
            const top = shifted.slice([height - yShift, 0, 0], [yShift, width, channels]);
            const bottom = shifted.slice([0, 0, 0], [height - yShift, width, channels]);
            shifted = tf.concat([top, bottom], 0);
        }
        if (xShift !== 0) {
            const left = shifted.slice([0, width - xShift, 0], [height, xShift, channels]);
            const right = shifted.slice([0, 0, 0], [height, width - xShift, channels]);
            shifted = tf.concat([left, right], 1);
        }
        return shifted;
    });
}

// Deep Dream Core Algorithm (single-scale, Lucid-inspired)
async function deepDream(inputTensor, iterations, options = {}) {
    const config = {
        ...DREAM_OPTIONS,
        ...options
    };
    config.layers = config.layers || activeLayers;

    const baseImage = tf.tidy(() => tf.cast(inputTensor, 'float32').div(255));
    const dreamed = await gradientAscent(baseImage, iterations, config);
    baseImage.dispose();

    updateProgress(85, 'Polishing details...');
    return dreamed;
}

// Gradient Ascent
async function gradientAscent(baseImage, steps, config) {
    const dreamVar = tf.variable(baseImage.clone());

    const computeGrad = tf.grad(image => tf.tidy(() => {
        const prepped = preprocessForInception(image);
        const featureLoss = computeLayerObjective(prepped, config.layers);

        let loss = featureLoss;
        if (config.tvStrength > 0) {
            const tv = totalVariation(image);
            loss = loss.sub(tv.mul(config.tvStrength));
        }
        if (config.contentStrength > 0) {
            const content = tf.mean(tf.square(image.sub(baseImage)));
            loss = loss.sub(content.mul(config.contentStrength));
        }

        return loss;
    }));

    for (let step = 0; step < steps; step++) {
        const shiftX = Math.floor(Math.random() * (config.jitter * 2 + 1)) - config.jitter;
        const shiftY = Math.floor(Math.random() * (config.jitter * 2 + 1)) - config.jitter;

        const grads = tf.tidy(() => {
            const rolled = rollImage(dreamVar, shiftY, shiftX);
            const gradTensor = computeGrad(rolled);
            return rollImage(gradTensor, -shiftY, -shiftX);
        });

        const stepUpdate = tf.tidy(() => {
            // Normalize gradients using standard deviation (TensorFlow tutorial method)
            const mean = tf.mean(grads);
            const variance = tf.mean(tf.square(tf.sub(grads, mean)));
            const std = tf.sqrt(variance).add(1e-8);
            const normalized = grads.div(std).mul(config.stepSize);
            return normalized;
        });

        dreamVar.assign(tf.tidy(() => tf.clipByValue(dreamVar.add(stepUpdate), 0, 1)));

        if (config.smoothing > 0) {
            const smoothed = tf.tidy(() => {
                const expanded = dreamVar.expandDims(0);
                const pooled = tf.avgPool(expanded, [3, 3], [1, 1], 'same');
                return pooled.squeeze();
            });
            dreamVar.assign(tf.tidy(() =>
                dreamVar.mul(1 - config.smoothing).add(smoothed.mul(config.smoothing))
            ));
            smoothed.dispose();
        }

        if (config.contentBlend > 0) {
            dreamVar.assign(tf.tidy(() =>
                dreamVar.mul(1 - config.contentBlend).add(baseImage.mul(config.contentBlend))
            ));
        }

        grads.dispose();
        stepUpdate.dispose();

        const progress = (step + 1) / steps;
        if (step % 4 === 0 || step === steps - 1) {
            updateProgress(
                20 + progress * 60,
                `Optimizing step ${step + 1}/${steps}`
            );
            await tf.nextFrame();
        }
    }

    const result = dreamVar.clone();
    dreamVar.dispose();

    return result;
}

// Generate Dream
async function generateDream() {
    if (!inputImage) return;

    try {
        // Show progress
        progressSection.classList.remove('hidden');
        resultsSection.classList.add('hidden');
        dreamBtn.disabled = true;

        // Load model
        await loadModel();

        // Get settings
        const iterations = parseInt(iterationsSlider.value);
        const intensityFactor = Math.max(0.5, iterations / 80);
        const dynamicOptions = {
            ...DREAM_OPTIONS,
            stepSize: DREAM_OPTIONS.stepSize * intensityFactor,
            contentBlend: Math.max(0.015, DREAM_OPTIONS.contentBlend / (1 + Math.max(0, intensityFactor - 1) * 1.8)),
            contentStrength: DREAM_OPTIONS.contentStrength / intensityFactor,
            smoothing: Math.max(0.05, DREAM_OPTIONS.smoothing / intensityFactor),
            layers: activeLayers.map(layer => ({ ...layer }))
        };

        // Run deep dream
        updateProgress(20, 'Dreaming...');
        const dreamedImage = await deepDream(inputImage, iterations, dynamicOptions);

        // Display results
        updateProgress(95, 'Finalizing...');
        displayResults(inputImage, dreamedImage);

        dreamedImage.dispose();

        updateProgress(100, 'Complete!');

        // Show results
        setTimeout(() => {
            progressSection.classList.add('hidden');
            resultsSection.classList.remove('hidden');
        }, 500);

    } catch (error) {
        console.error('Error generating dream:', error);
        alert('Error generating dream: ' + error.message);
        progressSection.classList.add('hidden');
    } finally {
        dreamBtn.disabled = false;
    }
}

// Display Results
function displayResults(original, dreamed) {
    // Only show the dreamed result at full resolution
    const processed = tf.tidy(() => {
        const clipped = tf.clipByValue(dreamed, 0, 1);
        const scaled = clipped.mul(255);
        return scaled.toInt();
    });

    outputCanvas.width = processed.shape[1];
    outputCanvas.height = processed.shape[0];
    tf.browser.toPixels(processed, outputCanvas);

    processed.dispose();
}

// Update Progress
function updateProgress(percent, message) {
    progressFill.style.width = percent + '%';
    progressText.textContent = message;
}

// Download Result
function downloadResult() {
    const link = document.createElement('a');
    link.download = 'neural-dream.png';
    link.href = outputCanvas.toDataURL();
    link.click();
}

// Reset
function reset() {
    resultsSection.classList.add('hidden');
    imagePreview.classList.remove('hidden');
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Initialize on load
init();
