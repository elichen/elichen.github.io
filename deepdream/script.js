// Global state
let inceptionModel = null;
let inputImage = null;
let stream = null;

// DOM Elements
const DEFAULT_IMAGE_PATH = 'doggy.png';
const QUERY_PARAMS = new URLSearchParams(window.location.search);
const FULL_PROFILE = {
    maxImageSize: 512,
    octaves: [-2, -1, 0, 1, 2],
    defaultSteps: 100,
    sliderMax: 150
};
const FAST_PROFILE = {
    maxImageSize: 320,
    octaves: [-1, 0, 1],
    defaultSteps: 20,
    sliderMax: 60
};
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
const outputCanvas = document.getElementById('outputCanvas');
const downloadBtn = document.getElementById('downloadBtn');
const resetBtn = document.getElementById('resetBtn');
const layerSelect = document.getElementById('layerSelect');

// Deep Dream configuration
const INCEPTION_PREFIX = 'module_apply_default/InceptionV3/InceptionV3/';
const RELU_SUFFIX = '/Branch_0/Conv2d_0a_1x1/Relu';
const LAYER_PRESETS = {
    multi: [
        { name: `${INCEPTION_PREFIX}Mixed_6a/concat`, weight: 1.0 },
        { name: `${INCEPTION_PREFIX}Mixed_6c${RELU_SUFFIX}`, weight: 1.0 }
    ],
    mixed3: [{ name: `${INCEPTION_PREFIX}Mixed_6a/concat`, weight: 1 }],
    mixed4: [{ name: `${INCEPTION_PREFIX}Mixed_6b${RELU_SUFFIX}`, weight: 1 }],
    mixed5: [{ name: `${INCEPTION_PREFIX}Mixed_6c${RELU_SUFFIX}`, weight: 1 }],
    mixed6: [{ name: `${INCEPTION_PREFIX}Mixed_6d${RELU_SUFFIX}`, weight: 1 }],
    mixed7: [{ name: `${INCEPTION_PREFIX}Mixed_6e${RELU_SUFFIX}`, weight: 1 }]
};

const DREAM_OPTIONS = {
    stepSize: 0.01,
    jitter: 0,
    tvStrength: 0,
    contentStrength: 0,
    contentBlend: 0,
    smoothing: 0
};

let activeLayers = LAYER_PRESETS.multi;
let runtimeProfile = { ...FULL_PROFILE };

function webglTensorSelfTest() {
    return tf.tidy(() => {
        const canvas = document.createElement('canvas');
        canvas.width = 1;
        canvas.height = 1;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = 'rgb(160, 165, 99)';
        ctx.fillRect(0, 0, 1, 1);

        const pixel = Array.from(tf.browser.fromPixels(canvas).dataSync());
        return pixel.length === 3
            && pixel[0] === 160
            && pixel[1] === 165
            && pixel[2] === 99;
    });
}

async function ensureStableBackend() {
    const requestedBackend = QUERY_PARAMS.get('backend');
    const supportedBackends = new Set(['webgl', 'cpu']);
    const backendCandidates = requestedBackend && supportedBackends.has(requestedBackend)
        ? [requestedBackend]
        : ['webgl', 'cpu'];

    let backendReady = false;
    for (const backendName of backendCandidates) {
        try {
            const switched = await tf.setBackend(backendName);
            if (!switched) {
                continue;
            }
            await tf.ready();
            backendReady = true;
            break;
        } catch (error) {
            console.warn(`Unable to initialize TensorFlow.js backend "${backendName}".`, error);
        }
    }

    if (!backendReady) {
        throw new Error(`Unable to initialize any TensorFlow.js backend (${backendCandidates.join(', ')}).`);
    }

    if (tf.getBackend() !== 'webgl') {
        return;
    }

    if (!webglTensorSelfTest()) {
        console.warn('TensorFlow.js WebGL tensor self-test failed; falling back to CPU backend.');
        const switched = await tf.setBackend('cpu');
        if (!switched) {
            throw new Error('WebGL backend failed self-test and CPU fallback was unavailable.');
        }
        await tf.ready();
    }
}

function configureRuntimeProfile() {
    const requestedProfile = QUERY_PARAMS.get('profile');

    if (requestedProfile === 'full') {
        runtimeProfile = { ...FULL_PROFILE };
    } else if (requestedProfile === 'fast') {
        runtimeProfile = { ...FAST_PROFILE };
    } else {
        runtimeProfile = tf.getBackend() === 'cpu'
            ? { ...FAST_PROFILE }
            : { ...FULL_PROFILE };
    }

    iterationsSlider.max = String(runtimeProfile.sliderMax);
    iterationsSlider.value = String(runtimeProfile.defaultSteps);
    iterationsValue.textContent = iterationsSlider.value;
}

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
        activeLayers = LAYER_PRESETS[key] || LAYER_PRESETS.multi;
    });
}

// Initialize
async function init() {
    console.log('Initializing Deep Dream...');
    await tf.ready();
    await ensureStableBackend();
    configureRuntimeProfile();
    console.log('TensorFlow.js backend:', tf.getBackend());
    await loadDefaultImage();
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
    if (!video.videoWidth || !video.videoHeight) {
        alert('Camera is still warming up. Try again in a moment.');
        return;
    }

    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);

    canvas.toBlob(blob => {
        if (!blob) {
            alert('Could not capture the current camera frame.');
            return;
        }
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
function displayImage(source) {
    const img = new Image();
    let objectURL = null;
    img.onload = () => {
        const maxSize = runtimeProfile.maxImageSize;
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
        ctx.clearRect(0, 0, width, height);
        ctx.drawImage(img, 0, 0, width, height);

        imagePreview.classList.remove('hidden');
        dreamBtn.disabled = false;

        // Store the tensor
        if (inputImage) inputImage.dispose();
        inputImage = tf.browser.fromPixels(inputCanvas);

        if (objectURL) {
            URL.revokeObjectURL(objectURL);
        }
    };
    if (source instanceof Blob) {
        objectURL = URL.createObjectURL(source);
        img.src = objectURL;
    } else if (typeof source === 'string') {
        img.src = source;
    } else {
        throw new Error('Unsupported image source for displayImage');
    }
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

// TF Hub's InceptionV3 graph performs the DeepDream x * 2 - 1 preprocessing internally.
function preprocessForInception(image) {
    return tf.tidy(() => {
        const rank = image.shape.length;
        // Only add a batch dimension; keep the image in [0, 1].
        return rank === 4 ? image : tf.expandDims(image, 0);
    });
}

function computeLayerObjective(batchedImage, layers = activeLayers) {
    return tf.tidy(() => {
        const outputs = inceptionModel.execute(
            batchedImage,
            layers.map(({ name }) => name)
        );
        const activations = Array.isArray(outputs) ? outputs : [outputs];
        const scores = activations.map((activation, index) => {
            // Match the TensorFlow DeepDream tutorial: maximize mean layer activation.
            return tf.mean(activation).mul(layers[index].weight);
        });

        if (scores.length === 1) {
            return scores[0];
        }

        return tf.addN(scores).div(tf.scalar(scores.length));
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

// Deep Dream with Octaves (multi-scale processing)
async function deepDreamWithOctaves(inputTensor, stepsPerOctave, options = {}) {
    const config = {
        ...DREAM_OPTIONS,
        ...options
    };
    config.layers = config.layers || activeLayers;

    // Octave parameters from TensorFlow tutorial
    const octaveScale = 1.3;
    const octaves = runtimeProfile.octaves;

    // Convert pixels to float32 [0, 1]. The TF Hub graph handles x * 2 - 1 internally.
    const baseImage = tf.tidy(() => {
        return tf.cast(inputTensor, 'float32').div(255);
    });
    const [originalHeight, originalWidth] = [baseImage.shape[0], baseImage.shape[1]];

    let img = baseImage.clone();

    for (let i = 0; i < octaves.length; i++) {
        const octave = octaves[i];

        // Calculate new size for this octave
        const newHeight = Math.round(originalHeight * Math.pow(octaveScale, octave));
        const newWidth = Math.round(originalWidth * Math.pow(octaveScale, octave));

        // Resize image for this octave
        const resized = tf.tidy(() => {
            const expanded = img.expandDims(0);
            const resizedExpanded = tf.image.resizeBilinear(expanded, [newHeight, newWidth]);
            return resizedExpanded.squeeze();
        });

        img.dispose();

        // Run gradient ascent for this octave
        updateProgress(
            20 + ((i + 1) / octaves.length) * 60,
            `Processing octave ${i + 1}/${octaves.length} (${newWidth}x${newHeight})`
        );

        img = await gradientAscent(resized, stepsPerOctave, config);
        resized.dispose();

        await tf.nextFrame();
    }

    // Resize back to original dimensions
    const final = tf.tidy(() => {
        const expanded = img.expandDims(0);
        const resizedExpanded = tf.image.resizeBilinear(expanded, [originalHeight, originalWidth]);
        return tf.clipByValue(resizedExpanded.squeeze(), 0, 1);
    });

    img.dispose();
    baseImage.dispose();

    updateProgress(85, 'Polishing details...');
    return final;
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

        const grads = shiftX === 0 && shiftY === 0
            ? computeGrad(dreamVar)
            : tf.tidy(() => {
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

        // Only update UI occasionally to avoid slowdown
        if (step % 10 === 0 || step === steps - 1) {
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
        const stepsPerOctave = parseInt(iterationsSlider.value);
        const dynamicOptions = {
            ...DREAM_OPTIONS,
            layers: activeLayers.map(layer => ({ ...layer }))
        };

        // Run deep dream with octaves
        updateProgress(20, 'Dreaming across octaves...');
        const dreamedImage = await deepDreamWithOctaves(inputImage, stepsPerOctave, dynamicOptions);

        // Display results
        updateProgress(95, 'Finalizing...');
        if (checkForNaNs(dreamedImage, 'dreamedImage')) {
            throw new Error('Dream result contains invalid values.');
        }
        await displayResults(dreamedImage);

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
async function displayResults(dreamed) {
    // Dreamed image is in [0, 1] range
    outputCanvas.width = dreamed.shape[1];
    outputCanvas.height = dreamed.shape[0];

    // tf.browser.toPixels expects [0, 1] and handles conversion to [0, 255]
    await tf.browser.toPixels(dreamed, outputCanvas);
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
    fileInput.value = '';
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

async function loadDefaultImage() {
    try {
        const response = await fetch(DEFAULT_IMAGE_PATH);
        if (!response.ok) {
            throw new Error(`Failed to fetch default image: ${response.status}`);
        }
        const blob = await response.blob();
        displayImage(blob);
    } catch (error) {
        console.error('Unable to load default image:', error);
    }
}

function checkForNaNs(tensor, label) {
    const hasNaN = tf.tidy(() => tf.any(tf.isNaN(tensor)).dataSync()[0]);
    if (hasNaN) {
        console.warn(`NaNs detected in ${label}`);
    }
    return hasNaN;
}

// Initialize on load
init();
