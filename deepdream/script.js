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
    steps: {
        classic: { value: 100, max: 150 },
        lucid: { value: 256, max: 512 }
    }
};
const FAST_PROFILE = {
    maxImageSize: 320,
    octaves: [-1, 0, 1],
    steps: {
        classic: { value: 20, max: 60 },
        lucid: { value: 64, max: 256 }
    }
};
const STEP_SLIDERS = {
    classic: { label: 'Steps per Octave', min: 10, step: 10 },
    lucid: { label: 'Steps', min: 32, step: 32 }
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
const iterationsLabel = document.getElementById('iterationsLabel');
const iterationsValue = document.getElementById('iterationsValue');
const methodSelect = document.getElementById('methodSelect');
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
// The converter hoisted each block's ReLU out of its concat, so `Mixed_6b/concat` is pre-activation.
// The block output (Keras `mixedN`) is the hoisted ReLU, which kept the first branch's name.
// Mixed_6a's concat is already post-ReLU because its branches end in ReLU/max-pool.
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

// TF DeepDream tutorial: gradient ascent in pixel space, run over octaves.
const CLASSIC = {
    // The tutorial steps 0.01 in [-1, 1] space; our image lives in [0, 1].
    stepSize: 0.005,
    octaveScale: 1.3,
    tileSize: 512
};

// Lucid: optimize a 1/f-scaled Fourier spectrum in decorrelated color space with Adam,
// under small random transforms (Olah et al., "Feature Visualization", Distill 2017).
const LUCID = {
    learningRate: 0.05,
    pad: 12,
    jitter: 8,
    scales: Array.from({ length: 11 }, (_, i) => 1 + (i - 5) / 50),
    jitterAfterScale: 4
};
// Square root of ImageNet's color covariance, from lucid.optvis.param.color.
const COLOR_CORRELATION_SVD_SQRT = [
    [0.26, 0.09, 0.02],
    [0.27, 0.00, -0.05],
    [0.27, -0.09, 0.03]
];

let activeLayers = LAYER_PRESETS.multi;
let activeMethod = 'classic';
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

    configureStepSlider();
}

function configureStepSlider() {
    const slider = STEP_SLIDERS[activeMethod];
    const steps = runtimeProfile.steps[activeMethod];
    iterationsLabel.textContent = slider.label;
    iterationsSlider.min = String(slider.min);
    iterationsSlider.step = String(slider.step);
    iterationsSlider.max = String(steps.max);
    iterationsSlider.value = String(steps.value);
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
methodSelect.addEventListener('change', (e) => {
    activeMethod = e.target.value;
    configureStepSlider();
});

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

// TF Hub's InceptionV3 graph takes [0, 1] images and applies the x * 2 - 1 preprocessing itself.
function computeLayerObjective(batchedImage, layers, squared = false) {
    return tf.tidy(() => {
        const outputs = inceptionModel.execute(
            batchedImage,
            layers.map(({ name }) => name)
        );
        const activations = Array.isArray(outputs) ? outputs : [outputs];
        // The tutorial maximizes each layer's mean activation; Lucid's deepdream objective uses the mean square.
        const scores = activations.map((activation, index) =>
            tf.mean(squared ? tf.square(activation) : activation).mul(layers[index].weight)
        );

        return scores.length === 1 ? scores[0] : tf.addN(scores);
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

// TF 2's tf.image.resize uses half-pixel centers; without them every octave nudges the image up and left.
function resizeImage(image, height, width) {
    return tf.tidy(() =>
        tf.image.resizeBilinear(image.expandDims(0), [height, width], false, true).squeeze([0])
    );
}

// ---------------------------------------------------------------------------
// Classic: the TensorFlow DeepDream tutorial
// ---------------------------------------------------------------------------

async function deepDreamWithOctaves(inputTensor, stepsPerOctave, layers) {
    const octaves = runtimeProfile.octaves;

    const baseImage = tf.tidy(() => tf.cast(inputTensor, 'float32').div(255));
    const [originalHeight, originalWidth] = baseImage.shape;

    let img = baseImage;

    for (let i = 0; i < octaves.length; i++) {
        const scale = Math.pow(CLASSIC.octaveScale, octaves[i]);
        const newHeight = Math.floor(originalHeight * scale);
        const newWidth = Math.floor(originalWidth * scale);

        const resized = resizeImage(img, newHeight, newWidth);
        img.dispose();

        const label = `Octave ${i + 1}/${octaves.length} (${newWidth}x${newHeight})`;
        img = await gradientAscent(resized, stepsPerOctave, layers, (step) => {
            const done = (i + step / stepsPerOctave) / octaves.length;
            updateProgress(20 + done * 65, `${label}: step ${step}/${stepsPerOctave}`);
        });
        resized.dispose();
    }

    const final = tf.tidy(() => tf.clipByValue(resizeImage(img, originalHeight, originalWidth), 0, 1));
    img.dispose();

    return final;
}

// Tile starts as in the tutorial: tf.range(0, size, tile)[:-1], or a single tile at 0.
function tileStarts(size, tileSize) {
    const starts = [];
    for (let start = 0; start < size; start += tileSize) {
        starts.push(start);
    }
    starts.pop();
    return starts.length ? starts : [0];
}

// Sum the objective over 512px tiles, which caps the network's input size on large octaves.
function tiledObjective(image, layers) {
    const [height, width] = image.shape;
    const { tileSize } = CLASSIC;
    const scores = [];

    for (const y of tileStarts(height, tileSize)) {
        for (const x of tileStarts(width, tileSize)) {
            const tile = image.slice(
                [y, x, 0],
                [Math.min(tileSize, height - y), Math.min(tileSize, width - x), 3]
            );
            scores.push(computeLayerObjective(tile.expandDims(0), layers));
        }
    }

    return scores.length === 1 ? scores[0] : tf.addN(scores);
}

async function gradientAscent(baseImage, steps, layers, onProgress) {
    const [height, width] = baseImage.shape;
    const computeGrad = tf.grad(image => tiledObjective(image, layers));
    let img = baseImage.clone();

    for (let step = 0; step < steps; step++) {
        // Roll the image randomly before each step so tile edges and the network's
        // stride grid don't imprint fixed artifacts.
        const shiftY = Math.floor(Math.random() * height);
        const shiftX = Math.floor(Math.random() * width);

        const next = tf.tidy(() => {
            const rolledGrads = computeGrad(rollImage(img, shiftY, shiftX));
            const grads = rollImage(rolledGrads, -shiftY, -shiftX);
            const std = tf.moments(grads).variance.sqrt().add(1e-8);
            return tf.clipByValue(img.add(grads.div(std).mul(CLASSIC.stepSize)), 0, 1);
        });
        img.dispose();
        img = next;

        if (step % 10 === 0 || step === steps - 1) {
            onProgress(step + 1);
            await tf.nextFrame();
        }
    }

    return img;
}

// ---------------------------------------------------------------------------
// Lucid: Fourier-space DeepDream
// ---------------------------------------------------------------------------

function invert3x3(m) {
    const [[a, b, c], [d, e, f], [g, h, i]] = m;
    const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    return [
        [(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det],
        [(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det],
        [(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det]
    ];
}

// rgb = decorrelated @ toRgb, with Lucid's matrix normalized by its largest column norm.
function colorMatrices() {
    const C = COLOR_CORRELATION_SVD_SQRT;
    const maxNorm = Math.max(...[0, 1, 2].map(j => Math.hypot(C[0][j], C[1][j], C[2][j])));
    const toRgb = [0, 1, 2].map(i => [0, 1, 2].map(o => C[o][i] / maxNorm));
    return { toRgb, fromRgb: invert3x3(toRgb) };
}

// DFT matrices for a real 2D FFT done as matrix products (tfjs has no gradient for its FFT ops).
function fourierBasis(height, width) {
    const freqWidth = Math.floor(width / 2) + 1;

    const cosH = new Float32Array(height * height);
    const sinH = new Float32Array(height * height);
    for (let k = 0; k < height; k++) {
        for (let y = 0; y < height; y++) {
            const angle = 2 * Math.PI * k * y / height;
            cosH[k * height + y] = Math.cos(angle);
            sinH[k * height + y] = Math.sin(angle);
        }
    }

    const cosW = new Float32Array(freqWidth * width);
    const sinW = new Float32Array(freqWidth * width);
    for (let k = 0; k < freqWidth; k++) {
        for (let x = 0; x < width; x++) {
            const angle = 2 * Math.PI * k * x / width;
            cosW[k * width + x] = Math.cos(angle);
            sinW[k * width + x] = Math.sin(angle);
        }
    }

    // Only non-negative x frequencies are stored, so every column but DC (and Nyquist,
    // for even widths) also stands in for its mirror image when inverting.
    const columnWeights = new Float32Array(freqWidth).map((_, k) =>
        (k === 0 || 2 * k === width ? 1 : 2) / width
    );

    // Lucid's 1/f spectrum scaling: low frequencies take bigger steps than high ones, so the
    // optimizer builds coherent structure instead of pixel noise. It also folds in Lucid's
    // divide-by-4 and the 1/height of the inverse DFT.
    const spectrumScale = new Float32Array(height * freqWidth);
    for (let y = 0; y < height; y++) {
        const fy = (y <= (height - 1) / 2 ? y : y - height) / height;
        for (let x = 0; x < freqWidth; x++) {
            const freq = Math.max(Math.hypot(x / width, fy), 1 / Math.max(width, height));
            spectrumScale[y * freqWidth + x] = Math.sqrt(width * height) / freq / (4 * height);
        }
    }

    return {
        height,
        width,
        freqWidth,
        // One copy per color channel for batched matmuls.
        cosH: tf.tidy(() => tf.tensor2d(cosH, [height, height]).expandDims(0).tile([3, 1, 1])),
        sinH: tf.tidy(() => tf.tensor2d(sinH, [height, height]).expandDims(0).tile([3, 1, 1])),
        cosW: tf.tensor2d(cosW, [freqWidth, width]),
        sinW: tf.tensor2d(sinW, [freqWidth, width]),
        columnWeights: tf.tensor1d(columnWeights),
        spectrumScale: tf.tensor2d(spectrumScale, [height, freqWidth])
    };
}

function disposeBasis(basis) {
    Object.values(basis).forEach(value => value instanceof tf.Tensor && value.dispose());
}

// [3, H, W] decorrelated image -> spectrum (real, imag) that spectrumToImage maps back to it.
function imageToSpectrum(decorrelated, basis) {
    return tf.tidy(() => {
        const { height, width, freqWidth } = basis;
        const rows = decorrelated.reshape([3 * height, width]);
        const rowReal = tf.matMul(rows, basis.cosW, false, true).reshape([3, height, freqWidth]);
        const rowImag = tf.matMul(rows, basis.sinW, false, true).neg().reshape([3, height, freqWidth]);
        const real = tf.matMul(basis.cosH, rowReal).add(tf.matMul(basis.sinH, rowImag));
        const imag = tf.matMul(basis.cosH, rowImag).sub(tf.matMul(basis.sinH, rowReal));
        const norm = basis.spectrumScale.mul(height);
        return [real.div(norm), imag.div(norm)];
    });
}

// Spectrum -> [H, W, 3] image in [0, 1]: inverse real FFT, recorrelate colors, sigmoid.
function spectrumToImage(real, imag, basis, toRgbFilter) {
    return tf.tidy(() => {
        const { height, width, freqWidth } = basis;
        const scaledReal = real.mul(basis.spectrumScale);
        const scaledImag = imag.mul(basis.spectrumScale);
        const colReal = tf.matMul(basis.cosH, scaledReal).sub(tf.matMul(basis.sinH, scaledImag)).mul(basis.columnWeights);
        const colImag = tf.matMul(basis.cosH, scaledImag).add(tf.matMul(basis.sinH, scaledReal)).mul(basis.columnWeights);
        const decorrelated = tf.matMul(colReal.reshape([3 * height, freqWidth]), basis.cosW)
            .sub(tf.matMul(colImag.reshape([3 * height, freqWidth]), basis.sinW))
            .reshape([3, height, width]);
        const rgb = tf.conv2d(decorrelated.transpose([1, 2, 0]).expandDims(0), toRgbFilter, 1, 'valid');
        return tf.sigmoid(rgb).squeeze([0]);
    });
}

// Pixels -> [3, H, W] decorrelated logits, so the sigmoid parameterization starts at the photo.
// Computed in plain JS: on WebGL, log(0) in packed-texture padding turns 3-channel matmuls into NaN.
function photoToDecorrelated(inputTensor, fromRgb) {
    const [height, width] = inputTensor.shape;
    const pixels = inputTensor.dataSync();
    const count = height * width;
    const out = new Float32Array(3 * count);

    for (let p = 0; p < count; p++) {
        const logits = [0, 1, 2].map(c => {
            const v = Math.min(0.98, Math.max(0.02, pixels[p * 3 + c] / 255));
            return Math.log(v / (1 - v));
        });
        for (let o = 0; o < 3; o++) {
            out[o * count + p] = logits[0] * fromRgb[0][o] + logits[1] * fromRgb[1][o] + logits[2] * fromRgb[2][o];
        }
    }

    return tf.tensor3d(out, [3, height, width]);
}

function randomCrop(image, amount) {
    const [height, width] = image.shape;
    const dy = Math.floor(Math.random() * (amount + 1));
    const dx = Math.floor(Math.random() * (amount + 1));
    return image.slice([dy, dx, 0], [height - amount, width - amount, 3]);
}

// Lucid's standard transforms, minus rotation (tfjs can't backprop through image rotation).
function lucidTransforms(image) {
    const { pad } = LUCID;
    let x = tf.pad(image, [[pad, pad], [pad, pad], [0, 0]], 0.5);
    x = randomCrop(x, LUCID.jitter);
    const scale = LUCID.scales[Math.floor(Math.random() * LUCID.scales.length)];
    const [height, width] = x.shape;
    x = tf.image.resizeBilinear(x, [Math.round(height * scale), Math.round(width * scale)], false, true);
    return randomCrop(x, LUCID.jitterAfterScale);
}

async function lucidDream(inputTensor, steps, layers) {
    const [height, width] = inputTensor.shape;
    const basis = fourierBasis(height, width);
    const { toRgb, fromRgb } = colorMatrices();
    const toRgbFilter = tf.tensor4d(toRgb.flat(), [1, 1, 3, 3]);

    const decorrelated = photoToDecorrelated(inputTensor, fromRgb);
    const [initReal, initImag] = imageToSpectrum(decorrelated, basis);
    decorrelated.dispose();
    const real = tf.variable(initReal);
    const imag = tf.variable(initImag);
    initReal.dispose();
    initImag.dispose();

    const optimizer = tf.train.adam(LUCID.learningRate);

    for (let step = 0; step < steps; step++) {
        optimizer.minimize(() => {
            const image = lucidTransforms(spectrumToImage(real, imag, basis, toRgbFilter));
            return computeLayerObjective(image.expandDims(0), layers, true).neg();
        }, false, [real, imag]);

        if (step % 5 === 0 || step === steps - 1) {
            updateProgress(20 + ((step + 1) / steps) * 65, `Optimizing in Fourier space: step ${step + 1}/${steps}`);
            await tf.nextFrame();
        }
    }

    const result = spectrumToImage(real, imag, basis, toRgbFilter);

    optimizer.dispose();
    real.dispose();
    imag.dispose();
    toRgbFilter.dispose();
    disposeBasis(basis);

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
        const steps = parseInt(iterationsSlider.value);
        const layers = activeLayers.map(layer => ({ ...layer }));

        updateProgress(20, 'Dreaming...');
        const dreamedImage = activeMethod === 'lucid'
            ? await lucidDream(inputImage, steps, layers)
            : await deepDreamWithOctaves(inputImage, steps, layers);

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
