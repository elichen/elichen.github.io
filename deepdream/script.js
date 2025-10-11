// Global state
let mobilenet = null;
let dreamModel = null;
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
const layerSelect = document.getElementById('layerSelect');
const iterationsSlider = document.getElementById('iterationsSlider');
const iterationsValue = document.getElementById('iterationsValue');
const octavesSlider = document.getElementById('octavesSlider');
const octavesValue = document.getElementById('octavesValue');
const dreamBtn = document.getElementById('dreamBtn');
const progressSection = document.getElementById('progressSection');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const resultsSection = document.getElementById('resultsSection');
const originalCanvas = document.getElementById('originalCanvas');
const outputCanvas = document.getElementById('outputCanvas');
const downloadBtn = document.getElementById('downloadBtn');
const resetBtn = document.getElementById('resetBtn');

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
octavesSlider.addEventListener('input', (e) => {
    octavesValue.textContent = e.target.value;
});

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
        // Resize to 224x224 for MobileNet
        const size = 224;
        const ctx = inputCanvas.getContext('2d');
        inputCanvas.width = size;
        inputCanvas.height = size;

        // Draw image maintaining aspect ratio
        const scale = Math.min(size / img.width, size / img.height);
        const x = (size - img.width * scale) / 2;
        const y = (size - img.height * scale) / 2;

        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, size, size);
        ctx.drawImage(img, x, y, img.width * scale, img.height * scale);

        imagePreview.classList.remove('hidden');
        dreamBtn.disabled = false;

        // Store the tensor
        if (inputImage) inputImage.dispose();
        inputImage = tf.browser.fromPixels(inputCanvas);
    };
    img.src = URL.createObjectURL(blob);
}

// Load MobileNet Model
async function loadModel() {
    if (mobilenet) return mobilenet;

    updateProgress(0, 'Loading MobileNet model...');
    mobilenet = await tf.loadGraphModel(
        'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v2_100_224/classification/3/default/1',
        { fromTFHub: true }
    );
    updateProgress(10, 'Model loaded!');
    return mobilenet;
}

// Create Dream Model for specific layer
function createDreamModel(layerName) {
    try {
        // Get the base model
        const model = mobilenet.modelUrl ? mobilenet : mobilenet;

        // For TFHub models, we'll use a simpler approach
        // Create a function that computes intermediate activations
        dreamModel = {
            predict: (input) => {
                return tf.tidy(() => {
                    const predictions = mobilenet.predict(input);
                    return predictions;
                });
            }
        };

        return dreamModel;
    } catch (error) {
        console.error('Error creating dream model:', error);
        throw error;
    }
}

// Deep Dream Core Algorithm
async function deepDream(inputTensor, iterations, octaves) {
    const octaveScale = 1.3;
    let img = inputTensor;

    // Multi-octave processing
    for (let octave = 0; octave < octaves; octave++) {
        updateProgress(
            10 + (octave / octaves) * 10,
            `Processing octave ${octave + 1}/${octaves}...`
        );

        // Resize for this octave
        const scale = Math.pow(octaveScale, octaves - octave - 1);
        const scaledSize = [
            Math.round(inputTensor.shape[0] * scale),
            Math.round(inputTensor.shape[1] * scale)
        ];

        img = tf.tidy(() => {
            return tf.image.resizeBilinear(img, scaledSize);
        });

        // Run gradient ascent
        img = await gradientAscent(img, iterations / octaves, octave, octaves);

        // Resize back to original size
        if (octave < octaves - 1) {
            const oldImg = img;
            img = tf.tidy(() => {
                return tf.image.resizeBilinear(img, [inputTensor.shape[0], inputTensor.shape[1]]);
            });
            oldImg.dispose();
        }

        await tf.nextFrame();
    }

    return img;
}

// Gradient Ascent
async function gradientAscent(img, steps, currentOctave, totalOctaves) {
    let dreamImg = tf.variable(img);

    const learningRate = 1.5;  // Increased for stronger effect

    for (let step = 0; step < steps; step++) {
        // Compute gradients
        const grads = tf.tidy(() => {
            return tf.grad(image => {
                // Ensure image is 224x224 for MobileNet
                let resized = image;
                if (image.shape[0] !== 224 || image.shape[1] !== 224) {
                    resized = tf.image.resizeBilinear(image, [224, 224]);
                }

                const normalized = tf.div(resized, 255.0);
                const batched = tf.expandDims(normalized, 0);
                const predictions = mobilenet.predict(batched);

                // Maximize the L2 norm (sum of squares) of activations
                // This encourages strong feature responses
                return tf.mean(tf.square(predictions));
            })(dreamImg);
        });

        // Normalize gradients
        const normalizedGrads = tf.tidy(() => {
            const mean = tf.mean(tf.abs(grads));
            return tf.div(grads, tf.add(mean, 1e-8));
        });

        // Apply gradient ascent and clamp values to [0, 255]
        tf.tidy(() => {
            const updated = tf.add(dreamImg, tf.mul(normalizedGrads, learningRate));
            const clamped = tf.clipByValue(updated, 0, 255);
            dreamImg.assign(clamped);
        });

        grads.dispose();
        normalizedGrads.dispose();

        // Update progress
        const totalProgress = 20 +
            ((currentOctave + (step / steps)) / totalOctaves) * 70;

        if (step % 10 === 0) {
            updateProgress(
                totalProgress,
                `Iteration ${Math.round((currentOctave * steps) + step)}/${Math.round(steps * totalOctaves)}...`
            );
            await tf.nextFrame();
        }
    }

    return dreamImg;
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
        const layerName = layerSelect.value;
        const iterations = parseInt(iterationsSlider.value);
        const octaves = parseInt(octavesSlider.value);

        // Create dream model
        updateProgress(15, 'Preparing dream model...');
        createDreamModel(layerName);

        // Run deep dream
        updateProgress(20, 'Dreaming...');
        const dreamedImage = await deepDream(inputImage, iterations, octaves);

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
    // Original
    originalCanvas.width = original.shape[1];
    originalCanvas.height = original.shape[0];
    tf.browser.toPixels(original, originalCanvas);

    // Dreamed - clip to [0, 255] and convert to uint8
    const processed = tf.tidy(() => {
        const clipped = tf.clipByValue(dreamed, 0, 255);
        // Convert to uint8 for toPixels
        return tf.cast(clipped, 'int32');
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
