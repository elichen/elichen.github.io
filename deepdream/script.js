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

// Load MobileNet Model
async function loadModel() {
    if (mobilenet) return mobilenet;

    updateProgress(0, 'Loading MobileNet model...');
    // Use @tensorflow-models/mobilenet which exposes intermediate layers
    mobilenet = await window.mobilenet.load({
        version: 2,
        alpha: 1.0
    });
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

                // Use infer() with embedding=true to get intermediate activations
                // This returns rich convolutional features instead of final classification
                const activations = mobilenet.infer(batched, true);

                // Maximize the L2 norm (sum of squares) of activations
                // This encourages strong feature responses
                return tf.mean(tf.square(activations));
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
        const iterations = parseInt(iterationsSlider.value);
        const octaves = 3; // Fixed at 3 octaves

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
    // Only show the dreamed result at full resolution
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
