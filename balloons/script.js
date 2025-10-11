// Game state
const gameState = {
    balloons: [],
    redScore: 0,
    blueScore: 0,
    isModelReady: false,
    hands: []
};

// Canvas and video elements
let video, canvas, ctx, statusElement;
let detector;

// Game settings
const BALLOON_RADIUS = 40;
const BALLOON_SPEED = 1;
const SPAWN_INTERVAL = 2000; // milliseconds
const POP_DISTANCE = 80; // Distance threshold for popping

// Balloon class
class Balloon {
    constructor(x, y, color) {
        this.x = x;
        this.y = y;
        this.color = color; // 'red' or 'blue'
        this.radius = BALLOON_RADIUS;
        this.speed = BALLOON_SPEED + Math.random() * 0.5;
        this.wobble = Math.random() * Math.PI * 2;
        this.wobbleSpeed = 0.05 + Math.random() * 0.05;
    }

    update() {
        // Move upward
        this.y -= this.speed;

        // Add wobble effect
        this.wobble += this.wobbleSpeed;
        this.x += Math.sin(this.wobble) * 0.5;
    }

    draw(ctx) {
        // Draw balloon body
        ctx.fillStyle = this.color;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
        ctx.fill();

        // Draw balloon highlight
        ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
        ctx.beginPath();
        ctx.arc(this.x - this.radius * 0.3, this.y - this.radius * 0.3, this.radius * 0.3, 0, Math.PI * 2);
        ctx.fill();

        // Draw string
        ctx.strokeStyle = this.color === 'red' ? '#8b0000' : '#00008b';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(this.x, this.y + this.radius);
        ctx.lineTo(this.x, this.y + this.radius + 30);
        ctx.stroke();
    }

    isOffScreen(canvasHeight) {
        return this.y + this.radius < 0;
    }
}

// Initialize webcam
async function setupCamera() {
    video = document.getElementById('video');

    const stream = await navigator.mediaDevices.getUserMedia({
        video: {
            width: { ideal: 1280 },
            height: { ideal: 720 }
        },
        audio: false
    });

    video.srcObject = stream;

    return new Promise((resolve) => {
        video.onloadedmetadata = () => {
            video.play();
            resolve(video);
        };
    });
}

// Initialize hand detector
async function loadHandDetector() {
    const model = handPoseDetection.SupportedModels.MediaPipeHands;
    const detectorConfig = {
        runtime: 'mediapipe',
        solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/hands',
        maxHands: 4, // Support multiple people
        modelType: 'full'
    };

    detector = await handPoseDetection.createDetector(model, detectorConfig);
    return detector;
}

// Setup canvas
function setupCanvas() {
    canvas = document.getElementById('canvas');
    ctx = canvas.getContext('2d');
    statusElement = document.getElementById('status');

    // Set canvas size to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
}

// Spawn balloons periodically
function spawnBalloon() {
    const color = Math.random() < 0.5 ? 'red' : 'blue';
    const x = Math.random() * (canvas.width - BALLOON_RADIUS * 2) + BALLOON_RADIUS;
    const y = canvas.height + BALLOON_RADIUS;

    gameState.balloons.push(new Balloon(x, y, color));
}

// Calculate distance between two points
function distance(x1, y1, x2, y2) {
    return Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
}

// Check collision between hand and balloon
function checkCollisions() {
    for (let i = gameState.balloons.length - 1; i >= 0; i--) {
        const balloon = gameState.balloons[i];

        for (const hand of gameState.hands) {
            // Get hand center position (palm center - keypoint 9)
            const palmCenter = hand.keypoints[9];

            // Don't mirror - CSS handles the visual transform
            const handX = palmCenter.x;
            const handY = palmCenter.y;

            const dist = distance(handX, handY, balloon.x, balloon.y);

            if (dist < POP_DISTANCE) {
                // Check if correct hand color combination
                // Swap handedness because video is mirrored
                const isRightHand = hand.handedness === 'Left';
                const isLeftHand = hand.handedness === 'Right';

                if ((isRightHand && balloon.color === 'red') ||
                    (isLeftHand && balloon.color === 'blue')) {
                    // Pop the balloon
                    createPopEffect(balloon.x, balloon.y, balloon.color);
                    gameState.balloons.splice(i, 1);

                    // Update score
                    if (balloon.color === 'red') {
                        gameState.redScore++;
                        document.getElementById('redScore').textContent = gameState.redScore;
                    } else {
                        gameState.blueScore++;
                        document.getElementById('blueScore').textContent = gameState.blueScore;
                    }

                    break; // Move to next balloon
                }
            }
        }
    }
}

// Create pop effect
function createPopEffect(x, y, color) {
    // Draw pop animation
    ctx.save();
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.5;

    for (let i = 0; i < 8; i++) {
        const angle = (Math.PI * 2 * i) / 8;
        const distance = 30;
        const particleX = x + Math.cos(angle) * distance;
        const particleY = y + Math.sin(angle) * distance;

        ctx.beginPath();
        ctx.arc(particleX, particleY, 5, 0, Math.PI * 2);
        ctx.fill();
    }

    ctx.restore();
}

// Draw hands on canvas
function drawHands(hands) {
    for (const hand of hands) {
        // Swap colors because video is mirrored (Left hand = red, Right hand = blue)
        const handColor = hand.handedness === 'Left' ? '#ff6b6b' : '#4dabf7';

        // Draw hand keypoints
        for (const keypoint of hand.keypoints) {
            const x = keypoint.x; // Don't mirror - CSS handles it
            const y = keypoint.y;

            ctx.fillStyle = handColor;
            ctx.beginPath();
            ctx.arc(x, y, 5, 0, Math.PI * 2);
            ctx.fill();
        }

        // Draw palm center (larger)
        const palmCenter = hand.keypoints[9];
        const x = palmCenter.x; // Don't mirror - CSS handles it
        const y = palmCenter.y;

        ctx.strokeStyle = handColor;
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.arc(x, y, 15, 0, Math.PI * 2);
        ctx.stroke();
    }
}

// Main detection and rendering loop
async function detectAndRender() {
    // Detect hands
    const hands = await detector.estimateHands(video);

    // Process hands and add handedness info
    gameState.hands = hands.map(hand => ({
        ...hand,
        handedness: hand.handedness // 'Left' or 'Right'
    }));

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Update and draw balloons
    for (let i = gameState.balloons.length - 1; i >= 0; i--) {
        const balloon = gameState.balloons[i];
        balloon.update();
        balloon.draw(ctx);

        // Remove off-screen balloons
        if (balloon.isOffScreen(canvas.height)) {
            gameState.balloons.splice(i, 1);
        }
    }

    // Check collisions
    checkCollisions();

    // Draw hands
    drawHands(gameState.hands);

    // Continue loop
    requestAnimationFrame(detectAndRender);
}

// Initialize the application
async function init() {
    try {
        // Get status element first
        statusElement = document.getElementById('status');

        // Setup camera
        statusElement.textContent = 'Accessing camera...';
        await setupCamera();

        // Setup canvas
        setupCanvas();

        // Load hand detector
        statusElement.textContent = 'Loading hand tracking model...';
        await loadHandDetector();

        // Hide status message
        statusElement.classList.add('hidden');
        gameState.isModelReady = true;

        // Start spawning balloons
        setInterval(spawnBalloon, SPAWN_INTERVAL);

        // Start detection loop
        detectAndRender();

    } catch (error) {
        console.error('Error initializing application:', error);
        if (statusElement) {
            statusElement.textContent = 'Error: ' + error.message;
        }
    }
}

// Start the application when page loads
window.addEventListener('load', init);
