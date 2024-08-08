// Game constants
const BIRD_SIZE = 20;
const GRAVITY = 0.5;
const FLAP_STRENGTH = -10;
const OBSTACLE_WIDTH = 50;
const OBSTACLE_GAP = 150;
const OBSTACLE_SPEED = 2;

// Matplotlib-inspired color scheme
const COLORS = {
    background: '#ffffff',
    bird: '#1f77b4',
    obstacle: '#2ca02c',
    trail: '#ff7f0e'
};

// Game variables
let CANVAS_WIDTH, CANVAS_HEIGHT;
let bird;
let obstacles = [];
let score = 0;
let highScore = 0;
let gameLoop;
let isGameOver = false;
let trail = [];

// DOM elements
const canvas = document.getElementById('game-canvas');
const ctx = canvas.getContext('2d');
const startButton = document.getElementById('start-button');
const restartButton = document.getElementById('restart-button');
const scoreDisplay = document.getElementById('score');
const highScoreDisplay = document.getElementById('high-score');

// Bird object
class Bird {
    constructor() {
        this.x = 50;
        this.y = CANVAS_HEIGHT / 2;
        this.velocity = 0;
        this.positionHistory = [];
        this.frameCount = 0;
    }

    flap() {
        this.velocity = FLAP_STRENGTH;
    }

    update() {
        this.velocity += GRAVITY;
        this.y += this.velocity;

        if (this.y < 0) this.y = 0;
        if (this.y > CANVAS_HEIGHT - BIRD_SIZE) this.y = CANVAS_HEIGHT - BIRD_SIZE;

        this.frameCount++;
        if (this.frameCount % 10 === 0) { // Save position every 10 frames
            this.positionHistory.push({ x: this.x, y: this.y });
        }
        
        trail.push({ x: this.x + BIRD_SIZE, y: this.y });
    }

    draw() {
        ctx.font = `${BIRD_SIZE * 1.5}px Arial`; // Set font size relative to bird size
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('🤖', this.x + BIRD_SIZE / 2, this.y + BIRD_SIZE / 2);
    }
}

// Obstacle object
class Obstacle {
    constructor() {
        this.x = CANVAS_WIDTH;
        this.topHeight = Math.random() * (CANVAS_HEIGHT - OBSTACLE_GAP);
        this.bottomY = this.topHeight + OBSTACLE_GAP;
    }

    update() {
        this.x -= OBSTACLE_SPEED;
    }

    draw() {
        ctx.fillStyle = COLORS.obstacle;
        ctx.fillRect(this.x, 0, OBSTACLE_WIDTH, this.topHeight);
        ctx.fillRect(this.x, this.bottomY, OBSTACLE_WIDTH, CANVAS_HEIGHT - this.bottomY);
    }
}

// Game functions
function startGame() {
    bird = new Bird();
    clearObstacles();
    score = 0;
    isGameOver = false;
    trail = [];
    startButton.style.display = 'none';
    restartButton.style.display = 'none';
    gameLoop = setInterval(updateGame, 20);
}

function updateGame() {
    if (isGameOver) return; // Skip updating if the game is over

    ctx.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT); // Clear canvas
    ctx.fillStyle = COLORS.background;
    ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

    bird.update();
    updateTrail();
    drawTrail();
    bird.draw();

    if (obstacles.length === 0 || obstacles[obstacles.length - 1].x < CANVAS_WIDTH - 300) {
        obstacles.push(new Obstacle());
    }

    obstacles.forEach((obstacle, index) => {
        obstacle.update();
        obstacle.draw();

        // Check collision
        if (
            bird.x < obstacle.x + OBSTACLE_WIDTH &&
            bird.x + BIRD_SIZE > obstacle.x &&
            (bird.y < obstacle.topHeight || bird.y + BIRD_SIZE > obstacle.bottomY)
        ) {
            gameOver();
        }

        // Remove off-screen obstacles and increase score
        if (obstacle.x + OBSTACLE_WIDTH < 0) {
            obstacles.splice(index, 1);
            score++;
            updateScore();
        }
    });
}

function updateTrail() {
    trail.forEach(point => point.x -= OBSTACLE_SPEED);
    trail = trail.filter(point => point.x > 0);
}

function clearObstacles() {
    while (obstacles.length > 0) {
        obstacles.splice(0, 1); // Remove each obstacle one by one
    }
}

function gameOver() {
    clearInterval(gameLoop);
    isGameOver = true;
    restartButton.style.display = 'block';

    clearObstacles();

    if (score > highScore) {
        highScore = score;
        highScoreDisplay.textContent = highScore;
        localStorage.setItem('highScore', highScore);
    }

    // Clear the canvas before drawing the final loss chart
    ctx.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
    showFinalLossChart();
}

function updateScore() {
    scoreDisplay.textContent = score;
}

function drawTrail() {
    ctx.beginPath();
    trail.forEach((point, index) => {
        if (index === 0) {
            ctx.moveTo(point.x, point.y);
        } else {
            ctx.lineTo(point.x, point.y);
        }
    });
    ctx.strokeStyle = COLORS.trail;
    ctx.lineWidth = 2;
    ctx.stroke();
}

function showFinalLossChart() {
    ctx.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT); // Clear the canvas again
    ctx.fillStyle = COLORS.background;
    ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

    const padding = 40;
    const chartWidth = CANVAS_WIDTH - 2 * padding;
    const chartHeight = CANVAS_HEIGHT - 2 * padding;

    // Draw axes
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, CANVAS_HEIGHT - padding);
    ctx.lineTo(CANVAS_WIDTH - padding, CANVAS_HEIGHT - padding);
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Draw dashed grid lines
    ctx.strokeStyle = '#cccccc';
    ctx.setLineDash([5, 5]);
    for (let i = padding; i <= CANVAS_WIDTH - padding; i += chartWidth / 10) {
        ctx.beginPath();
        ctx.moveTo(i, padding);
        ctx.lineTo(i, CANVAS_HEIGHT - padding);
        ctx.stroke();
    }
    for (let i = padding; i <= CANVAS_HEIGHT - padding; i += chartHeight / 10) {
        ctx.beginPath();
        ctx.moveTo(padding, i);
        ctx.lineTo(CANVAS_WIDTH - padding, i);
        ctx.stroke();
    }
    ctx.setLineDash([]); // Reset to solid lines

    // Draw trail
    ctx.beginPath();
    bird.positionHistory.forEach((point, index) => {
        const x = padding + (index / bird.positionHistory.length) * chartWidth;
        const y = CANVAS_HEIGHT - padding - ((CANVAS_HEIGHT - point.y) / CANVAS_HEIGHT) * chartHeight;
        if (index === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    });
    ctx.strokeStyle = COLORS.trail;
    ctx.lineWidth = 2;
    ctx.stroke();

    // Add labels and title
    ctx.fillStyle = '#000000';
    ctx.font = '16px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Training Iterations', CANVAS_WIDTH / 2, CANVAS_HEIGHT - 10);
    ctx.save();
    ctx.translate(20, CANVAS_HEIGHT / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Loss', 0, 0);
    ctx.restore();
    ctx.font = '20px Arial';
    ctx.fillText('Final Loss Chart', CANVAS_WIDTH / 2, 30);
}

// Event listeners
startButton.addEventListener('click', startGame);
restartButton.addEventListener('click', startGame);
document.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && !isGameOver) bird.flap();
});
canvas.addEventListener('click', () => {
    if (!isGameOver) bird.flap();
});

// Ensure the canvas is properly sized
function resizeCanvas() {
    const container = document.getElementById('game-container');
    canvas.width = container.clientWidth;
    canvas.height = container.clientWidth / 2; // Maintain 2:1 aspect ratio
    CANVAS_WIDTH = canvas.width;
    CANVAS_HEIGHT = canvas.height;
}

window.addEventListener('resize', resizeCanvas);
resizeCanvas(); // Initial sizing

// Load high score from local storage
const savedHighScore = localStorage.getItem('highScore');
if (savedHighScore) {
    highScore = parseInt(savedHighScore);
    highScoreDisplay.textContent = highScore;
}

// Call startGame when the page loads
window.addEventListener('load', startGame);
