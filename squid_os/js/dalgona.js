const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const instructions = document.getElementById('instructions');
const startButton = document.getElementById('start-button');
const status = document.getElementById('status');

let gameRunning = false;
let drawing = false;
let carvedPoints = new Set();
let totalPoints = 0;

// Define the triangle shape
const shape = new Path2D();
const p1 = { x: 200, y: 100 };
const p2 = { x: 250, y: 200 };
const p3 = { x: 150, y: 200 };
shape.moveTo(p1.x, p1.y);
shape.lineTo(p2.x, p2.y);
shape.lineTo(p3.x, p3.y);
shape.closePath();

// Create a slightly larger path for checking if the user goes too far
const outerShape = new Path2D();
const offset = 15;
outerShape.moveTo(p1.x, p1.y - offset);
outerShape.lineTo(p2.x + offset, p2.y + offset);
outerShape.lineTo(p3.x - offset, p3.y + offset);
outerShape.closePath();

function getPointsOnPath() {
    const points = new Set();
    const lines = [
        { p1: p1, p2: p2 },
        { p1: p2, p2: p3 },
        { p1: p3, p2: p1 },
    ];

    for (const line of lines) {
        const dx = line.p2.x - line.p1.x;
        const dy = line.p2.y - line.p1.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const steps = dist / 5; // Check every 5 pixels
        for (let i = 0; i < steps; i++) {
            const x = line.p1.x + (dx * i) / steps;
            const y = line.p1.y + (dy * i) / steps;
            points.add(`${Math.round(x)},${Math.round(y)}`);
        }
    }
    return points;
}

const shapePoints = getPointsOnPath();
totalPoints = shapePoints.size;

function drawHoneycomb() {
    ctx.fillStyle = '#f1c40f';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = '#e74c3c';
    ctx.lineWidth = 5;
    ctx.stroke(shape);
}

function startGame() {
    instructions.style.display = 'none';
    status.textContent = '';
    gameRunning = true;
    drawing = false;
    carvedPoints.clear();
    ctx.globalCompositeOperation = 'source-over';
    clearCanvas();
    drawHoneycomb();
}

function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function endGame(win) {
    gameRunning = false;
    if (win) {
        status.textContent = 'You Win!';
        status.style.color = '#2ecc71';
    } else {
        status.textContent = 'You broke the honeycomb! You lose!';
        status.style.color = '#e74c3c';
    }
    instructions.style.display = 'block';
    startButton.textContent = 'Play Again';
}

function handleMouseDown(e) {
    if (!gameRunning) return;
    drawing = true;
}

function handleMouseUp(e) {
    if (!gameRunning) return;
    drawing = false;
}

function handleMouseMove(e) {
    if (!gameRunning || !drawing) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (!ctx.isPointInPath(outerShape, x, y)) {
        endGame(false);
        return;
    }

    ctx.globalCompositeOperation = 'destination-out';
    ctx.beginPath();
    ctx.arc(x, y, 10, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalCompositeOperation = 'source-over';

    for (const pointStr of shapePoints) {
        const [px, py] = pointStr.split(',').map(Number);
        if (Math.sqrt((x - px) ** 2 + (y - py) ** 2) < 10) {
            carvedPoints.add(pointStr);
        }
    }

    if (carvedPoints.size >= totalPoints * 0.9) {
        endGame(true);
    }
}

startButton.addEventListener('click', startGame);
canvas.addEventListener('mousedown', handleMouseDown);
canvas.addEventListener('mouseup', handleMouseUp);
canvas.addEventListener('mousemove', handleMouseMove);