const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');
const cellSize = 10;
const rows = 50;
const cols = 50;

canvas.width = cols * cellSize;
canvas.height = rows * cellSize;

let grid = createGrid();
let isRunning = false;
let intervalId = null;

function createGrid() {
    return Array(rows).fill().map(() => Array(cols).fill().map(() => Math.random() > 0.7));
}

function drawGrid() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            if (grid[i][j]) {
                ctx.fillStyle = 'black';
                ctx.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);
            }
        }
    }
}

function updateGrid() {
    const newGrid = grid.map(arr => [...arr]);
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            const neighbors = countNeighbors(i, j);
            if (grid[i][j]) {
                newGrid[i][j] = neighbors === 2 || neighbors === 3;
            } else {
                newGrid[i][j] = neighbors === 3;
            }
        }
    }
    grid = newGrid;
}

function countNeighbors(x, y) {
    let count = 0;
    for (let i = -1; i <= 1; i++) {
        for (let j = -1; j <= 1; j++) {
            if (i === 0 && j === 0) continue;
            const newX = (x + i + rows) % rows;
            const newY = (y + j + cols) % cols;
            if (grid[newX][newY]) count++;
        }
    }
    return count;
}

function step() {
    updateGrid();
    drawGrid();
}

function startStop() {
    if (isRunning) {
        clearInterval(intervalId);
        isRunning = false;
    } else {
        intervalId = setInterval(step, 100);
        isRunning = true;
    }
}

function reset() {
    grid = createGrid();
    drawGrid();
    if (isRunning) {
        clearInterval(intervalId);
        isRunning = false;
    }
}

document.getElementById('startStop').addEventListener('click', startStop);
document.getElementById('reset').addEventListener('click', reset);

drawGrid();