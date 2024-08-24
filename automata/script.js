const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');
const cellSize = 10;
const rows = 50;
const cols = 50;
const MAX_AGE = 10; // Maximum age for color gradient

canvas.width = cols * cellSize;
canvas.height = rows * cellSize;

let grid = createGrid();
let isRunning = false;
let intervalId = null;

function createGrid() {
    return Array(rows).fill().map(() => Array(cols).fill().map(() => ({alive: Math.random() > 0.7, age: 0})));
}

function drawGrid() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            if (grid[i][j].alive) {
                ctx.fillStyle = getCellColor(grid[i][j].age);
                ctx.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);
            }
        }
    }
}

function getCellColor(age) {
    const normalizedAge = Math.min(age, MAX_AGE) / MAX_AGE;
    const r = Math.floor(165 * normalizedAge);
    const g = Math.floor(128 + 127 * (1 - normalizedAge)); // Start with a darker green
    const b = Math.floor(50 * normalizedAge);
    return `rgb(${r}, ${g}, ${b})`;
}

function updateGrid() {
    const newGrid = grid.map(row => row.map(cell => ({...cell})));
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            const neighbors = countNeighbors(i, j);
            if (grid[i][j].alive) {
                newGrid[i][j].alive = neighbors === 2 || neighbors === 3;
                if (newGrid[i][j].alive) {
                    newGrid[i][j].age = Math.min(grid[i][j].age + 1, MAX_AGE);
                } else {
                    newGrid[i][j].age = 0;
                }
            } else {
                newGrid[i][j].alive = neighbors === 3;
                newGrid[i][j].age = newGrid[i][j].alive ? 0 : grid[i][j].age;
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
            if (grid[newX][newY].alive) count++;
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

function createRuleDisplay() {
    const rulesDisplay = document.getElementById('rulesDisplay');
    const rules = [
        { before: [0,1,0,1,1,0,0,1,0], after: 1, description: "Live cell with 2-3 neighbors survives" },
        { before: [0,1,0,0,1,0,0,1,0], after: 0, description: "Live cell with <2 neighbors dies" },
        { before: [1,1,1,1,1,1,0,1,0], after: 0, description: "Live cell with >3 neighbors dies" },
        { before: [0,1,0,1,0,1,0,1,0], after: 1, description: "Dead cell with 3 neighbors becomes alive" }
    ];

    rules.forEach(rule => {
        const ruleElement = document.createElement('div');
        ruleElement.className = 'rule';
        
        const ruleContent = document.createElement('div');
        ruleContent.className = 'rule-content';
        
        const beforeGrid = document.createElement('div');
        beforeGrid.className = 'rule-grid';
        rule.before.forEach(cell => {
            const cellElement = document.createElement('div');
            cellElement.className = `rule-cell ${cell ? 'alive' : ''}`;
            beforeGrid.appendChild(cellElement);
        });
        
        const afterCell = document.createElement('div');
        afterCell.className = `rule-cell rule-after ${rule.after ? 'alive' : ''}`;
        
        const descriptionElement = document.createElement('div');
        descriptionElement.className = 'rule-description';
        descriptionElement.textContent = rule.description;
        
        ruleContent.appendChild(beforeGrid);
        ruleContent.appendChild(document.createElement('div')).className = 'rule-arrow';
        ruleContent.lastChild.textContent = '→';
        ruleContent.appendChild(afterCell);
        ruleContent.appendChild(descriptionElement);
        
        ruleElement.appendChild(ruleContent);
        rulesDisplay.appendChild(ruleElement);
    });
}

// Call this function after the page loads
window.addEventListener('load', createRuleDisplay);

// New function to handle canvas clicks
function handleCanvasClick(event) {
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const cellX = Math.floor(x / cellSize);
    const cellY = Math.floor(y / cellSize);

    if (cellX >= 0 && cellX < cols && cellY >= 0 && cellY < rows) {
        grid[cellY][cellX].alive = !grid[cellY][cellX].alive;
        grid[cellY][cellX].age = 0;
        drawGrid();
    }
}

// Add event listener for canvas clicks
canvas.addEventListener('click', handleCanvasClick);

document.getElementById('startStop').addEventListener('click', startStop);
document.getElementById('reset').addEventListener('click', reset);

drawGrid();