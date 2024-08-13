const GRID_SIZE = 10;
const COLORS = ['red', 'blue', 'green', 'yellow', 'purple'];

let gameBoard = [];
let moveCount = 0;
let bestScore = localStorage.getItem('bestScore') || Infinity;

const gameBoardElement = document.getElementById('game-board');
const moveCountElement = document.getElementById('move-count');
const bestScoreElement = document.getElementById('best-score');
const newGameButton = document.getElementById('new-game-btn');

function initializeGame() {
    gameBoard = [];
    moveCount = 0;
    updateMoveCount();
    updateBestScore();

    for (let i = 0; i < GRID_SIZE; i++) {
        const row = [];
        for (let j = 0; j < GRID_SIZE; j++) {
            const randomColor = COLORS[Math.floor(Math.random() * COLORS.length)];
            row.push(randomColor);
        }
        gameBoard.push(row);
    }

    renderBoard();
}

function renderBoard() {
    gameBoardElement.innerHTML = '';
    for (let i = 0; i < GRID_SIZE; i++) {
        for (let j = 0; j < GRID_SIZE; j++) {
            const cell = document.createElement('div');
            cell.classList.add('cell');
            cell.style.backgroundColor = gameBoard[i][j];
            gameBoardElement.appendChild(cell);
        }
    }
}

function floodFill(targetColor) {
    const startColor = gameBoard[0][0];
    if (startColor === targetColor) return;

    moveCount++;
    updateMoveCount();

    const stack = [[0, 0]];
    while (stack.length > 0) {
        const [x, y] = stack.pop();
        if (x < 0 || x >= GRID_SIZE || y < 0 || y >= GRID_SIZE || gameBoard[x][y] !== startColor) continue;

        gameBoard[x][y] = targetColor;
        stack.push([x + 1, y], [x - 1, y], [x, y + 1], [x, y - 1]);
    }

    renderBoard();
    checkWinCondition();
}

function checkWinCondition() {
    const winColor = gameBoard[0][0];
    const hasWon = gameBoard.every(row => row.every(cell => cell === winColor));

    if (hasWon) {
        if (moveCount < bestScore) {
            bestScore = moveCount;
            localStorage.setItem('bestScore', bestScore);
            updateBestScore();
        }
        alert(`Congratulations! You won in ${moveCount} moves!`);
    }
}

function updateMoveCount() {
    moveCountElement.textContent = moveCount;
}

function updateBestScore() {
    bestScoreElement.textContent = bestScore === Infinity ? '-' : bestScore;
}

// Event Listeners
document.querySelectorAll('.color-btn').forEach(button => {
    button.addEventListener('click', () => floodFill(button.dataset.color));
    button.style.backgroundColor = button.dataset.color;
});

newGameButton.addEventListener('click', initializeGame);

// Initialize the game
initializeGame();