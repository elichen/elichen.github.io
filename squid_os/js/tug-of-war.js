const rope = document.getElementById('rope');
const instructions = document.getElementById('instructions');
const startButton = document.getElementById('start-button');
const status = document.getElementById('status');

let gameRunning = false;
let ropePosition = 10;
let gameInterval;

function pullRope() {
    if (gameRunning) {
        ropePosition += 1;
        rope.style.left = `${ropePosition}%`;

        if (ropePosition >= 20) {
            endGame(true);
        }
    }
}

function opponentPull() {
    if (gameRunning) {
        ropePosition -= 0.5;
        rope.style.left = `${ropePosition}%`;

        if (ropePosition <= 0) {
            endGame(false);
        }
    }
}

function startGame() {
    instructions.style.display = 'none';
    status.textContent = '';
    ropePosition = 10;
    rope.style.left = '10%';
    gameRunning = true;

    gameInterval = setInterval(opponentPull, 100);
}

function endGame(win) {
    gameRunning = false;
    clearInterval(gameInterval);
    if (win) {
        status.textContent = 'You Win!';
        status.style.color = '#2ecc71';
    } else {
        status.textContent = 'You Lose!';
        status.style.color = '#e74c3c';
    }
    instructions.style.display = 'block';
    startButton.textContent = 'Play Again';
}

startButton.addEventListener('click', startGame);
document.addEventListener('keydown', (e) => {
    if (e.code === 'Space') {
        pullRope();
    }
});