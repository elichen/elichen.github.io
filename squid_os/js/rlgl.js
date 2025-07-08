const player = document.getElementById('player');
const status = document.getElementById('status');
const instructions = document.getElementById('instructions');
const startButton = document.getElementById('start-button');
const finishLine = document.getElementById('finish-line');

let gameRunning = false;
let greenLight = true;
let gameInterval;

function setLight(light) {
    greenLight = light;
    if (gameRunning) {
        status.textContent = light ? 'Green Light' : 'Red Light';
        status.style.color = light ? '#2ecc71' : '#e74c3c';
    }
}

function movePlayer(e) {
    if (gameRunning && e.key === 'ArrowUp') {
        if (!greenLight) {
            endGame(false);
        } else {
            const currentPosition = parseInt(player.style.bottom) || 10;
            const newPosition = currentPosition + 10;
            player.style.bottom = `${newPosition}px`;

            const gameContainer = document.getElementById('game-container');
            if (newPosition >= gameContainer.offsetHeight - player.offsetHeight) {
                endGame(true);
            }
        }
    }
}

function startGame() {
    instructions.style.display = 'none';
    player.style.bottom = '10px';
    gameRunning = true;
    setLight(true);

    gameInterval = setInterval(() => {
        setLight(!greenLight);
    }, Math.random() * 2000 + 1000);
}

function endGame(win) {
    gameRunning = false;
    clearInterval(gameInterval);
    if (win) {
        status.textContent = 'You Win!';
        status.style.color = '#2ecc71';
    } else {
        status.textContent = 'You moved on Red Light! You lose!';
        player.style.backgroundColor = '#e74c3c';
    }
    instructions.style.display = 'block';
    startButton.textContent = 'Play Again';
}

startButton.addEventListener('click', startGame);
document.addEventListener('keydown', movePlayer);