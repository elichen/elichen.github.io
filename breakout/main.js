const canvas = document.getElementById('gameCanvas');
canvas.width = 800;
canvas.height = 600;

const game = new Game(canvas);
const humanPlayer = new HumanPlayer(game);

const messageElement = document.getElementById('message');

function showMessage(text) {
    messageElement.textContent = text;
    messageElement.style.display = 'block';
}

function hideMessage() {
    messageElement.style.display = 'none';
}

function gameLoop() {
    game.update();
    game.draw();

    if (game.gameOver) {
        showMessage(`Game Over! Your score: ${game.score}\nPress any key to restart`);
        document.addEventListener('keydown', restartGame, { once: true });
    } else {
        requestAnimationFrame(gameLoop);
    }
}

function restartGame() {
    hideMessage();
    game.reset();
    gameLoop();
}

gameLoop();