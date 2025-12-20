// Main script for egocentric FC Snake AI

let game;
let maxScore = 0;
let episodeCount = 1;
let gameLoopId = null;
const GAME_SPEED = 50; // ms per frame

async function init() {
    const loadingOverlay = document.getElementById('loadingOverlay');

    game = new SnakeGame('gameCanvas', 20);
    game.maxMovesWithoutFood = 200;
    game.draw();

    try {
        await agent.load('web_model_egocentric/weights.json');
        agent.reset(game);
        loadingOverlay.classList.add('hidden');
        setTimeout(startGame, 100);
    } catch (error) {
        console.error('Failed to load model:', error);
        loadingOverlay.querySelector('.loading-text').textContent = 'Failed to load model';
        loadingOverlay.querySelector('.loading-subtext').textContent = error.message;
        loadingOverlay.querySelector('.loading-spinner').style.display = 'none';
    }
}

function startGame() {
    gameLoop();
}

function gameLoop() {
    if (game.gameOver) {
        // Update max score before reset
        if (game.score > maxScore) {
            maxScore = game.score;
            document.getElementById('maxScore').textContent = maxScore;
        }
        // Reset and start new episode
        episodeCount++;
        document.getElementById('episode').textContent = episodeCount;
        game.reset();
        agent.reset(game);  // Sync agent direction with game's random initial direction
    }

    // Get action from agent
    const action = agent.predictAction(game);

    // Take step
    const result = game.step(action);

    // Update stats
    document.getElementById('score').textContent = game.score;

    // Schedule next frame
    gameLoopId = setTimeout(gameLoop, GAME_SPEED);
}

// Start when page loads
window.addEventListener('load', init);
