// Main script for curriculum-trained Snake AI

let game;
let totalFoodEaten = 0;
let episodeCount = 1;
let gameLoopId = null;
const GAME_SPEED = 50; // ms per frame

async function init() {
    document.getElementById('modelStatus').textContent = 'Loading model...';

    game = new SnakeGame('gameCanvas', 20);
    game.maxMovesWithoutFood = 200;
    game.draw();

    try {
        await agent.load('web_model_new/weights.json');
        agent.reset(game);
        document.getElementById('modelStatus').textContent = 'Model loaded! Starting game...';
        setTimeout(startGame, 500);
    } catch (error) {
        console.error('Failed to load model:', error);
        document.getElementById('modelStatus').textContent = 'Failed to load model: ' + error.message;
    }
}

function startGame() {
    document.getElementById('modelStatus').textContent = '';
    gameLoop();
}

function gameLoop() {
    if (game.gameOver) {
        // Reset and start new episode
        episodeCount++;
        document.getElementById('episode').textContent = episodeCount;
        document.getElementById('foodEaten').textContent = 0;
        totalFoodEaten = 0;
        game.reset();
        agent.reset(game);  // Sync agent direction with game's random initial direction
    }

    // Get action from agent
    const action = agent.predictAction(game);

    // Take step
    const result = game.step(action);

    // Update stats
    document.getElementById('score').textContent = game.score;
    if (result.reward > 0) {
        totalFoodEaten++;
        document.getElementById('foodEaten').textContent = totalFoodEaten;
    }

    // Schedule next frame
    gameLoopId = setTimeout(gameLoop, GAME_SPEED);
}

// Start when page loads
window.addEventListener('load', init);
