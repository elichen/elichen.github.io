// Bootstrap for the cont_ctrl_s202r model (live, in-browser, deterministic argmax).

let game;
let maxScore = 0;
let episodeCount = 1;
let winCount = 0;
const TRAIL_LEN = 9;         // head positions kept for the glowing comet wake
let headTrail = [];

function pushTrail(head) {
    headTrail.push({ x: head.x, y: head.y });
    if (headTrail.length > TRAIL_LEN) headTrail.shift();
}
const STEPS_PER_FRAME = 5;   // advance several steps per animation tick so wins are watchable
const FRAME_MS = 16;
const WIN_PAUSE_MS = 1500;   // hold the filled board briefly on a win

// Match the training env's no-food limit: max(80 + 4*length, 2*n*n).
function starvationLimit(g) {
    return Math.max(80 + 4 * g.snake.length, 2 * g.gridSize * g.gridSize);
}

async function init() {
    const loadingOverlay = document.getElementById('loadingOverlay');
    game = new SnakeGame('gameCanvas', 20);
    game.maxMovesWithoutFood = starvationLimit(game);
    game.draw();

    try {
        await agent.load('web_model_v2/weights.bin');
        agent.reset(game);
        headTrail = [{ ...game.snake[0] }];
        loadingOverlay.classList.add('hidden');
        setTimeout(loop, 200);
    } catch (error) {
        console.error('Failed to load model:', error);
        loadingOverlay.querySelector('.loading-text').textContent = 'Failed to load model';
        loadingOverlay.querySelector('.loading-subtext').textContent = error.message;
        loadingOverlay.querySelector('.loading-spinner').style.display = 'none';
    }
}

function refreshStats() {
    document.getElementById('score').textContent = game.score;
    document.getElementById('episode').textContent = episodeCount;
    document.getElementById('maxScore').textContent = maxScore;
    const winEl = document.getElementById('winCount');
    if (winEl) winEl.textContent = winCount;
    const rateEl = document.getElementById('winRate');
    if (rateEl) rateEl.textContent = episodeCount > 1
        ? Math.round(100 * winCount / (episodeCount - 1)) + '%' : '—';
}

function newEpisode() {
    episodeCount++;
    game.reset();
    game.maxMovesWithoutFood = starvationLimit(game);
    agent.reset(game);
    headTrail = [{ ...game.snake[0] }];
    refreshStats();
}

function loop() {
    for (let i = 0; i < STEPS_PER_FRAME; i++) {
        if (game.gameOver) {
            if (game.score > maxScore) maxScore = game.score;
            if (game.won) winCount++;
            refreshStats();
            const pause = game.won ? WIN_PAUSE_MS : 250;
            game.draw();
            game.drawActionTrail(headTrail);
            setTimeout(() => { newEpisode(); setTimeout(loop, FRAME_MS); }, pause);
            return;
        }
        game.maxMovesWithoutFood = starvationLimit(game);
        const action = agent.predictAction(game);
        game.step(action);
        pushTrail(game.snake[0]);
    }
    refreshStats();
    game.drawActionTrail(headTrail);
    setTimeout(loop, FRAME_MS);
}

window.addEventListener('load', init);
