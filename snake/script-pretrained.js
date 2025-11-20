const GRID_SIZE = 20;
const STACK_SIZE = 4;
const STEP_DELAY = 30;

let game;
let agent;
let episode = 1;
let foodEaten = 0;

async function runEpisodes() {
    while (true) {
        game.reset();
        agent.bootstrap(game);
        foodEaten = 0;
        updateStats(episode, game.score, foodEaten);

        while (!game.gameOver) {
            const action = agent.predictAction(game);
            const result = game.step(action);
            if (result.reward > 0.9) {
                foodEaten++;
            }
            const obs = agent.computeObservation(game);
            agent.updateStack(obs.board, obs.stats);
            updateStats(episode, game.score, foodEaten);
            await new Promise(resolve => setTimeout(resolve, STEP_DELAY));
        }
        episode++;
        await tf.nextFrame();
    }
}

function updateStats(ep, score, food) {
    document.getElementById('episode').textContent = ep;
    document.getElementById('score').textContent = score;
    document.getElementById('foodEaten').textContent = food;
}

window.addEventListener('load', async () => {
    document.getElementById('modelStatus').textContent = 'Loading TensorFlow.js...';
    await tf.ready();
    game = new SnakeGame('gameCanvas', GRID_SIZE);
    agent = new PPOWebAgent(GRID_SIZE, STACK_SIZE);
    document.getElementById('modelStatus').textContent = 'Loading policy weights...';
    await agent.load();
    document.getElementById('modelStatus').textContent = 'Model loaded – running demo';
    await runEpisodes();
});
