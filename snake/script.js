let game, agent, visualization;
const gridSize = 20;
const episodes = 1000;
const maxSteps = 1000;
const maxMovesWithoutReward = 100;
const testEpisodes = 100;

let isTrainingMode = true;

async function initializeTensorFlow() {
    await tf.ready();
    console.log('TensorFlow.js initialized');
}

async function initializeGame() {
    game = new SnakeGame('gameCanvas', gridSize);
    agent = new SnakeAgent(gridSize);
    if (!visualization) {
        visualization = new Visualization();
    } else {
        visualization.reset();
    }
    agent.setTestingMode(!isTrainingMode);
}

function toggleMode() {
    isTrainingMode = !isTrainingMode;
    const toggleButton = document.getElementById('toggleMode');
    toggleButton.textContent = isTrainingMode ? 'Switch to Testing' : 'Switch to Training';

    const statusText = document.getElementById('modeStatus');
    if (statusText) {
        statusText.textContent = `Mode: ${isTrainingMode ? 'Training' : 'Testing'}`;
    }

    agent.setTestingMode(!isTrainingMode);
    initializeGame();
}

async function runEpisode() {
    game.reset();
    let state = agent.getState(game);
    let totalReward = 0;
    let movesWithoutFood = 0;
    let foodEaten = 0;

    for (let step = 0; step < maxSteps; step++) {
        const action = agent.getAction(state);
        const { state: nextState, reward, done } = game.step(action);
        totalReward += reward;

        if (isTrainingMode) {
            agent.remember(state, action, reward, agent.getState(game), done);
            await agent.replay();
        }

        state = agent.getState(game);

        if (reward >= 10) {
            movesWithoutFood = 0;
            foodEaten++;
        } else {
            movesWithoutFood++;
        }

        if (!isTrainingMode || (isTrainingMode && done)) {
            game.draw();
            await new Promise(resolve => setTimeout(resolve, isTrainingMode ? 50 : 100));
        }

        if (movesWithoutFood >= maxMovesWithoutReward || done) {
            break;
        }
    }

    if (isTrainingMode) {
        agent.incrementEpisodeCount();
    }
    updateStats(agent.episodeCount, totalReward, agent.epsilon, foodEaten);
    visualization.updateCharts(agent.episodeCount, totalReward, agent.epsilon);

    return totalReward;
}

async function run() {
    while (true) {
        if (isTrainingMode) {
            if (agent.episodeCount >= episodes) {
                toggleMode();
                continue;
            }
        } else {
            if (agent.episodeCount >= episodes + testEpisodes) {
                console.log('Testing completed.');
                break;
            }
        }

        await runEpisode();
        await tf.nextFrame();
    }
}

function updateStats(episode, score, epsilon, foodEaten) {
    document.getElementById('episode').textContent = episode;
    document.getElementById('score').textContent = score.toFixed(2);
    document.getElementById('epsilon').textContent = epsilon.toFixed(4);
    document.getElementById('foodEaten').textContent = foodEaten;
}

document.getElementById('toggleMode').addEventListener('click', () => {
    toggleMode();
    if (!isTrainingMode) {
        run(); // Only start running if switching to testing mode
    }
});

window.addEventListener('load', async () => {
    await initializeTensorFlow();
    await initializeGame();
    run(); // Start training automatically when the page loads
});