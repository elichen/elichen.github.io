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
    
    // Initialize the agent only once
    if (!agent) {
        agent = new SnakeAgent(gridSize);
    }
    
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

    // Set the testing mode without reinitializing the agent
    agent.setTestingMode(!isTrainingMode);
}

async function runEpisode() {
    game.reset();
    let state = agent.getState(game); // Initial flat state array
    let totalReward = 0;
    let movesWithoutFood = 0;
    let foodEaten = 0;

    for (let step = 0; step < maxSteps; step++) {
        const action = agent.getAction(state);
        const { reward, done } = game.step(action); // Retrieve only reward and done
        const nextState = agent.getState(game); // Get new flat state array
        totalReward += reward;

        if (isTrainingMode) {
            agent.remember(state, action, reward, nextState, done);
            await agent.trainShortTerm(state, action, reward, nextState, done);
        }

        state = nextState; // Update state for the next step

        if (reward >= 1) {
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
        await agent.replay(); // Perform long-term training after the episode
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
});

window.addEventListener('load', async () => {
    await initializeTensorFlow();
    // Initialize the agent and game once
    if (!agent) {
        agent = new SnakeAgent(gridSize);
    }
    await initializeGame();
    run(); // Start training automatically when the page loads
});