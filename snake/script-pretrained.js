let game, agent, visualization;
const gridSize = 20;
const maxSteps = 1000;
let gameSpeed = 100;
let isRunning = false;
let isTraining = false;

async function initializeTensorFlow() {
    await tf.ready();
    console.log('TensorFlow.js initialized');
}

async function initializeGame() {
    game = new SnakeGame('gameCanvas', gridSize);
    
    if (!agent) {
        agent = new SnakeAgent(gridSize);
        // Load pre-trained weights
        const loaded = await agent.loadPreTrainedModel();
        document.getElementById('modelStatus').textContent = loaded ? 'Pre-trained model loaded!' : 'Using random weights';
    }
    
    if (!visualization) {
        visualization = new Visualization();
    } else {
        visualization.reset();
    }
    
    agent.setTestingMode(!isTraining);
    game.draw();
}

async function runEpisode() {
    game.reset();
    let state = agent.getState(game);
    let totalReward = 0;
    let foodEaten = 0;

    for (let step = 0; step < maxSteps && isRunning; step++) {
        const action = agent.getAction(state);
        const { reward, done } = game.step(action);
        const nextState = agent.getState(game);
        totalReward += reward;

        if (isTraining) {
            agent.remember(state, action, reward, nextState, done);
            await agent.trainShortTerm(state, action, reward, nextState, done);
        }

        state = nextState;

        if (reward >= 1) {
            foodEaten++;
        }

        game.draw();
        await new Promise(resolve => setTimeout(resolve, gameSpeed));

        updateStats(agent.episodeCount, game.score, agent.epsilon, foodEaten);

        if (done) {
            break;
        }
    }

    if (isTraining) {
        await agent.replay();
        agent.incrementEpisodeCount();
    } else {
        agent.episodeCount++;
    }
    
    visualization.updateCharts(agent.episodeCount, game.score, agent.epsilon);

    return totalReward;
}

async function startDemo() {
    if (isRunning) return;
    
    isRunning = true;
    isTraining = false;
    agent.setTestingMode(true);
    document.getElementById('demoBtn').textContent = 'Stop Demo';
    document.getElementById('modeStatus').textContent = 'Mode: Demo';
    
    while (isRunning) {
        await runEpisode();
        await tf.nextFrame();
    }
}

async function startTraining() {
    if (isRunning) return;
    
    isRunning = true;
    isTraining = true;
    agent.setTestingMode(false);
    // Allow some exploration during continued training
    if (agent.epsilon < 0.1) {
        agent.epsilon = 0.1;
    }
    document.getElementById('trainBtn').textContent = 'Stop Training';
    document.getElementById('modeStatus').textContent = 'Mode: Training';
    
    while (isRunning) {
        await runEpisode();
        await tf.nextFrame();
    }
}

function stopRunning() {
    isRunning = false;
    document.getElementById('demoBtn').textContent = 'Start Demo';
    document.getElementById('trainBtn').textContent = 'Continue Training';
}

function updateStats(episode, score, epsilon, foodEaten) {
    document.getElementById('episode').textContent = episode;
    document.getElementById('score').textContent = score;
    document.getElementById('foodEaten').textContent = foodEaten;
}

// Event listeners
document.getElementById('demoBtn').addEventListener('click', () => {
    if (isRunning && !isTraining) {
        stopRunning();
    } else {
        stopRunning();
        startDemo();
    }
});

document.getElementById('trainBtn').addEventListener('click', () => {
    if (isRunning && isTraining) {
        stopRunning();
    } else {
        stopRunning();
        startTraining();
    }
});

document.getElementById('resetBtn').addEventListener('click', async () => {
    stopRunning();
    await initializeGame();
    updateStats(0, 0, agent.epsilon, 0);
});

document.getElementById('speedSlider').addEventListener('input', (e) => {
    gameSpeed = parseInt(e.target.value);
    document.getElementById('speedValue').textContent = gameSpeed;
});

// Initialize on load
window.addEventListener('load', async () => {
    await initializeTensorFlow();
    await initializeGame();
    updateStats(0, 0, agent.epsilon, 0);
    
    // Auto-start demo after a short delay
    setTimeout(() => {
        startDemo();
    }, 1000);
});