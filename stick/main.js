// Main application logic

let environment;
let agent;
let isTraining = true;
let episodeCount = 0;

function initializeApp() {
    environment = new StickBalancingEnv();
    agent = new PolicyGradientAgent(environment);
    
    setupEventListeners();
    resetEnvironment();
    initializeMetricsChart();
    updateModeDisplay();
    startTraining();
}

function setupEventListeners() {
    document.getElementById('toggleMode').addEventListener('click', toggleMode);
}

function toggleMode() {
    isTraining = !isTraining;
    updateModeDisplay();
    if (isTraining) {
        startTraining();
    } else {
        stopTraining();
    }
}

function startTraining() {
    trainLoop();
}

function stopTraining() {
    // Stop the training loop, but continue running in testing mode
    runTestingLoop();
}

function resetEnvironment() {
    environment.reset();
    agent.reset();
    episodeCount = 0;
    updateStats();
    drawEnvironment();
}

async function trainLoop() {
    while (isTraining) {
        let state = environment.reset();
        let episodeReward = 0;
        let done = false;
        let stepCount = 0;

        // Run episode
        while (!done && stepCount < 500) {
            const action = await agent.selectAction(state);
            const [nextState, reward, stepDone] = environment.step(action);
            
            await agent.update(state, action, reward, nextState, stepDone);
            
            state = nextState;
            episodeReward += reward;
            stepCount++;
            done = stepDone;
        }

        // Draw final state
        drawEnvironment();

        episodeCount++;
        updateStats();
        updateMetricsChart(episodeCount, episodeReward);

        if (episodeCount % 10 === 0) {
            console.log(`Episode ${episodeCount}, Episode Reward: ${episodeReward.toFixed(2)}, Steps: ${stepCount}`);
        }

        await new Promise(resolve => setTimeout(resolve, 0));
    }
}

async function runTestingLoop() {
    while (!isTraining) {
        let state = environment.reset();
        let episodeReward = 0;
        let done = false;
        let stepCount = 0;

        while (!done) {
            const action = await agent.selectAction(state, true);
            const [nextState, reward, stepDone] = environment.step(action);
            
            state = nextState;
            episodeReward += reward;
            stepCount++;
            done = stepDone;
            
            drawEnvironment();
            await new Promise(resolve => setTimeout(resolve, 10));
        }

        episodeCount++;
        updateStats();
        updateMetricsChart(episodeCount, episodeReward);

        if (episodeCount % 10 === 0) {
            console.log(`Test Episode ${episodeCount}, Episode Reward: ${episodeReward.toFixed(2)}, Steps: ${stepCount}`);
        }
    }
}

function updateModeDisplay() {
    const modeButton = document.getElementById('toggleMode');
    modeButton.textContent = isTraining ? 'Switch to Testing' : 'Switch to Training';
    document.getElementById('currentMode').textContent = isTraining ? 'Training' : 'Testing';
}

function updateStats() {
    document.getElementById('episodeCount').textContent = episodeCount;
}

// Initialize the application when the window loads
window.onload = initializeApp;