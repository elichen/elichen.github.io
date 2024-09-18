// Main application logic

let environment;
let agent;
let isTraining = true; // New variable to track the current mode
let episodeCount = 0;
let totalReward = 0;

function initializeApp() {
    environment = new StickBalancingEnv();
    agent = new RLAgent(environment);
    
    setupEventListeners();
    resetEnvironment();
    initializeMetricsChart(); // Initialize the metrics chart
    updateModeDisplay(); // New function to update mode display
    startTraining(); // Start in training mode by default
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
    totalReward = 0;
    updateStats();
    drawEnvironment();
}

async function trainLoop() {
    while (isTraining) {
        let state = environment.reset();
        let episodeReward = 0;
        let done = false;
        let stepCount = 0;

        while (!done && stepCount < 500) { // Add a maximum step count to prevent very long episodes
            const action = await agent.selectAction(state);
            const [nextState, reward, stepDone] = environment.step(action);
            
            await agent.update(state, action, reward, nextState, stepDone);
            
            state = nextState;
            episodeReward += reward;
            stepCount++;
            done = stepDone;
            
            drawEnvironment();
            // Remove the delay to speed up training
            // await new Promise(resolve => setTimeout(resolve, 10));
        }

        episodeCount++;
        totalReward += episodeReward;
        updateStats();
        updateMetricsChart(episodeCount, episodeReward, agent.epsilon);

        if (episodeCount % 10 === 0) {
            console.log(`Episode ${episodeCount}, Total Reward: ${totalReward}, Steps: ${stepCount}, Epsilon: ${agent.epsilon.toFixed(4)}`);
        }
    }
}

async function runTestingLoop() {
    while (!isTraining) {
        let state = environment.reset();
        let episodeReward = 0;
        let done = false;
        let stepCount = 0;

        while (!done) {
            const action = await agent.selectAction(state, true); // Pass true for testing mode
            const [nextState, reward, stepDone] = environment.step(action);
            
            state = nextState;
            episodeReward += reward;
            stepCount++;
            done = stepDone;
            
            drawEnvironment();
            await new Promise(resolve => setTimeout(resolve, 10));
        }

        episodeCount++;
        totalReward += episodeReward;
        updateStats();
        updateMetricsChart(episodeCount, episodeReward, 0); // Epsilon is always 0 in testing mode

        if (episodeCount % 10 === 0) {
            console.log(`Test Episode ${episodeCount}, Total Reward: ${totalReward}, Steps: ${stepCount}`);
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
    document.getElementById('totalReward').textContent = totalReward.toFixed(2);
}

// Initialize the application when the window loads
window.onload = initializeApp;