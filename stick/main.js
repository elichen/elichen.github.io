// Main application logic

let environment;
let agent;
let isTraining = false;
let episodeCount = 0;
let totalReward = 0;

function initializeApp() {
    environment = new StickBalancingEnv();
    agent = new RLAgent(environment);
    
    setupEventListeners();
    resetEnvironment();
}

function setupEventListeners() {
    document.getElementById('startTraining').addEventListener('click', startTraining);
    document.getElementById('stopTraining').addEventListener('click', stopTraining);
    document.getElementById('resetEnvironment').addEventListener('click', resetEnvironment);
}

function startTraining() {
    if (!isTraining) {
        isTraining = true;
        trainLoop();
    }
}

function stopTraining() {
    isTraining = false;
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

        while (!done) {
            const action = await agent.selectAction(state);
            const [nextState, reward, stepDone] = environment.step(action);
            
            await agent.update(state, action, reward, nextState, done);
            
            state = nextState;
            episodeReward += reward;
            stepCount++;
            done = stepDone;
            
            drawEnvironment();
            await new Promise(resolve => setTimeout(resolve, 10)); // Small delay for visualization
        }

        episodeCount++;
        totalReward += episodeReward;
        updateStats();

        if (episodeCount % 10 === 0) {
            console.log(`Episode ${episodeCount}, Total Reward: ${totalReward}, Steps: ${stepCount}`);
        }
    }
}

function updateStats() {
    document.getElementById('episodeCount').textContent = episodeCount;
    document.getElementById('totalReward').textContent = totalReward.toFixed(2);
}

// Initialize the application when the window loads
window.onload = initializeApp;