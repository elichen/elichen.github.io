let game, agent, visualization;
const gridSize = 20;
const episodes = 1000;
const maxSteps = 1000;
const maxMovesWithoutReward = 100; // New constant for max moves without reward
const testEpisodes = 100;

async function initializeTensorFlow() {
    await tf.ready();
    console.log('TensorFlow.js initialized');
}

async function initializeGame() {
    await initializeTensorFlow();
    game = new SnakeGame('gameCanvas', gridSize);
    agent = new SnakeAgent(gridSize);
    if (!visualization) {
        visualization = new Visualization();
    } else {
        visualization.reset();
    }
}

async function train() {
    for (let episode = 0; episode < episodes; episode++) {
        game.reset();
        let state = agent.getState(game);
        let totalReward = 0;
        let movesWithoutFood = 0; // Counter for moves without eating food

        for (let step = 0; step < maxSteps; step++) {
            const action = agent.getAction(state);
            const { state: nextState, reward, done } = game.step(action);
            totalReward += reward;

            agent.remember(state, action, reward, agent.getState(game), done);
            state = agent.getState(game);

            try {
                await agent.replay();
            } catch (error) {
                console.error('Error during replay:', error);
            }

            game.draw(); // Draw the game state at each step
            await new Promise(resolve => setTimeout(resolve, 50)); // Add a small delay to make the movement visible

            // Check if the snake ate food
            if (reward >= 10) { // Assuming 10 is the reward for eating food
                movesWithoutFood = 0; // Reset the counter if food was eaten
            } else {
                movesWithoutFood++; // Increment the counter if no food was eaten
            }

            // Check if max moves without food has been reached
            if (movesWithoutFood >= maxMovesWithoutReward) {
                console.log(`Episode ${episode} terminated due to lack of progress`);
                break; // End the episode
            }

            if (done) break;
        }

        updateStats(episode, totalReward, agent.epsilon);
        visualization.updateCharts(episode, totalReward, agent.epsilon);
        await tf.nextFrame(); // Allow UI to update
    }
}

async function test() {
    agent.setTestingMode(true);
    let totalScore = 0;

    for (let episode = 0; episode < testEpisodes; episode++) {
        game.reset();
        let state = agent.getState(game);
        let episodeScore = 0;

        for (let step = 0; step < maxSteps; step++) {
            const action = agent.getAction(state);
            const { state: nextState, reward, done } = game.step(action);
            episodeScore += reward;
            state = agent.getState(game);

            game.draw(); // Ensure the game is visually updated
            await new Promise(resolve => setTimeout(resolve, 50)); // Slow down the game for visibility

            if (done) break;
        }

        totalScore += episodeScore;
        updateStats(episode, episodeScore, 0);
        visualization.updateCharts(episode, episodeScore, 0);
        await tf.nextFrame();
    }

    const averageScore = totalScore / testEpisodes;
    console.log(`Testing completed. Average score: ${averageScore.toFixed(2)}`);
    agent.setTestingMode(false);
}

function updateStats(episode, score, epsilon) {
    document.getElementById('episode').textContent = episode;
    document.getElementById('score').textContent = score.toFixed(2);
    document.getElementById('epsilon').textContent = epsilon.toFixed(4);
}

document.getElementById('startTraining').addEventListener('click', async () => {
    try {
        await initializeGame();
        await train();
    } catch (error) {
        console.error('Error during training:', error);
        // Handle the error appropriately
    }
});

document.getElementById('startTesting').addEventListener('click', async () => {
    if (!agent) {
        alert('Please train the agent first!');
        return;
    }
    try {
        await test();
    } catch (error) {
        console.error('Error during testing:', error);
        // Handle the error appropriately
    }
});

// Initialize TensorFlow.js and the game when the page loads
window.addEventListener('load', async () => {
    await initializeTensorFlow();
    await initializeGame();
});