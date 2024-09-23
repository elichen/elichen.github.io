let game, agent, visualization;
const gridSize = 20;
const episodes = 1000;
const maxSteps = 1000;
const maxMovesWithoutReward = 100; // New constant for max moves without reward
const testEpisodes = 100;

// State variable for visualization
let isVisualizationOn = true;

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

// Function to toggle visualization
function toggleVisualization() {
    isVisualizationOn = !isVisualizationOn;
    const toggleButton = document.getElementById('toggleVisualization');
    toggleButton.textContent = isVisualizationOn ? 'Disable Visualization' : 'Enable Visualization';

    // Update visualization status text
    const statusText = document.getElementById('visualizationStatus');
    if (statusText) {
        statusText.textContent = `Visualization: ${isVisualizationOn ? 'On' : 'Off'}`;
    }
}

async function train() {
    for (let episode = 0; episode < episodes; episode++) {
        game.reset();
        let state = agent.getState(game);
        let totalReward = 0;
        let movesWithoutFood = 0;
        let foodEaten = 0;
        let totalFoodEaten = 0;

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

            if (isVisualizationOn) {
                game.draw();
                await new Promise(resolve => setTimeout(resolve, 50)); // Delay for visualization
            }

            if (reward >= 10) { // Assuming 10 is the reward for eating food
                movesWithoutFood = 0;
                foodEaten++;
                totalFoodEaten++;
                console.log(`Episode ${episode}: Food eaten! Total in this episode: ${foodEaten}`);
            } else {
                movesWithoutFood++;
            }

            if (movesWithoutFood >= maxMovesWithoutReward) {
                console.log(`Episode ${episode} terminated due to lack of progress. Moves without food: ${movesWithoutFood}, Food eaten this episode: ${foodEaten}, Total food eaten: ${totalFoodEaten}`);
                break;
            }

            if (done) {
                console.log(`Episode ${episode} completed. Food eaten this episode: ${foodEaten}, Total food eaten: ${totalFoodEaten}`);
                break;
            }
        }

        agent.incrementEpisodeCount();

        // Update statistics
        updateStats(episode, totalReward, agent.epsilon, foodEaten, totalFoodEaten);
        
        if (visualization) { // Ensure visualization exists
            console.log(`Updating charts for episode ${episode}`);
            visualization.updateCharts(episode, totalReward, agent.epsilon);
        }

        await tf.nextFrame();

        // Yield control to the browser to handle UI updates if visualization is on
        if (isVisualizationOn) {
            await new Promise(resolve => setTimeout(resolve, 10));
        }
    }

    console.log('Training completed.');
}

function updateStats(episode, score, epsilon, foodEaten, totalFoodEaten) {
    updateElementText('episode', episode);
    updateElementText('score', score.toFixed(2));
    updateElementText('epsilon', epsilon.toFixed(4));
    updateElementText('foodEaten', foodEaten);
    updateElementText('totalFoodEaten', totalFoodEaten);
}

function updateElementText(id, value) {
    const element = document.getElementById(id);
    if (element) {
        element.textContent = value;
    }
}

// Add event listener for the toggle visualization button
document.getElementById('toggleVisualization').addEventListener('click', toggleVisualization);

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