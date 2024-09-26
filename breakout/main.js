const canvas = document.getElementById('gameCanvas');
canvas.width = 800;
canvas.height = 600;

const game = new Game(canvas);
const agent = new DQNAgent([42, 42], 3); // Updated state shape to [42, 42]
const visualization = new Visualization();

let isTraining = true;
let episode = 0;
let totalReward = 0;

const toggleButton = document.getElementById('toggleMode');
toggleButton.addEventListener('click', () => {
    isTraining = !isTraining;
    toggleButton.textContent = isTraining ? 'Switch to Test Mode' : 'Switch to Train Mode';
});

async function runEpisode() {
    try {
        game.reset();
        let state = game.getState();
        let done = false;
        totalReward = 0;
        const episodeMemory = [];

        while (!done) {
            const action = agent.act(state, isTraining);
            
            switch(action) {
                case 0: game.movePaddle('left'); break;
                case 1: game.movePaddle('right'); break;
                // case 2 is 'stay', so we don't need to do anything
            }

            game.update();
            const nextState = game.getState();
            const reward = game.score - totalReward; // Reward is the change in score
            totalReward = game.score;
            done = game.gameOver;

            if (isTraining) {
                episodeMemory.push([state, action, reward, nextState, done]);
            }

            state = nextState;

            if (!isTraining || done) {
                game.draw();
            }

            await new Promise(resolve => setTimeout(resolve, 10)); // Small delay to control game speed
        }

        // Train at the end of the episode
        if (isTraining) {
            await agent.trainOnEpisode(episodeMemory);
        }

        // Add console log for game over with final score
        console.log(`Game Over! Episode: ${episode + 1}, Final Score: ${totalReward}`);

        episode++;
        visualization.updateChart(episode, totalReward, agent.epsilon);

        if (isTraining) {
            runEpisode();
        } else {
            setTimeout(runEpisode, 1000); // Wait a second before starting a new episode in test mode
        }
    } catch (error) {
        console.error('Error in runEpisode:', error);
    }
}

runEpisode();