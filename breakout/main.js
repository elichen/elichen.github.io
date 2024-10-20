const canvas = document.getElementById('gameCanvas');
canvas.width = 800;
canvas.height = 600;

const game = new Game(canvas);
const inputSize = 2 + 2 + (6 * 13 * 2); // paddle (2) + ball (2) + bricks (6 rows * 13 columns * 2 coordinates)
const agent = new DQNAgent(inputSize, 3);
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
        let loss = 0;

        while (!done) {
            const action = agent.act(state, isTraining);
            
            switch(action) {
                case 0: game.movePaddle('left'); break;
                case 1: game.movePaddle('right'); break;
                // case 2 is 'stay', so we don't need to do anything
            }

            game.update();
            const nextState = game.getState();
            const reward = game.getReward();
            totalReward += reward;
            done = game.gameOver;

            if (isTraining) {
                agent.remember(state, action, reward, nextState, done);
            }

            state = nextState;

            if (!isTraining || done) {
                game.draw();
            }

            await new Promise(resolve => setTimeout(resolve, 10));
        }

        console.log(`Game Over! Episode: ${episode + 1}, Final Score: ${totalReward}`);

        if (isTraining) {
            loss = await agent.replay();
            agent.incrementEpisode(); // Add this line to increment the episode count
        }

        episode++;
        visualization.updateChart(episode, totalReward, agent.epsilon, loss);

        if (isTraining) {
            runEpisode();
        } else {
            setTimeout(runEpisode, 1000);
        }
    } catch (error) {
        console.error('Error in runEpisode:', error);
    }
}

runEpisode();
