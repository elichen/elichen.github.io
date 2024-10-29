const canvas = document.getElementById('gameCanvas');
canvas.width = 800;
canvas.height = 600;

const game = new Game(canvas);
const inputSize = 5;  // [paddle_x, ball_x, ball_y, ball_dx, ball_dy]
const agent = new DQNAgent(inputSize, 3);
const humanPlayer = new HumanPlayer(game);

let isNeuralNetworkMode = true;
let episode = 0;
let totalReward = 0;
let isResetting = false;  // Add flag to track reset state

const toggleModeButton = document.getElementById('toggleMode');
toggleModeButton.addEventListener('click', toggleMode);

function toggleMode() {
    isNeuralNetworkMode = !isNeuralNetworkMode;
    toggleModeButton.textContent = isNeuralNetworkMode ? 'Switch to Human Mode' : 'Switch to Neural Network Mode';
    if (isNeuralNetworkMode) {
        runEpisode();
    } else {
        game.reset();
        game.draw();
    }
}

async function loadModelAndRun() {
    await agent.loadModel('./model.json');
    runEpisode();
}

async function runEpisode() {
    if (!isNeuralNetworkMode) return;

    game.reset();
    let state = game.getState();
    let done = false;
    totalReward = 0;

    while (!done && isNeuralNetworkMode) {
        const action = agent.act(state);
        
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

        state = nextState;
        game.draw();

        await new Promise(resolve => setTimeout(resolve, 10));
    }

    console.log(`Game Over! Episode: ${episode + 1}, Final Score: ${totalReward}`);
    episode++;

    if (isNeuralNetworkMode) {
        setTimeout(runEpisode, 1000);
    }
}

loadModelAndRun();

// Game loop for human mode
function gameLoop() {
    if (!isNeuralNetworkMode) {
        if (!isResetting) {  // Only update and draw if not in reset state
            humanPlayer.checkKeys();  // Add this line to check keys each frame
            game.update();
            game.draw();
            
            // Check if game is over in human mode
            if (game.gameOver) {
                isResetting = true;  // Set reset flag
                setTimeout(() => {
                    game.reset();
                    game.draw();
                    isResetting = false;  // Clear reset flag
                }, 1000);
            }
        }
    }
    requestAnimationFrame(gameLoop);
}

gameLoop();
