let env;
let agent;
let isTrainingMode = true;
let previousState = null;
let previousAction = null;
let episodeRewards = { top: 0, bottom: 0 };

// Mouse movement
let mouseX = 0;
let mouseY = 0;

// Constants
const ACTIONS = {
    STAY: 0,
    UP: 1,
    RIGHT: 2,
    DOWN: 3,
    LEFT: 4
};

function initializeGame() {
    const canvas = document.getElementById('gameCanvas');
    env = new AirHockeyEnvironment(canvas);
    
    canvas.addEventListener('mousemove', (e) => {
        const rect = canvas.getBoundingClientRect();
        mouseX = e.clientX - rect.left;
        mouseY = e.clientY - rect.top;
    });
}

function logTrainingMetrics() {
    console.log(`Step ${agent.frameCount}`);
    console.log(`Epsilon: ${agent.epsilon.toFixed(4)}`);
    console.log(`Episode Rewards - Top: ${episodeRewards.top.toFixed(2)}, Bottom: ${episodeRewards.bottom.toFixed(2)}`);
    console.log(`Game Score - Top: ${env.state.aiScore}, Bottom: ${env.state.playerScore}`);
    console.log('------------------------');
}

function moveAgentPaddle(paddle, action, isTopPlayer) {
    let dx = 0, dy = 0;
    
    switch(action) {
        case ACTIONS.UP: dy = -1; break;
        case ACTIONS.RIGHT: dx = 1; break;
        case ACTIONS.DOWN: dy = 1; break;
        case ACTIONS.LEFT: dx = -1; break;
    }

    // If top player, flip the y direction
    if (isTopPlayer) {
        dy = -dy;
    }

    if (dx !== 0 || dy !== 0) {
        paddle.x += dx * paddle.speed;
        paddle.y += dy * paddle.speed;
    }

    // Keep paddle in bounds
    paddle.x = Math.max(paddle.radius, Math.min(env.canvas.width - paddle.radius, paddle.x));
    const minY = isTopPlayer ? paddle.radius : env.canvas.height/2 + paddle.radius;
    const maxY = isTopPlayer ? env.canvas.height/2 - paddle.radius : env.canvas.height - paddle.radius;
    paddle.y = Math.max(minY, Math.min(maxY, paddle.y));
}

function moveAI() {
    if (!agent) return;

    if (!isTrainingMode) {
        // Use the agent in inference mode (training = false for no exploration)
        const state = agent.getState(env.puck, env.playerPaddle, env.aiPaddle, true, env.canvas.width, env.canvas.height);
        const action = agent.act(state, false);  // false = no exploration
        
        // Move AI paddle according to the agent's decision
        moveAgentPaddle(env.aiPaddle, action, true);
    } else {
        const topState = agent.getState(env.puck, env.playerPaddle, env.aiPaddle, true, env.canvas.width, env.canvas.height);
        const topAction = agent.act(topState, true);
        const bottomState = agent.getState(env.puck, env.playerPaddle, env.aiPaddle, false, env.canvas.width, env.canvas.height);
        const bottomAction = agent.act(bottomState, true);

        // Store current distances before moving
        const prevTopDistance = Math.sqrt(
            Math.pow(env.puck.x - env.aiPaddle.x, 2) + 
            Math.pow(env.puck.y - env.aiPaddle.y, 2)
        );
        const prevBottomDistance = Math.sqrt(
            Math.pow(env.puck.x - env.playerPaddle.x, 2) + 
            Math.pow(env.puck.y - env.playerPaddle.y, 2)
        );

        // Move paddles
        moveAgentPaddle(env.aiPaddle, topAction, true);
        moveAgentPaddle(env.playerPaddle, bottomAction, false);

        // Calculate new distances after moving
        const newTopDistance = Math.sqrt(
            Math.pow(env.puck.x - env.aiPaddle.x, 2) + 
            Math.pow(env.puck.y - env.aiPaddle.y, 2)
        );
        const newBottomDistance = Math.sqrt(
            Math.pow(env.puck.x - env.playerPaddle.x, 2) + 
            Math.pow(env.puck.y - env.playerPaddle.y, 2)
        );

        // Initialize rewards based on distance improvement
        let reward = {
            top: prevTopDistance > newTopDistance ? 0.1 : 0,
            bottom: prevBottomDistance > newBottomDistance ? 0.1 : 0
        };

        // Add goal rewards
        const goalHit = env.isInGoal();
        if (goalHit === 'top') {
            reward.bottom += 1.0;  // Positive reward for scoring
            reward.top -= 1.0;     // Negative reward for being scored on
        } else if (goalHit === 'bottom') {
            reward.top += 1.0;     // Positive reward for scoring
            reward.bottom -= 1.0;  // Negative reward for being scored on
        }

        // Add hit rewards
        if (env.checkCollision(env.puck, env.aiPaddle)) {
            reward.top += 0.1;
        }
        if (env.checkCollision(env.puck, env.playerPaddle)) {
            reward.bottom += 0.1;
        }

        // Store experiences
        if (previousState !== null) {
            agent.remember(
                previousState.top,
                previousAction.top,
                reward.top,
                topState,
                goalHit !== false
            );

            agent.remember(
                previousState.bottom,
                previousAction.bottom,
                reward.bottom,
                bottomState,
                goalHit !== false
            );
        }

        // Update previous states and actions
        previousState = { top: topState, bottom: bottomState };
        previousAction = { top: topAction, bottom: bottomAction };

        // Training code
        if (agent.frameCount % agent.updateFrequency === 0) {
            agent.train();
        }

        if (agent.frameCount % agent.targetUpdateFrequency === 0) {
            agent.updateTargetNetwork();
        }

        agent.frameCount++;

        // Add rewards to episode total
        episodeRewards.top += reward.top;
        episodeRewards.bottom += reward.bottom;

        // Log metrics every 1000 steps
        if (agent.frameCount % 1000 === 0) {
            logTrainingMetrics();
        }

        // Reset episode rewards when a goal is scored
        if (goalHit) {
            episodeRewards = { top: 0, bottom: 0 };
        }
    }
}

function gameLoop() {
    moveAI();
    const goalHit = env.update(mouseX, mouseY, isTrainingMode);
    env.draw();
    requestAnimationFrame(gameLoop);
}

async function initializeAI() {
    await tf.ready();
    agent = new DQNAgent(11, 5);  // 11 state inputs, 5 actions
    console.log("AI initialized and ready for training!");
}

async function init() {
    initializeGame();
    await initializeAI();
    env.resetPuck();
    env.state.playerScore = 0;
    env.state.aiScore = 0;
    gameLoop();
}

function toggleTrainingMode() {
    isTrainingMode = !isTrainingMode;
    
    // Reset game state when switching modes
    env.resetPuck();
    env.state.playerScore = 0;
    env.state.aiScore = 0;
    episodeRewards = { top: 0, bottom: 0 };
    previousState = null;
    previousAction = null;
    
    console.log(`Switched to ${isTrainingMode ? 'Training' : 'Play'} mode`);
}

// Start the game
init();