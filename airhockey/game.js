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

const MAX_EPISODE_FRAMES = 1000;  // About 16-17 seconds at 60fps
let currentEpisodeFrames = 0;

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
    const scaleFactor = 0.5;
    
    // Convert continuous actions (-1 to 1) to paddle movement
    let dx = action[0] * paddle.speed * scaleFactor;
    let dy = action[1] * paddle.speed * scaleFactor;

    // Flip y direction for top player BEFORE momentum calculation
    if (isTopPlayer) {
        dy = -dy;
    }

    // Add momentum to smooth out movement
    paddle.dx = (paddle.dx || 0) * 0.8 + dx * 0.2;
    paddle.dy = (paddle.dy || 0) * 0.8 + dy * 0.2;

    // Apply momentum-based movement
    paddle.x += paddle.dx;
    paddle.y += paddle.dy;

    // Keep paddle in bounds
    paddle.x = Math.max(paddle.radius, Math.min(env.canvas.width - paddle.radius, paddle.x));
    const minY = isTopPlayer ? paddle.radius : env.canvas.height/2 + paddle.radius;
    const maxY = isTopPlayer ? env.canvas.height/2 - paddle.radius : env.canvas.height - paddle.radius;
    paddle.y = Math.max(minY, Math.min(maxY, paddle.y));
}

// Add this function to handle full episode reset
function resetEpisode() {
    episodeRewards = { top: 0, bottom: 0 };
    currentEpisodeFrames = 0;
    
    // Reset puck
    env.resetPuck();
    
    // Reset paddles to starting positions
    env.aiPaddle.x = env.canvas.width / 2;
    env.aiPaddle.y = 50;  // Original y position for top paddle
    env.playerPaddle.x = env.canvas.width / 2;
    env.playerPaddle.y = env.canvas.height - 50;  // Original y position for bottom paddle
    
    // Reset paddle momentum
    env.aiPaddle.dx = 0;
    env.aiPaddle.dy = 0;
    env.playerPaddle.dx = 0;
    env.playerPaddle.dy = 0;
}

function moveAI() {
    if (!agent) return;

    // Increment episode frame counter
    currentEpisodeFrames++;

    // Check if episode should end due to length
    const shouldEndEpisode = currentEpisodeFrames >= MAX_EPISODE_FRAMES;
    
    if (!isTrainingMode) {
        const state = agent.getState(env.puck, env.playerPaddle, env.aiPaddle, true, env.canvas.width, env.canvas.height);
        const result = agent.act(state);
        moveAgentPaddle(env.aiPaddle, result.action, true);
    } else {
        const topState = agent.getState(env.puck, env.playerPaddle, env.aiPaddle, true, env.canvas.width, env.canvas.height);
        const bottomState = agent.getState(env.puck, env.playerPaddle, env.aiPaddle, false, env.canvas.width, env.canvas.height);
        
        const topResult = agent.act(topState);
        const bottomResult = agent.act(bottomState);

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
        moveAgentPaddle(env.aiPaddle, topResult.action, true);
        moveAgentPaddle(env.playerPaddle, bottomResult.action, false);

        // Calculate new distances after moving
        const newTopDistance = Math.sqrt(
            Math.pow(env.puck.x - env.aiPaddle.x, 2) + 
            Math.pow(env.puck.y - env.aiPaddle.y, 2)
        );
        const newBottomDistance = Math.sqrt(
            Math.pow(env.puck.x - env.playerPaddle.x, 2) + 
            Math.pow(env.puck.y - env.playerPaddle.y, 2)
        );

        // Add small time penalty to encourage faster play
        const timePenalty = -0.001;

        // Initialize rewards based on distance improvement
        let reward = {
            top: (prevTopDistance > newTopDistance ? 0.1 : -0.1) + timePenalty,
            bottom: (prevBottomDistance > newBottomDistance ? 0.1 : -0.1) + timePenalty
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

        // Add episode timeout penalty
        if (shouldEndEpisode) {
            reward.top -= 0.5;
            reward.bottom -= 0.5;
        }

        // Store experiences
        if (previousState !== null) {
            agent.remember(
                previousState.top,
                previousAction.top,
                reward.top,
                previousValue.top,
                previousLogProb.top,
                goalHit !== false || shouldEndEpisode  // End episode on goal or timeout
            );

            agent.remember(
                previousState.bottom,
                previousAction.bottom,
                reward.bottom,
                previousValue.bottom,
                previousLogProb.bottom,
                goalHit !== false || shouldEndEpisode  // End episode on goal or timeout
            );
        }

        // Update previous states and actions
        previousState = { top: topState, bottom: bottomState };
        previousAction = { 
            top: topResult.action, 
            bottom: bottomResult.action 
        };
        previousValue = {
            top: topResult.value,
            bottom: bottomResult.value
        };
        previousLogProb = {
            top: topResult.logProb,
            bottom: bottomResult.logProb
        };

        // Training code
        agent.frameCount++;
        if (agent.frameCount % 128 === 0) {
            agent.train();
        }

        // Add rewards to episode total
        episodeRewards.top += reward.top;
        episodeRewards.bottom += reward.bottom;

        // Log metrics every 1000 steps
        if (agent.frameCount % 1000 === 0) {
            logTrainingMetrics();
        }

        // Reset episode when needed
        if (goalHit || shouldEndEpisode) {
            resetEpisode();
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
    agent = new PPOAgent(11, 2);  // 11 state inputs, 2 continuous actions (x, y movement)
    console.log("PPO AI initialized and ready for training!");
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
    resetEpisode();
    env.state.playerScore = 0;
    env.state.aiScore = 0;
    previousState = null;
    previousAction = null;
    
    console.log(`Switched to ${isTrainingMode ? 'Training' : 'Play'} mode`);
}

// Start the game
init();