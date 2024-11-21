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

const MAX_EPISODE_FRAMES = 5000;
let currentEpisodeFrames = 0;

const CURRICULUM = {
    STAGE_1: 'HIT_PUCK',
    STAGE_2: 'SCORE_GOAL',
    STAGE_3: 'STRATEGY'
};

// Add curriculum tracking
let currentStage = CURRICULUM.STAGE_1;
let successfulHits = 0;
let successfulGoals = 0;
const HITS_TO_ADVANCE = 1000;  // Number of successful hits to move to stage 2
const GOALS_TO_ADVANCE = 100;  // Number of goals to move to stage 3

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
    console.log(`Current Stage: ${currentStage}`);
    console.log(`Successful Hits: ${successfulHits}`);
    console.log(`Successful Goals: ${successfulGoals}`);
    console.log(`Episode Rewards - Top: ${episodeRewards.top.toFixed(2)}, Bottom: ${episodeRewards.bottom.toFixed(2)}`);
    console.log(`Game Score - Top: ${env.state.aiScore}, Bottom: ${env.state.playerScore}`);
    console.log('------------------------');
}

function moveAgentPaddle(paddle, action, isTopPlayer) {
    const scaleFactor = 1.0;
    
    // Convert continuous actions (-1 to 1) to paddle movement
    let dx = action[0] * paddle.speed * scaleFactor;
    let dy = action[1] * paddle.speed * scaleFactor;

    // Flip y direction for top player BEFORE momentum calculation
    if (isTopPlayer) {
        dy = -dy;
    }

    // Add momentum with higher responsiveness
    paddle.dx = (paddle.dx || 0) * 0.6 + dx * 0.4;
    paddle.dy = (paddle.dy || 0) * 0.6 + dy * 0.4;

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

function calculateRewards(prevDistances, newDistances, goalHit, hitPuck) {
    const timePenalty = -0.001;
    let reward = { top: timePenalty, bottom: timePenalty };

    switch(currentStage) {
        case CURRICULUM.STAGE_1:  // Focus on hitting the puck
            // Strong rewards for getting closer to puck
            reward.top += (prevDistances.top > newDistances.top ? 0.5 : -0.2);
            reward.bottom += (prevDistances.bottom > newDistances.bottom ? 0.5 : -0.2);
            
            // Very strong reward for hitting puck
            if (hitPuck.top) {
                reward.top += 2.0;
                successfulHits++;
            }
            if (hitPuck.bottom) {
                reward.bottom += 2.0;
                successfulHits++;
            }

            // Check for advancement
            if (successfulHits >= HITS_TO_ADVANCE) {
                currentStage = CURRICULUM.STAGE_2;
                console.log("Advancing to SCORE_GOAL stage!");
            }
            break;

        case CURRICULUM.STAGE_2:  // Focus on scoring
            // Moderate rewards for puck control
            reward.top += (prevDistances.top > newDistances.top ? 0.2 : -0.1);
            reward.bottom += (prevDistances.bottom > newDistances.bottom ? 0.2 : -0.1);
            
            // Reward hitting toward opponent's goal
            if (hitPuck.top) {
                const towardGoal = env.puck.dy > 0;  // Top player wants positive dy
                reward.top += towardGoal ? 1.0 : 0.2;
            }
            if (hitPuck.bottom) {
                const towardGoal = env.puck.dy < 0;  // Bottom player wants negative dy
                reward.bottom += towardGoal ? 1.0 : 0.2;
            }

            // Strong reward for scoring
            if (goalHit === 'top') {
                reward.bottom += 5.0;
                successfulGoals++;
            } else if (goalHit === 'bottom') {
                reward.top += 5.0;
                successfulGoals++;
            }

            // Check for advancement
            if (successfulGoals >= GOALS_TO_ADVANCE) {
                currentStage = CURRICULUM.STAGE_3;
                console.log("Advancing to STRATEGY stage!");
            }
            break;

        case CURRICULUM.STAGE_3:  // Focus on strategy
            // Small rewards for puck control
            reward.top += (prevDistances.top > newDistances.top ? 0.1 : -0.05);
            reward.bottom += (prevDistances.bottom > newDistances.bottom ? 0.1 : -0.05);
            
            // Moderate rewards for good hits
            if (hitPuck.top) {
                const towardGoal = env.puck.dy > 0;
                reward.top += towardGoal ? 0.5 : 0.1;
            }
            if (hitPuck.bottom) {
                const towardGoal = env.puck.dy < 0;
                reward.bottom += towardGoal ? 0.5 : 0.1;
            }

            // Strong rewards for scoring and defense
            if (goalHit === 'top') {
                reward.bottom += 2.0;  // Scoring
                reward.top -= 1.0;     // Failed defense
            } else if (goalHit === 'bottom') {
                reward.top += 2.0;     // Scoring
                reward.bottom -= 1.0;  // Failed defense
            }
            break;
    }

    return reward;
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

        // Check for goals after movement
        const goalHit = env.isInGoal();

        // Calculate new distances after moving
        const newTopDistance = Math.sqrt(
            Math.pow(env.puck.x - env.aiPaddle.x, 2) + 
            Math.pow(env.puck.y - env.aiPaddle.y, 2)
        );
        const newBottomDistance = Math.sqrt(
            Math.pow(env.puck.x - env.playerPaddle.x, 2) + 
            Math.pow(env.puck.y - env.playerPaddle.y, 2)
        );

        const hitPuck = {
            top: env.checkCollision(env.puck, env.aiPaddle),
            bottom: env.checkCollision(env.puck, env.playerPaddle)
        };

        const prevDistances = {
            top: prevTopDistance,
            bottom: prevBottomDistance
        };

        const newDistances = {
            top: newTopDistance,
            bottom: newBottomDistance
        };

        const reward = calculateRewards(prevDistances, newDistances, goalHit, hitPuck);

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