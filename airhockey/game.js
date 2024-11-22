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
const HITS_TO_ADVANCE = 500;  // Number of successful hits to move to stage 2
const GOALS_TO_ADVANCE = 50;  // Number of goals to move to stage 3

let trainInterval = 1000; // Number of timesteps between training sessions

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

function resetEnvironment() {
    episodeRewards = { top: 0, bottom: 0 };
    currentEpisodeFrames = 0;

    // Reset environment without resetting the agent's internal state
    env.resetPuck();
    env.state.playerScore = 0;
    env.state.aiScore = 0;
    env.aiPaddle.x = env.canvas.width / 2;
    env.aiPaddle.y = 50;
    env.playerPaddle.x = env.canvas.width / 2;
    env.playerPaddle.y = env.canvas.height - 50;
    env.aiPaddle.dx = 0;
    env.aiPaddle.dy = 0;
    env.playerPaddle.dx = 0;
    env.playerPaddle.dy = 0;

    previousState = null;
    previousAction = null;
    previousValue = null;
    previousLogProb = null;
}

function calculateRewards(prevDistances, newDistances, goalHit, hitPuck) {
    const timePenalty = -0.001;
    let reward = { top: timePenalty, bottom: timePenalty };

    // Define stronger rewards for scoring goals
    const goalReward = 10.0;
    const goalPenalty = -10.0;

    // Add position-based defensive reward
    const defensivePositionReward = (paddle, puck, isTop) => {
        // Calculate ideal defensive position (between puck and goal)
        const goalY = isTop ? 0 : env.canvas.height;
        const idealX = puck.x;
        const idealY = isTop ? 
            Math.min(puck.y - 50, env.canvas.height/2 - paddle.radius) : 
            Math.max(puck.y + 50, env.canvas.height/2 + paddle.radius);
        
        // Calculate distance to ideal position
        const distToIdeal = Math.sqrt(
            Math.pow(paddle.x - idealX, 2) + 
            Math.pow(paddle.y - idealY, 2)
        );
        
        // Normalize and return reward
        return Math.max(0, 1 - (distToIdeal / (env.canvas.height/2)));
    };

    switch(currentStage) {
        case CURRICULUM.STAGE_1:  // Focus on hitting the puck
            // Strong rewards for reducing distance to puck
            reward.top += (prevDistances.top > newDistances.top ? 0.1 : -0.1);
            reward.bottom += (prevDistances.bottom > newDistances.bottom ? 0.1 : -0.1);
            
            // Strong reward for hitting puck
            if (hitPuck.top) {
                reward.top += 1.0;
                successfulHits++;
                console.log('AI Paddle hit the puck!');
            }
            if (hitPuck.bottom) {
                reward.bottom += 1.0;
                successfulHits++;
                console.log('Player Paddle hit the puck!');
            }

            if (successfulHits >= HITS_TO_ADVANCE) {
                currentStage = CURRICULUM.STAGE_2;
                console.log("Advancing to SCORE_GOAL stage!");
            }
            break;

        case CURRICULUM.STAGE_2:  // Focus on scoring goals
            // Basic rewards for hitting the puck
            if (hitPuck.top) {
                const towardGoal = env.puck.dy > 0;
                reward.top += towardGoal ? 2.0 : 0.5;
            }
            if (hitPuck.bottom) {
                const towardGoal = env.puck.dy < 0;
                reward.bottom += towardGoal ? 2.0 : 0.5;
            }

            // Provide stronger rewards for scoring
            if (goalHit === 'top') {
                reward.bottom += goalReward;
                reward.top += goalPenalty;
                successfulGoals++;
            } else if (goalHit === 'bottom') {
                reward.top += goalReward;
                reward.bottom += goalPenalty;
                successfulGoals++;
            }

            if (successfulGoals >= GOALS_TO_ADVANCE) {
                currentStage = CURRICULUM.STAGE_3;
                console.log("Advancing to STRATEGY stage!");
            }
            break;

        case CURRICULUM.STAGE_3:  // Advanced strategies
            // Defensive positioning reward
            const topDefenseReward = defensivePositionReward(env.aiPaddle, env.puck, true);
            const bottomDefenseReward = defensivePositionReward(env.playerPaddle, env.puck, false);
            reward.top += topDefenseReward * 0.3;
            reward.bottom += bottomDefenseReward * 0.3;

            // Offensive rewards
            if (hitPuck.top) {
                const towardGoal = env.puck.dy > 0;
                const puckSpeed = Math.sqrt(env.puck.dx * env.puck.dx + env.puck.dy * env.puck.dy);
                reward.top += towardGoal ? (1.0 + puckSpeed / maxSpeed) : 0.2;
            }
            if (hitPuck.bottom) {
                const towardGoal = env.puck.dy < 0;
                const puckSpeed = Math.sqrt(env.puck.dx * env.puck.dx + env.puck.dy * env.puck.dy);
                reward.bottom += towardGoal ? (1.0 + puckSpeed / maxSpeed) : 0.2;
            }

            // Strong rewards and penalties for goals
            if (goalHit === 'top') {
                reward.bottom += goalReward;
                reward.top += goalPenalty;
            } else if (goalHit === 'bottom') {
                reward.top += goalReward;
                reward.bottom += goalPenalty;
            }
            break;
    }

    return reward;
}

async function moveAI() {
    if (!agent) return;

    currentEpisodeFrames++;

    const maxEpisodeLength = 5000; // Maximum length of an episode
    const isDone = currentEpisodeFrames >= maxEpisodeLength;

    if (!isTrainingMode) {
        // Move the AI paddle
        const state = agent.getState(env.puck, env.playerPaddle, env.aiPaddle, true, env.canvas.width, env.canvas.height);
        const result = agent.act(state);
        moveAgentPaddle(env.aiPaddle, result.action, true);

        // Move the player paddle using mouse input
        let dx = mouseX - env.playerPaddle.x;
        let dy = mouseY - env.playerPaddle.y;

        // Normalize the direction vector
        const magnitude = Math.sqrt(dx * dx + dy * dy);
        if (magnitude > 0) {
            dx /= magnitude;
            dy /= magnitude;
        } else {
            dx = 0;
            dy = 0;
        }

        const action = [dx, dy];
        moveAgentPaddle(env.playerPaddle, action, false);
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

        // After updating the environment and collecting rewards
        // Store experiences
        if (previousState !== null) {
            agent.remember(
                previousState.top,
                previousAction.top,
                reward.top,
                previousValue.top,
                previousLogProb.top,
                isDone  // Mark as done if the maximum episode length is reached
            );

            agent.remember(
                previousState.bottom,
                previousAction.bottom,
                reward.bottom,
                previousValue.bottom,
                previousLogProb.bottom,
                isDone
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

        // Increment frame count
        agent.frameCount++;

        // Train at fixed intervals
        if (agent.frameCount % trainInterval === 0) {
            await agent.train();
            console.log(`Trained at frame ${agent.frameCount}`);

            // Optionally, reset the environment
            resetEnvironment();
        }

        // Reset the environment if done
        if (isDone) {
            resetEnvironment();
        }

        // Log metrics periodically
        if (agent.frameCount % 1000 === 0) {
            logTrainingMetrics();
        }

        // Accumulate rewards
        episodeRewards.top += reward.top;
        episodeRewards.bottom += reward.bottom;
    }
}

async function gameLoop() {
    await moveAI();
    const goalHit = env.update(mouseX, mouseY, isTrainingMode);
    env.draw();
    requestAnimationFrame(gameLoop);
}

async function initializeAI() {
    await tf.ready();
    agent = new PPOAgent(14, 2);  // 14 state inputs, 2 continuous actions (x, y movement)
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
    resetEnvironment();
    env.state.playerScore = 0;
    env.state.aiScore = 0;
    previousState = null;
    previousAction = null;
    
    console.log(`Switched to ${isTrainingMode ? 'Training' : 'Play'} mode`);
}

// Start the game
init();