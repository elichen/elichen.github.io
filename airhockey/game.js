const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');

// Set canvas size
canvas.width = 600;
canvas.height = 800;

// Game state
const state = {
    playerScore: 0,
    aiScore: 0,
    stuckTime: 0,
    lastPuckPos: { x: 0, y: 0 },
    samePositionTime: 0
};

const GOAL_WIDTH = 200;
const GOAL_POSTS = 20;

// Update colors for flat look
const TABLE_COLOR = '#2c3e50';
const TABLE_BORDER = '#34495e';
const CENTER_CIRCLE_RADIUS = 100;

// Game objects
const playerPaddle = {
    x: canvas.width / 2,
    y: canvas.height - 50,
    radius: 20,
    color: '#3498db'
};

const aiPaddle = {
    x: canvas.width / 2,
    y: 50,
    radius: 20,
    color: '#2ecc71',
    speed: 5
};

const puck = {
    x: canvas.width / 2,
    y: canvas.height / 2,
    radius: 15,
    dx: 0,
    dy: 0,
    color: '#e74c3c',
    isStuck: false,
    stuckEffectSize: 0
};

// Adjust physics for smoother movement
const friction = 0.99; // Was 0.98
const maxSpeed = 20; // Was 15

// Mouse movement
let mouseX = 0;
let mouseY = 0;

canvas.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    mouseX = e.clientX - rect.left;
    mouseY = e.clientY - rect.top;
});

function createGradient(x, y, radius, colorStart, colorEnd) {
    const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius);
    gradient.addColorStop(0, colorStart);
    gradient.addColorStop(1, colorEnd);
    return gradient;
}

function drawTableMarkings() {
    // Draw center circle
    ctx.beginPath();
    ctx.arc(canvas.width/2, canvas.height/2, CENTER_CIRCLE_RADIUS, 0, Math.PI * 2);
    ctx.strokeStyle = '#ffffff22';
    ctx.lineWidth = 4;
    ctx.stroke();

    // Draw center line
    ctx.beginPath();
    ctx.moveTo(0, canvas.height/2);
    ctx.lineTo(canvas.width, canvas.height/2);
    ctx.strokeStyle = '#ffffff22';
    ctx.stroke();
}

function drawGoals() {
    // Draw goals with flat colors
    ctx.fillStyle = '#ffffff22';
    
    // Top goal
    ctx.fillRect((canvas.width - GOAL_WIDTH) / 2, -GOAL_POSTS/2, GOAL_WIDTH, GOAL_POSTS);
    // Bottom goal
    ctx.fillRect((canvas.width - GOAL_WIDTH) / 2, canvas.height - GOAL_POSTS/2, GOAL_WIDTH, GOAL_POSTS);
}

function drawCircle(x, y, radius, color) {
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
}

function drawScore() {
    ctx.font = 'bold 48px Arial';
    ctx.fillStyle = '#ffffff44';
    ctx.textAlign = 'center';
    ctx.fillText(`${state.aiScore} - ${state.playerScore}`, canvas.width/2, canvas.height/2);
}

function checkCollision(circle1, circle2) {
    const dx = circle1.x - circle2.x;
    const dy = circle1.y - circle2.y;
    const distance = Math.sqrt(dx * dx + dy * dy);
    return distance < circle1.radius + circle2.radius;
}

function moveAI() {
    const dx = puck.x - aiPaddle.x;
    const dy = puck.y - aiPaddle.y;
    const angle = Math.atan2(dy, dx);
    
    aiPaddle.x += Math.cos(angle) * aiPaddle.speed;
    aiPaddle.y += Math.sin(angle) * aiPaddle.speed;

    // Keep AI in bounds
    aiPaddle.x = Math.max(aiPaddle.radius, Math.min(canvas.width - aiPaddle.radius, aiPaddle.x));
    aiPaddle.y = Math.max(aiPaddle.radius, Math.min(canvas.height/2 - aiPaddle.radius, aiPaddle.y));
}

function resetPuck(scoredOnTop = null) {
    puck.x = canvas.width / 2;
    puck.dx = 0;
    puck.dy = 0;

    if (scoredOnTop === true) {
        puck.y = canvas.height / 4; // Position in AI's side
    } else if (scoredOnTop === false) {
        puck.y = canvas.height * 3/4; // Position in player's side
    } else {
        puck.y = canvas.height / 2; // Center for other resets
    }
    
    // Reset stuck detection state
    state.samePositionTime = 0;
    state.lastPuckPos.x = puck.x;
    state.lastPuckPos.y = puck.y;
}

function isInGoal() {
    const inXRange = puck.x > (canvas.width - GOAL_WIDTH) / 2 && 
                    puck.x < (canvas.width + GOAL_WIDTH) / 2;
    
    if (puck.y - puck.radius < GOAL_POSTS && inXRange) return 'top';
    if (puck.y + puck.radius > canvas.height - GOAL_POSTS && inXRange) return 'bottom';
    return false;
}

function handleWallCollision() {
    // Left and right walls
    if (puck.x - puck.radius < 0) {
        puck.x = puck.radius;
        puck.dx *= -0.8;
    }
    if (puck.x + puck.radius > canvas.width) {
        puck.x = canvas.width - puck.radius;
        puck.dx *= -0.8;
    }

    // Top and bottom walls (except for goals)
    const goalHit = isInGoal();
    if (!goalHit) {
        if (puck.y - puck.radius < 0) {
            puck.y = puck.radius;
            puck.dy *= -0.8;
        }
        if (puck.y + puck.radius > canvas.height) {
            puck.y = canvas.height - puck.radius;
            puck.dy *= -0.8;
        }
    }
}

function handlePaddleCollision(paddle, prevX, prevY) {
    // Continuous collision detection
    const dx = puck.x - paddle.x;
    const dy = puck.y - paddle.y;
    const distance = Math.sqrt(dx * dx + dy * dy);

    if (distance < paddle.radius + puck.radius) {
        // Calculate collision point
        const angle = Math.atan2(dy, dx);
        const minDistance = paddle.radius + puck.radius;
        
        // Move puck outside paddle
        puck.x = paddle.x + Math.cos(angle) * minDistance;
        puck.y = paddle.y + Math.sin(angle) * minDistance;
        
        // Calculate new velocity based on paddle movement
        const paddleSpeed = {
            x: paddle.x - prevX,
            y: paddle.y - prevY
        };
        
        const dotProduct = (puck.dx * dx + puck.dy * dy) / distance;
        
        // Combine paddle momentum with puck direction
        puck.dx = (Math.cos(angle) * Math.abs(dotProduct) + paddleSpeed.x * 0.9);
        puck.dy = (Math.sin(angle) * Math.abs(dotProduct) + paddleSpeed.y * 0.9);
        
        // Add minimum speed after collision
        const speed = Math.sqrt(puck.dx * puck.dx + puck.dy * puck.dy);
        if (speed < 3) {
            puck.dx *= 3 / speed;
            puck.dy *= 3 / speed;
        }
        
        // Enforce speed limit
        if (speed > maxSpeed) {
            puck.dx = (puck.dx / speed) * maxSpeed;
            puck.dy = (puck.dy / speed) * maxSpeed;
        }
    }
}

function getRandomPosition() {
    return {
        x: Math.random() * (canvas.width - 2 * puck.radius) + puck.radius,
        y: Math.random() * (canvas.height - 2 * puck.radius) + puck.radius,
        dx: (Math.random() - 0.5) * 5,
        dy: (Math.random() - 0.5) * 5
    };
}

function isPuckStuck() {
    const speed = Math.sqrt(puck.dx * puck.dx + puck.dy * puck.dy);
    const isSlowMoving = Math.abs(puck.dx) < 0.1 && Math.abs(puck.dy) < 0.1;
    
    // Check if puck is near any wall
    const nearWall = (
        puck.x - puck.radius < 10 || // Left wall
        puck.x + puck.radius > canvas.width - 10 || // Right wall
        (puck.y - puck.radius < 10 && !isInGoal()) || // Top wall (not goal)
        (puck.y + puck.radius > canvas.height - 10 && !isInGoal()) // Bottom wall (not goal)
    );
    
    if (!nearWall) {
        return false;
    }

    const distFromLast = Math.sqrt(
        Math.pow(puck.x - state.lastPuckPos.x, 2) + 
        Math.pow(puck.y - state.lastPuckPos.y, 2)
    );
    
    if (distFromLast < 1) {
        state.samePositionTime++;
    } else {
        state.samePositionTime = 0;
        state.lastPuckPos.x = puck.x;
        state.lastPuckPos.y = puck.y;
    }

    return isSlowMoving || state.samePositionTime > 30;
}

function unstickPuck() {
    const newPos = getRandomPosition();
    puck.x = newPos.x;
    puck.y = newPos.y;
    puck.dx = newPos.dx;
    puck.dy = newPos.dy;
    state.samePositionTime = 0;
    puck.isStuck = true;
    puck.stuckEffectSize = 20;
}

function update() {
    // Store previous positions for collision detection
    const prevPlayerX = playerPaddle.x;
    const prevPlayerY = playerPaddle.y;
    const prevAIX = aiPaddle.x;
    const prevAIY = aiPaddle.y;

    // Update player paddle position
    playerPaddle.x = mouseX;
    playerPaddle.y = mouseY;

    // Restrict player to bottom half
    playerPaddle.x = Math.max(playerPaddle.radius, Math.min(canvas.width - playerPaddle.radius, playerPaddle.x));
    playerPaddle.y = Math.max(canvas.height/2 + playerPaddle.radius, Math.min(canvas.height - playerPaddle.radius, playerPaddle.y));

    // Move AI
    moveAI();

    // Update puck position
    puck.x += puck.dx;
    puck.y += puck.dy;

    // Apply friction
    puck.dx *= friction;
    puck.dy *= friction;

    // Handle collisions
    handleWallCollision();
    handlePaddleCollision(playerPaddle, prevPlayerX, prevPlayerY);
    handlePaddleCollision(aiPaddle, prevAIX, prevAIY);

    // Check for goals and stuck puck
    const goalHit = isInGoal();
    if (goalHit === 'top') {
        state.playerScore++;
        resetPuck(true);
    } else if (goalHit === 'bottom') {
        state.aiScore++;
        resetPuck(false);
    } else if (isPuckStuck()) {
        unstickPuck();
    }
}

function draw() {
    // Clear canvas with flat color
    ctx.fillStyle = TABLE_COLOR;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Add table border
    ctx.strokeStyle = TABLE_BORDER;
    ctx.lineWidth = 10;
    ctx.strokeRect(0, 0, canvas.width, canvas.height);
    
    // Draw table markings
    drawTableMarkings();
    
    // Draw goals
    drawGoals();
    
    // Draw score
    drawScore();
    
    // Draw game objects
    drawCircle(playerPaddle.x, playerPaddle.y, playerPaddle.radius, playerPaddle.color);
    drawCircle(aiPaddle.x, aiPaddle.y, aiPaddle.radius, aiPaddle.color);
    
    // Draw puck with stuck effect if needed
    drawCircle(puck.x, puck.y, puck.radius, puck.color);
    if (puck.isStuck) {
        ctx.beginPath();
        ctx.arc(puck.x, puck.y, puck.radius + puck.stuckEffectSize, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(255, 255, 255, ${puck.stuckEffectSize / 20})`;
        ctx.lineWidth = 2;
        ctx.stroke();
        
        puck.stuckEffectSize *= 0.9;
        if (puck.stuckEffectSize < 0.5) {
            puck.isStuck = false;
        }
    }
}

function gameLoop() {
    update();
    draw();
    requestAnimationFrame(gameLoop);
}

gameLoop();