let env, agent, mouseX = 0, mouseY = 0, aiOnTop = true;

function swapAI() {
    aiOnTop = !aiOnTop;
    env.state.playerScore = 0;
    env.state.aiScore = 0;
    env.resetPuck();

    // Position based on ROLE, not object name
    const aiPaddle = aiOnTop ? env.aiPaddle : env.playerPaddle;
    const playerPaddle = aiOnTop ? env.playerPaddle : env.aiPaddle;

    aiPaddle.x = env.canvas.width/2;
    playerPaddle.x = env.canvas.width/2;
    aiPaddle.y = aiOnTop ? 50 : env.canvas.height - 50;
    playerPaddle.y = aiOnTop ? env.canvas.height - 50 : 50;
}

function initializeGame() {
    const canvas = document.getElementById('gameCanvas');
    env = new AirHockeyEnvironment(canvas);
    canvas.addEventListener('mousemove', e => {
        const rect = canvas.getBoundingClientRect();
        mouseX = e.clientX - rect.left;
        mouseY = e.clientY - rect.top;
    });
}

function moveAgentPaddle(paddle, action, isTopPlayer) {
    let dx = action[0] * paddle.speed;
    let dy = action[1] * paddle.speed * (isTopPlayer ? -1 : 1);

    paddle.dx = (paddle.dx || 0) * 0.6 + dx * 0.4;
    paddle.dy = (paddle.dy || 0) * 0.6 + dy * 0.4;
    paddle.x = Math.max(paddle.radius, Math.min(env.canvas.width - paddle.radius, paddle.x + paddle.dx));

    const minY = isTopPlayer ? paddle.radius : env.canvas.height/2 + paddle.radius;
    const maxY = isTopPlayer ? env.canvas.height/2 - paddle.radius : env.canvas.height - paddle.radius;
    paddle.y = Math.max(minY, Math.min(maxY, paddle.y + paddle.dy));
}

async function moveAI() {
    const aiPaddle = aiOnTop ? env.aiPaddle : env.playerPaddle;
    const playerPaddle = aiOnTop ? env.playerPaddle : env.aiPaddle;

    const state = agent.getState(env.puck, env.playerPaddle, env.aiPaddle, aiOnTop, env.canvas.width, env.canvas.height);
    const result = await agent.act(state);
    moveAgentPaddle(aiPaddle, result.action, aiOnTop);

    const prevX = playerPaddle.x, prevY = playerPaddle.y;
    playerPaddle.x = Math.max(playerPaddle.radius, Math.min(env.canvas.width - playerPaddle.radius, mouseX));

    const minY = aiOnTop ? env.canvas.height/2 + playerPaddle.radius : playerPaddle.radius;
    const maxY = aiOnTop ? env.canvas.height - playerPaddle.radius : env.canvas.height/2 - playerPaddle.radius;
    playerPaddle.y = Math.max(minY, Math.min(maxY, mouseY));
    playerPaddle.dx = playerPaddle.x - prevX;
    playerPaddle.dy = playerPaddle.y - prevY;

    env.update(mouseX, mouseY, false);
}

async function gameLoop() {
    await moveAI();
    env.draw();
    requestAnimationFrame(gameLoop);
}

document.addEventListener('DOMContentLoaded', async () => {
    initializeGame();
    agent = new PPOAgent(8, 2);  // 8 features matching training environment
    await agent.loadONNXModel('model/ppo_selfplay_final.onnx');
    gameLoop();
});