let env, agent, mouseX = 0, mouseY = 0;

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
    let dy = action[1] * paddle.speed;

    paddle.dx = (paddle.dx || 0) * 0.6 + dx * 0.4;
    paddle.dy = (paddle.dy || 0) * 0.6 + dy * 0.4;
    paddle.x = Math.max(paddle.radius, Math.min(env.canvas.width - paddle.radius, paddle.x + paddle.dx));

    const minY = isTopPlayer ? paddle.radius : env.canvas.height/2 + paddle.radius;
    const maxY = isTopPlayer ? env.canvas.height/2 - paddle.radius : env.canvas.height - paddle.radius;
    paddle.y = Math.max(minY, Math.min(maxY, paddle.y + paddle.dy));
}

async function moveAI() {
    const state = agent.getState(env.puck, env.playerPaddle, env.aiPaddle, true, env.canvas.width, env.canvas.height);
    const result = await agent.act(state);
    moveAgentPaddle(env.aiPaddle, result.action, true);

    const prevX = env.playerPaddle.x, prevY = env.playerPaddle.y;
    env.playerPaddle.x = Math.max(env.playerPaddle.radius, Math.min(env.canvas.width - env.playerPaddle.radius, mouseX));
    env.playerPaddle.y = Math.max(env.canvas.height/2 + env.playerPaddle.radius, Math.min(env.canvas.height - env.playerPaddle.radius, mouseY));
    env.playerPaddle.dx = env.playerPaddle.x - prevX;
    env.playerPaddle.dy = env.playerPaddle.y - prevY;

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