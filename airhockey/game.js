let env, agent, mouseX = 0, mouseY = 0, aiOnTop = true, selfPlay = false, lastTime = null, acc = 0;
const STEP = 1000 / 60;

function updateModeHint() {
    document.getElementById('modeHint').textContent = selfPlay ? 'The policy plays both sides.' : 'Move your paddle with the mouse or a finger.';
}

function resetMatch() {
    env.reset();
    const humanPaddle = aiOnTop ? env.playerPaddle : env.aiPaddle;
    mouseX = humanPaddle.x;
    mouseY = humanPaddle.y;
}

function toggleSelfPlay() {
    selfPlay = !selfPlay;
    resetMatch();
    const toggle = document.getElementById('selfPlayBtn');
    toggle.setAttribute('aria-checked', String(selfPlay));
    document.getElementById('modeLabel').textContent = selfPlay ? 'AI self-play' : 'Human vs AI';
    document.getElementById('swapBtn').disabled = selfPlay;
    updateModeHint();
}

function swapAI() {
    if (selfPlay) return;
    aiOnTop = !aiOnTop;
    env.state.playerScore = 0;
    env.state.aiScore = 0;
    env.resetPuck(null, true);
}

function initializeGame() {
    const canvas = document.getElementById('gameCanvas');
    env = new AirHockeyEnvironment(canvas);
    mouseX = env.playerPaddle.x;
    mouseY = env.playerPaddle.y;
    const aim = e => {
        const rect = canvas.getBoundingClientRect();
        mouseX = (e.clientX - rect.left) * canvas.width / rect.width;
        mouseY = (e.clientY - rect.top) * canvas.height / rect.height;
    };
    canvas.addEventListener('pointermove', aim);   // mouse, pen or finger; touch-action: none keeps a drag from scrolling
    canvas.addEventListener('pointerdown', aim);
}

function moveAgentPaddle(paddle, action, isTopPlayer) {
    const requestedDx = action[0] * paddle.speed;
    const requestedDy = action[1] * paddle.speed * (isTopPlayer ? -1 : 1);
    const smoothedDx = (paddle.dx || 0) * 0.6 + requestedDx * 0.4;
    const smoothedDy = (paddle.dy || 0) * 0.6 + requestedDy * 0.4;
    const previousX = paddle.x;
    const previousY = paddle.y;

    const minY = isTopPlayer ? paddle.radius : env.canvas.height/2 + paddle.radius;
    const maxY = isTopPlayer ? env.canvas.height/2 - paddle.radius : env.canvas.height - paddle.radius;
    paddle.x = Math.max(paddle.radius, Math.min(env.canvas.width - paddle.radius, previousX + smoothedDx));
    paddle.y = Math.max(minY, Math.min(maxY, previousY + smoothedDy));
    paddle.dx = paddle.x - previousX;
    paddle.dy = paddle.y - previousY;
}

function policyAction(isTop) {
    return agent.act(agent.getState(env.puck, env.playerPaddle, env.aiPaddle, isTop, env.canvas.width, env.canvas.height));
}

function movePlayers() {
    if (selfPlay) {
        const jitter = a => a.map(v => Math.max(-1, Math.min(1, v + (Math.random() - 0.5) * 0.3)));   // a deterministic policy against itself would replay one point forever
        const top = jitter(policyAction(true)), bottom = jitter(policyAction(false));
        moveAgentPaddle(env.aiPaddle, top, true);
        moveAgentPaddle(env.playerPaddle, bottom, false);
        env.update();
        return;
    }

    const aiPaddle = aiOnTop ? env.aiPaddle : env.playerPaddle;
    const playerPaddle = aiOnTop ? env.playerPaddle : env.aiPaddle;
    moveAgentPaddle(aiPaddle, policyAction(aiOnTop), aiOnTop);

    const minY = aiOnTop ? env.canvas.height/2 + playerPaddle.radius : playerPaddle.radius;
    const maxY = aiOnTop ? env.canvas.height - playerPaddle.radius : env.canvas.height/2 - playerPaddle.radius;
    const targetX = Math.max(playerPaddle.radius, Math.min(env.canvas.width - playerPaddle.radius, mouseX));
    const targetY = Math.max(minY, Math.min(maxY, mouseY));
    moveAgentPaddle(playerPaddle, [
        Math.max(-1, Math.min(1, (targetX - playerPaddle.x) / playerPaddle.speed)),
        Math.max(-1, Math.min(1, (targetY - playerPaddle.y) / playerPaddle.speed)) * (aiOnTop ? 1 : -1)
    ], !aiOnTop);

    env.update();
}

// Physics runs at a fixed 60 Hz (the rate it was trained at), whatever the display refresh rate.
function gameLoop(now) {
    acc = Math.min(acc + (lastTime === null ? STEP : now - lastTime), 6 * STEP);
    lastTime = now;
    for (; acc >= STEP; acc -= STEP) movePlayers();
    env.draw();
    requestAnimationFrame(gameLoop);
}

document.addEventListener('DOMContentLoaded', async () => {
    initializeGame();
    agent = new PPOAgent();
    await agent.load('model/policy.bin');
    updateModeHint();
    requestAnimationFrame(gameLoop);
});
