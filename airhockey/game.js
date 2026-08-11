let env, agent, mouseX = 0, mouseY = 0, aiOnTop = true, selfPlay = false;
let matchPolicies = { top: 0, bottom: 0 };

function updateModeHint() {
    const hint = document.getElementById('modeHint');
    if (!agent?.policyCount) {
        hint.textContent = selfPlay ? 'Loading league policies…' : 'Move your paddle with the pointer.';
        return;
    }
    hint.textContent = selfPlay
        ? `${agent.getPolicyName(matchPolicies.top)} vs ${agent.getPolicyName(matchPolicies.bottom)}`
        : `Move your paddle with the pointer. Opponent: ${agent.getPolicyName(matchPolicies.top)}.`;
}

function selectMatchPolicies() {
    if (!agent?.policyCount) return;
    if (selfPlay) {
        matchPolicies.top = agent.samplePolicy();
        matchPolicies.bottom = agent.samplePolicy(matchPolicies.top);
    } else {
        const selected = agent.samplePolicy();
        matchPolicies = { top: selected, bottom: selected };
    }
    updateModeHint();
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
    selectMatchPolicies();

    const toggle = document.getElementById('selfPlayBtn');
    const swapButton = document.getElementById('swapBtn');
    toggle.setAttribute('aria-checked', String(selfPlay));
    document.getElementById('modeLabel').textContent = selfPlay ? 'AI self-play' : 'Human vs AI';
    swapButton.disabled = selfPlay;
    updateModeHint();
}

function swapAI() {
    if (selfPlay) return;
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
    selectMatchPolicies();
}

function initializeGame() {
    const canvas = document.getElementById('gameCanvas');
    env = new AirHockeyEnvironment(canvas);
    mouseX = env.playerPaddle.x;
    mouseY = env.playerPaddle.y;
    canvas.addEventListener('mousemove', e => {
        const rect = canvas.getBoundingClientRect();
        mouseX = e.clientX - rect.left;
        mouseY = e.clientY - rect.top;
    });
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

async function movePlayers() {
    if (selfPlay) {
        const topState = agent.getState(env.puck, env.playerPaddle, env.aiPaddle, true, env.canvas.width, env.canvas.height);
        const bottomState = agent.getState(env.puck, env.playerPaddle, env.aiPaddle, false, env.canvas.width, env.canvas.height);
        const topResult = await agent.act(topState, matchPolicies.top);
        const bottomResult = await agent.act(bottomState, matchPolicies.bottom);
        moveAgentPaddle(env.aiPaddle, topResult.action, true);
        moveAgentPaddle(env.playerPaddle, bottomResult.action, false);
        if (env.update()) selectMatchPolicies();
        return;
    }

    const aiPaddle = aiOnTop ? env.aiPaddle : env.playerPaddle;
    const playerPaddle = aiOnTop ? env.playerPaddle : env.aiPaddle;

    const state = agent.getState(env.puck, env.playerPaddle, env.aiPaddle, aiOnTop, env.canvas.width, env.canvas.height);
    const policy = aiOnTop ? matchPolicies.top : matchPolicies.bottom;
    const result = await agent.act(state, policy);
    moveAgentPaddle(aiPaddle, result.action, aiOnTop);

    const minY = aiOnTop ? env.canvas.height/2 + playerPaddle.radius : playerPaddle.radius;
    const maxY = aiOnTop ? env.canvas.height - playerPaddle.radius : env.canvas.height/2 - playerPaddle.radius;
    const targetX = Math.max(playerPaddle.radius, Math.min(env.canvas.width - playerPaddle.radius, mouseX));
    const targetY = Math.max(minY, Math.min(maxY, mouseY));
    moveAgentPaddle(playerPaddle, [
        Math.max(-1, Math.min(1, (targetX - playerPaddle.x) / playerPaddle.speed)),
        Math.max(-1, Math.min(1, (targetY - playerPaddle.y) / playerPaddle.speed)) * (aiOnTop ? 1 : -1)
    ], !aiOnTop);

    if (env.update(mouseX, mouseY, false)) selectMatchPolicies();
}

async function gameLoop() {
    await movePlayers();
    env.draw();
    requestAnimationFrame(gameLoop);
}

document.addEventListener('DOMContentLoaded', async () => {
    initializeGame();
    agent = new PPOAgent(12, 2);
    await agent.loadONNXLeague([
        { path: 'model/psro_v3_main_01.onnx', name: 'V3 Main', weight: .75 },
        { path: 'model/psro_v3_main_02.onnx', name: 'V3 Counter', weight: .20 },
        { path: 'model/psro_main_02.onnx', name: 'V2 Main', weight: .05 }
    ]);
    selectMatchPolicies();
    gameLoop();
});
