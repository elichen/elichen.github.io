const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const instructions = document.getElementById('instructions');
const startButton = document.getElementById('start-button');
const status = document.getElementById('status');
const healthDisplay = document.getElementById('health-display');
const playerHealthBar = document.getElementById('player-health');
const enemyHealthBar = document.getElementById('enemy-health');

let gameRunning = false;
let animationId;

const player = {
    x: 200,
    y: 430,
    width: 20,
    height: 20,
    speed: 4,
    health: 100,
    attacking: false,
    attackCooldown: 0
};

const enemy = {
    x: 200,
    y: 70,
    width: 25,
    height: 25,
    speed: 2,
    health: 100,
    attacking: false,
    attackCooldown: 0,
    targetX: 200,
    targetY: 70
};

const keys = {
    ArrowUp: false,
    ArrowDown: false,
    ArrowLeft: false,
    ArrowRight: false,
    Space: false
};

function drawSquidField() {
    ctx.fillStyle = '#d4a574';

    ctx.beginPath();
    ctx.arc(200, 80, 60, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillRect(160, 80, 80, 250);

    ctx.beginPath();
    ctx.arc(200, 380, 80, 0, Math.PI);
    ctx.fill();

    ctx.beginPath();
    ctx.moveTo(120, 380);
    ctx.lineTo(80, 450);
    ctx.lineTo(120, 450);
    ctx.lineTo(140, 400);
    ctx.closePath();
    ctx.fill();

    ctx.beginPath();
    ctx.moveTo(280, 380);
    ctx.lineTo(320, 450);
    ctx.lineTo(280, 450);
    ctx.lineTo(260, 400);
    ctx.closePath();
    ctx.fill();

    ctx.strokeStyle = '#8b4513';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(200, 80, 60, 0, Math.PI * 2);
    ctx.stroke();
    ctx.strokeRect(160, 80, 80, 250);
    ctx.beginPath();
    ctx.arc(200, 380, 80, 0, Math.PI);
    ctx.stroke();

    ctx.fillStyle = 'rgba(46, 204, 113, 0.3)';
    ctx.beginPath();
    ctx.arc(200, 80, 40, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#2ecc71';
    ctx.lineWidth = 2;
    ctx.stroke();
}

function isOnField(x, y, width, height) {
    const centerX = x + width / 2;
    const centerY = y + height / 2;

    const headDist = Math.sqrt((centerX - 200) ** 2 + (centerY - 80) ** 2);
    if (headDist < 60) return true;

    if (centerX >= 160 && centerX <= 240 && centerY >= 80 && centerY <= 330) return true;

    const bodyDist = Math.sqrt((centerX - 200) ** 2 + (centerY - 380) ** 2);
    if (bodyDist < 80 && centerY >= 300) return true;

    if (centerY >= 380 && centerY <= 450) {
        if (centerX >= 80 && centerX <= 140) return true;
        if (centerX >= 260 && centerX <= 320) return true;
    }

    return false;
}

function isInWinZone(x, y, width, height) {
    const centerX = x + width / 2;
    const centerY = y + height / 2;
    const dist = Math.sqrt((centerX - 200) ** 2 + (centerY - 80) ** 2);
    return dist < 40;
}

function drawPlayer() {
    ctx.fillStyle = player.attacking ? '#e74c3c' : '#f1c40f';
    ctx.fillRect(player.x, player.y, player.width, player.height);

    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.strokeRect(player.x, player.y, player.width, player.height);
}

function drawEnemy() {
    ctx.fillStyle = enemy.attacking ? '#ff6b6b' : '#e74c3c';
    ctx.fillRect(enemy.x, enemy.y, enemy.width, enemy.height);

    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.strokeRect(enemy.x, enemy.y, enemy.width, enemy.height);
}

function updatePlayer() {
    let newX = player.x;
    let newY = player.y;

    if (keys.ArrowUp) newY -= player.speed;
    if (keys.ArrowDown) newY += player.speed;
    if (keys.ArrowLeft) newX -= player.speed;
    if (keys.ArrowRight) newX += player.speed;

    if (isOnField(newX, newY, player.width, player.height)) {
        player.x = newX;
        player.y = newY;
    }

    if (player.attackCooldown > 0) {
        player.attackCooldown--;
        if (player.attackCooldown === 0) {
            player.attacking = false;
        }
    }

    if (keys.Space && player.attackCooldown === 0) {
        player.attacking = true;
        player.attackCooldown = 30;
        checkPlayerAttack();
    }
}

function updateEnemy() {
    if (Math.random() < 0.02) {
        enemy.targetX = player.x + (Math.random() - 0.5) * 100;
        enemy.targetY = player.y + (Math.random() - 0.5) * 100;
    }

    const dx = enemy.targetX - enemy.x;
    const dy = enemy.targetY - enemy.y;
    const dist = Math.sqrt(dx * dx + dy * dy);

    if (dist > 5) {
        const newX = enemy.x + (dx / dist) * enemy.speed;
        const newY = enemy.y + (dy / dist) * enemy.speed;

        if (isOnField(newX, newY, enemy.width, enemy.height)) {
            enemy.x = newX;
            enemy.y = newY;
        }
    }

    if (enemy.attackCooldown > 0) {
        enemy.attackCooldown--;
        if (enemy.attackCooldown === 0) {
            enemy.attacking = false;
        }
    }

    const playerDist = Math.sqrt((enemy.x - player.x) ** 2 + (enemy.y - player.y) ** 2);
    if (playerDist < 50 && enemy.attackCooldown === 0 && Math.random() < 0.05) {
        enemy.attacking = true;
        enemy.attackCooldown = 45;
        checkEnemyAttack();
    }
}

function checkPlayerAttack() {
    const dist = Math.sqrt((player.x - enemy.x) ** 2 + (player.y - enemy.y) ** 2);
    if (dist < 40) {
        enemy.health -= 20;
        enemyHealthBar.style.width = enemy.health + '%';
        status.innerHTML = '<span style="color: #2ecc71">Hit!</span>';
        setTimeout(() => { if (gameRunning) status.textContent = ''; }, 500);
    }
}

function checkEnemyAttack() {
    const dist = Math.sqrt((player.x - enemy.x) ** 2 + (player.y - enemy.y) ** 2);
    if (dist < 45) {
        player.health -= 15;
        playerHealthBar.style.width = player.health + '%';
        status.innerHTML = '<span style="color: #e74c3c">Ouch!</span>';
        setTimeout(() => { if (gameRunning) status.textContent = ''; }, 500);
    }
}

function checkWinConditions() {
    if (player.health <= 0) {
        endGame(false, 'You were defeated!');
        return true;
    }
    if (enemy.health <= 0) {
        endGame(true, 'You defeated the enemy!');
        return true;
    }
    if (isInWinZone(player.x, player.y, player.width, player.height)) {
        endGame(true, 'You reached the goal!');
        return true;
    }
    return false;
}

function gameLoop() {
    if (!gameRunning) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    drawSquidField();
    updatePlayer();
    updateEnemy();
    drawPlayer();
    drawEnemy();

    if (!checkWinConditions()) {
        animationId = requestAnimationFrame(gameLoop);
    }
}

function startGame() {
    instructions.style.display = 'none';
    canvas.style.display = 'block';
    healthDisplay.style.display = 'flex';

    player.x = 200;
    player.y = 430;
    player.health = 100;
    player.attacking = false;
    player.attackCooldown = 0;

    enemy.x = 200;
    enemy.y = 70;
    enemy.health = 100;
    enemy.attacking = false;
    enemy.attackCooldown = 0;

    playerHealthBar.style.width = '100%';
    enemyHealthBar.style.width = '100%';

    status.textContent = '';
    gameRunning = true;
    gameLoop();
}

function endGame(win, message) {
    gameRunning = false;
    cancelAnimationFrame(animationId);

    if (win) {
        status.innerHTML = `<span style="color: #2ecc71; font-size: 24px">You Win!</span><br>${message}`;
    } else {
        status.innerHTML = `<span style="color: #e74c3c; font-size: 24px">You Lose!</span><br>${message}`;
    }

    setTimeout(() => {
        canvas.style.display = 'none';
        healthDisplay.style.display = 'none';
        instructions.style.display = 'block';
        startButton.textContent = 'Play Again';
    }, 2500);
}

startButton.addEventListener('click', startGame);

document.addEventListener('keydown', (e) => {
    if (e.code in keys) {
        keys[e.code] = true;
        e.preventDefault();
    }
});

document.addEventListener('keyup', (e) => {
    if (e.code in keys) {
        keys[e.code] = false;
    }
});
