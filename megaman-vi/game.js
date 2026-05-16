const canvas = document.getElementById('game');
const ctx = canvas.getContext('2d');
const statusEl = document.getElementById('status');
const weaponPanelEl = document.getElementById('weaponSelect');

const W = canvas.width;
const H = canvas.height;
const GROUND = 456;
const GRAVITY = 1700;
const FRICTION = 0.83;
const CELL = 192;

const ASSETS = {
    actors: 'assets/sprites/actors.png?v=2',
    backgrounds: {
        foundry: 'assets/backgrounds/foundry.png',
        hydro: 'assets/backgrounds/hydro.png',
        sky: 'assets/backgrounds/sky.png',
        final: 'assets/backgrounds/final.png'
    }
};

const FRAMES = {
    heroIdle1: { sx: 0, sy: 0 },
    heroIdle2: { sx: 192, sy: 0 },
    heroRun1: { sx: 384, sy: 0 },
    heroRun2: { sx: 576, sy: 0 },
    heroJump: { sx: 768, sy: 0 },
    heroShoot: { sx: 960, sy: 0 },
    cinderIdle1: { sx: 1152, sy: 0 },
    cinderIdle2: { sx: 0, sy: 192 },
    tideIdle1: { sx: 192, sy: 192 },
    tideIdle2: { sx: 384, sy: 192 },
    voltIdle1: { sx: 576, sy: 192 },
    voltIdle2: { sx: 768, sy: 192 },
    nullIdle1: { sx: 960, sy: 192 },
    nullIdle2: { sx: 1152, sy: 192 }
};

const WEAPONS = [
    { id: 'buster', name: 'Pulse', color: '#7dfcff', damage: 7, cost: 0, speed: 620, cooldown: 0.18, hotkey: '1' },
    { id: 'ember', name: 'Ember Arc', color: '#ff8a3d', damage: 13, cost: 8, speed: 520, cooldown: 0.32, hotkey: '2' },
    { id: 'tide', name: 'Tide Bolt', color: '#5cc7ff', damage: 12, cost: 7, speed: 470, cooldown: 0.28, hotkey: '3' },
    { id: 'storm', name: 'Storm Mine', color: '#f3dd4e', damage: 15, cost: 10, speed: 280, cooldown: 0.44, hotkey: '4' }
];

const STAGES = [
    {
        id: 'foundry',
        name: 'Foundry',
        bossName: 'Cinder Ram',
        bossFrame: 'cinder',
        unlock: 'ember',
        weakness: 'tide',
        color: '#ff7c36',
        worldWidth: 2360,
        terrain: [
            [0, GROUND, 2360, 84],
            [310, 356, 210, 20],
            [680, 312, 180, 20],
            [1020, 372, 260, 20],
            [1430, 326, 190, 20],
            [1740, 390, 180, 20]
        ],
        hazards: [[560, GROUND - 12, 120, 12], [1290, GROUND - 12, 120, 12]],
        enemies: [[430, 316, 'turret'], [1180, 332, 'hopper'], [1560, 286, 'turret']]
    },
    {
        id: 'hydro',
        name: 'Hydro Lab',
        bossName: 'Tide Warden',
        bossFrame: 'tide',
        unlock: 'tide',
        weakness: 'storm',
        color: '#52cfff',
        worldWidth: 2320,
        terrain: [
            [0, GROUND, 2320, 84],
            [280, 378, 220, 20],
            [620, 326, 190, 20],
            [920, 282, 160, 20],
            [1260, 362, 260, 20],
            [1650, 310, 210, 20]
        ],
        hazards: [[810, GROUND - 12, 130, 12], [1540, GROUND - 12, 110, 12]],
        enemies: [[520, 338, 'drone'], [1030, 242, 'turret'], [1500, 322, 'drone']]
    },
    {
        id: 'sky',
        name: 'Sky Relay',
        bossName: 'Volt Heron',
        bossFrame: 'volt',
        unlock: 'storm',
        weakness: 'ember',
        color: '#f2df4d',
        worldWidth: 2400,
        terrain: [
            [0, GROUND, 2400, 84],
            [300, 360, 180, 20],
            [620, 302, 180, 20],
            [940, 250, 170, 20],
            [1250, 338, 230, 20],
            [1650, 286, 210, 20]
        ],
        hazards: [[820, GROUND - 12, 100, 12], [1480, GROUND - 12, 130, 12]],
        enemies: [[430, 320, 'drone'], [1080, 210, 'drone'], [1570, 246, 'turret']]
    }
];

const FINAL_STAGE = {
    id: 'final',
    name: 'Null Citadel',
    bossName: 'Null Regent',
    bossFrame: 'null',
    unlock: null,
    weakness: 'cycle',
    color: '#9f7cff',
    worldWidth: 2580,
    terrain: [
        [0, GROUND, 2580, 84],
        [360, 362, 230, 20],
        [760, 306, 210, 20],
        [1130, 378, 240, 20],
        [1510, 314, 220, 20],
        [1870, 272, 190, 20]
    ],
    hazards: [[610, GROUND - 12, 120, 12], [1380, GROUND - 12, 120, 12], [1760, GROUND - 12, 120, 12]],
    enemies: [[510, 322, 'drone'], [920, 266, 'turret'], [1340, 338, 'hopper'], [1780, 232, 'drone']]
};

const images = {
    actors: null,
    backgrounds: {}
};

const keys = new Map();
const pressed = new Set();

let mode = 'loading';
let selectedStage = 0;
let currentStage = null;
let cameraX = 0;
let boss = null;
let bossActive = false;
let message = '';
let messageTimer = 0;
let winTimer = 0;
let selectStartLocked = false;
let lastTime = performance.now();
let accumulator = 0;
let weaponPanelBuilt = false;

const progress = {
    defeated: new Set(),
    weapons: new Set(['buster'])
};

const player = {
    x: 88,
    y: GROUND - 70,
    w: 36,
    h: 64,
    vx: 0,
    vy: 0,
    face: 1,
    hp: 100,
    maxHp: 100,
    invuln: 0,
    cooldown: 0,
    dash: 0,
    shootTimer: 0,
    onGround: false,
    crouch: false,
    weaponIndex: 0,
    energy: { ember: 100, tide: 100, storm: 100 }
};

let bullets = [];
let enemyBullets = [];
let enemies = [];
let particles = [];

function loadImage(src) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = src;
    });
}

async function loadAssets() {
    images.actors = await loadImage(ASSETS.actors);
    const entries = Object.entries(ASSETS.backgrounds);
    await Promise.all(entries.map(async ([key, src]) => {
        images.backgrounds[key] = await loadImage(src);
    }));
    mode = 'select';
    statusEl.textContent = 'Choose a boss stage. Clear all three to open the final citadel.';
    buildWeaponPanel();
    updateWeaponPanel();
}

window.addEventListener('keydown', (event) => {
    const key = event.key.toLowerCase();
    if (['h', 'j', 'k', 'l', 'x', 'z', 'u', 'i', 'r', 'enter', ' ', '1', '2', '3', '4'].includes(key)) {
        event.preventDefault();
    }
    if (!keys.get(key)) {
        pressed.add(key);
    }
    keys.set(key, true);
});

window.addEventListener('keyup', (event) => {
    keys.set(event.key.toLowerCase(), false);
});

function just(key) {
    return pressed.has(key);
}

function down(key) {
    return keys.get(key) === true;
}

function startStage(stage) {
    currentStage = stage;
    cameraX = 0;
    boss = null;
    bossActive = false;
    bullets = [];
    enemyBullets = [];
    particles = [];
    enemies = stage.enemies.map(([x, y, type], index) => ({
        id: index,
        x,
        y,
        type,
        w: type === 'drone' ? 34 : 38,
        h: type === 'drone' ? 28 : 36,
        vx: type === 'drone' ? 55 : 0,
        vy: 0,
        hp: type === 'hopper' ? 24 : 18,
        maxHp: type === 'hopper' ? 24 : 18,
        cooldown: 0.6 + index * 0.17
    }));
    Object.assign(player, {
        x: 88,
        y: GROUND - player.h,
        vx: 0,
        vy: 0,
        hp: player.maxHp,
        invuln: 1,
        cooldown: 0,
        dash: 0,
        shootTimer: 0,
        onGround: false,
        crouch: false
    });
    for (const id of ['ember', 'tide', 'storm']) {
        player.energy[id] = progress.weapons.has(id) ? 100 : 0;
    }
    mode = 'play';
    message = `${stage.name}: ${stage.bossName}`;
    messageTimer = 2.4;
    statusEl.textContent = `${stage.name}. Defeat ${stage.bossName}.`;
    updateWeaponPanel();
}

function restart() {
    if (mode === 'play' && currentStage) {
        startStage(currentStage);
    } else if (mode === 'victory') {
        progress.defeated.clear();
        progress.weapons.clear();
        progress.weapons.add('buster');
        player.weaponIndex = 0;
        mode = 'select';
        statusEl.textContent = 'Run reset. Choose a boss stage.';
        updateWeaponPanel();
    }
}

function handleSelectInput() {
    const finalUnlocked = progress.defeated.size === STAGES.length;
    const choices = finalUnlocked ? [...STAGES, FINAL_STAGE] : STAGES;
    if (just('h')) selectedStage = (selectedStage - 1 + choices.length) % choices.length;
    if (just('l')) selectedStage = (selectedStage + 1) % choices.length;
    selectedStage = Math.min(selectedStage, choices.length - 1);
    const startHeld = down('enter') || down('x') || down(' ');
    if (!startHeld) selectStartLocked = false;
    if ((just('enter') || just('x') || just(' ') || startHeld) && !selectStartLocked) {
        selectStartLocked = true;
        const stage = choices[selectedStage];
        if (stage.id === 'final' || !progress.defeated.has(stage.id)) {
            startStage(stage);
            return;
        }
    }
    statusEl.textContent = finalUnlocked
        ? 'Final citadel open. Enter when ready.'
        : 'Choose an uncleared boss stage. Clear all three to unlock the final.';
}

function switchWeapon(dir) {
    const available = WEAPONS.filter((weapon) => progress.weapons.has(weapon.id));
    if (!available.length) return;
    const currentId = WEAPONS[player.weaponIndex].id;
    let index = Math.max(0, available.findIndex((weapon) => weapon.id === currentId));
    index = (index + dir + available.length) % available.length;
    player.weaponIndex = WEAPONS.findIndex((weapon) => weapon.id === available[index].id);
    updateWeaponPanel();
}

function selectWeapon(index) {
    const weapon = WEAPONS[index];
    if (!weapon || !progress.weapons.has(weapon.id)) return false;
    player.weaponIndex = index;
    updateWeaponPanel();
    return true;
}

function buildWeaponPanel() {
    if (!weaponPanelEl || weaponPanelBuilt) return;
    weaponPanelBuilt = true;
    weaponPanelEl.innerHTML = '';
    for (let i = 0; i < WEAPONS.length; i++) {
        const weapon = WEAPONS[i];
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'weapon-button';
        button.dataset.weaponIndex = String(i);
        button.style.setProperty('--weapon-color', weapon.color);
        button.innerHTML = `<span class="weapon-dot"></span><span class="weapon-name"></span><span class="weapon-meta"></span>`;
        button.addEventListener('click', () => selectWeapon(i));
        weaponPanelEl.appendChild(button);
    }
}

function updateWeaponPanel() {
    if (!weaponPanelEl) return;
    buildWeaponPanel();
    const buttons = weaponPanelEl.querySelectorAll('.weapon-button');
    buttons.forEach((button, index) => {
        const weapon = WEAPONS[index];
        const unlocked = progress.weapons.has(weapon.id);
        const selected = index === player.weaponIndex;
        const energy = weapon.id === 'buster' ? 'inf' : Math.max(0, Math.round(player.energy[weapon.id] || 0));
        button.disabled = !unlocked;
        button.classList.toggle('is-selected', selected);
        button.setAttribute('aria-pressed', selected ? 'true' : 'false');
        button.querySelector('.weapon-name').textContent = `${weapon.hotkey}. ${weapon.name}`;
        button.querySelector('.weapon-meta').textContent = unlocked ? energy : 'locked';
    });
}

function shoot() {
    const weapon = WEAPONS[player.weaponIndex];
    if (player.cooldown > 0) return;
    if (weapon.cost > 0 && player.energy[weapon.id] < weapon.cost) return;
    if (weapon.cost > 0) player.energy[weapon.id] -= weapon.cost;
    player.cooldown = weapon.cooldown;
    player.shootTimer = 0.18;
    updateWeaponPanel();
    const crouchOffset = player.crouch ? 16 : 0;
    bullets.push({
        x: player.x + player.w / 2 + player.face * 22,
        y: player.y + 23 + crouchOffset,
        vx: player.face * weapon.speed,
        vy: weapon.id === 'ember' ? -120 : 0,
        r: weapon.id === 'storm' ? 9 : 5,
        weapon: weapon.id,
        damage: weapon.damage,
        color: weapon.color,
        life: weapon.id === 'storm' ? 1.35 : 1.1,
        pierce: weapon.id === 'tide' ? 1 : 0
    });
}

function updatePlayer(dt) {
    if (just('u')) switchWeapon(-1);
    if (just('i')) switchWeapon(1);
    for (let i = 0; i < WEAPONS.length; i++) {
        if (just(WEAPONS[i].hotkey)) selectWeapon(i);
    }
    if (just('r')) restart();

    player.cooldown = Math.max(0, player.cooldown - dt);
    player.invuln = Math.max(0, player.invuln - dt);
    player.dash = Math.max(0, player.dash - dt);
    player.shootTimer = Math.max(0, player.shootTimer - dt);
    player.crouch = down('j') && player.onGround;

    const dir = (down('l') ? 1 : 0) - (down('h') ? 1 : 0);
    if (dir) player.face = dir;
    const accel = player.crouch ? 0 : 1250;
    const maxSpeed = player.crouch ? 80 : 250;
    player.vx += dir * accel * dt;
    if (!dir) player.vx *= Math.pow(FRICTION, dt * 60);
    player.vx = clamp(player.vx, -maxSpeed, maxSpeed);

    if (just('k') && player.onGround && !player.crouch) {
        player.vy = -610;
        player.onGround = false;
        puff(player.x + player.w / 2, player.y + player.h, '#aefcff', 8);
    }

    if (just('z') && player.dash <= 0 && !player.crouch) {
        player.dash = 0.28;
        player.vx = player.face * 520;
        puff(player.x + player.w / 2 - player.face * 12, player.y + 46, '#56f6e0', 12);
    }

    if (down('x') || down(' ')) shoot();

    player.vy += GRAVITY * dt;
    moveEntity(player, dt, currentStage.terrain);

    player.x = clamp(player.x, 0, currentStage.worldWidth - player.w);
    if (player.y > H + 180) damagePlayer(100);
    for (const hazard of currentStage.hazards) {
        if (rectsOverlap(player, rectFromArray(hazard))) {
            damagePlayer(14);
            player.vy = -420;
            player.vx = -player.face * 240;
        }
    }

    if (!bossActive && player.x > currentStage.worldWidth - 650) {
        activateBoss();
    }
}

function moveEntity(entity, dt, terrain) {
    entity.x += entity.vx * dt;
    for (const rect of terrain.map(rectFromArray)) {
        if (!rectsOverlap(entity, rect)) continue;
        if (entity.vx > 0) entity.x = rect.x - entity.w;
        if (entity.vx < 0) entity.x = rect.x + rect.w;
        entity.vx = 0;
    }

    entity.y += entity.vy * dt;
    entity.onGround = false;
    for (const rect of terrain.map(rectFromArray)) {
        if (!rectsOverlap(entity, rect)) continue;
        if (entity.vy > 0) {
            entity.y = rect.y - entity.h;
            entity.vy = 0;
            entity.onGround = true;
        } else if (entity.vy < 0) {
            entity.y = rect.y + rect.h;
            entity.vy = 0;
        }
    }
}

function activateBoss() {
    bossActive = true;
    const x = currentStage.worldWidth - 260;
    const isFinal = currentStage.id === 'final';
    boss = {
        x,
        y: GROUND - (isFinal ? 170 : 134),
        w: isFinal ? 142 : 130,
        h: isFinal ? 170 : 134,
        hp: isFinal ? 190 : 130,
        maxHp: isFinal ? 190 : 130,
        vx: 0,
        vy: 0,
        phase: 0,
        timer: 0.8,
        cooldown: 0.6,
        invuln: 0
    };
    message = `${currentStage.bossName}`;
    messageTimer = 1.9;
}

function updateEnemies(dt) {
    for (const enemy of enemies) {
        enemy.cooldown -= dt;
        if (enemy.type === 'drone') {
            enemy.x += enemy.vx * dt;
            if (enemy.x < 120 || enemy.x > currentStage.worldWidth - 780 || Math.abs(enemy.x - player.x) > 260) {
                enemy.vx *= -1;
            }
        } else if (enemy.type === 'hopper') {
            enemy.vy += GRAVITY * dt;
            if (enemy.onGround && Math.abs(enemy.x - player.x) < 300) {
                enemy.vy = -520;
                enemy.vx = enemy.x < player.x ? 150 : -150;
                enemy.onGround = false;
            }
            moveEntity(enemy, dt, currentStage.terrain);
            enemy.vx *= Math.pow(0.92, dt * 60);
        }

        if (enemy.cooldown <= 0 && Math.abs(enemy.x - player.x) < 560) {
            enemy.cooldown = enemy.type === 'turret' ? 1.45 : 1.1;
            const dx = player.x + player.w / 2 - (enemy.x + enemy.w / 2);
            const dy = player.y + player.h / 2 - (enemy.y + enemy.h / 2);
            const len = Math.hypot(dx, dy) || 1;
            enemyBullets.push({
                x: enemy.x + enemy.w / 2,
                y: enemy.y + enemy.h / 2,
                vx: dx / len * 260,
                vy: dy / len * 260,
                r: 5,
                damage: 8,
                color: currentStage.color,
                life: 2.4
            });
        }

        if (rectsOverlap(player, enemy)) damagePlayer(10);
    }
    enemies = enemies.filter((enemy) => enemy.hp > 0);
}

function updateBoss(dt) {
    if (!boss) return;
    boss.invuln = Math.max(0, boss.invuln - dt);
    boss.timer -= dt;
    boss.cooldown -= dt;

    if (currentStage.id === 'foundry') {
        if (boss.timer <= 0) {
            boss.timer = 1.8;
            boss.vx = boss.x > player.x ? -310 : 310;
            boss.vy = -260;
        }
        boss.vy += GRAVITY * dt;
        moveBoss(dt);
        if (boss.cooldown <= 0) {
            boss.cooldown = 0.9;
            bossShot(-320, -80, 8, '#ff7c36');
            bossShot(-280, -220, 7, '#ffb743');
        }
    } else if (currentStage.id === 'hydro') {
        boss.y = GROUND - boss.h - 46 + Math.sin(performance.now() / 360) * 28;
        boss.x = currentStage.worldWidth - 268 + Math.sin(performance.now() / 530) * 42;
        if (boss.cooldown <= 0) {
            boss.cooldown = 0.55;
            for (let i = -1; i <= 1; i++) bossShot(-300, i * 95, 6, '#5cc7ff');
        }
    } else if (currentStage.id === 'sky') {
        boss.y = GROUND - boss.h - 80 + Math.sin(performance.now() / 280) * 48;
        boss.x = currentStage.worldWidth - 270 + Math.sin(performance.now() / 440) * 80;
        if (boss.cooldown <= 0) {
            boss.cooldown = 0.72;
            bossShot(-330, 0, 6, '#f3dd4e');
            enemyBullets.push({
                x: player.x + player.w / 2,
                y: 92,
                vx: 0,
                vy: 360,
                r: 7,
                damage: 9,
                color: '#f3dd4e',
                life: 1.4
            });
        }
    } else {
        const phaseTwo = boss.hp < boss.maxHp * 0.52;
        boss.y = GROUND - boss.h - 46 + Math.sin(performance.now() / 300) * (phaseTwo ? 52 : 28);
        boss.x = currentStage.worldWidth - 282 + Math.sin(performance.now() / 520) * (phaseTwo ? 96 : 52);
        if (boss.cooldown <= 0) {
            boss.cooldown = phaseTwo ? 0.48 : 0.72;
            const palette = ['#ff8a3d', '#5cc7ff', '#f3dd4e'];
            for (let i = 0; i < (phaseTwo ? 5 : 3); i++) {
                const angle = -Math.PI + (i - 2) * 0.22;
                enemyBullets.push({
                    x: boss.x + 24,
                    y: boss.y + boss.h * 0.45,
                    vx: Math.cos(angle) * 310,
                    vy: Math.sin(angle) * 240,
                    r: 6,
                    damage: 10,
                    color: palette[i % palette.length],
                    life: 2.2
                });
            }
        }
    }

    if (rectsOverlap(player, boss)) damagePlayer(currentStage.id === 'final' ? 16 : 12);
    if (boss.hp <= 0) clearStage();
}

function moveBoss(dt) {
    boss.x += boss.vx * dt;
    if (boss.x < currentStage.worldWidth - 540 || boss.x > currentStage.worldWidth - 150) {
        boss.vx *= -1;
    }
    boss.y += boss.vy * dt;
    if (boss.y + boss.h > GROUND) {
        boss.y = GROUND - boss.h;
        boss.vy = 0;
        boss.vx *= 0.35;
        puff(boss.x + boss.w / 2, GROUND, currentStage.color, 16);
    }
}

function bossShot(vx, vy, r, color) {
    enemyBullets.push({
        x: boss.x + boss.w * 0.3,
        y: boss.y + boss.h * 0.48,
        vx,
        vy,
        r,
        damage: 10,
        color,
        life: 2.2
    });
}

function updateProjectiles(dt) {
    for (const bullet of bullets) {
        bullet.life -= dt;
        bullet.x += bullet.vx * dt;
        bullet.y += bullet.vy * dt;
        if (bullet.weapon === 'ember') bullet.vy += 420 * dt;
        if (bullet.weapon === 'storm') {
            bullet.vx *= Math.pow(0.95, dt * 60);
            bullet.y += Math.sin(performance.now() / 90) * 0.22;
        }
        for (const enemy of enemies) {
            if (circleRect(bullet, enemy)) {
                enemy.hp -= bullet.damage;
                bullet.life = bullet.pierce > 0 ? bullet.life : -1;
                bullet.pierce -= 1;
                puff(bullet.x, bullet.y, bullet.color, 6);
                break;
            }
        }
        if (boss && circleRect(bullet, boss) && boss.invuln <= 0) {
            const weak = currentStage.weakness === bullet.weapon || (currentStage.weakness === 'cycle' && bullet.weapon !== 'buster');
            boss.hp -= weak ? bullet.damage * 1.7 : bullet.damage;
            boss.invuln = 0.08;
            bullet.life = bullet.weapon === 'tide' ? bullet.life : -1;
            puff(bullet.x, bullet.y, weak ? '#ffffff' : bullet.color, weak ? 14 : 7);
        }
    }
    bullets = bullets.filter((bullet) => bullet.life > 0 && bullet.x > cameraX - 80 && bullet.x < cameraX + W + 120 && bullet.y > -80 && bullet.y < H + 80);

    for (const bullet of enemyBullets) {
        bullet.life -= dt;
        bullet.x += bullet.vx * dt;
        bullet.y += bullet.vy * dt;
        if (circleRect(bullet, player)) {
            damagePlayer(bullet.damage);
            bullet.life = -1;
        }
    }
    enemyBullets = enemyBullets.filter((bullet) => bullet.life > 0 && bullet.y < H + 90);
}

function updateParticles(dt) {
    for (const p of particles) {
        p.life -= dt;
        p.x += p.vx * dt;
        p.y += p.vy * dt;
        p.vy += 210 * dt;
    }
    particles = particles.filter((p) => p.life > 0);
}

function updateGame(dt) {
    if (mode === 'select') {
        handleSelectInput();
        return;
    }
    if (mode === 'victory') {
        winTimer += dt;
        if (just('r')) restart();
        return;
    }
    if (mode !== 'play') return;

    updatePlayer(dt);
    updateEnemies(dt);
    updateBoss(dt);
    updateProjectiles(dt);
    updateParticles(dt);
    messageTimer = Math.max(0, messageTimer - dt);
    cameraX = clamp(player.x - W * 0.36, 0, currentStage.worldWidth - W);
    if (bossActive) cameraX = clamp(currentStage.worldWidth - W, 0, currentStage.worldWidth - W);

    if (player.hp <= 0) {
        message = 'System reboot';
        messageTimer = 1.2;
        startStage(currentStage);
    }
}

function clearStage() {
    puff(boss.x + boss.w / 2, boss.y + boss.h / 2, '#ffffff', 48);
    const defeatedId = currentStage.id;
    const unlock = currentStage.unlock;
    if (defeatedId === 'final') {
        mode = 'victory';
        winTimer = 0;
        statusEl.textContent = 'Null Regent defeated. Press r for a fresh run.';
        return;
    }
    progress.defeated.add(defeatedId);
    if (unlock) {
        progress.weapons.add(unlock);
        player.energy[unlock] = 100;
        selectWeapon(WEAPONS.findIndex((weapon) => weapon.id === unlock));
    }
    mode = 'select';
    boss = null;
    selectedStage = progress.defeated.size === STAGES.length ? 3 : selectedStage;
    const weaponName = WEAPONS.find((weapon) => weapon.id === unlock)?.name || 'weapon';
    statusEl.textContent = `${currentStage.bossName} defeated. ${weaponName} acquired.`;
    updateWeaponPanel();
}

function damagePlayer(amount) {
    if (player.invuln > 0) return;
    player.hp = Math.max(0, player.hp - amount);
    player.invuln = 0.9;
    player.vx = -player.face * 260;
    puff(player.x + player.w / 2, player.y + player.h / 2, '#ff5d5d', 10);
}

function puff(x, y, color, count) {
    for (let i = 0; i < count; i++) {
        const angle = Math.random() * Math.PI * 2;
        const speed = 70 + Math.random() * 190;
        particles.push({
            x,
            y,
            vx: Math.cos(angle) * speed,
            vy: Math.sin(angle) * speed,
            life: 0.35 + Math.random() * 0.35,
            color,
            r: 2 + Math.random() * 4
        });
    }
}

function draw() {
    ctx.clearRect(0, 0, W, H);
    if (mode === 'loading') {
        drawLoading();
    } else if (mode === 'select') {
        drawSelect();
    } else if (mode === 'victory') {
        drawBackground(FINAL_STAGE, 0);
        drawVictory();
    } else {
        drawPlay();
    }
}

function drawLoading() {
    ctx.fillStyle = '#07090f';
    ctx.fillRect(0, 0, W, H);
    drawText('LOADING', W / 2, H / 2, 34, '#56f6e0', 'center');
}

function drawSelect() {
    const finalUnlocked = progress.defeated.size === STAGES.length;
    drawBackground(finalUnlocked ? FINAL_STAGE : STAGES[selectedStage], 0);
    ctx.fillStyle = 'rgba(4, 7, 14, 0.72)';
    ctx.fillRect(0, 0, W, H);
    drawText('VIM BLASTER', W / 2, 64, 42, '#56f6e0', 'center');
    drawText(finalUnlocked ? 'Final citadel unlocked' : 'Clear three bosses to unlock the final fight', W / 2, 104, 18, '#e8f7ff', 'center');

    const choices = finalUnlocked ? [...STAGES, FINAL_STAGE] : STAGES;
    const cardW = finalUnlocked ? 250 : 224;
    const cardH = finalUnlocked ? 142 : 218;
    const gap = finalUnlocked ? 22 : 26;
    const positions = finalUnlocked
        ? choices.map((_, i) => ({
            x: W / 2 - cardW - gap / 2 + (i % 2) * (cardW + gap),
            y: 138 + Math.floor(i / 2) * (cardH + 18)
        }))
        : choices.map((_, i) => ({
            x: (W - (choices.length * cardW + (choices.length - 1) * gap)) / 2 + i * (cardW + gap),
            y: 166
        }));
    for (let i = 0; i < choices.length; i++) {
        const stage = choices[i];
        drawStageCard(stage, positions[i].x, positions[i].y, cardW, cardH, i === selectedStage, progress.defeated.has(stage.id));
    }
    drawText('h/l choose   enter or x start', W / 2, finalUnlocked ? 486 : 446, 18, '#ffb743', 'center');
}

function drawStageCard(stage, x, y, w, h, selected, cleared) {
    ctx.fillStyle = selected ? 'rgba(86, 246, 224, 0.18)' : 'rgba(17, 24, 36, 0.9)';
    ctx.strokeStyle = selected ? '#56f6e0' : '#5f7894';
    ctx.lineWidth = selected ? 3 : 2;
    ctx.fillRect(x, y, w, h);
    ctx.strokeRect(x, y, w, h);
    ctx.fillStyle = stage.color;
    ctx.fillRect(x, y, w, 7);
    const compact = h < 180;
    drawText(stage.name.toUpperCase(), x + w / 2, y + (compact ? 30 : 44), compact ? 16 : 20, '#e8f7ff', 'center');
    drawText(stage.bossName, x + w / 2, y + (compact ? 56 : 76), compact ? 12 : 15, '#a7bdd0', 'center');
    drawActor(`${stage.bossFrame}Idle1`, x + w / 2, y + h - 18, compact ? 72 : stage.id === 'final' ? 124 : 108, compact ? 72 : stage.id === 'final' ? 124 : 108, -1, cleared ? 0.5 : 1);
    if (cleared) {
        ctx.fillStyle = 'rgba(7, 9, 15, 0.78)';
        ctx.fillRect(x + w - 78, y + 14, 62, 24);
        ctx.strokeStyle = '#56f6e0';
        ctx.strokeRect(x + w - 78, y + 14, 62, 24);
        drawText('CLEAR', x + w - 47, y + 27, 12, '#56f6e0', 'center');
    }
}

function drawPlay() {
    drawBackground(currentStage, cameraX);
    ctx.save();
    ctx.translate(-cameraX, 0);
    drawTerrain();
    drawHazards();
    drawEnemies();
    drawBullets();
    drawPlayer();
    drawBoss();
    drawParticles();
    ctx.restore();
    drawHud();
    if (messageTimer > 0) {
        ctx.fillStyle = `rgba(7, 9, 15, ${Math.min(0.78, messageTimer)})`;
        ctx.fillRect(0, 176, W, 72);
        drawText(message, W / 2, 222, 28, '#e8f7ff', 'center');
    }
}

function drawBackground(stage, cam) {
    const img = images.backgrounds[stage.id] || images.backgrounds.foundry;
    if (!img) {
        ctx.fillStyle = '#07090f';
        ctx.fillRect(0, 0, W, H);
        return;
    }
    const targetRatio = W / H;
    let srcW = img.width;
    let srcH = Math.round(srcW / targetRatio);
    if (srcH > img.height) {
        srcH = img.height;
        srcW = Math.round(srcH * targetRatio);
    }
    const travel = Math.max(1, stage.worldWidth - W);
    const sx = clamp((img.width - srcW) * (cam / travel), 0, img.width - srcW);
    const sy = Math.max(0, (img.height - srcH) * 0.52);
    ctx.drawImage(img, sx, sy, srcW, srcH, 0, 0, W, H);
    ctx.fillStyle = 'rgba(2, 5, 10, 0.28)';
    ctx.fillRect(0, 0, W, H);
}

function drawTerrain() {
    for (const rect of currentStage.terrain) {
        const [x, y, w, h] = rect;
        const grad = ctx.createLinearGradient(0, y, 0, y + h);
        grad.addColorStop(0, '#38485f');
        grad.addColorStop(0.32, '#1b2635');
        grad.addColorStop(1, '#0c121c');
        ctx.fillStyle = grad;
        ctx.fillRect(x, y, w, h);
        ctx.fillStyle = currentStage.color;
        ctx.globalAlpha = 0.72;
        ctx.fillRect(x, y, w, 4);
        ctx.globalAlpha = 1;
        for (let ix = x; ix < x + w; ix += 46) {
            ctx.strokeStyle = 'rgba(232,247,255,0.12)';
            ctx.beginPath();
            ctx.moveTo(ix, y + 8);
            ctx.lineTo(ix + 24, y + h - 8);
            ctx.stroke();
        }
    }
}

function drawHazards() {
    for (const [x, y, w, h] of currentStage.hazards) {
        const grad = ctx.createLinearGradient(0, y, 0, y + h);
        grad.addColorStop(0, currentStage.color);
        grad.addColorStop(1, '#ffffff');
        ctx.fillStyle = grad;
        ctx.fillRect(x, y, w, h);
        ctx.fillStyle = 'rgba(255,255,255,0.45)';
        for (let i = 0; i < w; i += 18) {
            ctx.fillRect(x + i, y, 9, h);
        }
    }
}

function drawEnemies() {
    for (const enemy of enemies) {
        const cx = enemy.x + enemy.w / 2;
        const cy = enemy.y + enemy.h / 2;
        ctx.save();
        ctx.translate(cx, cy);
        ctx.fillStyle = '#101722';
        ctx.strokeStyle = currentStage.color;
        ctx.lineWidth = 2;
        if (enemy.type === 'drone') {
            ctx.beginPath();
            ctx.ellipse(0, 0, enemy.w / 2, enemy.h / 2, 0, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
            ctx.fillStyle = '#e8f7ff';
            ctx.fillRect(-5, -3, 10, 6);
        } else if (enemy.type === 'hopper') {
            ctx.fillRect(-enemy.w / 2, -enemy.h / 2, enemy.w, enemy.h);
            ctx.strokeRect(-enemy.w / 2, -enemy.h / 2, enemy.w, enemy.h);
            ctx.fillStyle = currentStage.color;
            ctx.fillRect(-10, -20, 20, 8);
        } else {
            ctx.fillRect(-enemy.w / 2, -enemy.h / 2, enemy.w, enemy.h);
            ctx.strokeRect(-enemy.w / 2, -enemy.h / 2, enemy.w, enemy.h);
            ctx.fillStyle = currentStage.color;
            ctx.fillRect(-4, -8, 26, 10);
        }
        ctx.restore();
    }
}

function drawBullets() {
    for (const bullet of bullets) {
        ctx.fillStyle = bullet.color;
        ctx.shadowColor = bullet.color;
        ctx.shadowBlur = 12;
        ctx.beginPath();
        ctx.arc(bullet.x, bullet.y, bullet.r, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;
    }
    for (const bullet of enemyBullets) {
        ctx.fillStyle = bullet.color;
        ctx.shadowColor = bullet.color;
        ctx.shadowBlur = 10;
        ctx.beginPath();
        ctx.arc(bullet.x, bullet.y, bullet.r, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;
    }
}

function drawPlayer() {
    const flicker = player.invuln > 0 && Math.floor(performance.now() / 80) % 2 === 0;
    if (flicker) return;
    const drawH = player.crouch ? 62 : 82;
    drawActor(currentHeroFrame(), player.x + player.w / 2, player.y + player.h + 9, 78, drawH, player.face, player.dash > 0 ? 0.7 : 1);
    if (player.dash > 0) {
        ctx.globalAlpha = 0.26;
        drawActor('heroRun1', player.x + player.w / 2 - player.face * 28, player.y + player.h + 9, 78, drawH, player.face, 1);
        ctx.globalAlpha = 1;
    }
}

function currentHeroFrame() {
    if (!player.onGround) return 'heroJump';
    if (player.shootTimer > 0) return 'heroShoot';
    if (Math.abs(player.vx) > 35) return Math.floor(performance.now() / 120) % 2 ? 'heroRun1' : 'heroRun2';
    return Math.floor(performance.now() / 520) % 2 ? 'heroIdle1' : 'heroIdle2';
}

function drawBoss() {
    if (!boss) return;
    const flicker = boss.invuln > 0 && Math.floor(performance.now() / 45) % 2 === 0;
    if (!flicker) {
        const size = currentStage.id === 'final' ? 178 : currentStage.id === 'sky' ? 152 : 148;
        drawActor(currentBossFrame(), boss.x + boss.w / 2, boss.y + boss.h + 10, size, size, -1, 1);
    }
    ctx.fillStyle = 'rgba(7,9,15,0.78)';
    ctx.fillRect(currentStage.worldWidth - 500, 38, 390, 16);
    ctx.strokeStyle = '#e8f7ff';
    ctx.strokeRect(currentStage.worldWidth - 500, 38, 390, 16);
    ctx.fillStyle = currentStage.color;
    ctx.fillRect(currentStage.worldWidth - 498, 40, 386 * Math.max(0, boss.hp / boss.maxHp), 12);
}

function currentBossFrame() {
    return `${currentStage.bossFrame}Idle${Math.floor(performance.now() / 260) % 2 + 1}`;
}

function drawActor(frame, cx, bottom, width, height, face = 1, alpha = 1) {
    const f = FRAMES[frame];
    if (!f || !images.actors) return;
    ctx.save();
    ctx.globalAlpha *= alpha;
    ctx.translate(cx, bottom);
    ctx.scale(face < 0 ? -1 : 1, 1);
    ctx.drawImage(images.actors, f.sx, f.sy, CELL, CELL, -width / 2, -height, width, height);
    ctx.restore();
}

function drawParticles() {
    for (const p of particles) {
        ctx.globalAlpha = Math.max(0, p.life / 0.7);
        ctx.fillStyle = p.color;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fill();
    }
    ctx.globalAlpha = 1;
}

function drawHud() {
    ctx.fillStyle = 'rgba(7, 9, 15, 0.76)';
    ctx.fillRect(14, 14, 430, 88);
    ctx.strokeStyle = 'rgba(232,247,255,0.35)';
    ctx.strokeRect(14, 14, 430, 88);
    drawBar(30, 30, 132, 12, player.hp / player.maxHp, '#ff5d5d', 'HP');
    const weapon = WEAPONS[player.weaponIndex];
    const energy = weapon.id === 'buster' ? 1 : player.energy[weapon.id] / 100;
    drawBar(30, 58, 132, 12, energy, weapon.color, weapon.name);
    drawWeaponStrip(224, 28, true);
}

function drawWeaponStrip(x, y, showNumbers = false) {
    for (let i = 0; i < WEAPONS.length; i++) {
        const weapon = WEAPONS[i];
        const unlocked = progress.weapons.has(weapon.id);
        ctx.fillStyle = i === player.weaponIndex ? 'rgba(232,247,255,0.22)' : 'rgba(7,9,15,0.64)';
        ctx.strokeStyle = unlocked ? weapon.color : 'rgba(167,189,208,0.38)';
        ctx.fillRect(x + i * 48, y, 38, 38);
        ctx.strokeRect(x + i * 48, y, 38, 38);
        ctx.fillStyle = unlocked ? weapon.color : '#536273';
        ctx.beginPath();
        ctx.arc(x + i * 48 + 19, y + 19, weapon.id === 'storm' ? 9 : 7, 0, Math.PI * 2);
        ctx.fill();
        if (showNumbers) drawText(weapon.hotkey, x + i * 48 + 19, y + 53, 11, '#ffb743', 'center');
    }
}

function drawBar(x, y, w, h, pct, color, label) {
    ctx.fillStyle = '#05080d';
    ctx.fillRect(x, y, w, h);
    ctx.strokeStyle = 'rgba(232,247,255,0.38)';
    ctx.strokeRect(x, y, w, h);
    ctx.fillStyle = color;
    ctx.fillRect(x + 1, y + 1, Math.max(0, (w - 2) * pct), h - 2);
    drawText(label, x + w + 12, y + h, 12, '#e8f7ff', 'left');
}

function drawVictory() {
    ctx.fillStyle = 'rgba(4, 7, 14, 0.64)';
    ctx.fillRect(0, 0, W, H);
    drawText('NULL REGENT DOWN', W / 2, 214 + Math.sin(winTimer * 3) * 4, 40, '#56f6e0', 'center');
    drawText('Three weapons stabilized. Press r to run it again.', W / 2, 262, 18, '#e8f7ff', 'center');
    drawActor(Math.floor(performance.now() / 520) % 2 ? 'heroIdle1' : 'heroIdle2', W / 2, 400, 92, 96, 1, 1);
}

function drawText(text, x, y, size, color, align = 'left') {
    ctx.font = `700 ${size}px Trebuchet MS, Verdana, sans-serif`;
    ctx.textAlign = align;
    ctx.textBaseline = 'middle';
    ctx.fillStyle = color;
    ctx.shadowColor = 'rgba(0,0,0,0.7)';
    ctx.shadowBlur = 5;
    ctx.fillText(text, x, y);
    ctx.shadowBlur = 0;
}

function rectFromArray(rect) {
    return { x: rect[0], y: rect[1], w: rect[2], h: rect[3] };
}

function rectsOverlap(a, b) {
    return a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;
}

function circleRect(circle, rect) {
    const x = clamp(circle.x, rect.x, rect.x + rect.w);
    const y = clamp(circle.y, rect.y, rect.y + rect.h);
    return Math.hypot(circle.x - x, circle.y - y) <= circle.r;
}

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

function frame(now) {
    const dt = Math.min(0.05, (now - lastTime) / 1000);
    lastTime = now;
    accumulator += dt;
    let stepped = false;
    while (accumulator >= 1 / 60) {
        updateGame(1 / 60);
        accumulator -= 1 / 60;
        stepped = true;
    }
    draw();
    if (stepped) pressed.clear();
    requestAnimationFrame(frame);
}

loadAssets().catch((error) => {
    console.error(error);
    mode = 'loading';
    statusEl.textContent = 'Asset load failed. Check the browser console.';
});

requestAnimationFrame(frame);
