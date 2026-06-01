// Three.js Scene Setup
let scene, camera, renderer, raycaster, pointer;
let world = {};
let player = {
    height: 1.7,
    speed: 0.1,
    jumpVelocity: 0,
    onGround: false,
    hunger: 100,
    warmth: 100,
    wood: 0,
    stone: 0,
    apples: 0,
    unicornFeed: 0,
    crystals: 0,
    hasAxe: false,
    hasPickaxe: false,
    tamedUnicorns: 0,
    stableBlocksPlaced: 0,
    selectedBlock: 'grass',
    yaw: 0,
    pitch: 0
};
let crosshairElement;
let clickPromptElement;
let messageLogElement;
let suppressNextBlockAction = false;
let lastGeneratedChunkX = null;
let lastGeneratedChunkZ = null;
let ambientLight;
let directionalLight;
let timeOfDay = 0.22;
let dayCount = 1;
let elapsedSurvivalTime = 0;
let campfireLights = {};
let lastCampfireTip = 0;
let gameEnded = false;

const CHUNK_SIZE = 32;
const BLOCK_SIZE = 1;
const WORLD_HEIGHT = 16;
const RENDER_DISTANCE = 3;
const LOOK_SENSITIVITY = 0.002;

// Block types
const BLOCK_TYPES = {
    air: { id: 0, color: null, transparent: true },
    grass: { id: 1, color: 0x90EE90, name: 'Grass' },
    dirt: { id: 2, color: 0x8B4513, name: 'Dirt' },
    stone: { id: 3, color: 0x808080, name: 'Stone' },
    wood: { id: 4, color: 0xDEB887, name: 'Wood' },
    leaves: { id: 5, color: 0x228B22, name: 'Leaves' },
    campfire: { id: 6, color: 0xff8c1a, name: 'Campfire', cost: { wood: 3, stone: 1 } },
    crystal: { id: 7, color: 0xb968ff, name: 'Rainbow Crystal' }
};

const BLOCK_COSTS = {
    grass: {},
    dirt: {},
    stone: { stone: 1 },
    wood: { wood: 1 },
    leaves: {},
    campfire: { wood: 3, stone: 1 }
};

const OBJECTIVES = [
    { id: 'survive', label: 'Survive 3 days', complete: () => dayCount >= 4 },
    { id: 'befriend', label: 'Befriend 3 unicorns', complete: () => player.tamedUnicorns >= 3 },
    { id: 'stable', label: 'Build a 12-block stable', complete: () => player.stableBlocksPlaced >= 12 },
    { id: 'crystal', label: 'Find a rainbow crystal', complete: () => player.crystals >= 1 }
];

const DAY_LENGTH_SECONDS = 180;
const NIGHT_START = 0.72;
const NIGHT_END = 0.22;
const CAMPFIRE_RADIUS = 9;
const SHELTER_CHECK_HEIGHT = 4;

const UNICORN_COUNT = 6;
const UNICORN_WANDER_RADIUS = CHUNK_SIZE * (RENDER_DISTANCE - 0.5);
const UNICORN_MIN_TURN_TIME = 2;
const UNICORN_MAX_TURN_TIME = 5;
const UNICORN_BASE_SPEED = 0.8; // blocks per second

const unicorns = [];
const clock = new THREE.Clock();

function hashCoords(x, z, salt = 0) {
    const value = Math.sin(x * 127.1 + z * 311.7 + salt * 74.7) * 43758.5453;
    return value - Math.floor(value);
}

function showMessage(text) {
    if (!messageLogElement) return;

    const message = document.createElement('div');
    message.className = 'message';
    message.textContent = text;
    messageLogElement.appendChild(message);

    window.setTimeout(() => {
        message.remove();
    }, 3300);
}

function hasResources(cost = {}) {
    return Object.entries(cost).every(([resource, amount]) => player[resource] >= amount);
}

function spendResources(cost = {}) {
    if (!hasResources(cost)) return false;

    Object.entries(cost).forEach(([resource, amount]) => {
        player[resource] -= amount;
    });

    return true;
}

function formatCost(cost = {}) {
    const entries = Object.entries(cost);
    if (entries.length === 0) return 'free';
    return entries.map(([resource, amount]) => `${amount} ${resource}`).join(', ');
}

function isNight() {
    return timeOfDay >= NIGHT_START || timeOfDay < NIGHT_END;
}

function getChunkKey(chunkX, chunkZ) {
    return `${chunkX},${chunkZ}`;
}

function worldToChunkCoords(blockX, blockZ) {
    const flooredX = Math.floor(blockX);
    const flooredZ = Math.floor(blockZ);
    const chunkX = Math.floor(flooredX / CHUNK_SIZE);
    const chunkZ = Math.floor(flooredZ / CHUNK_SIZE);
    const localX = flooredX - chunkX * CHUNK_SIZE;
    const localZ = flooredZ - chunkZ * CHUNK_SIZE;

    return { chunkX, chunkZ, localX, localZ };
}

function getChunk(chunkX, chunkZ) {
    return world[getChunkKey(chunkX, chunkZ)];
}

function getBlockTypeAt(blockX, blockY, blockZ) {
    if (blockY < 0 || blockY >= WORLD_HEIGHT) return 'air';

    const { chunkX, chunkZ, localX, localZ } = worldToChunkCoords(blockX, blockZ);
    const chunk = getChunk(chunkX, chunkZ);
    if (!chunk) return 'air';

    return chunk.blocks[`${localX},${blockY},${localZ}`] || 'air';
}

function getProceduralSurfaceHeight(x, z) {
    const baseHeight = Math.floor(
        WORLD_HEIGHT / 2 +
        3 * Math.sin(x * 0.1) +
        3 * Math.cos(z * 0.1) +
        2 * Math.sin(x * 0.05 + z * 0.05)
    );

    return baseHeight + 1;
}

function getSurfaceHeightAt(worldX, worldZ) {
    const blockX = Math.floor(worldX);
    const blockZ = Math.floor(worldZ);
    const { chunkX, chunkZ, localX, localZ } = worldToChunkCoords(blockX, blockZ);
    const chunk = getChunk(chunkX, chunkZ);

    if (chunk) {
        for (let y = WORLD_HEIGHT - 1; y >= 0; y--) {
            const block = chunk.blocks[`${localX},${y},${localZ}`];
            if (block && block !== 'air') {
                return y + 1;
            }
        }
    }

    return getProceduralSurfaceHeight(blockX, blockZ);
}

function createUnicornMesh() {
    const unicorn = new THREE.Group();

    const bodyMaterial = new THREE.MeshLambertMaterial({ color: 0xffffff });
    const accentColor = new THREE.Color().setHSL(Math.random(), 0.6, 0.65);

    const body = new THREE.Mesh(new THREE.BoxGeometry(1.6, 0.9, 0.6), bodyMaterial);
    body.position.y = 0.55;
    body.castShadow = true;
    body.receiveShadow = true;
    unicorn.add(body);

    const head = new THREE.Mesh(new THREE.BoxGeometry(0.6, 0.6, 0.5), bodyMaterial);
    head.position.set(0.95, 0.9, 0);
    head.castShadow = true;
    head.receiveShadow = true;
    unicorn.add(head);

    const horn = new THREE.Mesh(new THREE.ConeGeometry(0.12, 0.4, 12), new THREE.MeshLambertMaterial({ color: 0xffd700 }));
    horn.position.set(1.25, 1.2, 0);
    horn.rotation.z = -Math.PI / 2;
    horn.castShadow = true;
    unicorn.add(horn);

    const mane = new THREE.Mesh(new THREE.BoxGeometry(0.25, 0.6, 0.6), new THREE.MeshLambertMaterial({ color: accentColor }));
    mane.position.set(0.4, 1.0, 0);
    mane.castShadow = true;
    unicorn.add(mane);

    const tail = new THREE.Mesh(new THREE.ConeGeometry(0.15, 0.5, 12), new THREE.MeshLambertMaterial({ color: accentColor.clone().offsetHSL(0.05, 0, 0) }));
    tail.position.set(-0.85, 0.7, 0);
    tail.rotation.z = Math.PI / 2;
    tail.castShadow = true;
    unicorn.add(tail);

    const legGeometry = new THREE.BoxGeometry(0.2, 0.6, 0.2);
    const legPositions = [
        [0.6, 0.3, 0.18],
        [0.6, 0.3, -0.18],
        [-0.6, 0.3, 0.18],
        [-0.6, 0.3, -0.18]
    ];

    legPositions.forEach(([x, y, z]) => {
        const leg = new THREE.Mesh(legGeometry, bodyMaterial);
        leg.position.set(x, y, z);
        leg.castShadow = true;
        leg.receiveShadow = true;
        unicorn.add(leg);
    });

    return unicorn;
}

function spawnUnicorns(count = UNICORN_COUNT) {
    const centerX = camera.position.x;
    const centerZ = camera.position.z;
    const spawnRadius = UNICORN_WANDER_RADIUS * 0.8;

    for (let i = 0; i < count; i++) {
        const angle = Math.random() * Math.PI * 2;
        const distance = Math.random() * spawnRadius;
        const x = centerX + Math.cos(angle) * distance;
        const z = centerZ + Math.sin(angle) * distance;
        const surface = getSurfaceHeightAt(x, z);

        const unicorn = createUnicornMesh();
        unicorn.position.set(x, surface + 0.45, z);
        unicorn.rotation.y = Math.random() * Math.PI * 2;
        unicorn.userData = {
            type: 'unicorn',
            heading: unicorn.rotation.y,
            speed: UNICORN_BASE_SPEED * (0.8 + Math.random() * 0.4),
            turnTimer: UNICORN_MIN_TURN_TIME + Math.random() * (UNICORN_MAX_TURN_TIME - UNICORN_MIN_TURN_TIME),
            bobPhase: Math.random() * Math.PI * 2,
            bobSpeed: 2 + Math.random(),
            scaredTimer: 0,
            tamed: false
        };

        unicorns.push(unicorn);
        scene.add(unicorn);
    }
}

function updateUnicorns(delta = 0) {
    const radiusSquared = UNICORN_WANDER_RADIUS * UNICORN_WANDER_RADIUS;

    unicorns.forEach(unicorn => {
        const data = unicorn.userData;
        if (!data) return;

        const toPlayerX = camera.position.x - unicorn.position.x;
        const toPlayerZ = camera.position.z - unicorn.position.z;
        const distToPlayerSq = toPlayerX * toPlayerX + toPlayerZ * toPlayerZ;

        if (data.tamed) {
            if (distToPlayerSq > 18) {
                data.heading = Math.atan2(toPlayerZ, toPlayerX);
            } else if (distToPlayerSq < 5) {
                data.heading = Math.atan2(-toPlayerZ, -toPlayerX);
            }
        } else if (data.scaredTimer > 0) {
            data.scaredTimer -= delta;
            data.heading = Math.atan2(-toPlayerZ, -toPlayerX);
        }

        data.turnTimer -= delta;
        if (data.turnTimer <= 0 && !data.tamed && data.scaredTimer <= 0) {
            data.turnTimer = UNICORN_MIN_TURN_TIME + Math.random() * (UNICORN_MAX_TURN_TIME - UNICORN_MIN_TURN_TIME);
            data.heading += (Math.random() - 0.5) * Math.PI * 0.6;
        }

        const moveDistance = data.speed * (data.scaredTimer > 0 ? 1.8 : 1) * delta;
        unicorn.position.x += Math.cos(data.heading) * moveDistance;
        unicorn.position.z += Math.sin(data.heading) * moveDistance;

        const dx = unicorn.position.x - camera.position.x;
        const dz = unicorn.position.z - camera.position.z;
        const distSq = dx * dx + dz * dz;
        if (distSq > radiusSquared) {
            data.heading = Math.atan2(-dz, -dx);
        }

        const surface = getSurfaceHeightAt(unicorn.position.x, unicorn.position.z);
        const targetY = surface + 0.45;
        const baseY = THREE.MathUtils.lerp(unicorn.position.y, targetY, 0.15);

        data.bobPhase += data.bobSpeed * delta;
        if (data.bobPhase > Math.PI * 2) {
            data.bobPhase -= Math.PI * 2;
        }
        const bobOffset = Math.sin(data.bobPhase) * 0.05;
        unicorn.position.y = baseY + bobOffset;

        const targetRotation = data.heading;
        unicorn.rotation.y = THREE.MathUtils.lerp(unicorn.rotation.y, targetRotation, 0.06);
    });
}

function getNearestUnicorn(maxDistance = 4) {
    let nearest = null;
    let nearestDistanceSq = maxDistance * maxDistance;

    unicorns.forEach(unicorn => {
        const dx = unicorn.position.x - camera.position.x;
        const dz = unicorn.position.z - camera.position.z;
        const distSq = dx * dx + dz * dz;
        if (distSq < nearestDistanceSq) {
            nearestDistanceSq = distSq;
            nearest = unicorn;
        }
    });

    return nearest;
}

function tameNearestUnicorn() {
    const unicorn = getNearestUnicorn(5);
    if (!unicorn) {
        showMessage('No unicorn is close enough to feed.');
        return;
    }

    if (unicorn.userData.tamed) {
        showMessage('This unicorn already trusts you.');
        return;
    }

    if (player.unicornFeed <= 0) {
        showMessage('Craft unicorn feed first: press R with 2 apples and 1 crystal.');
        return;
    }

    player.unicornFeed--;
    unicorn.userData.tamed = true;
    unicorn.userData.scaredTimer = 0;
    unicorn.userData.speed *= 1.1;
    player.tamedUnicorns++;

    unicorn.traverse(part => {
        if (part.material && part.material.color) {
            part.material = part.material.clone();
            part.material.color.lerp(new THREE.Color(0xffc6f7), 0.35);
        }
    });

    showMessage('A unicorn joined you. It will follow and steady your hunger.');
    updateUI();
}

function scareNearbyUnicorns(radius = 8) {
    unicorns.forEach(unicorn => {
        if (unicorn.userData.tamed) return;
        if (unicorn.position.distanceTo(camera.position) < radius) {
            unicorn.userData.scaredTimer = 4;
        }
    });
}

function setPointerLockState(isLocked) {
    controls.locked = isLocked;

    if (clickPromptElement) {
        clickPromptElement.style.display = isLocked ? 'none' : 'block';
    }

    if (crosshairElement) {
        crosshairElement.style.display = isLocked ? 'block' : 'none';
    }

    document.body.classList.toggle('pointer-locked', isLocked);
    updateCameraOrientation();

    if (isLocked) {
        document.addEventListener('mousemove', onMouseMove);
    } else {
        document.removeEventListener('mousemove', onMouseMove);
    }
}

function attemptPointerLock(event) {
    if (!renderer || !renderer.domElement) return;
    if (document.pointerLockElement === renderer.domElement) return;
    if (typeof renderer.domElement.requestPointerLock !== 'function') return;

    if (event && (event.type === 'pointerdown' || event.type === 'touchstart')) {
        suppressNextBlockAction = true;
        window.setTimeout(() => {
            suppressNextBlockAction = false;
        }, 250);
    }

    const lockRequest = renderer.domElement.requestPointerLock();
    if (lockRequest && typeof lockRequest.catch === 'function') {
        lockRequest.catch(() => {
            suppressNextBlockAction = false;
            if (clickPromptElement && !controls.locked) {
                clickPromptElement.textContent = 'Click inside the game window to lock the camera (Esc to release)';
            }
        });
    }
}

function updateCameraOrientation() {
    if (!camera) return;

    camera.rotation.order = 'YXZ';
    const maxPitch = Math.PI / 2 - 0.01;
    player.pitch = Math.max(-maxPitch, Math.min(maxPitch, player.pitch));

    camera.rotation.y = player.yaw;
    camera.rotation.x = player.pitch;
    camera.rotation.z = 0;
}

// Controls
const keys = {};
let controls = {
    locked: false,
    moveForward: false,
    moveBackward: false,
    moveLeft: false,
    moveRight: false,
    jump: false
};

// Initialize Three.js
function init() {
    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x87CEEB);
    scene.fog = new THREE.Fog(0x87CEEB, 0, CHUNK_SIZE * RENDER_DISTANCE);

    // Camera
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(CHUNK_SIZE / 2, WORLD_HEIGHT + 5, CHUNK_SIZE / 2);
    player.yaw = Math.PI;
    player.pitch = 0;
    updateCameraOrientation();

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    document.querySelector('.game-container').appendChild(renderer.domElement);

    crosshairElement = document.getElementById('crosshair');
    clickPromptElement = document.getElementById('clickPrompt');
    messageLogElement = document.getElementById('messageLog');
    if (crosshairElement) crosshairElement.style.display = 'none';
    if (clickPromptElement) {
        clickPromptElement.style.display = 'block';
        clickPromptElement.addEventListener('click', attemptPointerLock);
    }

    // Raycaster for block selection
    raycaster = new THREE.Raycaster();
    raycaster.far = 10;
    pointer = new THREE.Vector2(0, 0);

    // Lighting
    ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(50, 100, 50);
    directionalLight.castShadow = true;
    directionalLight.shadow.camera.left = -50;
    directionalLight.shadow.camera.right = 50;
    directionalLight.shadow.camera.top = 50;
    directionalLight.shadow.camera.bottom = -50;
    directionalLight.shadow.mapSize.width = 2048;
    directionalLight.shadow.mapSize.height = 2048;
    scene.add(directionalLight);

    // Event listeners
    document.addEventListener('keydown', onKeyDown);
    document.addEventListener('keyup', onKeyUp);
    window.addEventListener('resize', onWindowResize);

    // Pointer lock helpers
    renderer.domElement.addEventListener('click', onClick);
    renderer.domElement.addEventListener('contextmenu', onRightClick);
    renderer.domElement.addEventListener('pointerdown', attemptPointerLock);
    renderer.domElement.addEventListener('touchstart', attemptPointerLock);

    document.addEventListener('pointerlockchange', () => {
        const isLocked = document.pointerLockElement === renderer.domElement;
        setPointerLockState(isLocked);
    });

    document.addEventListener('pointerlockerror', () => {
        if (clickPromptElement) {
            clickPromptElement.textContent = 'Click inside the game window to lock the camera (Esc to release)';
        }
    });

    setPointerLockState(false);

    // Generate world
    generateWorld();
    movePlayerToSurface();

    // Spawn food and resources
    spawnItems();
    spawnUnicorns();

    // Start game loop
    animate();
}

// Generate voxel world
function generateWorld() {
    const centerChunkX = Math.floor(camera.position.x / CHUNK_SIZE);
    const centerChunkZ = Math.floor(camera.position.z / CHUNK_SIZE);

    for (let cx = centerChunkX - RENDER_DISTANCE; cx <= centerChunkX + RENDER_DISTANCE; cx++) {
        for (let cz = centerChunkZ - RENDER_DISTANCE; cz <= centerChunkZ + RENDER_DISTANCE; cz++) {
            generateChunk(cx, cz);
        }
    }

    lastGeneratedChunkX = centerChunkX;
    lastGeneratedChunkZ = centerChunkZ;
}

function generateChunk(chunkX, chunkZ) {
    const chunkKey = getChunkKey(chunkX, chunkZ);
    if (world[chunkKey]) return;

    const chunk = {
        blocks: [],
        mesh: null
    };

    // Generate terrain
    for (let x = 0; x < CHUNK_SIZE; x++) {
        for (let z = 0; z < CHUNK_SIZE; z++) {
            const worldX = chunkX * CHUNK_SIZE + x;
            const worldZ = chunkZ * CHUNK_SIZE + z;

            // Simple height map using sine waves
            const height = Math.floor(
                WORLD_HEIGHT / 2 +
                3 * Math.sin(worldX * 0.1) +
                3 * Math.cos(worldZ * 0.1) +
                2 * Math.sin(worldX * 0.05 + worldZ * 0.05)
            );

            for (let y = 0; y < WORLD_HEIGHT; y++) {
                let blockType;
                if (y < height - 3) {
                    blockType = 'stone';
                } else if (y < height) {
                    blockType = 'dirt';
                } else if (y === height) {
                    blockType = 'grass';
                } else {
                    blockType = 'air';
                }

                const blockKey = `${x},${y},${z}`;
                chunk.blocks[blockKey] = blockType;
            }

            // Add some trees
            if (Math.random() < 0.02 && height < WORLD_HEIGHT - 5) {
                addTree(chunk, x, height + 1, z);
            }
        }
    }

    addLandmarks(chunk, chunkX, chunkZ);

    chunk.mesh = createChunkMesh(chunk, chunkX, chunkZ);
    if (chunk.mesh) {
        scene.add(chunk.mesh);
    }

    world[chunkKey] = chunk;
}

function addLandmarks(chunk, chunkX, chunkZ) {
    const roll = hashCoords(chunkX, chunkZ, 1);
    const centerX = Math.floor(CHUNK_SIZE / 2);
    const centerZ = Math.floor(CHUNK_SIZE / 2);

    if (roll < 0.18) {
        // Apple groves are safer food pockets with several dense trees.
        for (let i = 0; i < 6; i++) {
            const localX = 5 + Math.floor(hashCoords(chunkX, chunkZ, 10 + i) * (CHUNK_SIZE - 10));
            const localZ = 5 + Math.floor(hashCoords(chunkX, chunkZ, 20 + i) * (CHUNK_SIZE - 10));
            const worldX = chunkX * CHUNK_SIZE + localX;
            const worldZ = chunkZ * CHUNK_SIZE + localZ;
            const baseY = Math.min(WORLD_HEIGHT - 6, getProceduralSurfaceHeight(worldX, worldZ));
            addTree(chunk, localX, baseY, localZ);
        }
    } else if (roll < 0.32) {
        // Crystal outcrops are rarer and exposed, rewarding exploration.
        for (let dx = -2; dx <= 2; dx++) {
            for (let dz = -2; dz <= 2; dz++) {
                if (Math.abs(dx) + Math.abs(dz) > 3) continue;
                const localX = centerX + dx;
                const localZ = centerZ + dz;
                const worldX = chunkX * CHUNK_SIZE + localX;
                const worldZ = chunkZ * CHUNK_SIZE + localZ;
                const y = getProceduralSurfaceHeight(worldX, worldZ);
                chunk.blocks[`${localX},${y},${localZ}`] = hashCoords(worldX, worldZ, 30) > 0.55 ? 'crystal' : 'stone';
            }
        }
    } else if (roll < 0.45) {
        // Dark woods offer more leaves/apples but are harder to move through at night.
        for (let i = 0; i < 12; i++) {
            const localX = 3 + Math.floor(hashCoords(chunkX, chunkZ, 40 + i) * (CHUNK_SIZE - 6));
            const localZ = 3 + Math.floor(hashCoords(chunkX, chunkZ, 60 + i) * (CHUNK_SIZE - 6));
            const worldX = chunkX * CHUNK_SIZE + localX;
            const worldZ = chunkZ * CHUNK_SIZE + localZ;
            const baseY = Math.min(WORLD_HEIGHT - 6, getProceduralSurfaceHeight(worldX, worldZ));
            addTree(chunk, localX, baseY, localZ);
        }
    }
}

function addTree(chunk, x, baseY, z) {
    // Trunk
    for (let y = 0; y < 4; y++) {
        chunk.blocks[`${x},${baseY + y},${z}`] = 'wood';
    }

    // Leaves
    for (let dx = -2; dx <= 2; dx++) {
        for (let dz = -2; dz <= 2; dz++) {
            for (let dy = 3; dy <= 5; dy++) {
                if (dx >= -1 && dx <= 1 && dz >= -1 && dz <= 1 && dy === 5) continue;
                const leafX = x + dx;
                const leafZ = z + dz;
                if (leafX >= 0 && leafX < CHUNK_SIZE && leafZ >= 0 && leafZ < CHUNK_SIZE) {
                    chunk.blocks[`${leafX},${baseY + dy},${leafZ}`] = 'leaves';
                }
            }
        }
    }
}

function createChunkMesh(chunk, chunkX, chunkZ) {
    const geometry = new THREE.BufferGeometry();
    const vertices = [];
    const colors = [];
    const indices = [];
    let vertexCount = 0;

    for (let x = 0; x < CHUNK_SIZE; x++) {
        for (let y = 0; y < WORLD_HEIGHT; y++) {
            for (let z = 0; z < CHUNK_SIZE; z++) {
                const blockKey = `${x},${y},${z}`;
                const blockType = chunk.blocks[blockKey];

                if (!blockType || blockType === 'air') continue;

                const block = BLOCK_TYPES[blockType];
                const worldX = chunkX * CHUNK_SIZE + x;
                const worldZ = chunkZ * CHUNK_SIZE + z;

                // Check each face
                const faces = [
                    { dir: [0, 1, 0], corners: [[0,1,0], [0,1,1], [1,1,1], [1,1,0]] },   // top (up)
                    { dir: [0, -1, 0], corners: [[0,0,0], [1,0,0], [1,0,1], [0,0,1]] }, // bottom (down)
                    { dir: [1, 0, 0], corners: [[1,0,0], [1,1,0], [1,1,1], [1,0,1]] },  // right (east)
                    { dir: [-1, 0, 0], corners: [[0,0,0], [0,0,1], [0,1,1], [0,1,0]] }, // left (west)
                    { dir: [0, 0, 1], corners: [[0,0,1], [1,0,1], [1,1,1], [0,1,1]] },  // front (south)
                    { dir: [0, 0, -1], corners: [[0,0,0], [0,1,0], [1,1,0], [1,0,0]] }  // back (north)
                ];

                faces.forEach(face => {
                    const neighborKey = `${x + face.dir[0]},${y + face.dir[1]},${z + face.dir[2]}`;
                    const neighbor = chunk.blocks[neighborKey];

                    if (!neighbor || neighbor === 'air' || BLOCK_TYPES[neighbor]?.transparent) {
                        // Add face
                        const color = new THREE.Color(block.color);

                        face.corners.forEach(corner => {
                            vertices.push(
                                worldX + corner[0],
                                y + corner[1],
                                worldZ + corner[2]
                            );
                            colors.push(color.r, color.g, color.b);
                        });

                        indices.push(
                            vertexCount, vertexCount + 1, vertexCount + 2,
                            vertexCount, vertexCount + 2, vertexCount + 3
                        );
                        vertexCount += 4;
                    }
                });
            }
        }
    }

    if (vertices.length === 0) return null;

    geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    geometry.setIndex(indices);
    geometry.computeVertexNormals();

    const material = new THREE.MeshLambertMaterial({
        vertexColors: true
    });

    const mesh = new THREE.Mesh(geometry, material);
    mesh.receiveShadow = true;
    mesh.castShadow = true;

    return mesh;
}

// Spawn items
function spawnItems() {
    // Spawn food items as 3D objects
    const spawnRange = CHUNK_SIZE * (RENDER_DISTANCE - 1);
    const halfRange = spawnRange / 2;
    const centerX = camera.position.x;
    const centerZ = camera.position.z;

    for (let i = 0; i < 10; i++) {
        const x = centerX - halfRange + Math.random() * spawnRange;
        const z = centerZ - halfRange + Math.random() * spawnRange;
        const surfaceY = getSurfaceHeightAt(x, z);
        const y = surfaceY + 0.25;

        spawnFoodAt(x, y, z, 20);
    }
}

function spawnFoodAt(x, y, z, value = 20) {
    const geometry = new THREE.BoxGeometry(0.5, 0.5, 0.5);
    const material = new THREE.MeshLambertMaterial({ color: 0xFF0000 });
    const food = new THREE.Mesh(geometry, material);
    food.position.set(x, y, z);
    food.userData = { type: 'food', value };
    scene.add(food);
}

function getBlockKey(x, y, z) {
    return `${x},${y},${z}`;
}

// Input handlers
function onKeyDown(e) {
    const lowerKey = e.key.toLowerCase();
    keys[lowerKey] = true;

    if (!controls.locked) {
        const isSystemShortcut = e.metaKey || e.ctrlKey || e.altKey;
        if (e.key !== 'Escape' && !isSystemShortcut) {
            attemptPointerLock();
        }
    }

    if (e.key === ' ') {
        e.preventDefault();
        controls.jump = true;
    }

    // Block selection
    if (e.key >= '1' && e.key <= '6') {
        const blockTypes = ['grass', 'dirt', 'stone', 'wood', 'leaves', 'campfire'];
        player.selectedBlock = blockTypes[parseInt(e.key) - 1];
        document.getElementById('selectedBlock').textContent = BLOCK_TYPES[player.selectedBlock].name;
    }

    if (lowerKey === 'f') tameNearestUnicorn();
    if (lowerKey === 'c') craftCampfire();
    if (lowerKey === 'x') craftAxe();
    if (lowerKey === 'p') craftPickaxe();
    if (lowerKey === 'r') craftUnicornFeed();
}

function onKeyUp(e) {
    keys[e.key.toLowerCase()] = false;
    if (e.key === ' ') {
        controls.jump = false;
    }
}

function craftCampfire() {
    const cost = BLOCK_COSTS.campfire;
    if (!hasResources(cost)) {
        showMessage(`Campfire needs ${formatCost(cost)}.`);
        return;
    }

    player.selectedBlock = 'campfire';
    showMessage('Campfire selected. Right click to place it where night can reach you.');
    updateUI();
}

function craftAxe() {
    if (player.hasAxe) {
        showMessage('You already have an axe.');
        return;
    }

    const cost = { wood: 2, stone: 1 };
    if (!spendResources(cost)) {
        showMessage(`Axe needs ${formatCost(cost)}.`);
        return;
    }

    player.hasAxe = true;
    showMessage('Axe crafted. Wood and leaves now harvest faster.');
    updateUI();
}

function craftPickaxe() {
    if (player.hasPickaxe) {
        showMessage('You already have a pickaxe.');
        return;
    }

    const cost = { wood: 2, stone: 3 };
    if (!spendResources(cost)) {
        showMessage(`Pickaxe needs ${formatCost(cost)}.`);
        return;
    }

    player.hasPickaxe = true;
    showMessage('Pickaxe crafted. Stone and crystals now yield more.');
    updateUI();
}

function craftUnicornFeed() {
    const cost = { apples: 2, crystals: 1 };
    if (!spendResources(cost)) {
        showMessage(`Unicorn feed needs ${formatCost(cost)}.`);
        return;
    }

    player.unicornFeed++;
    showMessage('Unicorn feed crafted. Press F near a unicorn.');
    updateUI();
}

function onMouseMove(e) {
    if (!controls.locked) return;

    const movementX = e.movementX || 0;
    const movementY = e.movementY || 0;

    player.yaw -= movementX * LOOK_SENSITIVITY;
    player.pitch -= movementY * LOOK_SENSITIVITY;

    updateCameraOrientation();
}

function onClick(e) {
    if (suppressNextBlockAction) {
        suppressNextBlockAction = false;
        return;
    }

    if (!controls.locked) return;

    // Break block
    const intersection = getTargetBlock();
    if (intersection) {
        const pos = intersection.point.clone();
        pos.sub(intersection.face.normal.clone().multiplyScalar(0.5));

        const blockX = Math.floor(pos.x);
        const blockY = Math.floor(pos.y);
        const blockZ = Math.floor(pos.z);

        removeBlock(blockX, blockY, blockZ);
    }
}

function onRightClick(e) {
    e.preventDefault();
    if (suppressNextBlockAction) {
        suppressNextBlockAction = false;
        return;
    }

    if (!controls.locked) return;

    // Place block
    const intersection = getTargetBlock();
    if (intersection) {
        const pos = intersection.point.clone();
        pos.add(intersection.face.normal.clone().multiplyScalar(0.5));

        const blockX = Math.floor(pos.x);
        const blockY = Math.floor(pos.y);
        const blockZ = Math.floor(pos.z);

        placeBlock(blockX, blockY, blockZ, player.selectedBlock);
    }
}

function getTargetBlock() {
    const direction = new THREE.Vector3();
    camera.getWorldDirection(direction);

    raycaster.set(camera.position, direction);

    const meshes = [];
    Object.values(world).forEach(chunk => {
        if (chunk.mesh) meshes.push(chunk.mesh);
    });

    const intersects = raycaster.intersectObjects(meshes);
    return intersects.length > 0 ? intersects[0] : null;
}

function removeBlock(x, y, z) {
    if (y < 0 || y >= WORLD_HEIGHT) return;

    const { chunkX, chunkZ, localX, localZ } = worldToChunkCoords(x, z);
    const chunk = getChunk(chunkX, chunkZ);

    if (!chunk) return;

    const blockKey = `${localX},${y},${localZ}`;

    const blockType = chunk.blocks[blockKey];
    if (blockType && blockType !== 'air') {
        if (blockType === 'wood') {
            player.wood += player.hasAxe ? 2 : 1;
            showMessage(player.hasAxe ? '+2 wood' : '+1 wood');
        }
        if (blockType === 'stone') {
            player.stone += player.hasPickaxe ? 2 : 1;
            showMessage(player.hasPickaxe ? '+2 stone' : '+1 stone');
        }
        if (blockType === 'leaves') {
            const appleChance = player.hasAxe ? 0.45 : 0.25;
            if (Math.random() < appleChance) {
                player.apples++;
                spawnFoodAt(x + 0.5, y + 0.8, z + 0.5, 15);
                showMessage('+1 apple. Apples can be eaten or crafted into feed.');
            }
        }
        if (blockType === 'crystal') {
            player.crystals += player.hasPickaxe ? 2 : 1;
            showMessage(player.hasPickaxe ? '+2 rainbow crystals' : '+1 rainbow crystal');
        }
        if (blockType === 'campfire') {
            removeCampfireLight(x, y, z);
            showMessage('Campfire removed.');
        }

        chunk.blocks[blockKey] = 'air';
        updateChunkMesh(chunkX, chunkZ);
        updateAdjacentChunkMeshes(chunkX, chunkZ, localX, localZ);
        scareNearbyUnicorns();
        updateUI();
    }
}

function placeBlock(x, y, z, blockType) {
    if (y < 0 || y >= WORLD_HEIGHT) return;

    const { chunkX, chunkZ, localX, localZ } = worldToChunkCoords(x, z);
    const chunk = getChunk(chunkX, chunkZ);

    if (!chunk) return;

    const blockKey = `${localX},${y},${localZ}`;

    if (chunk.blocks[blockKey] === 'air') {
        const cost = BLOCK_COSTS[blockType] || {};
        if (!spendResources(cost)) {
            showMessage(`${BLOCK_TYPES[blockType].name} needs ${formatCost(cost)}.`);
            return;
        }

        chunk.blocks[blockKey] = blockType;
        if (blockType === 'wood' || blockType === 'leaves') {
            player.stableBlocksPlaced++;
        }
        if (blockType === 'campfire') {
            addCampfireLight(x, y, z);
            showMessage('Campfire placed. Stay nearby at night to keep warm.');
        }

        updateChunkMesh(chunkX, chunkZ);
        updateAdjacentChunkMeshes(chunkX, chunkZ, localX, localZ);
        updateUI();
    }
}

function addCampfireLight(x, y, z) {
    const key = getBlockKey(x, y, z);
    if (campfireLights[key]) return;

    const light = new THREE.PointLight(0xff9d38, 1.4, CAMPFIRE_RADIUS * 2);
    light.position.set(x + 0.5, y + 0.8, z + 0.5);
    scene.add(light);
    campfireLights[key] = light;
}

function removeCampfireLight(x, y, z) {
    const key = getBlockKey(x, y, z);
    const light = campfireLights[key];
    if (!light) return;

    scene.remove(light);
    delete campfireLights[key];
}

function updateChunkMesh(chunkX, chunkZ) {
    const chunk = getChunk(chunkX, chunkZ);

    if (!chunk) return;

    if (chunk.mesh) {
        scene.remove(chunk.mesh);
        chunk.mesh.geometry.dispose();
        chunk.mesh.material.dispose();
    }

    chunk.mesh = createChunkMesh(chunk, chunkX, chunkZ);
    if (chunk.mesh) {
        scene.add(chunk.mesh);
    }
}

function updateAdjacentChunkMeshes(chunkX, chunkZ, localX, localZ) {
    if (localX === 0) updateChunkMesh(chunkX - 1, chunkZ);
    if (localX === CHUNK_SIZE - 1) updateChunkMesh(chunkX + 1, chunkZ);
    if (localZ === 0) updateChunkMesh(chunkX, chunkZ - 1);
    if (localZ === CHUNK_SIZE - 1) updateChunkMesh(chunkX, chunkZ + 1);
}

function updateVisibleWorld() {
    const centerChunkX = Math.floor(camera.position.x / CHUNK_SIZE);
    const centerChunkZ = Math.floor(camera.position.z / CHUNK_SIZE);

    if (centerChunkX !== lastGeneratedChunkX || centerChunkZ !== lastGeneratedChunkZ) {
        generateWorld();
    }
}

function movePlayerToSurface() {
    if (!camera) return;

    const surfaceHeight = getSurfaceHeightAt(camera.position.x, camera.position.z);
    camera.position.y = surfaceHeight + player.height;
    player.jumpVelocity = 0;
    player.onGround = true;
}

function isNearCampfire() {
    return Object.values(campfireLights).some(light => {
        const dx = light.position.x - camera.position.x;
        const dz = light.position.z - camera.position.z;
        return dx * dx + dz * dz <= CAMPFIRE_RADIUS * CAMPFIRE_RADIUS;
    });
}

function isSheltered() {
    const blockX = Math.floor(camera.position.x);
    const blockZ = Math.floor(camera.position.z);
    const headY = Math.floor(camera.position.y);

    for (let y = headY + 1; y <= Math.min(WORLD_HEIGHT - 1, headY + SHELTER_CHECK_HEIGHT); y++) {
        const block = getBlockTypeAt(blockX, y, blockZ);
        if (block && block !== 'air' && block !== 'leaves') {
            return true;
        }
    }

    return false;
}

function updateWorldTime(delta) {
    const previousTime = timeOfDay;
    timeOfDay += delta / DAY_LENGTH_SECONDS;

    if (timeOfDay >= 1) {
        timeOfDay -= 1;
        dayCount++;
        showMessage(`Day ${dayCount}. The world feels a little less forgiving.`);
    }

    if (previousTime < NIGHT_START && timeOfDay >= NIGHT_START) {
        showMessage('Night falls. Find shelter or stand near a campfire.');
    }

    if (previousTime < NIGHT_END && timeOfDay >= NIGHT_END) {
        showMessage('Morning light returns.');
    }

    const night = isNight();
    const lightFactor = night ? 0.24 : 0.75 + Math.sin(timeOfDay * Math.PI) * 0.2;

    if (ambientLight) ambientLight.intensity = night ? 0.22 : 0.55;
    if (directionalLight) directionalLight.intensity = lightFactor;

    if (scene) {
        const sky = night ? new THREE.Color(0x101936) : new THREE.Color(0x87CEEB);
        scene.background.lerp(sky, 0.02);
        scene.fog.color.lerp(sky, 0.02);
    }
}

function updatePlayer(delta = 0) {
    if (!controls.locked) return;

    const direction = new THREE.Vector3();
    const right = new THREE.Vector3();
    const frameFactor = delta > 0 ? delta * 60 : 1;

    // Get forward and right vectors
    camera.getWorldDirection(direction);
    direction.y = 0;
    direction.normalize();

    right.crossVectors(camera.up, direction).normalize();

    // Movement
    const velocity = new THREE.Vector3();

    if (keys['w']) velocity.add(direction);
    if (keys['s']) velocity.sub(direction);
    if (keys['a']) velocity.add(right);
    if (keys['d']) velocity.sub(right);

    if (velocity.length() > 0) {
        velocity.normalize().multiplyScalar(player.speed * frameFactor);
        camera.position.x += velocity.x;
        camera.position.z += velocity.z;
        updateVisibleWorld();
    }

    // Gravity and jumping
    if (controls.jump && player.onGround) {
        player.jumpVelocity = 0.15;
        player.onGround = false;
    }

    player.jumpVelocity -= 0.01 * frameFactor; // gravity
    camera.position.y += player.jumpVelocity * frameFactor;

    // Ground collision
    const surfaceHeight = getSurfaceHeightAt(camera.position.x, camera.position.z);
    const groundHeight = surfaceHeight + player.height;
    if (camera.position.y <= groundHeight) {
        camera.position.y = groundHeight;
        player.jumpVelocity = 0;
        player.onGround = true;
    }

    const night = isNight();
    const nearCampfire = isNearCampfire();
    const sheltered = isSheltered();
    const tamedSupport = Math.min(player.tamedUnicorns * 0.004 * frameFactor, 0.02 * frameFactor);

    if (night && !nearCampfire && !sheltered) {
        player.warmth -= 0.035 * frameFactor;
        if (Date.now() - lastCampfireTip > 8000) {
            showMessage('You are getting cold. Build cover or place a campfire.');
            lastCampfireTip = Date.now();
        }
    } else {
        player.warmth += (nearCampfire ? 0.08 : 0.025) * frameFactor;
    }
    player.warmth = Math.max(0, Math.min(100, player.warmth));

    // Hunger
    player.hunger -= (0.01 + (player.warmth <= 0 ? 0.02 : 0)) * frameFactor;
    player.hunger += tamedSupport;
    player.hunger = Math.max(0, player.hunger);

    if (player.hunger <= 0) {
        gameOver('You starved!');
    }

    if (player.warmth <= 0) {
        gameOver('You froze in the night.');
    }

    // Collect nearby items
    scene.children.forEach(obj => {
        if (obj.userData.type === 'food') {
            const dist = camera.position.distanceTo(obj.position);
            if (dist < 2) {
                player.hunger = Math.min(100, player.hunger + obj.userData.value);
                showMessage(`+${obj.userData.value} hunger`);
                scene.remove(obj);
            }
        }
    });
}

function updateUI() {
    document.getElementById('hungerBar').style.width = player.hunger + '%';
    document.getElementById('hungerValue').textContent = Math.floor(player.hunger);
    document.getElementById('woodCount').textContent = player.wood;
    document.getElementById('stoneCount').textContent = player.stone;
    document.getElementById('appleCount').textContent = player.apples;
    document.getElementById('feedCount').textContent = player.unicornFeed;
    document.getElementById('crystalCount').textContent = player.crystals;
    document.getElementById('warmthBar').style.width = player.warmth + '%';
    document.getElementById('warmthValue').textContent = Math.floor(player.warmth);
    document.getElementById('dayValue').textContent = dayCount;
    document.getElementById('timeValue').textContent = getTimeLabel();
    document.getElementById('toolStatus').textContent = getToolStatus();
    document.getElementById('selectedBlock').textContent = BLOCK_TYPES[player.selectedBlock].name;

    const hungerBar = document.getElementById('hungerBar');
    if (player.hunger < 30) {
        hungerBar.style.background = 'linear-gradient(90deg, #ff0000 0%, #ff4444 100%)';
    } else if (player.hunger < 60) {
        hungerBar.style.background = 'linear-gradient(90deg, #ffa500 0%, #ffb732 100%)';
    } else {
        hungerBar.style.background = 'linear-gradient(90deg, #f093fb 0%, #f5576c 100%)';
    }

    const warmthBar = document.getElementById('warmthBar');
    if (player.warmth < 30) {
        warmthBar.style.background = 'linear-gradient(90deg, #68a4ff 0%, #a6d8ff 100%)';
    } else {
        warmthBar.style.background = 'linear-gradient(90deg, #ffd166 0%, #f8961e 100%)';
    }

    updateObjectivesUI();
}

function getTimeLabel() {
    if (isNight()) return 'Night';
    if (timeOfDay < 0.35) return 'Morning';
    if (timeOfDay < 0.58) return 'Midday';
    return 'Evening';
}

function getToolStatus() {
    const tools = [];
    if (player.hasAxe) tools.push('Axe');
    if (player.hasPickaxe) tools.push('Pickaxe');
    return tools.length ? tools.join(' + ') : 'Hands';
}

function updateObjectivesUI() {
    const container = document.getElementById('objectives');
    if (!container) return;

    container.innerHTML = '';
    OBJECTIVES.forEach(objective => {
        const item = document.createElement('div');
        item.className = `objective${objective.complete() ? ' complete' : ''}`;
        item.textContent = `${objective.complete() ? 'Done: ' : ''}${objective.label}`;
        container.appendChild(item);
    });
}

function gameOver(message = 'You starved!') {
    if (gameEnded) return;
    gameEnded = true;
    controls.locked = false;
    document.exitPointerLock();
    document.getElementById('gameOverMessage').textContent = message;
    document.getElementById('gameOver').style.display = 'block';
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

// Game loop
function animate() {
    requestAnimationFrame(animate);

    const delta = clock.getDelta();

    elapsedSurvivalTime += delta;
    updateWorldTime(delta);
    updatePlayer(delta);
    updateUnicorns(delta);
    updateUI();

    renderer.render(scene, camera);
}

// Restart
document.getElementById('restartBtn').addEventListener('click', () => {
    location.reload();
});

// Initialize game
init();
