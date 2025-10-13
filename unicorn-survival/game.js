// Three.js Scene Setup
let scene, camera, renderer, raycaster, pointer;
let world = {};
let player = {
    height: 1.7,
    speed: 0.1,
    jumpVelocity: 0,
    onGround: false,
    hunger: 100,
    wood: 0,
    stone: 0,
    selectedBlock: 'grass',
    yaw: 0,
    pitch: 0
};
let crosshairElement;
let clickPromptElement;

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
    leaves: { id: 5, color: 0x228B22, name: 'Leaves' }
};

const UNICORN_COUNT = 6;
const UNICORN_WANDER_RADIUS = CHUNK_SIZE * (RENDER_DISTANCE - 0.5);
const UNICORN_MIN_TURN_TIME = 2;
const UNICORN_MAX_TURN_TIME = 5;
const UNICORN_BASE_SPEED = 0.8; // blocks per second

const unicorns = [];
const clock = new THREE.Clock();

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
            bobSpeed: 2 + Math.random()
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

        data.turnTimer -= delta;
        if (data.turnTimer <= 0) {
            data.turnTimer = UNICORN_MIN_TURN_TIME + Math.random() * (UNICORN_MAX_TURN_TIME - UNICORN_MIN_TURN_TIME);
            data.heading += (Math.random() - 0.5) * Math.PI * 0.6;
        }

        const moveDistance = data.speed * delta;
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

function attemptPointerLock() {
    if (!renderer || !renderer.domElement) return;
    if (document.pointerLockElement === renderer.domElement) return;
    if (typeof renderer.domElement.requestPointerLock !== 'function') return;

    renderer.domElement.requestPointerLock();
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
    player.yaw = 0;
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
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
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
    document.addEventListener('click', onClick);
    document.addEventListener('contextmenu', onRightClick);
    document.addEventListener('keydown', onKeyDown);
    document.addEventListener('keyup', onKeyUp);
    window.addEventListener('resize', onWindowResize);

    // Pointer lock helpers
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

    chunk.mesh = createChunkMesh(chunk, chunkX, chunkZ);
    if (chunk.mesh) {
        scene.add(chunk.mesh);
    }

    world[chunkKey] = chunk;
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
        vertexColors: true,
        flatShading: true
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

        const geometry = new THREE.BoxGeometry(0.5, 0.5, 0.5);
        const material = new THREE.MeshLambertMaterial({ color: 0xFF0000 });
        const food = new THREE.Mesh(geometry, material);
        food.position.set(x, y, z);
        food.userData = { type: 'food', value: 20 };
        scene.add(food);
    }
}

// Input handlers
function onKeyDown(e) {
    const lowerKey = e.key.toLowerCase();
    keys[lowerKey] = true;

    if (!controls.locked) {
        const lockableKeys = ['w', 'a', 's', 'd', ' ', 'arrowup', 'arrowdown', 'arrowleft', 'arrowright'];
        if (lockableKeys.includes(lowerKey)) {
            attemptPointerLock();
        }
    }

    if (e.key === ' ') {
        controls.jump = true;
    }

    // Block selection
    if (e.key >= '1' && e.key <= '5') {
        const blockTypes = ['grass', 'dirt', 'stone', 'wood', 'leaves'];
        player.selectedBlock = blockTypes[parseInt(e.key) - 1];
        document.getElementById('selectedBlock').textContent = BLOCK_TYPES[player.selectedBlock].name;
    }
}

function onKeyUp(e) {
    keys[e.key.toLowerCase()] = false;
    if (e.key === ' ') {
        controls.jump = false;
    }
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
        // Add resources
        if (blockType === 'wood') player.wood++;
        if (blockType === 'stone') player.stone++;

        chunk.blocks[blockKey] = 'air';
        updateChunkMesh(chunkX, chunkZ);
        updateAdjacentChunkMeshes(chunkX, chunkZ, localX, localZ);
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
        chunk.blocks[blockKey] = blockType;
        updateChunkMesh(chunkX, chunkZ);
        updateAdjacentChunkMeshes(chunkX, chunkZ, localX, localZ);
    }
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

    // Hunger
    player.hunger -= 0.01 * frameFactor;
    player.hunger = Math.max(0, player.hunger);

    if (player.hunger <= 0) {
        gameOver();
    }

    // Collect nearby items
    scene.children.forEach(obj => {
        if (obj.userData.type === 'food') {
            const dist = camera.position.distanceTo(obj.position);
            if (dist < 2) {
                player.hunger = Math.min(100, player.hunger + obj.userData.value);
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

    const hungerBar = document.getElementById('hungerBar');
    if (player.hunger < 30) {
        hungerBar.style.background = 'linear-gradient(90deg, #ff0000 0%, #ff4444 100%)';
    } else if (player.hunger < 60) {
        hungerBar.style.background = 'linear-gradient(90deg, #ffa500 0%, #ffb732 100%)';
    } else {
        hungerBar.style.background = 'linear-gradient(90deg, #f093fb 0%, #f5576c 100%)';
    }
}

function gameOver() {
    controls.locked = false;
    document.exitPointerLock();
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
