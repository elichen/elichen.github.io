let scene, camera, renderer, saddleGrid, marble, hoveredPosition;
let marbleVelocity = new THREE.Vector3();
let isSimulationRunning = false;
let rotationAngle = 0;

const surfaceData = {
    size: 25,
    divisions: 20,
    getHeight: function(x, z) {
        return (x * x / 25) - (z * z / 25);
    }
};

function initSimulation() {
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x87CEEB);
    
    const aspect = window.innerWidth / window.innerHeight;
    const frustumSize = 20;
    camera = new THREE.OrthographicCamera(frustumSize * aspect / -2, frustumSize * aspect / 2, frustumSize / 2, frustumSize / -2, 0.1, 1000);
    camera.position.set(0, 30, 30);
    camera.lookAt(0, 0, 0);
    
    renderer = new THREE.WebGLRenderer({ canvas: document.getElementById('game-canvas'), antialias: true });
    renderer.setSize(800, 600); // Set to fixed size

    generateSaddleGrid();
    createMarble();

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(10, 20, 10);
    scene.add(directionalLight);

    document.getElementById('reset-button').addEventListener('click', resetSimulation);
    renderer.domElement.addEventListener('mousemove', onMouseMove);
    renderer.domElement.addEventListener('click', onMouseClick);

    animate();
}

function generateSaddleGrid() {
    if (saddleGrid) {
        scene.remove(saddleGrid);
    }
    
    const geometry = new THREE.BufferGeometry();
    const material = new THREE.LineBasicMaterial({ color: 0xffffff });

    const vertices = [];
    const step = surfaceData.size / surfaceData.divisions;

    // Create horizontal lines
    for (let i = 0; i <= surfaceData.divisions; i++) {
        const z = (i * step) - (surfaceData.size / 2);
        for (let j = 0; j < surfaceData.divisions; j++) {
            const x1 = (j * step) - (surfaceData.size / 2);
            const x2 = ((j + 1) * step) - (surfaceData.size / 2);
            const y1 = surfaceData.getHeight(x1, z);
            const y2 = surfaceData.getHeight(x2, z);
            vertices.push(x1, y1, z, x2, y2, z);
        }
    }

    // Create vertical lines
    for (let j = 0; j <= surfaceData.divisions; j++) {
        const x = (j * step) - (surfaceData.size / 2);
        for (let i = 0; i < surfaceData.divisions; i++) {
            const z1 = (i * step) - (surfaceData.size / 2);
            const z2 = ((i + 1) * step) - (surfaceData.size / 2);
            const y1 = surfaceData.getHeight(x, z1);
            const y2 = surfaceData.getHeight(x, z2);
            vertices.push(x, y1, z1, x, y2, z2);
        }
    }

    geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
    saddleGrid = new THREE.LineSegments(geometry, material);
    scene.add(saddleGrid);

    console.log("Saddle-shaped grid generated and added to scene");
}

function createMarble() {
    const geometry = new THREE.SphereGeometry(0.5, 32, 32);
    const material = new THREE.MeshPhongMaterial({ color: 0xFFFFFF });
    marble = new THREE.Mesh(geometry, material);
    marble.visible = false;
    scene.add(marble);
}

function updateMarblePhysics() {
    if (!isSimulationRunning) return;

    const gravity = 3;
    const friction = 0.98;

    const gradientX = -(marble.position.x / 12.5);
    const gradientZ = (marble.position.z / 12.5);

    marbleVelocity.x += gradientX * 0.1;
    marbleVelocity.z += gradientZ * 0.1;
    
    marbleVelocity.y = 0;
    marbleVelocity.multiplyScalar(friction);

    marble.position.add(marbleVelocity);
    marble.position.y = surfaceData.getHeight(marble.position.x, marble.position.z);

    const maxDistance = surfaceData.size / 2 - 1;
    if (marble.position.length() > maxDistance) {
        marble.position.setLength(maxDistance);
        marbleVelocity.multiplyScalar(0.5);
    }

    if (marbleVelocity.length() < 0.001) {
        isSimulationRunning = false;
        console.log("Marble came to rest at:", marble.position);
    }
}

function onMouseMove(event) {
    if (isSimulationRunning) return;

    const rect = renderer.domElement.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    const y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(new THREE.Vector2(x, y), camera);

    const intersects = raycaster.intersectObject(saddleGrid);
    if (intersects.length > 0) {
        hoveredPosition = intersects[0].point;
        marble.position.copy(hoveredPosition);
        marble.visible = true;
    } else {
        marble.visible = false;
    }
}

function onMouseClick() {
    if (isSimulationRunning || !hoveredPosition) return;

    marble.position.copy(hoveredPosition);
    marble.visible = true;
    marbleVelocity.set(0, 0, 0);
    isSimulationRunning = true;
}

function resetSimulation() {
    isSimulationRunning = false;
    marble.visible = false;
    marbleVelocity.set(0, 0, 0);
}

function animate() {
    requestAnimationFrame(animate);
    
    // Rotate camera
    rotationAngle += 0.005; // Adjust this value to change rotation speed
    const radius = 80; // Adjust this value to change camera distance
    camera.position.x = radius * Math.cos(rotationAngle);
    camera.position.z = radius * Math.sin(rotationAngle);
    camera.lookAt(0, 0, 0);
    
    updateMarblePhysics();
    renderer.render(scene, camera);
}

window.onload = initSimulation;

