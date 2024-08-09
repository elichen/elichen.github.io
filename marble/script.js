// Surface data
const surfaceData = {
    size: 50,
    divisions: 50,
    getHeight: function(x, z) {
        return (x * x / 15) - (z * z / 15); // Changed from 25 to 15 to increase curvature
    }
};

let scene, camera, renderer, saddleGrid, marble;
let marbleVelocity = new THREE.Vector3();

function initGame() {
    console.log("Initializing game...");
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x87CEEB);
    
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 30, 30);
    camera.lookAt(0, 0, 0);
    
    renderer = new THREE.WebGLRenderer({ canvas: document.getElementById('game-canvas'), antialias: true });
    renderer.setSize(document.getElementById('game-area').offsetWidth, document.getElementById('game-area').offsetHeight);

    console.log("Canvas size:", renderer.domElement.width, renderer.domElement.height);

    generateSaddleGrid();
    createMarble();

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(10, 20, 10);
    scene.add(directionalLight);

    console.log("Scene setup complete");

    document.addEventListener('keydown', handleKeyDown);

    animate();
}

function generateSaddleGrid() {
    const geometry = new THREE.BufferGeometry();
    const material = new THREE.LineBasicMaterial({ color: 0xffffff });

    const vertices = [];
    const step = surfaceData.size / surfaceData.divisions;

    // Create saddle shape
    for (let i = 0; i <= surfaceData.divisions; i++) {
        const x = (i * step) - (surfaceData.size / 2);
        for (let j = 0; j <= surfaceData.divisions; j++) {
            const z = (j * step) - (surfaceData.size / 2);
            const y = surfaceData.getHeight(x, z);
            vertices.push(x, y, z);
        }
    }

    // Create grid lines
    for (let i = 0; i <= surfaceData.divisions; i++) {
        for (let j = 0; j <= surfaceData.divisions; j++) {
            const index = (i * (surfaceData.divisions + 1) + j) * 3;
            if (i < surfaceData.divisions) {
                vertices.push(
                    vertices[index], vertices[index + 1], vertices[index + 2],
                    vertices[index + 3], vertices[index + 4], vertices[index + 5]
                );
            }
            if (j < surfaceData.divisions) {
                vertices.push(
                    vertices[index], vertices[index + 1], vertices[index + 2],
                    vertices[index + (surfaceData.divisions + 1) * 3], vertices[index + (surfaceData.divisions + 1) * 3 + 1], vertices[index + (surfaceData.divisions + 1) * 3 + 2]
                );
            }
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
    marble.position.set(0, surfaceData.getHeight(0, 0), 0);
    scene.add(marble);
    console.log("Marble added to scene");
}

function updateMarblePhysics() {
    const gravity = 9.8; // Changed to positive
    const friction = 0.98;

    // Calculate gradient at current position
    const gradientX = -(marble.position.x / 12.5); // Negated
    const gradientZ = (marble.position.z / 12.5);  // Removed negation

    // Apply forces
    marbleVelocity.x += gradientX * 0.1;
    marbleVelocity.z += gradientZ * 0.1;
    
    // Remove vertical velocity component
    marbleVelocity.y = 0;

    // Apply friction
    marbleVelocity.multiplyScalar(friction);

    // Update position
    marble.position.add(marbleVelocity);

    // Keep marble on the surface
    marble.position.y = surfaceData.getHeight(marble.position.x, marble.position.z);

    // Boundary check
    const maxDistance = surfaceData.size / 2 - 1;
    if (marble.position.length() > maxDistance) {
        marble.position.setLength(maxDistance);
        marbleVelocity.multiplyScalar(0.5); // Reduce velocity on collision
    }
}

function handleKeyDown(event) {
    const force = 0.1;
    switch(event.key) {
        case 'ArrowUp':
            marbleVelocity.z -= force;
            break;
        case 'ArrowDown':
            marbleVelocity.z += force;
            break;
        case 'ArrowLeft':
            marbleVelocity.x -= force;
            break;
        case 'ArrowRight':
            marbleVelocity.x += force;
            break;
    }
}

function animate() {
    requestAnimationFrame(animate);
    updateMarblePhysics();
    renderer.render(scene, camera);
}

// Initialize the game when the page loads
window.onload = initGame;