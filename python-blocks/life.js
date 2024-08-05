window.initGameOfLife3D = function() {
    let container = document.getElementById('game-of-life-3d');
    if (!container) {
        container = document.createElement('div');
        container.id = 'game-of-life-3d';
        document.body.appendChild(container);
    }

    container.style.position = 'fixed';
    container.style.top = '0';
    container.style.left = '0';
    container.style.width = '100vw';
    container.style.height = '100vh';
    container.style.zIndex = '1000';

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111111);

    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    
    renderer.setSize(window.innerWidth, window.innerHeight);
    container.appendChild(renderer.domElement);

    window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    });

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);
    const pointLight = new THREE.PointLight(0xffffff, 1);
    pointLight.position.set(25, 25, 25);
    scene.add(pointLight);

    const geometry = new THREE.BoxGeometry(0.5, 0.5, 0.5);
    const material = new THREE.MeshPhongMaterial({ color: 0x00ff00 });

    const gridSize = 20;
    let grid = initializeGrid(gridSize);

    let cubes = [];

    function initializeGrid(size) {
        return new Array(size).fill(null).map(() => 
            new Array(size).fill(null).map(() => 
                new Array(size).fill(null).map(() => Math.random() > 0.7)
            )
        );
    }

    function createCubes() {
        cubes.forEach(cube => scene.remove(cube));
        cubes = [];
        for (let x = 0; x < gridSize; x++) {
            for (let y = 0; y < gridSize; y++) {
                for (let z = 0; z < gridSize; z++) {
                    if (grid[x][y][z]) {
                        const cube = new THREE.Mesh(geometry, material);
                        cube.position.set(x - gridSize/2, y - gridSize/2, z - gridSize/2);
                        scene.add(cube);
                        cubes.push(cube);
                    }
                }
            }
        }
    }

    createCubes();

    camera.position.set(15, 15, 15);
    camera.lookAt(scene.position);

    function updateGrid() {
        const newGrid = grid.map(plane => plane.map(row => [...row]));

        for (let x = 0; x < gridSize; x++) {
            for (let y = 0; y < gridSize; y++) {
                for (let z = 0; z < gridSize; z++) {
                    let liveNeighbors = 0;
                    for (let dx = -1; dx <= 1; dx++) {
                        for (let dy = -1; dy <= 1; dy++) {
                            for (let dz = -1; dz <= 1; dz++) {
                                if (dx === 0 && dy === 0 && dz === 0) continue;
                                const nx = (x + dx + gridSize) % gridSize;
                                const ny = (y + dy + gridSize) % gridSize;
                                const nz = (z + dz + gridSize) % gridSize;
                                if (grid[nx][ny][nz]) liveNeighbors++;
                            }
                        }
                    }
                    if (grid[x][y][z]) {
                        newGrid[x][y][z] = liveNeighbors >= 4 && liveNeighbors <= 6;
                    } else {
                        newGrid[x][y][z] = liveNeighbors === 5;
                    }
                }
            }
        }

        return newGrid;
    }

    let animationId;
    let frameCount = 0;

    function animate() {
        animationId = requestAnimationFrame(animate);

        if (frameCount % 10 === 0) {  // Update grid every 10 frames
            grid = updateGrid();
            createCubes();
        }

        camera.position.x = 15 * Math.sin(Date.now() * 0.0005) + 15;
        camera.position.z = 15 * Math.cos(Date.now() * 0.0005) + 15;
        camera.lookAt(scene.position);

        renderer.render(scene, camera);

        frameCount++;
    }
    
    animate();

    // Add controls
    const controls = document.createElement('div');
    controls.style.position = 'fixed';
    controls.style.top = '10px';
    controls.style.left = '10px';
    controls.style.zIndex = '1001';
    container.appendChild(controls);

    const resetButton = document.createElement('button');
    resetButton.textContent = 'Reset';
    resetButton.onclick = () => {
        grid = initializeGrid(gridSize);
        createCubes();
    };
    controls.appendChild(resetButton);

    return function cleanUp() {
        cancelAnimationFrame(animationId);
        if (container.parentNode) {
            container.parentNode.removeChild(container);
        }
    };
};