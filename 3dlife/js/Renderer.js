class Renderer {
    constructor() {
        // Configuration constants
        this.CELL_SIZE = 4;  // Base size for all geometry calculations
        this.WORLD_SIZE = 50;  // Grid dimensions (10x10x10, 20x20x20, etc)
        
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 1, 2000);
        
        this.renderer = new THREE.WebGLRenderer({
            canvas: document.getElementById('gameCanvas'),
            antialias: true
        });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setClearColor(0x000000);
        
        // Create geometry with age attribute
        this.cellGeometry = new THREE.BoxGeometry(
            this.CELL_SIZE * 0.8, 
            this.CELL_SIZE * 0.8, 
            this.CELL_SIZE * 0.8
        );
        
        // Create a reusable buffer for age attribute
        this.ageAttribute = new THREE.BufferAttribute(
            new Float32Array(this.cellGeometry.attributes.position.count),
            1
        );
        this.cellGeometry.setAttribute('age', this.ageAttribute);
        
        this.material = new THREE.ShaderMaterial({
            uniforms: {
                maxAge: { value: 30.0 }
            },
            vertexShader: document.getElementById('vertexShader').textContent,
            fragmentShader: document.getElementById('fragmentShader').textContent,
            transparent: true,
            side: THREE.DoubleSide
        });
        
        // Store active cells and their positions
        this.activeCells = new Map();
        this.meshPool = [];  // Pool of reusable meshes
        
        // Add lighting for better visibility
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(1, 1, 1);
        
        const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.5);
        directionalLight2.position.set(-1, -1, -1);
        
        this.scene.add(ambientLight);
        this.scene.add(directionalLight);
        this.scene.add(directionalLight2);

        // Add grid boxes only for exterior faces
        const gridBoxGeometry = new THREE.BoxGeometry(
            this.CELL_SIZE * 0.9, 
            this.CELL_SIZE * 0.9, 
            this.CELL_SIZE * 0.9
        );
        const gridBoxMaterial = new THREE.LineBasicMaterial({ 
            color: 0x404040,
            transparent: true,
            opacity: 0.15
        });
        this.cellBoxes = new THREE.Group();
        
        const center = new THREE.Vector3(this.WORLD_SIZE/2, this.WORLD_SIZE/2, this.WORLD_SIZE/2);
        
        // Create grid boxes only for the six faces
        for (let z = 0; z < this.WORLD_SIZE; z++) {
            for (let y = 0; y < this.WORLD_SIZE; y++) {
                for (let x = 0; x < this.WORLD_SIZE; x++) {
                    // Only create boxes if we're on any of the six faces
                    if (x === 0 || x === this.WORLD_SIZE - 1 ||
                        y === 0 || y === this.WORLD_SIZE - 1 ||
                        z === 0 || z === this.WORLD_SIZE - 1) {
                        
                        const cellBox = new THREE.LineSegments(
                            new THREE.WireframeGeometry(gridBoxGeometry),
                            gridBoxMaterial
                        );
                        cellBox.position.set(
                            (x - center.x) * this.CELL_SIZE,
                            (y - center.y) * this.CELL_SIZE,
                            (z - center.z) * this.CELL_SIZE
                        );
                        this.cellBoxes.add(cellBox);
                    }
                }
            }
        }
        this.scene.add(this.cellBoxes);

        window.addEventListener('resize', this.onWindowResize.bind(this));
    }

    onWindowResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }

    render(grid) {
        const center = new THREE.Vector3(grid.width/2, grid.height/2, grid.depth/2);
        const newActiveCells = new Set();
        
        // Hide all existing meshes (will be reused or remain hidden)
        for (let mesh of this.meshPool) {
            mesh.visible = false;
        }
        
        let meshIndex = 0;
        
        // Update or create cells
        for (let z = 0; z < grid.depth; z++) {
            for (let y = 0; y < grid.height; y++) {
                for (let x = 0; x < grid.width; x++) {
                    const age = grid.getCellAge(x, y, z);
                    if (age > 0) {
                        let mesh;
                        
                        // Reuse or create mesh
                        if (meshIndex < this.meshPool.length) {
                            mesh = this.meshPool[meshIndex];
                            mesh.visible = true;
                        } else {
                            mesh = new THREE.Mesh(this.cellGeometry, this.material);
                            this.scene.add(mesh);
                            this.meshPool.push(mesh);
                        }
                        
                        // Update position and age
                        mesh.position.set(
                            (x - center.x) * this.CELL_SIZE,
                            (y - center.y) * this.CELL_SIZE,
                            (z - center.z) * this.CELL_SIZE
                        );
                        
                        // Update age attribute for this mesh
                        const ageArray = mesh.geometry.attributes.age.array;
                        ageArray.fill(age);
                        mesh.geometry.attributes.age.needsUpdate = true;
                        
                        meshIndex++;
                    }
                }
            }
        }

        this.renderer.render(this.scene, this.camera);
    }
} 