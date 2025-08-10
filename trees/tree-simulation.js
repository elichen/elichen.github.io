import { applyCSS } from './css.js';

class TreeSimulation extends HTMLElement {
    constructor() {
        super();
        
        // Create shadow DOM
        this.attachShadow({ mode: 'open' });
        
        // Simulation state
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.animationId = null;
        this.clock = new THREE.Clock();
        
        // Tree data
        this.tree = null;
        this.branches = [];
        this.leaves = [];
        
        // Wind simulation
        this.windDirection = new THREE.Vector3(1, 0, 0.3).normalize();
        this.windStrength = 0.5;
        this.windEnabled = true;
        this.windTime = 0;
        
        // Performance tracking
        this.fpsCounter = 0;
        this.lastFpsUpdate = 0;
        
        // Tree parameters
        this.treeHeight = 8;
        this.treeFullness = 15;
        
        // Initialize component
        this.init();
    }
    
    init() {
        applyCSS(this.shadowRoot);
        this.setupThreeJS();
        this.generateTree();
        this.setupEventListeners();
        this.animate();
        
        // Emit ready event
        setTimeout(() => {
            this.dispatchEvent(new CustomEvent('ready'));
        }, 100);
    }
    
    setupThreeJS() {
        const canvas = this.shadowRoot.getElementById('canvas');
        
        // Scene setup
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x87CEEB);
        this.scene.fog = new THREE.Fog(0x87CEEB, 50, 200);
        
        // Camera setup
        this.camera = new THREE.PerspectiveCamera(
            75, 
            window.innerWidth / window.innerHeight, 
            0.1, 
            1000
        );
        // Position camera to see full tree (trunk height 8 + branches ~4-5 = ~12-13 total height)
        this.camera.position.set(15, 6, 15);
        this.camera.lookAt(0, 6, 0); // Look at tree center, not ground
        
        // Renderer setup
        this.renderer = new THREE.WebGLRenderer({ 
            canvas: canvas, 
            antialias: true,
            alpha: true
        });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        
        // Controls setup
        this.controls = new THREE.OrbitControls(this.camera, canvas);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.maxDistance = 50;
        this.controls.minDistance = 5;
        this.controls.maxPolarAngle = Math.PI * 0.48;
        // Center controls on tree middle instead of ground
        this.controls.target.set(0, 6, 0);
        this.controls.update();
        
        // Lighting setup
        this.setupLighting();
        
        // Ground setup
        this.createGround();
    }
    
    setupLighting() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(0x87CEEB, 0.4);
        this.scene.add(ambientLight);
        
        // Directional light (sun)
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(50, 100, 30);
        directionalLight.castShadow = true;
        
        // Shadow camera setup
        directionalLight.shadow.camera.near = 0.1;
        directionalLight.shadow.camera.far = 200;
        directionalLight.shadow.camera.left = -50;
        directionalLight.shadow.camera.right = 50;
        directionalLight.shadow.camera.top = 50;
        directionalLight.shadow.camera.bottom = -50;
        directionalLight.shadow.mapSize.width = 2048;
        directionalLight.shadow.mapSize.height = 2048;
        
        this.scene.add(directionalLight);
        
        // Secondary fill light
        const fillLight = new THREE.DirectionalLight(0x87CEEB, 0.3);
        fillLight.position.set(-30, 40, -20);
        this.scene.add(fillLight);
    }
    
    createGround() {
        const groundGeometry = new THREE.PlaneGeometry(100, 100);
        const groundMaterial = new THREE.MeshLambertMaterial({ 
            color: 0x7CBB7C,
            transparent: true,
            opacity: 0.8
        });
        const ground = new THREE.Mesh(groundGeometry, groundMaterial);
        ground.rotation.x = -Math.PI / 2;
        ground.position.y = -0.5;
        ground.receiveShadow = true;
        this.scene.add(ground);
    }
    
    generateTree() {
        // Clear existing tree
        if (this.tree) {
            this.scene.remove(this.tree);
        }
        
        this.tree = new THREE.Group();
        this.branches = [];
        this.leaves = [];
        
        // Tree parameters (now dynamic based on user controls)
        const treeParams = {
            trunkHeight: this.treeHeight,
            trunkRadius: 0.5,
            maxDepth: 5,
            branchAngleVariation: Math.PI / 6,
            branchLengthRatio: 0.7,
            branchRadiusRatio: 0.7,
            leafDensity: this.treeFullness, // User-controlled fullness
            leafSize: 0.4,
            leafClusters: Math.max(2, Math.floor(this.treeFullness / 8)) // More clusters for fuller trees
        };
        
        // Generate trunk and branches recursively
        this.generateBranch(
            new THREE.Vector3(0, 0, 0),
            new THREE.Vector3(0, 1, 0),
            treeParams.trunkHeight,
            treeParams.trunkRadius,
            0,
            treeParams
        );
        
        this.scene.add(this.tree);
    }
    
    generateBranch(startPos, direction, length, radius, depth, params) {
        if (depth >= params.maxDepth) return;
        
        const endPos = startPos.clone().add(direction.clone().multiplyScalar(length));
        
        // Create branch geometry
        const segments = Math.max(8, 16 - depth * 2);
        const geometry = new THREE.CylinderGeometry(
            radius * 0.8, // top radius (taper)
            radius,        // bottom radius
            length,
            segments
        );
        
        // Branch material with realistic bark
        const material = new THREE.MeshLambertMaterial({
            color: new THREE.Color().setHSL(0.08, 0.6, 0.3 - depth * 0.05)
        });
        
        const branch = new THREE.Mesh(geometry, material);
        branch.position.copy(startPos.clone().add(direction.clone().multiplyScalar(length * 0.5)));
        branch.lookAt(endPos);
        branch.rotateX(Math.PI / 2);
        branch.castShadow = true;
        
        // Store branch data for wind animation
        const branchData = {
            mesh: branch,
            startPos: startPos.clone(),
            originalDirection: direction.clone(),
            length: length,
            flexibility: Math.max(0.1, 1 - depth * 0.15), // More flexible at ends
            depth: depth
        };
        this.branches.push(branchData);
        this.tree.add(branch);
        
        // Generate child branches
        if (depth < params.maxDepth - 1) {
            const numBranches = depth === 0 ? 4 : Math.max(2, 4 - depth);
            
            for (let i = 0; i < numBranches; i++) {
                // Branch positioning along parent
                const branchPoint = 0.6 + (Math.random() * 0.3);
                const branchStart = startPos.clone().add(
                    direction.clone().multiplyScalar(length * branchPoint)
                );
                
                // Branch direction with natural variation
                const branchDirection = direction.clone();
                const randomAxis = new THREE.Vector3(
                    (Math.random() - 0.5) * 2,
                    Math.random() * 0.5,
                    (Math.random() - 0.5) * 2
                ).normalize();
                
                const angle = params.branchAngleVariation + Math.random() * params.branchAngleVariation;
                branchDirection.applyAxisAngle(randomAxis, angle);
                branchDirection.normalize();
                
                // Recursive branch generation
                this.generateBranch(
                    branchStart,
                    branchDirection,
                    length * params.branchLengthRatio * (0.8 + Math.random() * 0.4),
                    radius * params.branchRadiusRatio,
                    depth + 1,
                    params
                );
            }
        }
        
        // Add leaves to branches (not just end branches for fuller tree)
        if (depth >= params.maxDepth - 3) { // Start adding leaves earlier (3 levels instead of 2)
            // Add multiple leaf clusters along the branch for fuller coverage
            for (let cluster = 0; cluster < params.leafClusters; cluster++) {
                const clusterPos = startPos.clone().add(
                    direction.clone().multiplyScalar(length * (0.4 + cluster * 0.3))
                );
                // More leaves per cluster at higher levels
                const clusterDensity = Math.floor(params.leafDensity * (1 + (params.maxDepth - depth) * 0.3));
                this.generateLeaves(clusterPos, clusterDensity, params.leafSize, depth);
            }
            
            // Extra leaves at branch end
            this.generateLeaves(endPos, params.leafDensity, params.leafSize, depth);
        }
    }
    
    generateLeaves(position, density, size, depth) {
        // Create varied leaf geometries for more realism
        const leafGeometries = [
            new THREE.PlaneGeometry(size, size * 1.2),      // Oval leaves
            new THREE.PlaneGeometry(size * 1.1, size),      // Wide leaves
            new THREE.PlaneGeometry(size * 0.8, size * 1.4) // Narrow leaves
        ];
        
        for (let i = 0; i < density; i++) {
            const leafGeometry = leafGeometries[Math.floor(Math.random() * leafGeometries.length)];
            const leaf = new THREE.Mesh(leafGeometry, this.getLeafMaterial());
            
            // Create more natural clustering around the branch
            const clusterRadius = size * (2 + Math.random() * 2); // Larger spread
            const angle = Math.random() * Math.PI * 2;
            const distance = Math.random() * clusterRadius;
            
            const offset = new THREE.Vector3(
                Math.cos(angle) * distance,
                (Math.random() - 0.3) * clusterRadius * 0.8, // Bias slightly downward for natural droop
                Math.sin(angle) * distance
            );
            
            leaf.position.copy(position).add(offset);
            
            // More natural leaf orientations (not completely random)
            leaf.rotation.set(
                (Math.random() - 0.5) * Math.PI * 0.6, // More horizontal orientation
                Math.random() * Math.PI * 2,           // Random twist
                (Math.random() - 0.5) * Math.PI * 0.4  // Slight tilt
            );
            
            // Scale leaves slightly for variety
            const scale = 0.8 + Math.random() * 0.4;
            leaf.scale.setScalar(scale);
            
            // Store leaf data for wind animation
            const leafData = {
                mesh: leaf,
                originalPosition: position.clone(),
                offset: offset.clone(),
                flexibility: 1.0 + Math.random() * 0.5, // Varied flexibility
                phase: Math.random() * Math.PI * 2
            };
            this.leaves.push(leafData);
            this.tree.add(leaf);
        }
    }
    
    getLeafMaterial() {
        // Use natural green with slight variation for realism
        const hue = 0.25 + (Math.random() - 0.5) * 0.1; // Green hues with variation
        const saturation = 0.6 + Math.random() * 0.3;   // Varied saturation
        const lightness = 0.3 + Math.random() * 0.4;    // Varied brightness
        
        const color = new THREE.Color().setHSL(hue, saturation, lightness);
        
        return new THREE.MeshLambertMaterial({ 
            color: color,
            side: THREE.DoubleSide,
            transparent: true,
            opacity: 0.8
        });
    }
    
    setupEventListeners() {
        const controls = this.shadowRoot.querySelector('.controls');
        
        // Wind strength
        const windStrengthSlider = controls.querySelector('#windStrength');
        const windStrengthValue = controls.querySelector('#windStrengthValue');
        windStrengthSlider.addEventListener('input', (e) => {
            this.windStrength = parseFloat(e.target.value);
            windStrengthValue.textContent = e.target.value;
            this.updateWindInfo();
        });
        
        // Wind direction
        const windDirectionSlider = controls.querySelector('#windDirection');
        windDirectionSlider.addEventListener('input', (e) => {
            const angle = (parseFloat(e.target.value) * Math.PI) / 180;
            this.windDirection.set(Math.cos(angle), 0, Math.sin(angle)).normalize();
        });
        
        // Toggle wind
        controls.querySelector('#toggleWind').addEventListener('click', () => {
            this.toggleWind();
        });
        
        // Regenerate tree
        controls.querySelector('#regenerate').addEventListener('click', () => {
            this.regenerateTree();
        });
        
        // Tree height control
        const treeHeightSlider = controls.querySelector('#treeHeight');
        const treeHeightValue = controls.querySelector('#treeHeightValue');
        treeHeightSlider.addEventListener('input', (e) => {
            this.treeHeight = parseInt(e.target.value);
            treeHeightValue.textContent = e.target.value;
            this.generateTree(); // Regenerate tree with new height
        });
        
        // Tree fullness control
        const treeFullnessSlider = controls.querySelector('#treeFullness');
        const treeFullnessValue = controls.querySelector('#treeFullnessValue');
        treeFullnessSlider.addEventListener('input', (e) => {
            this.treeFullness = parseInt(e.target.value);
            treeFullnessValue.textContent = e.target.value;
            this.generateTree(); // Regenerate tree with new fullness
        });
    }
    
    animate() {
        this.animationId = requestAnimationFrame(() => this.animate());
        
        const deltaTime = this.clock.getDelta();
        this.windTime += deltaTime;
        
        if (this.windEnabled) {
            this.updateWindAnimation(deltaTime);
        }
        
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
        
        // FPS tracking
        this.fpsCounter++;
        if (this.windTime - this.lastFpsUpdate > 1) {
            this.dispatchEvent(new CustomEvent('fps-update', {
                detail: { fps: this.fpsCounter }
            }));
            this.fpsCounter = 0;
            this.lastFpsUpdate = this.windTime;
        }
    }
    
    updateWindAnimation(deltaTime) {
        // Multi-layer wind simulation
        const primaryWind = Math.sin(this.windTime * 0.5) * 0.3;
        const secondaryWind = Math.sin(this.windTime * 1.2) * 0.15;
        const turbulence = Math.sin(this.windTime * 3) * 0.05;
        
        const totalWindStrength = this.windStrength * (primaryWind + secondaryWind + turbulence + 0.5);
        
        // Animate branches
        this.branches.forEach(branchData => {
            const { mesh, originalDirection, flexibility, depth } = branchData;
            
            // Wind effect decreases with depth (trunk moves less)
            const depthFactor = Math.max(0.1, 1 - depth * 0.2);
            const windEffect = totalWindStrength * flexibility * depthFactor;
            
            // Calculate wind force
            const windForce = this.windDirection.clone().multiplyScalar(windEffect * 0.1);
            
            // Apply wind rotation
            const windQuaternion = new THREE.Quaternion().setFromAxisAngle(
                new THREE.Vector3(-windForce.z, 0, windForce.x).normalize(),
                windForce.length() * 2
            );
            
            // Smooth interpolation back to original direction
            const currentQuaternion = mesh.quaternion.clone();
            const targetQuaternion = new THREE.Quaternion().setFromUnitVectors(
                new THREE.Vector3(0, 1, 0),
                originalDirection.clone()
            ).multiply(windQuaternion);
            
            mesh.quaternion.slerp(targetQuaternion, deltaTime * 2);
        });
        
        // Animate leaves
        this.leaves.forEach(leafData => {
            const { mesh, originalPosition, offset, flexibility, phase } = leafData;
            
            const windPhase = this.windTime * 4 + phase;
            const windOffset = new THREE.Vector3(
                Math.sin(windPhase) * totalWindStrength * 0.5,
                Math.sin(windPhase * 1.3) * totalWindStrength * 0.2,
                Math.cos(windPhase * 0.8) * totalWindStrength * 0.3
            );
            
            mesh.position.copy(originalPosition).add(offset).add(windOffset);
            
            // Leaf rotation in wind
            mesh.rotation.x += Math.sin(windPhase) * deltaTime * totalWindStrength;
            mesh.rotation.z += Math.cos(windPhase * 1.2) * deltaTime * totalWindStrength;
        });
    }
    
    toggleWind() {
        this.windEnabled = !this.windEnabled;
        this.updateWindInfo();
        
        const button = this.shadowRoot.querySelector('#toggleWind');
        button.textContent = this.windEnabled ? 'Stop Wind' : 'Start Wind';
        button.classList.toggle('active', this.windEnabled);
    }
    
    regenerateTree() {
        this.generateTree();
    }
    
    
    updateWindInfo() {
        let description;
        if (!this.windEnabled) {
            description = 'Calm';
        } else if (this.windStrength < 0.3) {
            description = 'Light Breeze';
        } else if (this.windStrength < 0.7) {
            description = 'Moderate Breeze';
        } else if (this.windStrength < 1.2) {
            description = 'Strong Breeze';
        } else {
            description = 'Gale Force';
        }
        
        this.dispatchEvent(new CustomEvent('wind-update', {
            detail: { description }
        }));
    }
    
    handleResize() {
        if (this.camera && this.renderer) {
            this.camera.aspect = window.innerWidth / window.innerHeight;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(window.innerWidth, window.innerHeight);
        }
    }
    
    connectedCallback() {
        // Component connected to DOM
    }
    
    disconnectedCallback() {
        // Cleanup
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        if (this.renderer) {
            this.renderer.dispose();
        }
    }
}

// Register the custom element
customElements.define('tree-simulation', TreeSimulation);