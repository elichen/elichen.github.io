import { applyCSS } from './css.js';

// Tree species definitions
const TREE_SPECIES = {
    oak: {
        name: 'Oak',
        trunkColor: { h: 0.08, s: 0.6, l: 0.25 },
        leafColor: { h: 0.28, s: 0.6, l: 0.35 },
        leafColorVariation: 0.08,
        heightRange: [6, 12],
        trunkThickness: 0.05,
        branchAngle: Math.PI / 5,
        leafDensity: 1.2,
        leafSize: 0.4,
        crownShape: 'round'
    },
    pine: {
        name: 'Pine',
        trunkColor: { h: 0.06, s: 0.5, l: 0.2 },
        leafColor: { h: 0.35, s: 0.5, l: 0.25 },
        leafColorVariation: 0.04,
        heightRange: [8, 16],
        trunkThickness: 0.035,
        branchAngle: Math.PI / 3,
        leafDensity: 0.8,
        leafSize: 0.25,
        crownShape: 'conical'
    },
    birch: {
        name: 'Birch',
        trunkColor: { h: 0.1, s: 0.1, l: 0.85 },
        leafColor: { h: 0.22, s: 0.5, l: 0.45 },
        leafColorVariation: 0.1,
        heightRange: [7, 14],
        trunkThickness: 0.025,
        branchAngle: Math.PI / 4,
        leafDensity: 1.0,
        leafSize: 0.3,
        crownShape: 'oval'
    },
    maple: {
        name: 'Maple',
        trunkColor: { h: 0.07, s: 0.55, l: 0.3 },
        leafColor: { h: 0.25, s: 0.65, l: 0.4 },
        leafColorVariation: 0.12,
        heightRange: [5, 10],
        trunkThickness: 0.045,
        branchAngle: Math.PI / 4.5,
        leafDensity: 1.4,
        leafSize: 0.45,
        crownShape: 'round'
    },
    willow: {
        name: 'Willow',
        trunkColor: { h: 0.09, s: 0.45, l: 0.35 },
        leafColor: { h: 0.24, s: 0.4, l: 0.4 },
        leafColorVariation: 0.06,
        heightRange: [6, 11],
        trunkThickness: 0.04,
        branchAngle: Math.PI / 3.5,
        leafDensity: 1.5,
        leafSize: 0.2,
        crownShape: 'weeping'
    }
};

class ForestSimulation extends HTMLElement {
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });

        // Scene components
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.animationId = null;
        this.clock = new THREE.Clock();

        // Forest data
        this.trees = [];
        this.allBranches = [];
        this.allLeaves = [];
        this.leafInstancedMesh = null;
        this.grassInstancedMesh = null;
        this.clouds = [];
        this.terrain = null;

        // Forest parameters
        this.treeCount = 25;
        this.forestRadius = 40;
        this.treeVariety = 0.5;
        this.hillHeight = 3;
        this.grassDensity = 1000;

        // Wind simulation
        this.windDirection = new THREE.Vector3(1, 0, 0.3).normalize();
        this.windStrength = 0.5;
        this.windEnabled = true;
        this.windTime = 0;

        // Time of day
        this.timeOfDay = 12;
        this.dayCycleEnabled = false;
        this.dayCycleSpeed = 0.5; // Hours per second
        this.sunLight = null;
        this.ambientLight = null;
        this.skyColor = new THREE.Color(0x87CEEB);

        // Atmosphere
        this.fogDensity = 0.3;
        this.cloudsEnabled = true;

        // Performance
        this.fpsCounter = 0;
        this.lastFpsUpdate = 0;

        // LOD distances
        this.lodDistances = {
            high: 30,
            medium: 60,
            low: 100
        };

        this.init();
    }

    init() {
        applyCSS(this.shadowRoot);
        this.setupThreeJS();
        this.createTerrain();
        this.createSkybox();
        this.createClouds();
        this.generateForest();
        this.createGrass();
        this.setupEventListeners();
        this.updateTimeOfDay();
        this.animate();

        setTimeout(() => {
            this.dispatchEvent(new CustomEvent('ready'));
        }, 100);
    }

    setupThreeJS() {
        const canvas = this.shadowRoot.getElementById('canvas');

        this.scene = new THREE.Scene();
        this.scene.background = this.skyColor;
        this.scene.fog = new THREE.FogExp2(0x87CEEB, 0.008);

        this.camera = new THREE.PerspectiveCamera(
            60,
            window.innerWidth / window.innerHeight,
            0.1,
            500
        );
        this.camera.position.set(30, 15, 30);
        this.camera.lookAt(0, 5, 0);

        this.renderer = new THREE.WebGLRenderer({
            canvas: canvas,
            antialias: true,
            alpha: true
        });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.0;

        this.controls = new THREE.OrbitControls(this.camera, canvas);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.maxDistance = 150;
        this.controls.minDistance = 5;
        this.controls.maxPolarAngle = Math.PI * 0.48;
        this.controls.target.set(0, 5, 0);
        this.controls.update();

        this.setupLighting();
    }

    setupLighting() {
        // Ambient light (sky color)
        this.ambientLight = new THREE.AmbientLight(0x87CEEB, 0.4);
        this.scene.add(this.ambientLight);

        // Main sun light
        this.sunLight = new THREE.DirectionalLight(0xffffff, 0.8);
        this.sunLight.position.set(50, 100, 30);
        this.sunLight.castShadow = true;
        this.sunLight.shadow.camera.near = 0.1;
        this.sunLight.shadow.camera.far = 300;
        this.sunLight.shadow.camera.left = -80;
        this.sunLight.shadow.camera.right = 80;
        this.sunLight.shadow.camera.top = 80;
        this.sunLight.shadow.camera.bottom = -80;
        this.sunLight.shadow.mapSize.width = 2048;
        this.sunLight.shadow.mapSize.height = 2048;
        this.sunLight.shadow.bias = -0.0001;
        this.scene.add(this.sunLight);

        // Hemisphere light for natural sky/ground color
        const hemiLight = new THREE.HemisphereLight(0x87CEEB, 0x556B2F, 0.3);
        this.scene.add(hemiLight);
    }

    createTerrain() {
        // Create terrain with hills using noise
        const size = 200;
        const segments = 100;
        const geometry = new THREE.PlaneGeometry(size, size, segments, segments);

        const positions = geometry.attributes.position.array;
        for (let i = 0; i < positions.length; i += 3) {
            const x = positions[i];
            const z = positions[i + 1];
            // Multi-octave noise for natural terrain
            const height = this.getTerrainHeight(x, z);
            positions[i + 2] = height;
        }
        geometry.computeVertexNormals();

        // Terrain material with vertex colors for variety
        const colors = [];
        for (let i = 0; i < positions.length; i += 3) {
            const height = positions[i + 2];
            const hue = 0.28 + (Math.random() - 0.5) * 0.05;
            const sat = 0.5 + height * 0.02;
            const light = 0.35 + height * 0.02 + Math.random() * 0.1;
            const color = new THREE.Color().setHSL(hue, sat, light);
            colors.push(color.r, color.g, color.b);
        }
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

        const material = new THREE.MeshLambertMaterial({
            vertexColors: true,
            side: THREE.DoubleSide
        });

        this.terrain = new THREE.Mesh(geometry, material);
        this.terrain.rotation.x = -Math.PI / 2;
        this.terrain.receiveShadow = true;
        this.scene.add(this.terrain);
    }

    getTerrainHeight(x, z) {
        // Multi-octave Perlin-like noise
        const scale1 = 0.02;
        const scale2 = 0.05;
        const scale3 = 0.1;

        const noise1 = Math.sin(x * scale1) * Math.cos(z * scale1) * this.hillHeight;
        const noise2 = Math.sin(x * scale2 + 1.5) * Math.cos(z * scale2 + 2.3) * this.hillHeight * 0.5;
        const noise3 = Math.sin(x * scale3 + 3.7) * Math.cos(z * scale3 + 4.1) * this.hillHeight * 0.25;

        return noise1 + noise2 + noise3;
    }

    createSkybox() {
        // Gradient sky dome
        const skyGeometry = new THREE.SphereGeometry(400, 32, 32);
        const skyMaterial = new THREE.ShaderMaterial({
            uniforms: {
                topColor: { value: new THREE.Color(0x0077be) },
                bottomColor: { value: new THREE.Color(0x87CEEB) },
                offset: { value: 20 },
                exponent: { value: 0.6 }
            },
            vertexShader: `
                varying vec3 vWorldPosition;
                void main() {
                    vec4 worldPosition = modelMatrix * vec4(position, 1.0);
                    vWorldPosition = worldPosition.xyz;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform vec3 topColor;
                uniform vec3 bottomColor;
                uniform float offset;
                uniform float exponent;
                varying vec3 vWorldPosition;
                void main() {
                    float h = normalize(vWorldPosition + offset).y;
                    gl_FragColor = vec4(mix(bottomColor, topColor, max(pow(max(h, 0.0), exponent), 0.0)), 1.0);
                }
            `,
            side: THREE.BackSide
        });

        this.skyDome = new THREE.Mesh(skyGeometry, skyMaterial);
        this.scene.add(this.skyDome);
    }

    createClouds() {
        const cloudGeometry = new THREE.SphereGeometry(1, 8, 8);
        const cloudMaterial = new THREE.MeshLambertMaterial({
            color: 0xffffff,
            transparent: true,
            opacity: 0.9
        });

        // Create cloud clusters
        for (let i = 0; i < 15; i++) {
            const cloudGroup = new THREE.Group();
            const cloudX = (Math.random() - 0.5) * 300;
            const cloudY = 50 + Math.random() * 30;
            const cloudZ = (Math.random() - 0.5) * 300;

            // Each cloud is made of several spheres
            const puffCount = 5 + Math.floor(Math.random() * 8);
            for (let j = 0; j < puffCount; j++) {
                const puff = new THREE.Mesh(cloudGeometry, cloudMaterial);
                const scale = 3 + Math.random() * 5;
                puff.scale.set(scale * 1.5, scale, scale);
                puff.position.set(
                    (Math.random() - 0.5) * 15,
                    (Math.random() - 0.5) * 3,
                    (Math.random() - 0.5) * 8
                );
                cloudGroup.add(puff);
            }

            cloudGroup.position.set(cloudX, cloudY, cloudZ);
            cloudGroup.userData = {
                speed: 0.5 + Math.random() * 1,
                originalX: cloudX
            };

            this.clouds.push(cloudGroup);
            this.scene.add(cloudGroup);
        }
    }

    generateForest() {
        // Clear existing trees
        this.trees.forEach(tree => this.scene.remove(tree.group));
        this.trees = [];
        this.allBranches = [];
        this.allLeaves = [];

        // Remove old instanced mesh
        if (this.leafInstancedMesh) {
            this.scene.remove(this.leafInstancedMesh);
        }

        // Generate tree positions using Poisson disk sampling approximation
        const positions = this.generateTreePositions();

        // Determine species distribution
        const speciesKeys = Object.keys(TREE_SPECIES);

        // Collect all leaf data for instancing
        const allLeafData = [];

        positions.forEach((pos, index) => {
            // Select species based on variety parameter
            let speciesKey;
            if (this.treeVariety < 0.2) {
                speciesKey = speciesKeys[0]; // Mostly oak
            } else {
                speciesKey = speciesKeys[Math.floor(Math.random() * speciesKeys.length)];
            }

            const species = TREE_SPECIES[speciesKey];
            const treeData = this.generateTree(pos, species, index);
            this.trees.push(treeData);

            // Collect leaf data
            treeData.leafData.forEach(leaf => allLeafData.push(leaf));
        });

        // Create instanced mesh for all leaves
        this.createInstancedLeaves(allLeafData);

        this.updateStats();
    }

    generateTreePositions() {
        const positions = [];
        const minDistance = 4; // Minimum distance between trees
        const attempts = this.treeCount * 20;

        for (let i = 0; i < attempts && positions.length < this.treeCount; i++) {
            const angle = Math.random() * Math.PI * 2;
            const radius = Math.sqrt(Math.random()) * this.forestRadius;
            const x = Math.cos(angle) * radius;
            const z = Math.sin(angle) * radius;

            // Check distance from other trees
            let valid = true;
            for (const pos of positions) {
                const dx = pos.x - x;
                const dz = pos.z - z;
                if (Math.sqrt(dx * dx + dz * dz) < minDistance) {
                    valid = false;
                    break;
                }
            }

            if (valid) {
                const y = this.getTerrainHeight(x, z);
                positions.push({ x, y, z });
            }
        }

        return positions;
    }

    generateTree(position, species, treeIndex) {
        const group = new THREE.Group();
        group.position.set(position.x, position.y, position.z);

        const branches = [];
        const leafData = [];

        // Tree parameters with species variation
        const heightRange = species.heightRange;
        const height = heightRange[0] + Math.random() * (heightRange[1] - heightRange[0]);
        const variety = this.treeVariety;

        const params = {
            trunkHeight: height * (0.9 + Math.random() * 0.2 * variety),
            trunkRadius: Math.max(0.2, height * species.trunkThickness),
            maxDepth: Math.min(5, Math.floor(height * 0.25) + 2),
            branchAngle: species.branchAngle * (1 + (Math.random() - 0.5) * variety * 0.3),
            branchLengthRatio: 0.65 + Math.random() * 0.15,
            branchRadiusRatio: 0.7,
            leafDensity: Math.floor(8 * species.leafDensity),
            leafSize: species.leafSize,
            leafClusters: 2,
            species: species,
            treeIndex: treeIndex,
            crownShape: species.crownShape
        };

        // Generate trunk and branches
        this.generateBranch(
            new THREE.Vector3(0, 0, 0),
            new THREE.Vector3(0, 1, 0),
            params.trunkHeight,
            params.trunkRadius,
            0,
            params,
            branches,
            leafData,
            group
        );

        this.scene.add(group);
        this.allBranches.push(...branches);

        return {
            group,
            branches,
            leafData,
            species,
            position,
            height
        };
    }

    generateBranch(startPos, direction, length, radius, depth, params, branches, leafData, group) {
        if (depth >= params.maxDepth || length < 0.3) return;

        const endPos = startPos.clone().add(direction.clone().multiplyScalar(length));

        // LOD: Reduce segments for deeper branches
        const segments = Math.max(4, 8 - depth);
        const geometry = new THREE.CylinderGeometry(
            radius * 0.75,
            radius,
            length,
            segments
        );

        const species = params.species;
        const depthDarken = depth * 0.03;
        const material = new THREE.MeshLambertMaterial({
            color: new THREE.Color().setHSL(
                species.trunkColor.h,
                species.trunkColor.s,
                Math.max(0.15, species.trunkColor.l - depthDarken)
            )
        });

        const branch = new THREE.Mesh(geometry, material);
        branch.position.copy(startPos.clone().add(direction.clone().multiplyScalar(length * 0.5)));
        branch.lookAt(endPos);
        branch.rotateX(Math.PI / 2);
        branch.castShadow = true;

        const branchData = {
            mesh: branch,
            startPos: startPos.clone(),
            originalDirection: direction.clone(),
            length,
            flexibility: Math.max(0.1, 1 - depth * 0.15),
            depth,
            treeIndex: params.treeIndex
        };
        branches.push(branchData);
        group.add(branch);

        // Child branches
        if (depth < params.maxDepth - 1) {
            let numBranches;
            if (depth === 0) {
                numBranches = Math.floor(3 + Math.random() * 3);
            } else {
                numBranches = Math.max(2, 3 - depth);
            }

            for (let i = 0; i < numBranches; i++) {
                const branchPoint = 0.4 + Math.random() * 0.5;
                const branchStart = startPos.clone().add(
                    direction.clone().multiplyScalar(length * branchPoint)
                );

                const branchDirection = direction.clone();

                // Apply crown shape
                if (params.crownShape === 'conical') {
                    // Pine-like upward branches
                    const upBias = 0.3 + depth * 0.1;
                    branchDirection.y += upBias;
                } else if (params.crownShape === 'weeping') {
                    // Willow-like drooping branches
                    if (depth > 1) {
                        branchDirection.y -= 0.5;
                    }
                }

                const randomAxis = new THREE.Vector3(
                    (Math.random() - 0.5) * 2,
                    Math.random() * 0.3,
                    (Math.random() - 0.5) * 2
                ).normalize();

                const angle = params.branchAngle + Math.random() * 0.3;
                branchDirection.applyAxisAngle(randomAxis, angle);
                branchDirection.normalize();

                const lengthScale = 0.6 + Math.random() * 0.5;

                this.generateBranch(
                    branchStart,
                    branchDirection,
                    length * params.branchLengthRatio * lengthScale,
                    radius * params.branchRadiusRatio,
                    depth + 1,
                    params,
                    branches,
                    leafData,
                    group
                );
            }
        }

        // Add leaves
        if (depth >= params.maxDepth - 2) {
            for (let cluster = 0; cluster < params.leafClusters; cluster++) {
                const clusterPos = startPos.clone().add(
                    direction.clone().multiplyScalar(length * (0.5 + cluster * 0.3))
                );
                this.generateLeafData(clusterPos, params.leafDensity, params.leafSize, depth, params, leafData, group);
            }
            this.generateLeafData(endPos, params.leafDensity, params.leafSize, depth, params, leafData, group);
        }
    }

    generateLeafData(position, density, size, depth, params, leafData, group) {
        const species = params.species;
        const worldPos = group.position.clone().add(position);

        for (let i = 0; i < density; i++) {
            const clusterRadius = size * 2.5;
            const angle = Math.random() * Math.PI * 2;
            const distance = Math.random() * clusterRadius;

            const offset = new THREE.Vector3(
                Math.cos(angle) * distance,
                (Math.random() - 0.3) * clusterRadius * 0.8,
                Math.sin(angle) * distance
            );

            // Weeping adjustment
            if (species.crownShape === 'weeping' && depth > 1) {
                offset.y -= Math.random() * 2;
            }

            const leafWorldPos = worldPos.clone().add(offset);

            const hue = species.leafColor.h + (Math.random() - 0.5) * species.leafColorVariation;
            const sat = species.leafColor.s + (Math.random() - 0.5) * 0.2;
            const light = species.leafColor.l + (Math.random() - 0.5) * 0.15;

            leafData.push({
                position: leafWorldPos,
                localOffset: offset.clone(),
                basePosition: position.clone(),
                groupPosition: group.position.clone(),
                rotation: new THREE.Euler(
                    (Math.random() - 0.5) * Math.PI * 0.5,
                    Math.random() * Math.PI * 2,
                    (Math.random() - 0.5) * Math.PI * 0.3
                ),
                scale: (0.7 + Math.random() * 0.6) * size,
                color: new THREE.Color().setHSL(hue, sat, light),
                flexibility: 1.0 + Math.random() * 0.5,
                phase: Math.random() * Math.PI * 2,
                treeIndex: params.treeIndex
            });
        }
    }

    createInstancedLeaves(allLeafData) {
        if (allLeafData.length === 0) return;

        // Create instanced mesh for leaves
        const leafGeometry = new THREE.PlaneGeometry(1, 1.2);
        const leafMaterial = new THREE.MeshLambertMaterial({
            side: THREE.DoubleSide,
            transparent: true,
            opacity: 0.85,
            vertexColors: true
        });

        // We need to use a different approach for per-instance colors
        // Create a basic material and update matrices
        const basicLeafMaterial = new THREE.MeshLambertMaterial({
            side: THREE.DoubleSide,
            transparent: true,
            opacity: 0.85,
            color: 0x4a7c4a
        });

        this.leafInstancedMesh = new THREE.InstancedMesh(
            leafGeometry,
            basicLeafMaterial,
            allLeafData.length
        );

        const matrix = new THREE.Matrix4();
        const quaternion = new THREE.Quaternion();
        const scale = new THREE.Vector3();
        const color = new THREE.Color();

        allLeafData.forEach((leaf, index) => {
            quaternion.setFromEuler(leaf.rotation);
            scale.set(leaf.scale, leaf.scale, leaf.scale);
            matrix.compose(leaf.position, quaternion, scale);
            this.leafInstancedMesh.setMatrixAt(index, matrix);
            this.leafInstancedMesh.setColorAt(index, leaf.color);
        });

        this.leafInstancedMesh.instanceMatrix.needsUpdate = true;
        if (this.leafInstancedMesh.instanceColor) {
            this.leafInstancedMesh.instanceColor.needsUpdate = true;
        }

        this.allLeaves = allLeafData;
        this.scene.add(this.leafInstancedMesh);
    }

    createGrass() {
        if (this.grassInstancedMesh) {
            this.scene.remove(this.grassInstancedMesh);
        }

        if (this.grassDensity === 0) return;

        // Grass blade geometry
        const grassGeometry = new THREE.PlaneGeometry(0.1, 0.5);
        grassGeometry.translate(0, 0.25, 0);

        const grassMaterial = new THREE.MeshLambertMaterial({
            color: 0x4a7c4a,
            side: THREE.DoubleSide,
            transparent: true,
            opacity: 0.9
        });

        this.grassInstancedMesh = new THREE.InstancedMesh(
            grassGeometry,
            grassMaterial,
            this.grassDensity
        );

        const matrix = new THREE.Matrix4();
        const position = new THREE.Vector3();
        const quaternion = new THREE.Quaternion();
        const scale = new THREE.Vector3();
        const color = new THREE.Color();

        for (let i = 0; i < this.grassDensity; i++) {
            const angle = Math.random() * Math.PI * 2;
            const radius = Math.sqrt(Math.random()) * this.forestRadius * 1.2;
            const x = Math.cos(angle) * radius;
            const z = Math.sin(angle) * radius;
            const y = this.getTerrainHeight(x, z);

            position.set(x, y, z);
            quaternion.setFromEuler(new THREE.Euler(0, Math.random() * Math.PI * 2, 0));
            const grassScale = 0.5 + Math.random() * 1;
            scale.set(grassScale, grassScale, grassScale);

            matrix.compose(position, quaternion, scale);
            this.grassInstancedMesh.setMatrixAt(i, matrix);

            // Vary grass color
            const hue = 0.28 + (Math.random() - 0.5) * 0.08;
            const sat = 0.5 + Math.random() * 0.2;
            const light = 0.3 + Math.random() * 0.15;
            color.setHSL(hue, sat, light);
            this.grassInstancedMesh.setColorAt(i, color);
        }

        this.grassInstancedMesh.instanceMatrix.needsUpdate = true;
        if (this.grassInstancedMesh.instanceColor) {
            this.grassInstancedMesh.instanceColor.needsUpdate = true;
        }

        this.scene.add(this.grassInstancedMesh);
    }

    updateTimeOfDay() {
        // Calculate sun position based on time
        const sunAngle = ((this.timeOfDay - 6) / 12) * Math.PI; // 6am = horizon, 12pm = zenith, 6pm = horizon
        const sunHeight = Math.sin(sunAngle);
        const sunDistance = 100;

        this.sunLight.position.set(
            Math.cos(sunAngle) * sunDistance,
            Math.max(5, sunHeight * sunDistance),
            30
        );

        // Adjust light intensity and color based on time
        let sunIntensity, ambientIntensity;
        let sunColor, ambientColor, skyColor, fogColor;

        if (this.timeOfDay < 5 || this.timeOfDay > 20) {
            // Night
            sunIntensity = 0.1;
            ambientIntensity = 0.2;
            sunColor = new THREE.Color(0x4444aa);
            ambientColor = new THREE.Color(0x111133);
            skyColor = new THREE.Color(0x0a0a20);
            fogColor = new THREE.Color(0x0a0a20);
        } else if (this.timeOfDay < 7 || this.timeOfDay > 18) {
            // Dawn/Dusk
            const t = this.timeOfDay < 12 ? (this.timeOfDay - 5) / 2 : (20 - this.timeOfDay) / 2;
            sunIntensity = 0.3 + t * 0.4;
            ambientIntensity = 0.2 + t * 0.2;
            sunColor = new THREE.Color().setHSL(0.08, 0.8, 0.5 + t * 0.2);
            ambientColor = new THREE.Color().setHSL(0.08, 0.5, 0.2 + t * 0.1);
            skyColor = new THREE.Color().setHSL(0.08 + t * 0.1, 0.6, 0.4 + t * 0.2);
            fogColor = skyColor.clone();
        } else {
            // Day
            sunIntensity = 0.8;
            ambientIntensity = 0.4;
            sunColor = new THREE.Color(0xfffaf0);
            ambientColor = new THREE.Color(0x87CEEB);
            skyColor = new THREE.Color(0x87CEEB);
            fogColor = new THREE.Color(0x87CEEB);
        }

        this.sunLight.intensity = sunIntensity;
        this.sunLight.color.copy(sunColor);
        this.ambientLight.intensity = ambientIntensity;
        this.ambientLight.color.copy(ambientColor);
        this.scene.background = skyColor;
        this.scene.fog.color.copy(fogColor);

        // Update sky dome colors
        if (this.skyDome) {
            this.skyDome.material.uniforms.bottomColor.value.copy(skyColor);
            const topColor = skyColor.clone();
            topColor.offsetHSL(0, 0, -0.3);
            this.skyDome.material.uniforms.topColor.value.copy(topColor);
        }

        // Update UI
        const hours = Math.floor(this.timeOfDay);
        const minutes = Math.floor((this.timeOfDay % 1) * 60);
        const timeStr = `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}`;
        const timeDisplay = this.shadowRoot.querySelector('#timeDisplay');
        if (timeDisplay) {
            timeDisplay.textContent = timeStr;
        }
    }

    updateFog() {
        this.scene.fog.density = 0.002 + this.fogDensity * 0.015;
    }

    setupEventListeners() {
        const controls = this.shadowRoot.querySelector('.controls');

        // Tree count
        const treeCountSlider = controls.querySelector('#treeCount');
        treeCountSlider.addEventListener('input', (e) => {
            this.treeCount = parseInt(e.target.value);
            controls.querySelector('#treeCountValue').textContent = e.target.value;
        });
        treeCountSlider.addEventListener('change', () => this.generateForest());

        // Forest radius
        const forestRadiusSlider = controls.querySelector('#forestRadius');
        forestRadiusSlider.addEventListener('input', (e) => {
            this.forestRadius = parseInt(e.target.value);
            controls.querySelector('#forestRadiusValue').textContent = e.target.value;
        });
        forestRadiusSlider.addEventListener('change', () => {
            this.generateForest();
            this.createGrass();
        });

        // Tree variety
        const treeVarietySlider = controls.querySelector('#treeVariety');
        treeVarietySlider.addEventListener('input', (e) => {
            this.treeVariety = parseFloat(e.target.value);
            controls.querySelector('#treeVarietyValue').textContent = e.target.value;
        });
        treeVarietySlider.addEventListener('change', () => this.generateForest());

        // Regenerate button
        controls.querySelector('#regenerate').addEventListener('click', () => {
            this.generateForest();
            this.createGrass();
        });

        // Wind strength
        const windStrengthSlider = controls.querySelector('#windStrength');
        windStrengthSlider.addEventListener('input', (e) => {
            this.windStrength = parseFloat(e.target.value);
            controls.querySelector('#windStrengthValue').textContent = e.target.value;
        });

        // Wind direction
        const windDirectionSlider = controls.querySelector('#windDirection');
        windDirectionSlider.addEventListener('input', (e) => {
            const angle = (parseFloat(e.target.value) * Math.PI) / 180;
            this.windDirection.set(Math.cos(angle), 0, Math.sin(angle)).normalize();
            controls.querySelector('#windDirectionValue').textContent = e.target.value + '°';
        });

        // Toggle wind
        const toggleWindBtn = controls.querySelector('#toggleWind');
        toggleWindBtn.addEventListener('click', () => this.toggleWind());

        // Time of day
        const timeSlider = controls.querySelector('#timeOfDay');
        timeSlider.addEventListener('input', (e) => {
            this.timeOfDay = parseFloat(e.target.value);
            this.updateTimeOfDay();
        });

        // Toggle day cycle
        const toggleDayCycleBtn = controls.querySelector('#toggleDayCycle');
        toggleDayCycleBtn.addEventListener('click', () => {
            this.dayCycleEnabled = !this.dayCycleEnabled;
            toggleDayCycleBtn.textContent = this.dayCycleEnabled ? 'Stop Day Cycle' : 'Start Day Cycle';
            toggleDayCycleBtn.classList.toggle('active', this.dayCycleEnabled);
        });

        // Reset time
        controls.querySelector('#resetTime').addEventListener('click', () => {
            this.timeOfDay = 12;
            timeSlider.value = 12;
            this.updateTimeOfDay();
        });

        // Hill height
        const hillHeightSlider = controls.querySelector('#hillHeight');
        hillHeightSlider.addEventListener('input', (e) => {
            this.hillHeight = parseFloat(e.target.value);
            controls.querySelector('#hillHeightValue').textContent = e.target.value;
        });
        hillHeightSlider.addEventListener('change', () => {
            this.scene.remove(this.terrain);
            this.createTerrain();
            this.generateForest();
            this.createGrass();
        });

        // Grass density
        const grassDensitySlider = controls.querySelector('#grassDensity');
        grassDensitySlider.addEventListener('input', (e) => {
            this.grassDensity = parseInt(e.target.value);
            controls.querySelector('#grassDensityValue').textContent = e.target.value;
        });
        grassDensitySlider.addEventListener('change', () => this.createGrass());

        // Fog density
        const fogDensitySlider = controls.querySelector('#fogDensity');
        fogDensitySlider.addEventListener('input', (e) => {
            this.fogDensity = parseFloat(e.target.value);
            controls.querySelector('#fogDensityValue').textContent = e.target.value;
            this.updateFog();
        });

        // Toggle clouds
        const toggleCloudsBtn = controls.querySelector('#toggleClouds');
        toggleCloudsBtn.addEventListener('click', () => {
            this.cloudsEnabled = !this.cloudsEnabled;
            this.clouds.forEach(cloud => {
                cloud.visible = this.cloudsEnabled;
            });
            toggleCloudsBtn.textContent = this.cloudsEnabled ? 'Hide Clouds' : 'Show Clouds';
            toggleCloudsBtn.classList.toggle('active', this.cloudsEnabled);
        });
    }

    animate() {
        this.animationId = requestAnimationFrame(() => this.animate());

        const deltaTime = this.clock.getDelta();
        this.windTime += deltaTime;

        // Update day cycle
        if (this.dayCycleEnabled) {
            this.timeOfDay += deltaTime * this.dayCycleSpeed;
            if (this.timeOfDay >= 24) this.timeOfDay -= 24;
            this.updateTimeOfDay();

            const timeSlider = this.shadowRoot.querySelector('#timeOfDay');
            if (timeSlider) timeSlider.value = this.timeOfDay;
        }

        // Update wind
        if (this.windEnabled) {
            this.updateWindAnimation(deltaTime);
        }

        // Animate clouds
        if (this.cloudsEnabled) {
            this.clouds.forEach(cloud => {
                cloud.position.x += cloud.userData.speed * deltaTime;
                if (cloud.position.x > 200) {
                    cloud.position.x = -200;
                }
            });
        }

        // Update grass animation
        this.updateGrassAnimation(deltaTime);

        this.controls.update();
        this.renderer.render(this.scene, this.camera);

        // FPS tracking
        this.fpsCounter++;
        if (this.windTime - this.lastFpsUpdate > 1) {
            this.updateStats();
            this.fpsCounter = 0;
            this.lastFpsUpdate = this.windTime;
        }
    }

    updateWindAnimation(deltaTime) {
        const primaryWind = Math.sin(this.windTime * 0.5) * 0.3;
        const secondaryWind = Math.sin(this.windTime * 1.2) * 0.15;
        const turbulence = Math.sin(this.windTime * 3) * 0.05;
        const totalWindStrength = this.windStrength * (primaryWind + secondaryWind + turbulence + 0.5);

        // Animate branches
        this.allBranches.forEach(branchData => {
            const { mesh, originalDirection, flexibility, depth } = branchData;
            const depthFactor = Math.max(0.1, 1 - depth * 0.2);
            const windEffect = totalWindStrength * flexibility * depthFactor;

            const windForce = this.windDirection.clone().multiplyScalar(windEffect * 0.1);

            const windQuaternion = new THREE.Quaternion().setFromAxisAngle(
                new THREE.Vector3(-windForce.z, 0, windForce.x).normalize(),
                windForce.length() * 2
            );

            const targetQuaternion = new THREE.Quaternion().setFromUnitVectors(
                new THREE.Vector3(0, 1, 0),
                originalDirection.clone()
            ).multiply(windQuaternion);

            mesh.quaternion.slerp(targetQuaternion, deltaTime * 2);
        });

        // Animate leaves using instanced mesh
        if (this.leafInstancedMesh && this.allLeaves.length > 0) {
            const matrix = new THREE.Matrix4();
            const position = new THREE.Vector3();
            const quaternion = new THREE.Quaternion();
            const scale = new THREE.Vector3();

            this.allLeaves.forEach((leaf, index) => {
                const windPhase = this.windTime * 4 + leaf.phase;
                const windOffset = new THREE.Vector3(
                    Math.sin(windPhase) * totalWindStrength * 0.5,
                    Math.sin(windPhase * 1.3) * totalWindStrength * 0.2,
                    Math.cos(windPhase * 0.8) * totalWindStrength * 0.3
                );

                position.copy(leaf.groupPosition)
                    .add(leaf.basePosition)
                    .add(leaf.localOffset)
                    .add(windOffset);

                // Update rotation
                const rotX = leaf.rotation.x + Math.sin(windPhase) * totalWindStrength * 0.3;
                const rotZ = leaf.rotation.z + Math.cos(windPhase * 1.2) * totalWindStrength * 0.2;
                quaternion.setFromEuler(new THREE.Euler(rotX, leaf.rotation.y, rotZ));

                scale.set(leaf.scale, leaf.scale, leaf.scale);
                matrix.compose(position, quaternion, scale);
                this.leafInstancedMesh.setMatrixAt(index, matrix);
            });

            this.leafInstancedMesh.instanceMatrix.needsUpdate = true;
        }
    }

    updateGrassAnimation(deltaTime) {
        if (!this.grassInstancedMesh || !this.windEnabled) return;

        const matrix = new THREE.Matrix4();
        const position = new THREE.Vector3();
        const quaternion = new THREE.Quaternion();
        const scale = new THREE.Vector3();

        // Only update a subset of grass per frame for performance
        const updateCount = Math.min(500, this.grassDensity);
        const startIndex = Math.floor(Math.random() * (this.grassDensity - updateCount));

        for (let i = startIndex; i < startIndex + updateCount; i++) {
            this.grassInstancedMesh.getMatrixAt(i, matrix);
            matrix.decompose(position, quaternion, scale);

            const windPhase = this.windTime * 3 + position.x * 0.1 + position.z * 0.1;
            const windBend = Math.sin(windPhase) * this.windStrength * 0.3;

            quaternion.setFromEuler(new THREE.Euler(windBend, quaternion.y, 0));
            matrix.compose(position, quaternion, scale);
            this.grassInstancedMesh.setMatrixAt(i, matrix);
        }

        this.grassInstancedMesh.instanceMatrix.needsUpdate = true;
    }

    toggleWind() {
        this.windEnabled = !this.windEnabled;
        const button = this.shadowRoot.querySelector('#toggleWind');
        button.textContent = this.windEnabled ? 'Stop Wind' : 'Start Wind';
        button.classList.toggle('active', this.windEnabled);
    }

    updateStats() {
        const fpsValue = this.shadowRoot.querySelector('#fpsValue');
        const triangleCount = this.shadowRoot.querySelector('#triangleCount');
        const drawCalls = this.shadowRoot.querySelector('#drawCalls');

        if (fpsValue) fpsValue.textContent = this.fpsCounter;
        if (triangleCount) {
            const info = this.renderer.info;
            triangleCount.textContent = this.formatNumber(info.render.triangles);
        }
        if (drawCalls) {
            const info = this.renderer.info;
            drawCalls.textContent = info.render.calls;
        }
    }

    formatNumber(num) {
        if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
        if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
        return num.toString();
    }

    handleResize() {
        if (this.camera && this.renderer) {
            this.camera.aspect = window.innerWidth / window.innerHeight;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(window.innerWidth, window.innerHeight);
        }
    }

    connectedCallback() {}

    disconnectedCallback() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        if (this.renderer) {
            this.renderer.dispose();
        }
    }
}

customElements.define('tree-simulation', ForestSimulation);
