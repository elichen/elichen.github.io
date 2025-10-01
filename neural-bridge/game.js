import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

class AntModel {
    constructor() {
        this.group = new THREE.Group();

        // Ant color - reddish brown
        const antMaterial = new THREE.MeshStandardMaterial({
            color: 0x8b4513,
            roughness: 0.8,
            metalness: 0.2
        });

        const darkAntMaterial = new THREE.MeshStandardMaterial({
            color: 0x5c2e0f,
            roughness: 0.9,
            metalness: 0.1
        });

        // Head (front)
        const headGeometry = new THREE.SphereGeometry(0.15, 8, 8);
        const head = new THREE.Mesh(headGeometry, antMaterial);
        head.position.set(0.25, 0, 0);
        head.scale.set(1, 0.8, 0.8);
        this.group.add(head);

        // Thorax (middle)
        const thoraxGeometry = new THREE.SphereGeometry(0.12, 8, 8);
        const thorax = new THREE.Mesh(thoraxGeometry, antMaterial);
        thorax.scale.set(1.2, 1, 1);
        this.group.add(thorax);

        // Abdomen (back)
        const abdomenGeometry = new THREE.SphereGeometry(0.18, 8, 8);
        const abdomen = new THREE.Mesh(abdomenGeometry, darkAntMaterial);
        abdomen.position.set(-0.25, 0, 0);
        abdomen.scale.set(1.3, 0.9, 0.9);
        this.group.add(abdomen);

        // Legs (simplified)
        const legGeometry = new THREE.CylinderGeometry(0.015, 0.015, 0.15, 4);
        const legMaterial = new THREE.MeshStandardMaterial({
            color: 0x5c2e0f,
            roughness: 0.9
        });

        // 6 legs total (3 per side)
        for (let side = 0; side < 2; side++) {
            const sideSign = side === 0 ? 1 : -1;
            for (let i = 0; i < 3; i++) {
                const leg = new THREE.Mesh(legGeometry, legMaterial);
                leg.position.set(
                    (i - 1) * 0.15,
                    -0.075,
                    sideSign * 0.12
                );
                leg.rotation.z = sideSign * Math.PI / 6;
                leg.rotation.x = 0.3;
                this.group.add(leg);
            }
        }

        // Antennae
        const antennaGeometry = new THREE.CylinderGeometry(0.008, 0.008, 0.2, 4);
        for (let side = 0; side < 2; side++) {
            const sideSign = side === 0 ? 1 : -1;
            const antenna = new THREE.Mesh(antennaGeometry, legMaterial);
            antenna.position.set(0.35, 0.08, sideSign * 0.08);
            antenna.rotation.z = sideSign * Math.PI / 4;
            antenna.rotation.y = sideSign * Math.PI / 6;
            this.group.add(antenna);
        }
    }

    getMesh() {
        return this.group;
    }
}

class AntBridgeSimulation3D {
    constructor() {
        this.container = document.getElementById('container');
        this.scene = new THREE.Scene();

        // Camera setup - positioned to view the bridge formation
        this.camera = new THREE.PerspectiveCamera(
            60,
            window.innerWidth / window.innerHeight,
            0.1,
            1000
        );
        this.camera.position.set(8, 6, 12);
        this.camera.lookAt(0, 2, 0);

        // Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.container.appendChild(this.renderer.domElement);

        // Orbit controls
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.target.set(0, 2, 0);
        this.controls.minDistance = 5;
        this.controls.maxDistance = 30;

        // Lighting
        this.setupLighting();

        // Simulation state
        this.ants = [];
        this.bridgeLeft = [];
        this.bridgeRight = [];
        this.bridgeComplete = false;
        this.linkLength = 0.6;
        this.maxSag = 1.5;

        // Cliff positions in 3D space
        this.leftCliff = { x: -4, y: 2, z: 0 };
        this.rightCliff = { x: 4, y: 2, z: 0 };
        this.gapDistance = this.distance3D(this.leftCliff, this.rightCliff);

        // Environment
        this.createEnvironment();

        // Create ants
        this.reset();

        // Handle window resize
        window.addEventListener('resize', () => this.onWindowResize());

        // Start animation loop
        this.lastTime = performance.now();
        this.animate();
    }

    setupLighting() {
        // Ambient light
        const ambient = new THREE.AmbientLight(0xffffff, 0.4);
        this.scene.add(ambient);

        // Main directional light
        const sun = new THREE.DirectionalLight(0xfff5e6, 1.2);
        sun.position.set(10, 15, 8);
        sun.castShadow = true;
        sun.shadow.camera.left = -15;
        sun.shadow.camera.right = 15;
        sun.shadow.camera.top = 15;
        sun.shadow.camera.bottom = -15;
        sun.shadow.mapSize.width = 2048;
        sun.shadow.mapSize.height = 2048;
        this.scene.add(sun);

        // Fill light
        const fill = new THREE.DirectionalLight(0x8ba7c9, 0.5);
        fill.position.set(-8, 8, -5);
        this.scene.add(fill);

        // Rim light
        const rim = new THREE.DirectionalLight(0xffffff, 0.6);
        rim.position.set(0, 5, -10);
        this.scene.add(rim);
    }

    createEnvironment() {
        // Sky color
        this.scene.background = new THREE.Color(0x87ceeb);
        this.scene.fog = new THREE.Fog(0x87ceeb, 15, 50);

        // Left cliff platform
        const cliffGeometry = new THREE.BoxGeometry(3, 0.5, 2);
        const cliffMaterial = new THREE.MeshStandardMaterial({
            color: 0x8b7355,
            roughness: 0.9,
            metalness: 0.1
        });

        this.leftPlatform = new THREE.Mesh(cliffGeometry, cliffMaterial);
        this.leftPlatform.position.set(this.leftCliff.x, this.leftCliff.y - 0.25, 0);
        this.leftPlatform.castShadow = true;
        this.leftPlatform.receiveShadow = true;
        this.scene.add(this.leftPlatform);

        // Right cliff platform
        this.rightPlatform = new THREE.Mesh(cliffGeometry, cliffMaterial);
        this.rightPlatform.position.set(this.rightCliff.x, this.rightCliff.y - 0.25, 0);
        this.rightPlatform.castShadow = true;
        this.rightPlatform.receiveShadow = true;
        this.scene.add(this.rightPlatform);

        // Chasm/void below
        const waterGeometry = new THREE.PlaneGeometry(20, 20);
        const waterMaterial = new THREE.MeshStandardMaterial({
            color: 0x2c5a7a,
            roughness: 0.3,
            metalness: 0.6
        });
        const water = new THREE.Mesh(waterGeometry, waterMaterial);
        water.rotation.x = -Math.PI / 2;
        water.position.y = -2;
        water.receiveShadow = true;
        this.scene.add(water);

        // Support pillars for cliffs to show they're elevated
        const pillarGeometry = new THREE.CylinderGeometry(0.3, 0.4, 4, 8);
        const pillarMaterial = new THREE.MeshStandardMaterial({
            color: 0x6b5b4a,
            roughness: 0.95
        });

        const leftPillar = new THREE.Mesh(pillarGeometry, pillarMaterial);
        leftPillar.position.set(this.leftCliff.x, 0, 0);
        leftPillar.castShadow = true;
        this.scene.add(leftPillar);

        const rightPillar = new THREE.Mesh(pillarGeometry, pillarMaterial);
        rightPillar.position.set(this.rightCliff.x, 0, 0);
        rightPillar.castShadow = true;
        this.scene.add(rightPillar);
    }

    reset() {
        // Clear existing ants
        for (const ant of this.ants) {
            this.scene.remove(ant.mesh);
        }

        this.ants = [];
        this.bridgeLeft = [];
        this.bridgeRight = [];
        this.bridgeComplete = false;

        // Create new ants on both platforms
        for (let i = 0; i < 50; i++) {
            this.createAnt('left');
        }
        for (let i = 0; i < 50; i++) {
            this.createAnt('right');
        }
    }

    createAnt(side) {
        const model = new AntModel();
        const mesh = model.getMesh();
        mesh.castShadow = true;

        // Position on appropriate platform
        const platform = side === 'left' ? this.leftCliff : this.rightCliff;
        const x = platform.x + (Math.random() - 0.5) * 2;
        const z = (Math.random() - 0.5) * 1.5;

        mesh.position.set(x, platform.y, z);
        mesh.rotation.y = Math.random() * Math.PI * 2;

        this.scene.add(mesh);

        const ant = {
            mesh,
            state: 'foraging',
            side,
            velocity: new THREE.Vector3(),
            targetRotation: mesh.rotation.y,
            bridgeIndex: -1,
            chain: null
        };

        this.ants.push(ant);
        return ant;
    }

    animate() {
        requestAnimationFrame(() => this.animate());

        const now = performance.now();
        const dt = Math.min((now - this.lastTime) / 1000, 0.033);
        this.lastTime = now;

        this.update(dt);
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    update(dt) {
        for (const ant of this.ants) {
            if (ant.state === 'foraging') {
                this.updateForagingAnt(ant, dt);
            } else if (ant.state === 'bridge') {
                this.updateBridgeAnt(ant, dt);
            } else if (ant.state === 'crossing') {
                this.updateCrossingAnt(ant, dt);
            }
        }

        this.updateBridgeMetrics();
    }

    updateForagingAnt(ant, dt) {
        const platform = ant.side === 'left' ? this.leftCliff : this.rightCliff;
        const edgeX = ant.side === 'left' ? platform.x + 1.2 : platform.x - 1.2;

        // Move towards edge
        const toEdgeX = edgeX - ant.mesh.position.x;
        ant.velocity.x = toEdgeX * 1.5 + (Math.random() - 0.5) * 2;
        ant.velocity.z = (Math.random() - 0.5) * 1.5;

        // Clamp to platform
        ant.mesh.position.x += ant.velocity.x * dt;
        ant.mesh.position.z += ant.velocity.z * dt;

        // Keep on platform
        const platformLeft = platform.x - 1.5;
        const platformRight = platform.x + 1.5;
        ant.mesh.position.x = this.clamp(ant.mesh.position.x, platformLeft, platformRight);
        ant.mesh.position.z = this.clamp(ant.mesh.position.z, -0.8, 0.8);
        ant.mesh.position.y = platform.y;

        // Update rotation based on movement
        if (Math.abs(ant.velocity.x) > 0.1 || Math.abs(ant.velocity.z) > 0.1) {
            ant.targetRotation = Math.atan2(ant.velocity.x, ant.velocity.z);
        }
        ant.mesh.rotation.y += (ant.targetRotation - ant.mesh.rotation.y) * 5 * dt;

        // Try to attach to bridge
        if (!this.bridgeComplete) {
            const attachmentPoint = this.getTailPosition(ant.side);
            const distToTail = this.distance3D(ant.mesh.position, attachmentPoint);

            const ownChain = ant.side === 'left' ? this.bridgeLeft : this.bridgeRight;
            const otherChain = ant.side === 'left' ? this.bridgeRight : this.bridgeLeft;
            const chainBalanceOK = ownChain.length <= otherChain.length + 2;

            if (chainBalanceOK && distToTail < 0.4 && ownChain.length < 30) {
                this.attachAntToBridge(ant, ant.side);
            }
        } else {
            // Bridge is complete, start crossing
            const distToPlatformEdge = Math.abs(ant.mesh.position.x - edgeX);
            if (distToPlatformEdge < 0.3 && Math.random() < 0.3 * dt) {
                ant.state = 'crossing';
                ant.crossingT = 0;
                ant.crossingDirection = ant.side;
            }
        }
    }

    attachAntToBridge(ant, side) {
        const chain = side === 'left' ? this.bridgeLeft : this.bridgeRight;
        ant.state = 'bridge';
        ant.chain = side;
        ant.bridgeIndex = chain.length;
        ant.velocity.set(0, 0, 0);
        chain.push(ant);
    }

    updateBridgeAnt(ant, dt) {
        const totalLinks = this.bridgeLeft.length + this.bridgeRight.length;
        if (!totalLinks) return;

        const segments = totalLinks + 1;
        const dir = new THREE.Vector3()
            .subVectors(this.rightCliff, this.leftCliff)
            .normalize();

        const globalIndex = ant.chain === 'left'
            ? ant.bridgeIndex + 1
            : segments - ant.bridgeIndex - 1;

        const t = globalIndex / segments;

        // Calculate position along bridge span
        const spanPos = new THREE.Vector3().lerpVectors(
            new THREE.Vector3(this.leftCliff.x, this.leftCliff.y, this.leftCliff.z),
            new THREE.Vector3(this.rightCliff.x, this.rightCliff.y, this.rightCliff.z),
            t
        );

        // Add sag (gravity effect)
        const bridgeProgress = this.getBridgeProgress();
        const sag = (1 - bridgeProgress) * this.maxSag * Math.sin(Math.PI * t);
        spanPos.y -= sag;

        // Move ant towards target position
        ant.mesh.position.lerp(spanPos, 5 * dt);

        // Update rotation to face along bridge
        ant.mesh.lookAt(
            ant.mesh.position.x + dir.x,
            ant.mesh.position.y,
            ant.mesh.position.z + dir.z
        );
        ant.mesh.rotation.y += Math.PI / 2;
    }

    updateCrossingAnt(ant, dt) {
        if (!this.bridgeComplete) {
            ant.state = 'foraging';
            return;
        }

        ant.crossingT = ant.crossingT || 0;
        ant.crossingT += dt * 0.4;

        if (ant.crossingT >= 1) {
            // Finished crossing
            ant.side = ant.crossingDirection === 'left' ? 'right' : 'left';
            ant.state = 'foraging';
            ant.crossingT = 0;

            const newPlatform = ant.side === 'left' ? this.leftCliff : this.rightCliff;
            ant.mesh.position.set(
                newPlatform.x + (Math.random() - 0.5) * 2,
                newPlatform.y,
                (Math.random() - 0.5) * 1.5
            );
            return;
        }

        // Interpolate along bridge
        const t = ant.crossingDirection === 'left' ? ant.crossingT : 1 - ant.crossingT;
        const bridgePath = this.getBridgeNodes();
        const pos = this.interpolateBridgePath(bridgePath, t);

        ant.mesh.position.copy(pos);
        ant.mesh.position.y += 0.08; // Slightly above bridge

        // Face direction of travel
        const dir = ant.crossingDirection === 'left' ? 1 : -1;
        ant.mesh.rotation.y = dir > 0 ? Math.PI / 2 : -Math.PI / 2;
    }

    updateBridgeMetrics() {
        const leftTail = this.getTailPosition('left');
        const rightTail = this.getTailPosition('right');
        const tipGap = this.distance3D(leftTail, rightTail);

        if (!this.bridgeComplete &&
            this.bridgeLeft.length > 0 &&
            this.bridgeRight.length > 0 &&
            tipGap <= this.linkLength * 1.3) {
            this.bridgeComplete = true;
        }
    }

    getTailPosition(side) {
        if (side === 'left') {
            if (!this.bridgeLeft.length) {
                return new THREE.Vector3(this.leftCliff.x, this.leftCliff.y, this.leftCliff.z);
            }
            return this.bridgeLeft[this.bridgeLeft.length - 1].mesh.position.clone();
        } else {
            if (!this.bridgeRight.length) {
                return new THREE.Vector3(this.rightCliff.x, this.rightCliff.y, this.rightCliff.z);
            }
            return this.bridgeRight[this.bridgeRight.length - 1].mesh.position.clone();
        }
    }

    getBridgeProgress() {
        const leftTail = this.getTailPosition('left');
        const rightTail = this.getTailPosition('right');
        const tipGap = this.distance3D(leftTail, rightTail);
        const coverage = this.gapDistance - tipGap;
        return this.clamp(coverage / this.gapDistance, 0, 1);
    }

    getBridgeNodes() {
        if (!this.bridgeComplete) return [];

        const nodes = [new THREE.Vector3(this.leftCliff.x, this.leftCliff.y, this.leftCliff.z)];

        for (const ant of this.bridgeLeft) {
            nodes.push(ant.mesh.position.clone());
        }

        for (let i = this.bridgeRight.length - 1; i >= 0; i--) {
            nodes.push(this.bridgeRight[i].mesh.position.clone());
        }

        nodes.push(new THREE.Vector3(this.rightCliff.x, this.rightCliff.y, this.rightCliff.z));

        return nodes;
    }

    interpolateBridgePath(nodes, t) {
        if (nodes.length < 2) return new THREE.Vector3();

        let totalDist = 0;
        const segments = [];

        for (let i = 0; i < nodes.length - 1; i++) {
            const dist = nodes[i].distanceTo(nodes[i + 1]);
            segments.push({ start: nodes[i], end: nodes[i + 1], dist });
            totalDist += dist;
        }

        const targetDist = t * totalDist;
        let accum = 0;

        for (const seg of segments) {
            if (accum + seg.dist >= targetDist) {
                const localT = (targetDist - accum) / seg.dist;
                return new THREE.Vector3().lerpVectors(seg.start, seg.end, localT);
            }
            accum += seg.dist;
        }

        return nodes[nodes.length - 1].clone();
    }

    distance3D(a, b) {
        if (a.x !== undefined && b.x !== undefined) {
            const dx = a.x - b.x;
            const dy = a.y - b.y;
            const dz = a.z - b.z;
            return Math.sqrt(dx * dx + dy * dy + dz * dz);
        }
        return a.distanceTo(b);
    }

    clamp(value, min, max) {
        return Math.max(min, Math.min(max, value));
    }

    onWindowResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }
}

// Start simulation when page loads
window.addEventListener('DOMContentLoaded', () => {
    new AntBridgeSimulation3D();
});
