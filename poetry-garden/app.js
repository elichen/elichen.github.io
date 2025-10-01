import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// Poetry word database
const POETRY_WORDS = {
    moonlight: { color: 0xc8b7ff, words: ['silver', 'gentle', 'whispers', 'shadows', 'dreams'] },
    ocean: { color: 0x4fb3ff, words: ['waves', 'endless', 'deep', 'horizon', 'blue'] },
    forest: { color: 0x4dff88, words: ['ancient', 'green', 'silence', 'trees', 'mystery'] },
    starlight: { color: 0xffffaa, words: ['distant', 'bright', 'guide', 'eternal', 'cosmos'] },
    wind: { color: 0xb8e6ff, words: ['gentle', 'carries', 'stories', 'freedom', 'touch'] },
    sunrise: { color: 0xffd700, words: ['golden', 'hope', 'awakening', 'warmth', 'light'] },
    longing: { color: 0xff9bce, words: ['distant', 'heart', 'yearning', 'memory', 'reach'] },
    joy: { color: 0xffeb3b, words: ['laughter', 'bright', 'dancing', 'celebration', 'light'] },
    melancholy: { color: 0x9c88aa, words: ['autumn', 'rain', 'quiet', 'thoughtful', 'solitude'] },
    wonder: { color: 0x80d8ff, words: ['curious', 'magic', 'questions', 'amazement', 'awe'] },
    peace: { color: 0xc5e1a5, words: ['calm', 'stillness', 'harmony', 'breath', 'centered'] },
    hope: { color: 0xffcc80, words: ['tomorrow', 'rising', 'possibility', 'faith', 'bloom'] },
    time: { color: 0xb39ddb, words: ['flowing', 'endless', 'moments', 'eternal', 'passing'] },
    memory: { color: 0xf48fb1, words: ['fading', 'precious', 'golden', 'whispers', 'treasured'] },
    dream: { color: 0xce93d8, words: ['floating', 'colors', 'impossible', 'flight', 'wonder'] },
    silence: { color: 0x90a4ae, words: ['profound', 'speaks', 'empty', 'listening', 'space'] },
    infinity: { color: 0x7986cb, words: ['endless', 'circle', 'beyond', 'limitless', 'vast'] },
    shadow: { color: 0x616161, words: ['dancing', 'following', 'mystery', 'depth', 'contrast'] }
};

const WORD_KEYS = Object.keys(POETRY_WORDS);

class Flower {
    constructor(word, position, color) {
        this.word = word;
        this.position = position;
        this.color = color;
        this.group = new THREE.Group();
        this.group.position.copy(position);

        this.bloomProgress = 0;
        this.targetBloom = 1;
        this.isFullyBloomed = false;

        this.createFlower();
        this.createParticles();
        this.poem = this.generatePoem();
        this.poemMesh = null;
    }

    createFlower() {
        const stemGeometry = new THREE.CylinderGeometry(0.02, 0.03, 0.8, 8);
        const stemMaterial = new THREE.MeshStandardMaterial({
            color: 0x4a7c59,
            roughness: 0.8
        });
        const stem = new THREE.Mesh(stemGeometry, stemMaterial);
        stem.position.y = 0.4;
        this.group.add(stem);

        // Flower head (multiple petals)
        this.petalGroup = new THREE.Group();
        this.petalGroup.position.y = 0.8;

        const petalCount = 8;
        this.petals = [];

        for (let i = 0; i < petalCount; i++) {
            const angle = (i / petalCount) * Math.PI * 2;
            const petal = this.createPetal(angle);
            this.petals.push(petal);
            this.petalGroup.add(petal);
        }

        // Center of flower
        const centerGeometry = new THREE.SphereGeometry(0.08, 16, 16);
        const centerMaterial = new THREE.MeshStandardMaterial({
            color: 0xffeb3b,
            emissive: 0xffeb3b,
            emissiveIntensity: 0.3
        });
        const center = new THREE.Mesh(centerGeometry, centerMaterial);
        this.petalGroup.add(center);

        this.group.add(this.petalGroup);

        // Start fully closed
        this.petalGroup.scale.set(0.01, 0.01, 0.01);
    }

    createPetal(angle) {
        const petalGeometry = new THREE.SphereGeometry(0.15, 16, 16);
        petalGeometry.scale(1, 0.3, 0.6);

        const petalMaterial = new THREE.MeshStandardMaterial({
            color: this.color,
            emissive: this.color,
            emissiveIntensity: 0.2,
            roughness: 0.4,
            metalness: 0.1
        });

        const petal = new THREE.Mesh(petalGeometry, petalMaterial);
        petal.position.x = Math.cos(angle) * 0.15;
        petal.position.z = Math.sin(angle) * 0.15;
        petal.rotation.y = angle;
        petal.rotation.x = Math.PI / 6;

        return petal;
    }

    createParticles() {
        this.particles = [];
        const particleGeometry = new THREE.SphereGeometry(0.02, 8, 8);
        const particleMaterial = new THREE.MeshBasicMaterial({
            color: this.color,
            transparent: true,
            opacity: 0.6
        });

        for (let i = 0; i < 20; i++) {
            const particle = new THREE.Mesh(particleGeometry, particleMaterial.clone());
            particle.position.set(
                (Math.random() - 0.5) * 0.3,
                Math.random() * 0.5,
                (Math.random() - 0.5) * 0.3
            );
            particle.userData = {
                velocity: new THREE.Vector3(
                    (Math.random() - 0.5) * 0.02,
                    Math.random() * 0.03 + 0.01,
                    (Math.random() - 0.5) * 0.02
                ),
                lifetime: Math.random() * 2 + 1
            };
            this.particles.push(particle);
            this.group.add(particle);
        }
    }

    generatePoem() {
        const wordData = POETRY_WORDS[this.word];
        const words = wordData ? wordData.words : ['beauty', 'wonder', 'magic', 'light', 'dream'];

        const templates = [
            [
                `${this.capitalize(this.word)} ${this.pick(words)}`,
                `${this.pick(words)} ${this.pick(words)} through the ${this.pick(['night', 'day', 'moment'])}`,
                `${this.pick(words)} ${this.pick(['remains', 'echoes', 'flows'])}`
            ],
            [
                `In the ${this.word}`,
                `${this.pick(words)} moments`,
                `that ${this.pick(words)} and ${this.pick(words)}`,
                `like ${this.pick(['memories', 'dreams', 'whispers'])}`
            ],
            [
                `${this.capitalize(this.word)}.`,
                `${this.pick(words)}.`,
                `${this.pick(['silence', 'space', 'breath'])}.`
            ]
        ];

        return this.pick(templates);
    }

    capitalize(word) {
        return word.charAt(0).toUpperCase() + word.slice(1);
    }

    pick(array) {
        return array[Math.floor(Math.random() * array.length)];
    }

    update(dt) {
        // Bloom animation
        if (this.bloomProgress < this.targetBloom) {
            this.bloomProgress = Math.min(this.targetBloom, this.bloomProgress + dt * 0.5);
            const scale = this.bloomProgress;
            this.petalGroup.scale.set(scale, scale, scale);

            if (this.bloomProgress >= 0.99 && !this.isFullyBloomed) {
                this.isFullyBloomed = true;
            }
        }

        // Gentle swaying
        const time = performance.now() * 0.001;
        this.group.rotation.z = Math.sin(time + this.position.x) * 0.1;
        this.petalGroup.rotation.y = time * 0.2;

        // Update particles
        for (let i = this.particles.length - 1; i >= 0; i--) {
            const particle = this.particles[i];
            particle.position.add(particle.userData.velocity);
            particle.userData.lifetime -= dt;

            particle.material.opacity = Math.max(0, particle.userData.lifetime * 0.3);

            if (particle.userData.lifetime <= 0) {
                particle.position.set(
                    (Math.random() - 0.5) * 0.3,
                    0.5,
                    (Math.random() - 0.5) * 0.3
                );
                particle.userData.lifetime = Math.random() * 2 + 1;
                particle.material.opacity = 0.6;
            }
        }
    }

    createPoemText(scene) {
        if (this.poemMesh) return;

        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = 512;
        canvas.height = 512;

        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        ctx.fillStyle = 'white';
        ctx.font = 'italic 32px Georgia';
        ctx.textAlign = 'center';

        let y = 180;
        for (const line of this.poem) {
            ctx.fillText(line, canvas.width / 2, y);
            y += 50;
        }

        const texture = new THREE.CanvasTexture(canvas);
        const material = new THREE.MeshBasicMaterial({
            map: texture,
            transparent: true,
            opacity: 0,
            side: THREE.DoubleSide
        });

        const geometry = new THREE.PlaneGeometry(1.5, 1.5);
        this.poemMesh = new THREE.Mesh(geometry, material);
        this.poemMesh.position.copy(this.group.position);
        this.poemMesh.position.y += 1.5;

        scene.add(this.poemMesh);

        // Fade in
        let opacity = 0;
        const fadeIn = () => {
            opacity += 0.02;
            if (opacity < 1) {
                this.poemMesh.material.opacity = opacity;
                requestAnimationFrame(fadeIn);
            }
        };
        setTimeout(fadeIn, 1000);
    }

    updatePoemFacing(camera) {
        if (this.poemMesh) {
            this.poemMesh.lookAt(camera.position);
        }
    }

    getMesh() {
        return this.group;
    }
}

class PoetryGarden3D {
    constructor() {
        this.container = document.getElementById('container');
        this.scene = new THREE.Scene();

        // Camera
        this.camera = new THREE.PerspectiveCamera(
            60,
            window.innerWidth / window.innerHeight,
            0.1,
            1000
        );
        this.camera.position.set(0, 3, 8);
        this.camera.lookAt(0, 0, 0);

        // Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.container.appendChild(this.renderer.domElement);

        // Controls
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.minDistance = 3;
        this.controls.maxDistance = 20;
        this.controls.maxPolarAngle = Math.PI / 2.2;

        // Setup scene
        this.setupLighting();
        this.createEnvironment();

        // Flowers
        this.flowers = [];

        // Raycaster for clicking
        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();

        // Event listeners
        window.addEventListener('resize', () => this.onWindowResize());
        this.renderer.domElement.addEventListener('click', (e) => this.onMouseClick(e));

        // Ambient particles
        this.createAmbientParticles();

        // Start animation
        this.lastTime = performance.now();
        this.animate();
    }

    setupLighting() {
        // Ambient light
        const ambient = new THREE.AmbientLight(0xffffff, 0.3);
        this.scene.add(ambient);

        // Main sun light
        const sun = new THREE.DirectionalLight(0xfff5e1, 1.5);
        sun.position.set(10, 20, 5);
        sun.castShadow = true;
        sun.shadow.camera.left = -15;
        sun.shadow.camera.right = 15;
        sun.shadow.camera.top = 15;
        sun.shadow.camera.bottom = -15;
        sun.shadow.mapSize.width = 2048;
        sun.shadow.mapSize.height = 2048;
        this.scene.add(sun);

        // Colored accent lights
        const light1 = new THREE.PointLight(0xff6ec7, 0.5, 10);
        light1.position.set(-5, 2, 5);
        this.scene.add(light1);

        const light2 = new THREE.PointLight(0x6ea8ff, 0.5, 10);
        light2.position.set(5, 2, -5);
        this.scene.add(light2);

        // Atmospheric fog
        this.scene.fog = new THREE.Fog(0x1a1a2e, 10, 30);
        this.scene.background = new THREE.Color(0x1a1a2e);
    }

    createEnvironment() {
        // Ground
        const groundGeometry = new THREE.CircleGeometry(12, 64);
        const groundMaterial = new THREE.MeshStandardMaterial({
            color: 0x2d4a3e,
            roughness: 0.9,
            metalness: 0.1
        });
        const ground = new THREE.Mesh(groundGeometry, groundMaterial);
        ground.rotation.x = -Math.PI / 2;
        ground.receiveShadow = true;
        this.scene.add(ground);
        this.ground = ground;

        // Add grass texture with simple geometry
        const grassGeometry = new THREE.CylinderGeometry(0.01, 0.01, 0.15, 4);
        const grassMaterial = new THREE.MeshStandardMaterial({
            color: 0x3d6b4a,
            roughness: 1
        });

        for (let i = 0; i < 200; i++) {
            const grass = new THREE.Mesh(grassGeometry, grassMaterial);
            const angle = Math.random() * Math.PI * 2;
            const radius = Math.random() * 11;
            grass.position.x = Math.cos(angle) * radius;
            grass.position.z = Math.sin(angle) * radius;
            grass.position.y = 0.075;
            grass.rotation.y = Math.random() * Math.PI;
            grass.rotation.z = (Math.random() - 0.5) * 0.3;
            this.scene.add(grass);
        }
    }

    createAmbientParticles() {
        this.ambientParticles = [];
        const particleGeometry = new THREE.SphereGeometry(0.03, 8, 8);

        for (let i = 0; i < 50; i++) {
            const color = new THREE.Color().setHSL(Math.random(), 0.7, 0.7);
            const material = new THREE.MeshBasicMaterial({
                color: color,
                transparent: true,
                opacity: 0.4
            });

            const particle = new THREE.Mesh(particleGeometry, material);
            particle.position.set(
                (Math.random() - 0.5) * 20,
                Math.random() * 5 + 0.5,
                (Math.random() - 0.5) * 20
            );

            particle.userData = {
                velocity: new THREE.Vector3(
                    (Math.random() - 0.5) * 0.02,
                    (Math.random() - 0.5) * 0.01,
                    (Math.random() - 0.5) * 0.02
                ),
                baseOpacity: 0.4
            };

            this.ambientParticles.push(particle);
            this.scene.add(particle);
        }
    }

    onMouseClick(event) {
        this.mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
        this.mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

        this.raycaster.setFromCamera(this.mouse, this.camera);
        const intersects = this.raycaster.intersectObject(this.ground);

        if (intersects.length > 0) {
            const point = intersects[0].point;
            this.plantFlower(point);
        }
    }

    plantFlower(position) {
        const word = WORD_KEYS[Math.floor(Math.random() * WORD_KEYS.length)];
        const color = POETRY_WORDS[word].color;

        const flower = new Flower(word, position, color);
        this.flowers.push(flower);
        this.scene.add(flower.getMesh());

        // Create poem after bloom
        setTimeout(() => {
            flower.createPoemText(this.scene);
        }, 2000);

        // Sparkle effect at plant location
        this.createSparkles(position);
    }

    createSparkles(position) {
        const sparkleCount = 20;
        const sparkles = [];

        for (let i = 0; i < sparkleCount; i++) {
            const geometry = new THREE.SphereGeometry(0.05, 8, 8);
            const material = new THREE.MeshBasicMaterial({
                color: new THREE.Color().setHSL(Math.random(), 1, 0.7),
                transparent: true,
                opacity: 1
            });

            const sparkle = new THREE.Mesh(geometry, material);
            sparkle.position.copy(position);
            sparkle.position.y = 0.1;

            sparkle.userData = {
                velocity: new THREE.Vector3(
                    (Math.random() - 0.5) * 0.1,
                    Math.random() * 0.15,
                    (Math.random() - 0.5) * 0.1
                ),
                lifetime: 1
            };

            sparkles.push(sparkle);
            this.scene.add(sparkle);
        }

        // Animate and remove sparkles
        const animateSparkles = () => {
            let allDone = true;

            for (const sparkle of sparkles) {
                sparkle.userData.lifetime -= 0.016;
                if (sparkle.userData.lifetime > 0) {
                    allDone = false;
                    sparkle.position.add(sparkle.userData.velocity);
                    sparkle.userData.velocity.y -= 0.005;
                    sparkle.material.opacity = sparkle.userData.lifetime;
                }
            }

            if (!allDone) {
                requestAnimationFrame(animateSparkles);
            } else {
                sparkles.forEach(s => this.scene.remove(s));
            }
        };

        animateSparkles();
    }

    animate() {
        requestAnimationFrame(() => this.animate());

        const now = performance.now();
        const dt = Math.min((now - this.lastTime) / 1000, 0.033);
        this.lastTime = now;

        // Update flowers
        for (const flower of this.flowers) {
            flower.update(dt);
            flower.updatePoemFacing(this.camera);
        }

        // Update ambient particles
        for (const particle of this.ambientParticles) {
            particle.position.add(particle.userData.velocity);

            // Bounds check and respawn
            if (Math.abs(particle.position.x) > 10 ||
                Math.abs(particle.position.z) > 10 ||
                particle.position.y > 6 ||
                particle.position.y < 0.5) {
                particle.position.set(
                    (Math.random() - 0.5) * 20,
                    Math.random() * 5 + 0.5,
                    (Math.random() - 0.5) * 20
                );
            }

            // Gentle pulsing
            const pulse = Math.sin(now * 0.002 + particle.position.x) * 0.2 + 0.8;
            particle.material.opacity = particle.userData.baseOpacity * pulse;
        }

        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    onWindowResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }
}

// Start the garden
window.addEventListener('DOMContentLoaded', () => {
    new PoetryGarden3D();
});
