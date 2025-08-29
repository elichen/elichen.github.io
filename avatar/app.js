import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRM, VRMLoaderPlugin, VRMExpressionPresetName } from '@pixiv/three-vrm';
import { VRMAnimationLoaderPlugin } from '@pixiv/three-vrm-animation';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { AudioProcessor } from './audio-processor.js';

class AvatarLipSync {
    constructor() {
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.vrm = null;
        this.animationMixer = null;
        this.currentAnimation = null;
        this.isInitialized = false;
        this.isRecording = false;
        this.audioProcessor = null;
        
        // Debug mode for manual expression testing
        this.debugMode = false;
        this.debugExpressions = ['REST', 'A', 'E', 'I', 'O', 'U'];
        this.currentDebugIndex = 0;
        
        // Performance tracking
        this.lastFrameTime = 0;
        this.frameCount = 0;
        this.fpsHistory = [];
        this.latencyStart = 0;
        
        // Expression smoothing
        this.currentExpressions = new Map();
        this.targetExpressions = new Map();
        this.expressionSmoothingFactor = 0.18; // Slightly faster response
        
        // UI elements
        this.startBtn = document.getElementById('start-btn');
        this.statusEl = document.getElementById('status');
        this.webgpuStatusEl = document.getElementById('webgpu-status');
        this.audioStatusEl = document.getElementById('audio-status');
        this.rhubarbStatusEl = document.getElementById('rhubarb-status');
        this.avatarStatusEl = document.getElementById('avatar-status');
        this.latencyEl = document.getElementById('latency');
        this.fpsEl = document.getElementById('fps');
        this.visemeEl = document.getElementById('viseme');
        
        this.init();
    }
    
    async init() {
        try {
            const webgpuSupport = await this.initWebGPU();
            await this.initThreeJS();
            await this.loadAvatar();
            await this.initAudioProcessor();
            this.setupEventListeners();
            this.animate();
            this.isInitialized = true;
            
            if (this.debugMode) {
                this.updateStatus('DEBUG MODE: Press spacebar to cycle through expressions', 'ready');
                // Start with first debug expression
                setTimeout(() => this.nextDebugExpression(), 1000);
            } else {
                this.updateStatus('Ready to start', 'ready');
            }
        } catch (error) {
            console.error('Initialization failed:', error);
            this.updateStatus(`Error: ${error.message}`, 'error');
        }
    }
    
    async initWebGPU() {
        if (!navigator.gpu) {
            this.webgpuStatusEl.textContent = 'Not Supported';
            console.log('WebGPU not supported, will use WebGL fallback');
            return null;
        }
        
        try {
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) {
                this.webgpuStatusEl.textContent = 'No Adapter';
                console.log('No WebGPU adapter found, will use WebGL fallback');
                return null;
            }
            
            const device = await adapter.requestDevice();
            this.webgpuStatusEl.textContent = 'Available';
            return { adapter, device };
        } catch (error) {
            this.webgpuStatusEl.textContent = 'Failed';
            console.log('WebGPU initialization failed, will use WebGL fallback:', error);
            return null;
        }
    }
    
    async initThreeJS() {
        const canvas = document.getElementById('avatar-canvas');
        
        // For now, let's use WebGL renderer for better compatibility
        this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
        console.log('Using WebGL renderer for maximum compatibility');
        
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        
        // Scene setup
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x1a1a1a);
        
        // Camera setup - centered view of full avatar
        this.camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100);
        this.camera.position.set(0, 0.6, -1.4); // Very close
        
        // Controls
        this.controls = new OrbitControls(this.camera, canvas);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.target.set(0, 1.0, 0); // Target center of avatar
        this.controls.minDistance = 1.5;
        this.controls.maxDistance = 5; 
        this.controls.maxPolarAngle = Math.PI * 0.9;
        
        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(1, 2, 1);
        directionalLight.castShadow = true;
        directionalLight.shadow.mapSize.setScalar(1024);
        this.scene.add(directionalLight);
        
        // Enable shadows
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        
        console.log('Three.js WebGL renderer initialized successfully');
    }
    
    async loadAvatar() {
        try {
            const loader = new GLTFLoader();
            loader.register((parser) => new VRMLoaderPlugin(parser));
            loader.register((parser) => new VRMAnimationLoaderPlugin(parser));
            
            console.log('Loading VRM avatar...');
            
            // Try loading with proper path
            const gltf = await new Promise((resolve, reject) => {
                loader.load(
                    '2407317377392965971.vrm',
                    (gltf) => resolve(gltf),
                    (progress) => {
                        const percent = Math.round((progress.loaded / progress.total) * 100);
                        if (percent % 25 === 0) console.log(`Loading VRM: ${percent}%`);
                    },
                    (error) => reject(error)
                );
            });
            
            this.vrm = gltf.userData.vrm;
            
            if (!this.vrm) {
                throw new Error('No VRM data found in loaded file');
            }
            
            // Add to scene
            this.scene.add(this.vrm.scene);
            
            // Initialize expression system
            this.initExpressions();
            
            // Initialize animation system with proper error handling
            this.animationMixer = new THREE.AnimationMixer(this.vrm.scene);
            this.addIdleAnimation();
            
            // Position avatar
            this.vrm.scene.position.y = 0.6;
            this.vrm.scene.rotation.y = 0;
            
            this.avatarStatusEl.textContent = 'Loaded';
            console.log('VRM avatar loaded successfully');
            console.log('Available expressions:', this.vrm.expressionManager?.expressions);
            console.log('Expression Manager:', this.vrm.expressionManager);
            
            // Debug: Check all available expressions
            if (this.vrm.expressionManager) {
                console.log('Expression presets available:');
                for (const [name, expression] of Object.entries(this.vrm.expressionManager.expressions || {})) {
                    console.log(`  - ${name}:`, expression);
                }
            }
            
            // Debug: Check mesh morph targets
            this.vrm.scene.traverse((child) => {
                if (child.isMesh && child.morphTargetInfluences) {
                    console.log('Mesh with morph targets found:', child.name);
                    console.log('Morph target names:', child.morphTargetDictionary);
                    console.log('Morph target count:', child.morphTargetInfluences.length);
                }
            });
            
        } catch (error) {
            this.avatarStatusEl.textContent = 'Error';
            console.error('Avatar loading error:', error);
            throw new Error(`Failed to load avatar: ${error.message}`);
        }
    }
    
    initExpressions() {
        if (!this.vrm?.expressionManager) {
            console.warn('No expression manager found in VRM');
            return;
        }
        
        console.log('Mouth expression names:', this.vrm.expressionManager.mouthExpressionNames);
        console.log('Blink expression names:', this.vrm.expressionManager.blinkExpressionNames);
        console.log('LookAt expression names:', this.vrm.expressionManager.lookAtExpressionNames);
        
        // Initialize expression weights using the actual mouth expression names from the VRM
        const mouthExpressions = this.vrm.expressionManager.mouthExpressionNames || [];
        
        // Create a mapping from our visemes to the available mouth expressions
        this.expressionMapping = new Map();
        
        // Try to find common mouth shapes
        mouthExpressions.forEach((expressionName, index) => {
            console.log(`Available mouth expression ${index}: ${expressionName}`);
            
            // Map common expression names to our visemes
            const lowerName = expressionName.toLowerCase();
            if (lowerName === 'aa') {
                this.expressionMapping.set('A', expressionName);
            } else if (lowerName === 'ee') {
                this.expressionMapping.set('E', expressionName);
            } else if (lowerName === 'ih') {
                this.expressionMapping.set('I', expressionName);
            } else if (lowerName === 'oh') {
                this.expressionMapping.set('O', expressionName);
            } else if (lowerName === 'ou') {
                this.expressionMapping.set('U', expressionName);
            }
        });
        
        console.log('Expression mapping:', this.expressionMapping);
        
        // Initialize current and target expression weights
        ['A', 'E', 'I', 'O', 'U'].forEach(viseme => {
            this.currentExpressions.set(viseme, 0);
            this.targetExpressions.set(viseme, 0);
        });
        
        console.log('Expression system initialized with direct mapping');
    }
    
    addIdleAnimation() {
        if (!this.vrm || !this.animationMixer) return;
        
        console.log('🎭 Creating VRoid Hub-style idle animation...');
        
        const humanoid = this.vrm.humanoid;
        if (!humanoid) {
            console.log('❌ No humanoid system found');
            return;
        }
        
        // Debug: Comprehensive bone analysis
        console.log('🔍 VRM Humanoid bone analysis:');
        const allBoneNames = [
            'hips', 'spine', 'chest', 'upperChest', 'neck', 'head',
            'leftShoulder', 'leftUpperArm', 'leftLowerArm', 'leftHand',
            'rightShoulder', 'rightUpperArm', 'rightLowerArm', 'rightHand',
            'leftUpperLeg', 'leftLowerLeg', 'leftFoot',
            'rightUpperLeg', 'rightLowerLeg', 'rightFoot'
        ];
        
        const foundBones = [];
        allBoneNames.forEach(boneName => {
            const bone = humanoid.getNormalizedBoneNode(boneName);
            if (bone) {
                foundBones.push({ name: boneName, bone });
                console.log(`  ✅ ${boneName}: "${bone.name}" (uuid: ${bone.uuid.substring(0,8)}...)`);
            } else {
                console.log(`  ❌ ${boneName}: Not found`);
            }
        });
        
        console.log(`📊 Found ${foundBones.length}/${allBoneNames.length} bones`);
        
        // Debug: Show raw humanoid structure
        console.log('🦴 Raw humanoid mapping:', humanoid.humanBones);
        console.log('🎯 Normalized bones available:', Object.keys(humanoid.normalizedHumanBones || {}));
        
        // 4-second breathing cycle with seamless looping
        const times = [0, 1, 2, 3, 4];
        const tracks = [];
        
        // Fixed smooth loop function for true seamless looping
        const createSmoothLoop = (amplitude, frequency = 1, phase = 0) => {
            return times.map(t => amplitude * Math.sin((t * frequency * 2 * Math.PI / 4) + phase));
        };
        
        // Chest breathing animation (subtle expansion)
        const chestBone = humanoid.getNormalizedBoneNode('chest');
        if (chestBone) {
            
            // Smooth breathing with perfect loop using sine wave
            const breathingCurve = createSmoothLoop(0.015, 1, 0); // 1.5% amplitude
            const chestScale = [];
            
            for (let i = 0; i < times.length; i++) {
                const scale = 1.0 + Math.abs(breathingCurve[i]); // Always positive for expansion
                chestScale.push(scale, 1.0 + Math.abs(breathingCurve[i]) * 0.5, 1.0);
            }
            
            const chestTrack = new THREE.VectorKeyframeTrack(
                chestBone.name + '.scale',
                times,
                chestScale
            );
            tracks.push(chestTrack);
        }
        
        // Spine/torso gentle sway
        const spineBone = humanoid.getNormalizedBoneNode('spine');
        if (spineBone) {
            
            // Synchronized spine sway for perfect loop
            const spineQuaternions = [];
            const swayX = createSmoothLoop(0.01, 1, 0);           // Forward/back - base frequency
            const swayY = createSmoothLoop(0.008, 1, Math.PI/3);  // Side sway - SAME frequency
            const swayZ = createSmoothLoop(0.005, 1, Math.PI/6);  // Twist - SAME frequency
            
            for (let i = 0; i < times.length; i++) {
                const euler = new THREE.Euler(swayX[i], swayY[i], swayZ[i]);
                const quat = new THREE.Quaternion().setFromEuler(euler);
                spineQuaternions.push(quat.x, quat.y, quat.z, quat.w);
            }
            
            const spineTrack = new THREE.QuaternionKeyframeTrack(
                spineBone.name + '.quaternion',
                times,
                spineQuaternions
            );
            tracks.push(spineTrack);
        }
        
        // Simple head breathing movement (single track to avoid conflicts)
        const headBone = humanoid.getNormalizedBoneNode('head');
        if (headBone) {
            
            // Use only position-based movement for smooth breathing
            const headPosY = createSmoothLoop(0.01, 1, 0); // Simple up/down breathing movement
            const headPositions = [];
            
            for (let i = 0; i < times.length; i++) {
                headPositions.push(0, headPosY[i], 0); // Only Y movement
            }
            
            const headPosTrack = new THREE.VectorKeyframeTrack(
                headBone.name + '.position',
                times,
                headPositions
            );
            tracks.push(headPosTrack);
        }
        
        // Add shoulder movement for more lifelike animation
        const leftShoulder = humanoid.getNormalizedBoneNode('leftShoulder');
        const rightShoulder = humanoid.getNormalizedBoneNode('rightShoulder');
        
        if (leftShoulder && rightShoulder) {
            
            // Left shoulder quaternions (slight rise and fall)
            const leftShoulderQuats = [];
            const rightShoulderQuats = [];
            
            for (let i = 0; i < times.length; i++) {
                const t = times[i];
                // Shoulders move opposite to each other for natural asymmetry
                const leftRotZ = Math.sin(t * Math.PI * 0.5 + Math.PI * 0.2) * 0.01; // Slight roll
                const rightRotZ = Math.sin(t * Math.PI * 0.5 - Math.PI * 0.2) * 0.01; // Opposite
                
                const leftEuler = new THREE.Euler(0, 0, leftRotZ);
                const rightEuler = new THREE.Euler(0, 0, rightRotZ);
                
                const leftQuat = new THREE.Quaternion().setFromEuler(leftEuler);
                const rightQuat = new THREE.Quaternion().setFromEuler(rightEuler);
                
                leftShoulderQuats.push(leftQuat.x, leftQuat.y, leftQuat.z, leftQuat.w);
                rightShoulderQuats.push(rightQuat.x, rightQuat.y, rightQuat.z, rightQuat.w);
            }
            
            const leftShoulderTrack = new THREE.QuaternionKeyframeTrack(
                leftShoulder.name + '.quaternion',
                times,
                leftShoulderQuats
            );
            
            const rightShoulderTrack = new THREE.QuaternionKeyframeTrack(
                rightShoulder.name + '.quaternion',
                times,
                rightShoulderQuats
            );
            
            tracks.push(leftShoulderTrack, rightShoulderTrack);
        }
        
        // Natural arm positioning (fix T-pose)
        const leftUpperArm = humanoid.getNormalizedBoneNode('leftUpperArm');
        const rightUpperArm = humanoid.getNormalizedBoneNode('rightUpperArm');
        
        if (leftUpperArm && rightUpperArm) {
            
            // Arms hang naturally at sides with slight breathing movement
            const leftArmQuats = [];
            const rightArmQuats = [];
            
            for (let i = 0; i < times.length; i++) {
                const t = times[i];
                
                // Base pose: arms hanging straight down  
                const baseRotX = 0.3; // More forward rotation 
                const baseRotZ_left = 1.2; // Much stronger downward rotation for left arm
                const baseRotZ_right = -1.2; // Much stronger downward rotation for right arm
                
                // Add subtle breathing movement
                const breathingOffset = Math.sin(t * Math.PI * 0.5) * 0.02;
                
                const leftEuler = new THREE.Euler(
                    baseRotX + breathingOffset * 0.5,
                    0, 
                    baseRotZ_left + breathingOffset
                );
                
                const rightEuler = new THREE.Euler(
                    baseRotX + breathingOffset * 0.5,
                    0,
                    baseRotZ_right - breathingOffset
                );
                
                const leftQuat = new THREE.Quaternion().setFromEuler(leftEuler);
                const rightQuat = new THREE.Quaternion().setFromEuler(rightEuler);
                
                leftArmQuats.push(leftQuat.x, leftQuat.y, leftQuat.z, leftQuat.w);
                rightArmQuats.push(rightQuat.x, rightQuat.y, rightQuat.z, rightQuat.w);
            }
            
            const leftArmTrack = new THREE.QuaternionKeyframeTrack(
                leftUpperArm.name + '.quaternion',
                times,
                leftArmQuats
            );
            
            const rightArmTrack = new THREE.QuaternionKeyframeTrack(
                rightUpperArm.name + '.quaternion',
                times,
                rightArmQuats
            );
            
            tracks.push(leftArmTrack, rightArmTrack);
        }
        
        if (tracks.length > 0) {
            const clip = new THREE.AnimationClip('idle', 4, tracks);
            const action = this.animationMixer.clipAction(clip);
            action.loop = THREE.LoopRepeat;
            action.play();
            
            console.log('Idle animation started');
            
        } else {
            console.log('❌ No suitable bones found for idle animation');
        }
    }
    
    async initAudioProcessor() {
        try {
            this.audioProcessor = new AudioProcessor();
            
            // Set up callbacks
            this.audioProcessor.onVisemeChange = (viseme, timestamp) => {
                const processingStart = performance.now();
                this.setViseme(viseme);
                
                // Calculate end-to-end latency
                const audioLatency = this.audioProcessor.getLatency();
                const processingTime = performance.now() - processingStart;
                const totalLatency = audioLatency + processingTime;
                
                this.latencyEl.textContent = `${Math.round(totalLatency)} ms`;
                
                // Update latency status color
                const latencyEl = this.latencyEl.parentElement;
                if (totalLatency < 80) {
                    latencyEl.style.color = '#4CAF50';
                } else if (totalLatency < 120) {
                    latencyEl.style.color = '#ff9800';
                } else {
                    latencyEl.style.color = '#f44336';
                }
            };
            
            this.audioProcessor.onVolumeChange = (volume) => {
                // Could be used for volume-based effects
            };
            
            this.rhubarbStatusEl.textContent = 'Advanced Audio Ready';
            console.log('Advanced audio processor initialized');
        } catch (error) {
            this.rhubarbStatusEl.textContent = 'Error';
            throw error;
        }
    }
    
    setupEventListeners() {
        this.startBtn.addEventListener('click', () => {
            if (this.isRecording) {
                this.stopRecording();
            } else {
                this.startRecording();
            }
        });
        
        window.addEventListener('resize', () => {
            this.camera.aspect = window.innerWidth / window.innerHeight;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(window.innerWidth, window.innerHeight);
        });
        
        // Debug: Spacebar to cycle through expressions
        window.addEventListener('keydown', (event) => {
            if (event.code === 'Space') {
                event.preventDefault();
                this.nextDebugExpression();
            }
        });
    }
    
    async startRecording() {
        if (!this.isInitialized) {
            this.updateStatus('System not ready', 'error');
            return;
        }
        
        try {
            // Initialize and start audio processor
            await this.audioProcessor.initialize();
            this.audioProcessor.start();
            
            this.isRecording = true;
            this.startBtn.textContent = 'Stop Microphone';
            this.startBtn.classList.add('stop');
            this.updateStatus('Recording and analyzing speech...', 'active');
            this.audioStatusEl.textContent = 'Recording';
            
            console.log('Advanced audio processing started');
            
        } catch (error) {
            console.error('Failed to start recording:', error);
            this.updateStatus(`Microphone error: ${error.message}`, 'error');
        }
    }
    
    
    setViseme(viseme) {
        // Reset all expressions
        this.targetExpressions.forEach((_, name) => {
            this.targetExpressions.set(name, 0);
        });
        
        // Set target expression based on viseme using our mapped system
        if (viseme !== 'REST') {
            this.targetExpressions.set(viseme, 2.0); // Amplify for more visible expression
        }
        
        // Update UI
        this.visemeEl.textContent = viseme;
    }
    
    stopRecording() {
        this.isRecording = false;
        
        if (this.audioProcessor) {
            this.audioProcessor.stop();
        }
        
        this.startBtn.textContent = 'Start Microphone';
        this.startBtn.classList.remove('stop');
        this.updateStatus('Ready to start', 'ready');
        this.audioStatusEl.textContent = 'Ready';
        this.visemeEl.textContent = 'REST';
        this.latencyEl.textContent = '-- ms';
        
        // Reset expressions
        this.targetExpressions.forEach((_, name) => {
            this.targetExpressions.set(name, 0);
        });
        
        console.log('Recording stopped');
    }
    
    nextDebugExpression() {
        this.currentDebugIndex = (this.currentDebugIndex + 1) % this.debugExpressions.length;
        const expression = this.debugExpressions[this.currentDebugIndex];
        
        console.log(`🔧 DEBUG: Cycling to expression ${this.currentDebugIndex}: ${expression}`);
        
        // Force set the expression
        this.setViseme(expression);
        
        // Also try direct VRM expression setting
        if (this.vrm?.expressionManager && this.expressionMapping) {
            const expressionName = this.expressionMapping.get(expression);
            if (expressionName) {
                console.log(`🔧 DEBUG: Directly setting VRM expression "${expressionName}" to 1.0`);
                this.vrm.expressionManager.setValue(expressionName, 1.0);
                this.vrm.expressionManager.update();
                
                // Check if it was actually set
                const currentValue = this.vrm.expressionManager.getValue(expressionName);
                console.log(`🔧 DEBUG: Confirmed value for "${expressionName}": ${currentValue}`);
                
                // Log all current expression values
                console.log('🔧 DEBUG: All current expression values:');
                ['aa', 'ee', 'ih', 'oh', 'ou'].forEach(name => {
                    const value = this.vrm.expressionManager.getValue(name);
                    if (value > 0.01) {
                        console.log(`  ${name}: ${value.toFixed(3)}`);
                    }
                });
            } else {
                console.log(`🔧 DEBUG: No mapping found for viseme "${expression}"`);
            }
        }
        
        // Update UI
        this.statusEl.textContent = `DEBUG MODE: ${expression} (Press spacebar to cycle)`;
    }
    
    updateExpressions() {
        if (!this.vrm?.expressionManager || !this.expressionMapping) return;
        
        let hasActiveExpression = false;
        
        // Smooth interpolation towards target expressions
        this.currentExpressions.forEach((current, viseme) => {
            const target = this.targetExpressions.get(viseme) || 0;
            const smooth = current + (target - current) * this.expressionSmoothingFactor;
            this.currentExpressions.set(viseme, smooth);
            
            // Apply to VRM using the mapped expression name
            const expressionName = this.expressionMapping.get(viseme);
            if (expressionName && smooth > 0.01) {
                this.vrm.expressionManager.setValue(expressionName, smooth);
                hasActiveExpression = true;
                
            }
        });
        
        // Update expression manager
        this.vrm.expressionManager.update();
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());
        
        const now = performance.now();
        
        // Update FPS
        this.frameCount++;
        if (now - this.lastFrameTime >= 1000) {
            const fps = Math.round((this.frameCount * 1000) / (now - this.lastFrameTime));
            this.fpsEl.textContent = fps;
            this.frameCount = 0;
            this.lastFrameTime = now;
        }
        
        // Update avatar
        if (this.vrm) {
            this.updateExpressions();
            this.vrm.update(0.016); // ~60fps delta
        }
        
        // Update animations
        if (this.animationMixer) {
            this.animationMixer.update(0.016);
        }
        
        // Update controls
        this.controls.update();
        
        // Render
        this.renderer.render(this.scene, this.camera);
    }
    
    updateStatus(message, type = 'normal') {
        this.statusEl.textContent = message;
        this.statusEl.className = `status ${type}`;
    }
}

// Initialize the application
const app = new AvatarLipSync();