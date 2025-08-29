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
        this.debugExpressions = ['A', 'E', 'I', 'O', 'U'];
        this.currentDebugIndex = 0;
        
        // Performance tracking
        this.lastFrameTime = 0;
        this.frameCount = 0;
        this.fpsHistory = [];
        this.latencyStart = 0;
        
        // Expression smoothing
        this.currentExpressions = new Map();
        this.targetExpressions = new Map();
        this.currentActiveViseme = 'REST';
        this.previousViseme = 'REST';
        this.transitionProgress = 1.0; // 0 = start of transition, 1 = complete
        this.expressionSmoothingFactor = 0.35; // Fast enough for phoneme changes
        
        // Mode management
        this.currentMode = 'microphone'; // 'microphone' or 'tts'
        
        // Sample texts for TTS demonstration
        this.sampleTexts = {
            demo: "Welcome to the future of avatar technology! This real-time lip sync system uses advanced audio processing and three-dimensional rendering to create lifelike facial animations. Experience the seamless integration of artificial intelligence and interactive media.",
            story: "Once upon a time, in a digital realm where pixels danced and code came alive, there lived a virtual avatar who could speak with the voice of anyone who gave her words. She existed between ones and zeros, yet felt as real as any breathing soul.",
            technical: "This system leverages the Web Speech API for text-to-speech synthesis, combined with real-time frequency analysis using Fast Fourier Transform algorithms. The audio pipeline processes spectral data to classify vowel sounds into visemes, which are then mapped to facial expression morphs in the VRM avatar format."
        };
        
        // UI elements
        this.startBtn = document.getElementById('start-btn');
        this.modeToggleBtn = document.getElementById('mode-toggle');
        this.micControls = document.getElementById('mic-controls');
        this.ttsControls = document.getElementById('tts-controls');
        this.sampleTextSelect = document.getElementById('sample-text');
        this.customTextArea = document.getElementById('custom-text');
        this.voiceSelect = document.getElementById('voice-select');
        this.speechRateSlider = document.getElementById('speech-rate');
        this.speechPitchSlider = document.getElementById('speech-pitch');
        this.speakBtn = document.getElementById('speak-btn');
        this.statusEl = document.getElementById('status');
        this.audioStatusEl = document.getElementById('audio-status');
        this.rhubarbStatusEl = document.getElementById('rhubarb-status');
        this.avatarStatusEl = document.getElementById('avatar-status');
        this.latencyEl = document.getElementById('latency');
        this.fpsEl = document.getElementById('fps');
        this.visemeEl = document.getElementById('viseme');
        
        this.init();
    }
    
    async init() {
        await this.initThreeJS();
        await this.loadAvatar();
        await this.initAudioProcessor();
        this.setupEventListeners();
        this.animate();
        this.isInitialized = true;
        
        if (this.debugMode) {
            this.updateStatus('DEBUG MODE: Press spacebar to cycle through expressions', 'ready');
            setTimeout(() => this.nextDebugExpression(), 1000);
        } else {
            this.updateStatus('Ready to start', 'ready');
        }
    }
    
    
    async initThreeJS() {
        const canvas = document.getElementById('avatar-canvas');
        
        this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
        
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
        
    }
    
    async loadAvatar() {
        const loader = new GLTFLoader();
        loader.register((parser) => new VRMLoaderPlugin(parser));
        loader.register((parser) => new VRMAnimationLoaderPlugin(parser));
        
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
        
        this.scene.add(this.vrm.scene);
        this.initExpressions();
        
        this.animationMixer = new THREE.AnimationMixer(this.vrm.scene);
        this.addIdleAnimation();
        
        this.vrm.scene.position.y = 0.6;
        this.vrm.scene.rotation.y = 0;
        
        this.avatarStatusEl.textContent = 'Loaded';
    }
    
    initExpressions() {
        
        console.log('Mouth expression names:', this.vrm.expressionManager.mouthExpressionNames);
        console.log('Blink expression names:', this.vrm.expressionManager.blinkExpressionNames);
        console.log('LookAt expression names:', this.vrm.expressionManager.lookAtExpressionNames);
        
        const mouthExpressions = this.vrm.expressionManager.mouthExpressionNames;
        
        // Create a mapping from our visemes to the available mouth expressions
        this.expressionMapping = new Map();
        
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
        const humanoid = this.vrm.humanoid;
        
        
        // 4-second breathing cycle with seamless looping
        const times = [0, 1, 2, 3, 4];
        const tracks = [];
        
        // Fixed smooth loop function for true seamless looping
        const createSmoothLoop = (amplitude, frequency = 1, phase = 0) => {
            return times.map(t => amplitude * Math.sin((t * frequency * 2 * Math.PI / 4) + phase));
        };
        
        const chestBone = humanoid.getNormalizedBoneNode('chest');
            
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
        
        const spineBone = humanoid.getNormalizedBoneNode('spine');
        const spineQuaternions = [];
        const swayX = createSmoothLoop(0.01, 1, 0);
        const swayY = createSmoothLoop(0.008, 1, Math.PI/3);
        const swayZ = createSmoothLoop(0.005, 1, Math.PI/6);
        
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
        
        const headBone = humanoid.getNormalizedBoneNode('head');
        const headPosY = createSmoothLoop(0.01, 1, 0);
        const headPositions = [];
        
        for (let i = 0; i < times.length; i++) {
            headPositions.push(0, headPosY[i], 0);
        }
        
        const headPosTrack = new THREE.VectorKeyframeTrack(
            headBone.name + '.position',
            times,
            headPositions
        );
        tracks.push(headPosTrack);
        
        const leftShoulder = humanoid.getNormalizedBoneNode('leftShoulder');
        const rightShoulder = humanoid.getNormalizedBoneNode('rightShoulder');
        const leftShoulderQuats = [];
        const rightShoulderQuats = [];
        
        for (let i = 0; i < times.length; i++) {
            const t = times[i];
            const leftRotZ = Math.sin(t * Math.PI * 0.5 + Math.PI * 0.2) * 0.01;
            const rightRotZ = Math.sin(t * Math.PI * 0.5 - Math.PI * 0.2) * 0.01;
            
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
        
        const leftUpperArm = humanoid.getNormalizedBoneNode('leftUpperArm');
        const rightUpperArm = humanoid.getNormalizedBoneNode('rightUpperArm');
        const leftArmQuats = [];
        const rightArmQuats = [];
        
        for (let i = 0; i < times.length; i++) {
            const t = times[i];
            const baseRotX = 0.3;
            const baseRotZ_left = 1.2;
            const baseRotZ_right = -1.2;
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
        
        const clip = new THREE.AnimationClip('idle', 4, tracks);
        const action = this.animationMixer.clipAction(clip);
        action.loop = THREE.LoopRepeat;
        action.play();
    }
    
    async initAudioProcessor() {
        this.audioProcessor = new AudioProcessor();
        
        this.audioProcessor.onVisemeChange = (viseme, timestamp) => {
            const processingStart = performance.now();
            this.setViseme(viseme);
            
            const audioLatency = this.audioProcessor.getLatency();
            const processingTime = performance.now() - processingStart;
            const totalLatency = audioLatency + processingTime;
            
            this.latencyEl.textContent = `${Math.round(totalLatency)} ms`;
            
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
        };
        
        this.rhubarbStatusEl.textContent = 'Advanced Audio Ready';
    }
    
    setupEventListeners() {
        this.startBtn.addEventListener('click', () => {
            if (this.isRecording) {
                this.stopRecording();
            } else {
                this.startRecording();
            }
        });
        
        // Mode toggle button
        this.modeToggleBtn.addEventListener('click', () => {
            this.toggleMode();
        });
        
        // TTS controls
        this.sampleTextSelect.addEventListener('change', (e) => {
            if (e.target.value === 'custom') {
                this.customTextArea.style.display = 'block';
            } else {
                this.customTextArea.style.display = 'none';
                if (e.target.value && this.sampleTexts[e.target.value]) {
                    this.customTextArea.value = this.sampleTexts[e.target.value];
                }
            }
        });
        
        this.speakBtn.addEventListener('click', () => {
            this.handleTTSSpeak();
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
        
        // Initialize TTS voices
        this.initializeTTSVoices();
    }
    
    async startRecording() {
        await this.audioProcessor.initialize();
        this.audioProcessor.start();
        
        this.isRecording = true;
        this.startBtn.textContent = 'Stop Microphone';
        this.startBtn.classList.add('stop');
        this.updateStatus('Recording and analyzing speech...', 'active');
        this.audioStatusEl.textContent = 'Recording';
    }
    
    
    setViseme(viseme) {
        // Track transition
        if (viseme !== this.currentActiveViseme) {
            this.previousViseme = this.currentActiveViseme;
            this.transitionProgress = 0; // Start new transition
        }
        
        // Reset ALL expressions to prevent additive blending
        this.targetExpressions.forEach((_, name) => {
            this.targetExpressions.set(name, 0);
        });
        
        // Set only the new target viseme
        if (viseme !== 'REST') {
            this.targetExpressions.set(viseme, 2.0);
        }
        
        this.currentActiveViseme = viseme;
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
    
    toggleMode() {
        if (this.currentMode === 'microphone') {
            this.currentMode = 'tts';
            this.audioProcessor.setMode('tts');
            this.modeToggleBtn.textContent = 'Switch to Microphone';
            this.micControls.style.display = 'none';
            this.ttsControls.style.display = 'block';
            this.updateStatus('TTS mode active - select text to speak', 'ready');
        } else {
            this.currentMode = 'microphone';
            this.audioProcessor.setMode('microphone');
            this.modeToggleBtn.textContent = 'Switch to TTS Mode';
            this.micControls.style.display = 'block';
            this.ttsControls.style.display = 'none';
            this.updateStatus('Microphone mode active', 'ready');
        }
    }
    
    initializeTTSVoices() {
        const updateVoices = () => {
            const voices = this.audioProcessor.getAvailableVoices();
            this.voiceSelect.innerHTML = '';
            
            if (voices.length === 0) {
                this.voiceSelect.innerHTML = '<option>No voices available</option>';
                return;
            }
            
            // Add default option
            this.voiceSelect.innerHTML = '<option value="">Default Voice</option>';
            
            // Add available voices
            voices.forEach((voice, index) => {
                const option = document.createElement('option');
                option.value = index;
                option.textContent = `${voice.name} (${voice.lang})`;
                if (voice.default) {
                    option.textContent += ' - Default';
                    option.selected = true;
                }
                this.voiceSelect.appendChild(option);
            });
        };
        
        // Initial load
        updateVoices();
        
        // Some browsers need to wait for voices to load
        if (window.speechSynthesis.onvoiceschanged !== undefined) {
            window.speechSynthesis.onvoiceschanged = updateVoices;
        }
    }
    
    async handleTTSSpeak() {
        let textToSpeak = '';
        const selectedSample = this.sampleTextSelect.value;
        
        if (selectedSample === 'custom') {
            textToSpeak = this.customTextArea.value.trim();
        } else if (selectedSample && this.sampleTexts[selectedSample]) {
            textToSpeak = this.sampleTexts[selectedSample];
        }
        
        const voices = this.audioProcessor.getAvailableVoices();
        const selectedVoiceIndex = this.voiceSelect.value;
        const voice = selectedVoiceIndex ? voices[parseInt(selectedVoiceIndex)] : null;
        
        const rate = parseFloat(this.speechRateSlider.value);
        const pitch = parseFloat(this.speechPitchSlider.value);
        
        this.speakBtn.textContent = 'Speaking...';
        this.speakBtn.disabled = true;
        this.updateStatus('Speaking text with lip sync...', 'active');
        this.audioStatusEl.textContent = 'TTS Active';
        
        await this.audioProcessor.speakText(textToSpeak, voice, rate, pitch);
        
        this.audioProcessor.currentUtterance.onend = () => {
            this.speakBtn.textContent = 'Speak Text';
            this.speakBtn.disabled = false;
            this.updateStatus('TTS speech completed', 'ready');
            this.audioStatusEl.textContent = 'Ready';
            this.visemeEl.textContent = 'REST';
        };
    }
    
    nextDebugExpression() {
        this.currentDebugIndex = (this.currentDebugIndex + 1) % this.debugExpressions.length;
        const expression = this.debugExpressions[this.currentDebugIndex];
        
        console.log(`🔧 DEBUG: Cycling to expression ${this.currentDebugIndex}: ${expression}`);
        
        // Only use the smooth interpolation system - no direct VRM setting
        this.setViseme(expression);
        
        // Update UI
        this.statusEl.textContent = `DEBUG MODE: ${expression} (Press spacebar to cycle)`;
    }
    
    updateExpressions() {
        // Update transition progress
        if (this.transitionProgress < 1.0) {
            this.transitionProgress = Math.min(1.0, this.transitionProgress + this.expressionSmoothingFactor);
        }
        
        // For cross-fade: during transition, scale expressions to prevent additive blending
        const isTransitioning = this.transitionProgress < 0.95;
        
        this.currentExpressions.forEach((current, viseme) => {
            const target = this.targetExpressions.get(viseme);
            if (target !== undefined) {
                let finalValue = current + (target - current) * this.expressionSmoothingFactor;
                
                // During transitions between different visemes, apply cross-fade scaling
                if (isTransitioning && this.previousViseme !== 'REST' && this.currentActiveViseme !== 'REST' 
                    && this.previousViseme !== this.currentActiveViseme) {
                    
                    // If this is the outgoing viseme, fade it out faster
                    if (viseme === this.previousViseme) {
                        finalValue *= (1.0 - this.transitionProgress);
                    }
                    // If this is the incoming viseme, fade it in gradually
                    else if (viseme === this.currentActiveViseme) {
                        finalValue *= this.transitionProgress;
                    }
                    // All other visemes should be at 0
                    else {
                        finalValue *= 0.1; // Quick fade to prevent interference
                    }
                }
                
                this.currentExpressions.set(viseme, finalValue);
                
                const expressionName = this.expressionMapping.get(viseme);
                if (expressionName) {
                    this.vrm.expressionManager.setValue(expressionName, Math.max(0, finalValue));
                }
            }
        });
        
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