export class AudioProcessor {
    constructor() {
        this.audioContext = null;
        this.mediaStream = null;
        this.analyser = null;
        this.bufferLength = 0;
        this.dataArray = null;
        this.isProcessing = false;
        
        // Advanced audio analysis parameters
        this.sampleRate = 16000;
        this.fftSize = 2048;
        this.smoothingTimeConstant = 0.2;
        
        // Viseme detection parameters
        this.voiceActivityThreshold = 35;
        this.frequencyBands = {
            low: { start: 0, end: 8 },      // 0-600Hz - Open vowels (A, O)
            lowMid: { start: 8, end: 20 },  // 600-1500Hz - Mid vowels
            mid: { start: 20, end: 40 },    // 1500-3000Hz - Front vowels (E, I)
            high: { start: 40, end: 80 }    // 3000-6000Hz - Consonants, sibilants
        };
        
        // Hysteresis and smoothing  
        this.currentViseme = 'REST';
        this.visemeHistory = [];
        this.lastVisemeChange = 0;
        this.minDwellTime = 60; // minimum ms to stay on a viseme (reduced for better responsiveness)
        this.smoothingBuffer = [];
        this.maxSmoothingFrames = 3; // reduced for lower latency
        
        // Callbacks
        this.onVisemeChange = null;
        this.onVolumeChange = null;
    }
    
    async initialize() {
        try {
            // Request microphone with optimal settings
            this.mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    sampleRate: this.sampleRate,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });
            
            // Create audio context
            this.audioContext = new AudioContext({ 
                sampleRate: this.sampleRate 
            });
            
            // Set up audio analysis
            this.setupAudioAnalysis();
            
            console.log('AudioProcessor initialized successfully');
            return true;
            
        } catch (error) {
            console.error('Failed to initialize AudioProcessor:', error);
            throw error;
        }
    }
    
    setupAudioAnalysis() {
        const source = this.audioContext.createMediaStreamSource(this.mediaStream);
        
        // Create analyser node
        this.analyser = this.audioContext.createAnalyser();
        this.analyser.fftSize = this.fftSize;
        this.analyser.smoothingTimeConstant = this.smoothingTimeConstant;
        this.analyser.minDecibels = -90;
        this.analyser.maxDecibels = -10;
        
        // Connect audio graph
        source.connect(this.analyser);
        
        // Set up data arrays
        this.bufferLength = this.analyser.frequencyBinCount;
        this.dataArray = new Uint8Array(this.bufferLength);
        
        console.log('Audio analysis setup complete');
    }
    
    start() {
        if (!this.analyser) {
            throw new Error('AudioProcessor not initialized');
        }
        
        this.isProcessing = true;
        this.processAudio();
        console.log('Audio processing started');
    }
    
    stop() {
        this.isProcessing = false;
        
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
        }
        
        if (this.audioContext && this.audioContext.state !== 'closed') {
            this.audioContext.close();
        }
        
        // Reset state
        this.currentViseme = 'REST';
        this.visemeHistory = [];
        this.smoothingBuffer = [];
        
        console.log('Audio processing stopped');
    }
    
    processAudio() {
        if (!this.isProcessing) return;
        
        // Get frequency data
        this.analyser.getByteFrequencyData(this.dataArray);
        
        // Calculate band energies
        const bandEnergies = this.calculateBandEnergies();
        
        // Calculate overall volume (RMS-like)
        const volume = this.calculateVolume();
        
        // Voice Activity Detection
        const isVoiceActive = this.detectVoiceActivity(volume, bandEnergies);
        
        // Determine viseme
        let detectedViseme = 'REST';
        if (isVoiceActive) {
            detectedViseme = this.classifyViseme(bandEnergies);
        }
        
        // Apply smoothing and hysteresis
        const finalViseme = this.smoothViseme(detectedViseme);
        
        // Update if changed
        if (finalViseme !== this.currentViseme) {
            const now = performance.now();
            if (now - this.lastVisemeChange >= this.minDwellTime) {
                this.currentViseme = finalViseme;
                this.lastVisemeChange = now;
                
                if (this.onVisemeChange) {
                    this.onVisemeChange(finalViseme, now);
                }
            }
        }
        
        // Update volume callback
        if (this.onVolumeChange) {
            this.onVolumeChange(volume);
        }
        
        // Continue processing
        requestAnimationFrame(() => this.processAudio());
    }
    
    calculateBandEnergies() {
        const bands = {};
        
        Object.keys(this.frequencyBands).forEach(bandName => {
            const band = this.frequencyBands[bandName];
            let energy = 0;
            let count = 0;
            
            for (let i = band.start; i < Math.min(band.end, this.bufferLength); i++) {
                energy += this.dataArray[i];
                count++;
            }
            
            bands[bandName] = count > 0 ? energy / count : 0;
        });
        
        return bands;
    }
    
    calculateVolume() {
        let sum = 0;
        for (let i = 0; i < this.bufferLength; i++) {
            sum += this.dataArray[i] * this.dataArray[i];
        }
        return Math.sqrt(sum / this.bufferLength);
    }
    
    detectVoiceActivity(volume, bandEnergies) {
        // Voice activity detection based on volume and spectral content
        const totalEnergy = Object.values(bandEnergies).reduce((sum, energy) => sum + energy, 0);
        const spectralCentroid = this.calculateSpectralCentroid(bandEnergies);
        
        return volume > this.voiceActivityThreshold && 
               totalEnergy > 20 && 
               spectralCentroid > 0.1;
    }
    
    calculateSpectralCentroid(bandEnergies) {
        const weights = { low: 1, lowMid: 2, mid: 3, high: 4 };
        let weightedSum = 0;
        let totalEnergy = 0;
        
        Object.keys(bandEnergies).forEach(band => {
            const energy = bandEnergies[band];
            weightedSum += energy * weights[band];
            totalEnergy += energy;
        });
        
        return totalEnergy > 0 ? weightedSum / totalEnergy : 0;
    }
    
    classifyViseme(bandEnergies) {
        const { low, lowMid, mid, high } = bandEnergies;
        const total = low + lowMid + mid + high;
        
        if (total < 10) return 'REST';
        
        // Normalize energies
        const normLow = low / total;
        const normLowMid = lowMid / total;
        const normMid = mid / total;
        const normHigh = high / total;
        
        // Advanced viseme classification using spectral characteristics
        if (normLow > 0.4 && normMid < 0.25) {
            // Strong low frequencies, weak mid - open back vowels
            return normLowMid > 0.25 ? 'O' : 'A';
        } else if (normMid > 0.35) {
            // Strong mid frequencies - front vowels
            return normHigh > 0.2 ? 'E' : 'I';
        } else if (normLowMid > 0.35 && normLow < 0.3) {
            // Mid-low dominance - rounded vowels
            return 'U';
        } else if (normHigh > 0.3) {
            // High frequency content - likely consonants or 'E'
            return 'E';
        } else {
            // Default classification
            const maxBand = Object.keys(bandEnergies).reduce((a, b) => 
                bandEnergies[a] > bandEnergies[b] ? a : b
            );
            
            switch (maxBand) {
                case 'low': return 'A';
                case 'lowMid': return 'O';
                case 'mid': return 'I';
                case 'high': return 'E';
                default: return 'A';
            }
        }
    }
    
    smoothViseme(detectedViseme) {
        // Add to smoothing buffer
        this.smoothingBuffer.push(detectedViseme);
        if (this.smoothingBuffer.length > this.maxSmoothingFrames) {
            this.smoothingBuffer.shift();
        }
        
        // Find most common viseme in buffer
        const counts = {};
        this.smoothingBuffer.forEach(viseme => {
            counts[viseme] = (counts[viseme] || 0) + 1;
        });
        
        let maxCount = 0;
        let mostCommon = 'REST';
        Object.keys(counts).forEach(viseme => {
            if (counts[viseme] > maxCount) {
                maxCount = counts[viseme];
                mostCommon = viseme;
            }
        });
        
        return mostCommon;
    }
    
    // Utility methods
    getLatency() {
        return this.audioContext ? this.audioContext.baseLatency * 1000 : 0;
    }
    
    getSampleRate() {
        return this.audioContext ? this.audioContext.sampleRate : 0;
    }
    
    getBufferSize() {
        return this.fftSize;
    }
}