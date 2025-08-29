export class AudioProcessor {
    constructor() {
        this.audioContext = null;
        this.mediaStream = null;
        this.analyser = null;
        this.bufferLength = 0;
        this.dataArray = null;
        this.isProcessing = false;
        
        // TTS mode properties
        this.mode = 'microphone'; // 'microphone' or 'tts'
        this.speechSynthesis = window.speechSynthesis;
        this.currentUtterance = null;
        this.ttsCharacterMap = null;
        this.currentText = '';
        this.ttsIsActive = false;
        
        // Word animation tracking
        this.currentWordData = {
            text: '',
            visemes: [],
            startTime: 0,
            charIndex: 0,
            duration: 0
        };
        
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
        this.mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                sampleRate: this.sampleRate,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true
            }
        });
        
        this.audioContext = new AudioContext({ 
            sampleRate: this.sampleRate 
        });
        
        this.setupAudioAnalysis();
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
        this.isProcessing = true;
        this.processAudio();
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
        }
        
        return 'REST';
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
    
    // TTS Mode Methods
    async initializeTTS() {
        this.ttsTimingData = null;
        this.ttsStartTime = 0;
        this.ttsIsActive = false;
    }
    
    setMode(mode) {
        if (mode === this.mode) return;
        
        // Stop current processing
        this.stop();
        
        this.mode = mode;
        console.log(`Audio processor mode set to: ${mode}`);
    }
    
    async speakText(text, voice = null, rate = 1, pitch = 1) {
        this.speechSynthesis.cancel();
        
        this.ttsCharacterMap = this.generateCharacterMap(text);
        this.currentText = text;
        
        this.currentUtterance = new SpeechSynthesisUtterance(text);
        
        if (voice) this.currentUtterance.voice = voice;
        this.currentUtterance.rate = rate;
        this.currentUtterance.pitch = pitch;
        this.currentUtterance.volume = 1.0;
        
        this.currentUtterance.onstart = () => {
            this.ttsIsActive = true;
            this.isProcessing = true;
            console.log('TTS speech started - real-time sync active');
        };
        
        this.currentUtterance.onend = () => {
            this.stopTTSRealTimeSync();
            console.log('TTS speech ended');
        };
        
        this.currentUtterance.onerror = (event) => {
            this.stopTTSRealTimeSync();
        };
        
        // Real-time viseme sync using boundary events
        this.currentUtterance.onboundary = (event) => {
            if (event.name === 'word') {
                const word = this.extractWordAt(this.currentText, event.charIndex);
                
                // Generate viseme sequence for entire word
                this.currentWordData = {
                    text: word,
                    visemes: word.split('').map(c => this.getVisemeForChar(c)),
                    startTime: performance.now(),
                    charIndex: event.charIndex,
                    duration: word.length * 60 // Roughly 60ms per character
                };
                
                console.log(`Word boundary: "${word}" → visemes: [${this.currentWordData.visemes.join(', ')}]`);
                
                // Start animating through the word
                this.animateWord();
            }
        };
        
        // Start speaking
        this.speechSynthesis.speak(this.currentUtterance);
    }
    
    getVisemeForChar(char) {
        const c = char.toLowerCase();
        
        // Vowels - clear mappings
        if ('aáàâä'.includes(c)) return 'A';
        if ('eéèêë'.includes(c)) return 'E'; 
        if ('iíìîï'.includes(c)) return 'I';
        if ('oóòôö'.includes(c)) return 'O';
        if ('uúùûü'.includes(c)) return 'U';
        
        // Consonants with distinct mouth shapes
        if ('pbm'.includes(c)) return 'A';  // Bilabial - lips together then open
        if ('w'.includes(c)) return 'O';     // Rounded lips like 'oo'
        if ('fv'.includes(c)) return 'U';    // Labiodental - teeth on lower lip
        if ('tdnl'.includes(c)) return 'I';  // Alveolar - tongue to roof
        if ('sz'.includes(c)) return 'E';    // Sibilant - teeth showing
        if ('kgq'.includes(c)) return 'I';   // Velar - back of tongue
        if ('r'.includes(c)) return 'O';     // Retroflex - rounded
        if ('ch'.includes(c)) return 'E';    // Affricate - teeth
        if ('sh'.includes(c)) return 'U';    // Fricative - pursed
        if ('th'.includes(c)) return 'I';    // Dental - tongue between teeth
        if ('y'.includes(c)) return 'E';     // Palatal - like 'ee'
        if ('h'.includes(c)) return 'A';     // Glottal - open
        if ('j'.includes(c)) return 'E';     // Like 'dzh'
        if ('x'.includes(c)) return 'I';     // Like 'ks'
        if ('c'.includes(c)) return 'I';     // Can be 'k' or 's'
        
        // Default neutral position
        return 'I';
    }
    
    generateCharacterMap(text) {
        const characterMap = [];
        
        for (let i = 0; i < text.length; i++) {
            characterMap[i] = this.getVisemeForChar(text[i]);
        }
        
        console.log('Generated character-to-viseme map:', characterMap.length, 'characters');
        return characterMap;
    }
    
    
    extractWordAt(text, charIndex) {
        // Find word boundaries
        let start = charIndex;
        let end = charIndex;
        
        // Find start of word
        while (start > 0 && !/\s/.test(text[start - 1])) {
            start--;
        }
        
        // Find end of word
        while (end < text.length && !/\s/.test(text[end])) {
            end++;
        }
        
        return text.substring(start, end);
    }
    
    animateWord() {
        if (!this.ttsIsActive || !this.currentWordData.visemes.length) return;
        
        const elapsed = performance.now() - this.currentWordData.startTime;
        const progress = Math.min(elapsed / this.currentWordData.duration, 1);
        const charPos = Math.floor(progress * this.currentWordData.visemes.length);
        
        const viseme = this.currentWordData.visemes[charPos] || this.currentWordData.visemes[this.currentWordData.visemes.length - 1];
        
        if (viseme !== this.currentViseme) {
            this.currentViseme = viseme;
            if (this.onVisemeChange) {
                this.onVisemeChange(viseme, performance.now());
            }
        }
        
        // Continue animating if word is not complete
        if (progress < 1) {
            requestAnimationFrame(() => this.animateWord());
        }
    }
    
    stopTTSRealTimeSync() {
        this.ttsIsActive = false;
        this.isProcessing = false;
        
        // Reset to rest state
        this.currentViseme = 'REST';
        if (this.onVisemeChange) {
            this.onVisemeChange('REST', performance.now());
        }
        
        console.log('TTS real-time sync stopped');
    }
    
    
    stopTTS() {
        if (this.speechSynthesis) {
            this.speechSynthesis.cancel();
        }
        this.stopTTSRealTimeSync();
    }
    
    getAvailableVoices() {
        return this.speechSynthesis.getVoices();
    }
    
    // Utility methods
    getLatency() {
        if (this.mode === 'tts') {
            // TTS latency is primarily the browser's speech synthesis latency
            return 50; // Estimated TTS latency
        } else {
            return this.audioContext ? this.audioContext.baseLatency * 1000 : 0;
        }
    }
    
    getSampleRate() {
        return this.audioContext ? this.audioContext.sampleRate : 0;
    }
    
    getBufferSize() {
        return this.fftSize;
    }
}