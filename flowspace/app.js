class FlowSpace {
    constructor() {
        this.canvas = document.getElementById('flowCanvas');
        this.gl = this.canvas.getContext('webgl');
        
        if (!this.gl) {
            alert('WebGL not supported');
            return;
        }

        // Get required extensions
        this.ext = {
            floatTexture: this.gl.getExtension('OES_texture_float'),
            floatLinear: this.gl.getExtension('OES_texture_float_linear')
        };

        if (!this.ext.floatTexture) {
            alert('Float textures not supported');
            return;
        }
        
        this.startTime = Date.now();
        this.setupCanvas();
        this.loadShaders();
        this.setupBuffers();
        this.setupTextures();
        this.setupMouseEvents();
        this.setupAudioButton();
        
        // Initialize audio values with defaults
        this.audioValues = {
            bass: 0.5,
            mid: 0.3,
            high: 0.2,
            energy: 0.4
        };
        
        this.render();
    }

    setupAudioButton() {
        this.startButton = document.getElementById('startAudio');
        this.startButton.addEventListener('click', async () => {
            if (!this.audioReady) {
                await this.setupAudio();
                if (this.audioReady) {
                    this.startButton.classList.add('hidden');
                }
            }
        });
    }

    async setupAudio() {
        try {
            // Initialize audio context
            this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            
            // Create analyzer node
            this.analyzer = this.audioCtx.createAnalyser();
            this.analyzer.fftSize = 1024;
            
            // Get audio input
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const source = this.audioCtx.createMediaStreamSource(stream);
            source.connect(this.analyzer);

            // Create additional analyzers for different frequency ranges
            this.bassAnalyzer = this.audioCtx.createBiquadFilter();
            this.midAnalyzer = this.audioCtx.createBiquadFilter();
            this.highAnalyzer = this.audioCtx.createBiquadFilter();

            // Configure filters
            this.bassAnalyzer.type = 'lowpass';
            this.bassAnalyzer.frequency.value = 150;

            this.midAnalyzer.type = 'bandpass';
            this.midAnalyzer.frequency.value = 1000;
            this.midAnalyzer.Q.value = 1;

            this.highAnalyzer.type = 'highpass';
            this.highAnalyzer.frequency.value = 4000;

            // Connect filters
            source.connect(this.bassAnalyzer);
            source.connect(this.midAnalyzer);
            source.connect(this.highAnalyzer);

            // Create analyzers for each frequency range
            this.bassAnalyzerNode = this.audioCtx.createAnalyser();
            this.midAnalyzerNode = this.audioCtx.createAnalyser();
            this.highAnalyzerNode = this.audioCtx.createAnalyser();

            this.bassAnalyzer.connect(this.bassAnalyzerNode);
            this.midAnalyzer.connect(this.midAnalyzerNode);
            this.highAnalyzer.connect(this.highAnalyzerNode);

            // Initialize data arrays
            this.dataArray = new Uint8Array(this.analyzer.frequencyBinCount);
            this.bassArray = new Uint8Array(this.bassAnalyzerNode.frequencyBinCount);
            this.midArray = new Uint8Array(this.midAnalyzerNode.frequencyBinCount);
            this.highArray = new Uint8Array(this.highAnalyzerNode.frequencyBinCount);

            // Audio is ready
            this.audioReady = true;

        } catch (err) {
            console.error('Error accessing microphone:', err);
            this.audioReady = false;
            this.startButton.textContent = 'Microphone access denied';
            this.startButton.style.background = 'rgba(255, 50, 50, 0.2)';
        }
    }

    updateAudioValues() {
        if (!this.audioReady) return;

        try {
            // Get frequency data
            this.analyzer.getByteFrequencyData(this.dataArray);
            this.bassAnalyzerNode.getByteFrequencyData(this.bassArray);
            this.midAnalyzerNode.getByteFrequencyData(this.midArray);
            this.highAnalyzerNode.getByteFrequencyData(this.highArray);

            // Calculate average values for each frequency range
            const getAverage = (array) => {
                const sum = array.reduce((a, b) => a + b, 0);
                return sum / array.length / 255; // Normalize to 0-1
            };

            const newValues = {
                bass: getAverage(this.bassArray),
                mid: getAverage(this.midArray),
                high: getAverage(this.highArray),
                energy: getAverage(this.dataArray)
            };

            // Apply smoothing
            const smooth = 0.8;
            this.audioValues = {
                bass: newValues.bass * smooth + (this.audioValues.bass || 0) * (1 - smooth),
                mid: newValues.mid * smooth + (this.audioValues.mid || 0) * (1 - smooth),
                high: newValues.high * smooth + (this.audioValues.high || 0) * (1 - smooth),
                energy: newValues.energy * smooth + (this.audioValues.energy || 0) * (1 - smooth)
            };
        } catch (err) {
            console.error('Error updating audio values:', err);
        }
    }
    
    setupCanvas() {
        const dpr = window.devicePixelRatio || 1;
        const rect = this.canvas.getBoundingClientRect();
        
        this.canvas.width = rect.width * dpr;
        this.canvas.height = rect.height * dpr;
        
        this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    }
    
    loadShaders() {
        const vertexSource = document.getElementById('vertexShader').textContent;
        const fragmentSource = document.getElementById('fragmentShader').textContent;
        
        const vertexShader = this.createShader(this.gl.VERTEX_SHADER, vertexSource);
        const fragmentShader = this.createShader(this.gl.FRAGMENT_SHADER, fragmentSource);
        
        this.program = this.createProgram(vertexShader, fragmentShader);
        this.gl.useProgram(this.program);
        
        // Get attribute and uniform locations
        this.positionLocation = this.gl.getAttribLocation(this.program, 'position');
        this.mouseLocation = this.gl.getUniformLocation(this.program, 'uMouse');
        this.resolutionLocation = this.gl.getUniformLocation(this.program, 'uResolution');
        this.timeLocation = this.gl.getUniformLocation(this.program, 'uTime');
        
        // Audio uniform locations
        this.bassLocation = this.gl.getUniformLocation(this.program, 'uBass');
        this.midLocation = this.gl.getUniformLocation(this.program, 'uMid');
        this.highLocation = this.gl.getUniformLocation(this.program, 'uHigh');
        this.energyLocation = this.gl.getUniformLocation(this.program, 'uEnergy');
    }
    
    createShader(type, source) {
        const shader = this.gl.createShader(type);
        this.gl.shaderSource(shader, source);
        this.gl.compileShader(shader);
        
        if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
            console.error('Shader compile error:', this.gl.getShaderInfoLog(shader));
            this.gl.deleteShader(shader);
            return null;
        }
        
        return shader;
    }
    
    createProgram(vertexShader, fragmentShader) {
        const program = this.gl.createProgram();
        this.gl.attachShader(program, vertexShader);
        this.gl.attachShader(program, fragmentShader);
        this.gl.linkProgram(program);
        
        if (!this.gl.getProgramParameter(program, this.gl.LINK_STATUS)) {
            console.error('Program link error:', this.gl.getProgramInfoLog(program));
            return null;
        }
        
        return program;
    }
    
    setupBuffers() {
        // Create a buffer for the quad that fills the screen
        const positions = new Float32Array([
            -1, -1,
            1, -1,
            -1, 1,
            1, 1
        ]);
        
        const buffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, buffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, positions, this.gl.STATIC_DRAW);
        
        this.gl.enableVertexAttribArray(this.positionLocation);
        this.gl.vertexAttribPointer(this.positionLocation, 2, this.gl.FLOAT, false, 0, 0);
    }
    
    setupTextures() {
        // Create textures for velocity and pressure
        this.velocityTexture = this.createTexture();
        this.pressureTexture = this.createTexture();

        // Set initial texture data
        const width = this.canvas.width;
        const height = this.canvas.height;
        const initialData = new Float32Array(width * height * 4);
        
        this.gl.bindTexture(this.gl.TEXTURE_2D, this.velocityTexture);
        this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA, width, height, 0, this.gl.RGBA, this.gl.FLOAT, initialData);
        
        this.gl.bindTexture(this.gl.TEXTURE_2D, this.pressureTexture);
        this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA, width, height, 0, this.gl.RGBA, this.gl.FLOAT, initialData);
    }
    
    createTexture() {
        const texture = this.gl.createTexture();
        this.gl.bindTexture(this.gl.TEXTURE_2D, texture);
        
        // Use LINEAR if we have the linear extension, otherwise fall back to NEAREST
        const filter = this.ext.floatLinear ? this.gl.LINEAR : this.gl.NEAREST;
        
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, filter);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, filter);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);
        
        return texture;
    }
    
    setupMouseEvents() {
        this.mousePosition = { x: 0, y: 0 };
        
        this.canvas.addEventListener('mousemove', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            this.mousePosition.x = (e.clientX - rect.left) / rect.width;
            this.mousePosition.y = 1.0 - (e.clientY - rect.top) / rect.height;
        });
    }
    
    render() {
        // Update audio analysis
        this.updateAudioValues();

        const time = (Date.now() - this.startTime) * 0.001; // Convert to seconds
        
        // Update uniforms
        this.gl.uniform1f(this.timeLocation, time);
        this.gl.uniform2f(this.mouseLocation, this.mousePosition.x, this.mousePosition.y);
        this.gl.uniform2f(this.resolutionLocation, this.canvas.width, this.canvas.height);
        
        // Update audio uniforms
        this.gl.uniform1f(this.bassLocation, this.audioValues.bass);
        this.gl.uniform1f(this.midLocation, this.audioValues.mid);
        this.gl.uniform1f(this.highLocation, this.audioValues.high);
        this.gl.uniform1f(this.energyLocation, this.audioValues.energy);
        
        this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
        requestAnimationFrame(() => this.render());
    }
}

// Start the application when the window loads
window.addEventListener('load', () => {
    new FlowSpace();
}); 