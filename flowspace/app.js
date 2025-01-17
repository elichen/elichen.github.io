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
        
        this.render();
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
        const time = (Date.now() - this.startTime) * 0.001; // Convert to seconds
        this.gl.uniform1f(this.timeLocation, time);
        this.gl.uniform2f(this.mouseLocation, this.mousePosition.x, this.mousePosition.y);
        this.gl.uniform2f(this.resolutionLocation, this.canvas.width, this.canvas.height);
        
        this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
        requestAnimationFrame(() => this.render());
    }
}

// Start the application when the window loads
window.addEventListener('load', () => {
    new FlowSpace();
}); 