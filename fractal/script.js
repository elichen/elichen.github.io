let gl;
let program;
let zoomCenter = { x: -0.745, y: 0.1 }; // Interesting starting point for Mandelbrot
let zoomLevel = 0.5; // Start a bit zoomed in
let fractalType = 'mandelbrot';

const vertexShaderSource = `
    attribute vec2 a_position;
    void main() {
        gl_Position = vec4(a_position, 0.0, 1.0);
    }
`;

const fragmentShaderSource = `
    precision highp float;
    uniform vec2 u_resolution;
    uniform vec2 u_zoomCenter;
    uniform float u_zoomLevel;
    uniform int u_fractalType;

    vec3 getPsychedelicColor(float t) {
        vec3 c1 = vec3(0.1, 0.1, 0.4);  // Deep blue
        vec3 c2 = vec3(0.0, 0.7, 0.7);  // Turquoise
        vec3 c3 = vec3(0.9, 1.0, 0.3);  // Lime yellow
        vec3 c4 = vec3(1.0, 0.4, 0.0);  // Orange
        vec3 c5 = vec3(0.8, 0.0, 0.5);  // Magenta

        float s = mod(t * 5.0, 5.0);
        if (s < 1.0) return mix(c1, c2, s);
        else if (s < 2.0) return mix(c2, c3, s - 1.0);
        else if (s < 3.0) return mix(c3, c4, s - 2.0);
        else if (s < 4.0) return mix(c4, c5, s - 3.0);
        else return mix(c5, c1, s - 4.0);
    }

    vec2 complexMul(vec2 a, vec2 b) {
        return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
    }

    float mandelbrot(vec2 c) {
        vec2 z = vec2(0.0);
        for (int i = 0; i < 1000; i++) {
            z = complexMul(z, z) + c;
            if (dot(z, z) > 4.0) return float(i) / 1000.0;
        }
        return 0.0;
    }

    float julia(vec2 z) {
        vec2 c = vec2(-0.4, 0.6);
        for (int i = 0; i < 1000; i++) {
            z = complexMul(z, z) + c;
            if (dot(z, z) > 4.0) return float(i) / 1000.0;
        }
        return 0.0;
    }

    float sierpinski(vec2 p) {
        float scale = 1.0;
        for (int i = 0; i < 20; i++) { // Increased from 10 to 20 iterations
            p *= 2.0;
            p -= floor(p * 0.5) * 2.0;
            if (p.x > 1.0 && p.y > 1.0) {
                return float(i) / 20.0; // Updated to match the new iteration count
            }
            scale *= 2.0;
        }
        return 1.0;
    }

    void main() {
        vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / min(u_resolution.x, u_resolution.y);
        uv = (uv - u_zoomCenter) * u_zoomLevel;

        float value;
        if (u_fractalType == 0) {
            value = mandelbrot(uv);
        } else if (u_fractalType == 1) {
            value = julia(uv);
        } else {
            value = sierpinski(uv + vec2(0.5));
        }

        vec3 color;
        if (value == 0.0) {
            color = vec3(0.0, 0.0, 0.0); // Black for inside the fractal
        } else {
            color = getPsychedelicColor(value);
        }
        gl_FragColor = vec4(color, 1.0);
    }
`;

let maxLateralZoom = 0.9; // Adjust this value to control the maximum lateral movement

let mousePosition = { x: -0.745, y: 0.1 }; // Match initial zoom center

function updateZoom() {
    const zoomFactor = zoomSpeed;
    const newZoomLevel = zoomLevel * zoomFactor;
    
    // Calculate new zoom center
    zoomCenter.x = mousePosition.x - (mousePosition.x - zoomCenter.x) * (zoomLevel / newZoomLevel);
    zoomCenter.y = mousePosition.y - (mousePosition.y - zoomCenter.y) * (zoomLevel / newZoomLevel);
    
    zoomLevel = newZoomLevel;
}

function initWebGL() {
    const canvas = document.getElementById('fractalCanvas');
    gl = canvas.getContext('webgl');

    if (!gl) {
        alert('WebGL not supported');
        return;
    }

    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    gl.viewport(0, 0, canvas.width, canvas.height);

    const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
    const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSource);

    program = createProgram(gl, vertexShader, fragmentShader);

    const positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);

    const positionAttributeLocation = gl.getAttribLocation(program, 'a_position');
    gl.enableVertexAttribArray(positionAttributeLocation);
    gl.vertexAttribPointer(positionAttributeLocation, 2, gl.FLOAT, false, 0, 0);
}

function createShader(gl, type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        console.error('Shader compilation error:', gl.getShaderInfoLog(shader));
        gl.deleteShader(shader);
        return null;
    }
    return shader;
}

function createProgram(gl, vertexShader, fragmentShader) {
    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        console.error('Program linking error:', gl.getProgramInfoLog(program));
        gl.deleteProgram(program);
        return null;
    }
    return program;
}

function draw() {
    gl.useProgram(program);

    const resolutionUniformLocation = gl.getUniformLocation(program, 'u_resolution');
    gl.uniform2f(resolutionUniformLocation, gl.canvas.width, gl.canvas.height);

    const zoomCenterUniformLocation = gl.getUniformLocation(program, 'u_zoomCenter');
    gl.uniform2f(zoomCenterUniformLocation, zoomCenter.x, zoomCenter.y);

    const zoomLevelUniformLocation = gl.getUniformLocation(program, 'u_zoomLevel');
    gl.uniform1f(zoomLevelUniformLocation, zoomLevel);

    const fractalTypeUniformLocation = gl.getUniformLocation(program, 'u_fractalType');
    gl.uniform1i(fractalTypeUniformLocation, ['mandelbrot', 'julia', 'sierpinski'].indexOf(fractalType));

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
}

let zoomSpeed = 0.995; // Adjust this value to control zoom speed (closer to 1 means slower zoom)
// To change the zoom speed, modify this value:
// - Values closer to 1 (e.g., 0.999) result in slower zooming
// - Values further from 1 (e.g., 0.99) result in faster zooming

function animate() {
    updateZoom();
    draw();
    requestAnimationFrame(animate);
}

function handleMouseMove(event) {
    const rect = gl.canvas.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / gl.canvas.width) * 2 - 1;
    const y = -((event.clientY - rect.top) / gl.canvas.height) * 2 + 1;
    mousePosition = { x, y };
}

function handleFractalTypeChange(event) {
    fractalType = event.target.value;
    if (fractalType === 'mandelbrot') {
        zoomCenter = { x: -0.745, y: 0.1 };
        zoomLevel = 0.5;
    } else if (fractalType === 'sierpinski') {
        zoomCenter = { x: 0, y: -0.2 }; // Adjusted center for Sierpinski
        zoomLevel = 1.5; // Zoomed out a bit to show more of the pattern
    } else {
        zoomCenter = { x: 0, y: 0 };
        zoomLevel = 1;
    }
    mousePosition = { ...zoomCenter };
}

function preventZoom(e) {
    e.preventDefault();
    e.stopPropagation();
}

window.onload = () => {
    initWebGL();
    gl.canvas.addEventListener('mousemove', handleMouseMove);
    document.getElementById('fractalType').addEventListener('change', handleFractalTypeChange);
    
    // Prevent scroll wheel and pinch zoom
    document.addEventListener('wheel', preventZoom, { passive: false });
    document.addEventListener('touchmove', preventZoom, { passive: false });
    
    // Set initial mouse and zoom center positions to the interesting point
    mousePosition = { x: -0.745, y: 0.1 };
    zoomCenter = { x: -0.745, y: 0.1 };
    
    animate();
};

window.onresize = () => {
    gl.canvas.width = window.innerWidth;
    gl.canvas.height = window.innerHeight;
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
};