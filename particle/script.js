let gl;
let program;
let positionBuffer;
let densityBuffer;
let particleCount = 3000;
let isRunning = false;
let time = 0;
let gravityStrength = 0.001;
let hubbleConstant = 0.0000;

let positions;
let velocities;
let densities;

const vertexShaderSource = `
    attribute vec2 a_position;
    attribute float a_density;
    varying float v_density;
    uniform vec2 u_resolution;
    void main() {
        vec2 zeroToOne = a_position / u_resolution;
        vec2 zeroToTwo = zeroToOne * 2.0;
        vec2 clipSpace = zeroToTwo - 1.0;
        gl_Position = vec4(clipSpace * vec2(1, -1), 0, 1);
        gl_PointSize = 2.0;
        v_density = a_density;
    }
`;

const fragmentShaderSource = `
    precision mediump float;
    varying float v_density;
    void main() {
        float intensity = min(1.0, v_density * 0.2);
        gl_FragColor = vec4(intensity, intensity * 0.7, 0.0, 1.0);
    }
`;

function initializeSimulation() {
    const canvas = document.getElementById('simulationCanvas');
    gl = canvas.getContext('webgl');
    if (!gl) {
        alert('WebGL not supported');
        return;
    }

    canvas.width = 200;
    canvas.height = 200;
    gl.viewport(0, 0, canvas.width, canvas.height);

    program = createProgram(gl, vertexShaderSource, fragmentShaderSource);
    gl.useProgram(program);

    initBuffers();

    document.getElementById('startPauseBtn').addEventListener('click', toggleSimulation);
    document.getElementById('resetBtn').addEventListener('click', resetSimulation);

    resetSimulation();
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

function createProgram(gl, vertexSource, fragmentSource) {
    const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexSource);
    const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentSource);
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

function initBuffers() {
    positionBuffer = gl.createBuffer();
    densityBuffer = gl.createBuffer();
}

function resetSimulation() {
    positions = new Float32Array(particleCount * 2);
    velocities = new Float32Array(particleCount * 2);
    densities = new Float32Array(particleCount);

    // Create a uniform distribution across the square
    for (let i = 0; i < particleCount; i++) {
        positions[i * 2] = Math.random() * gl.canvas.width;
        positions[i * 2 + 1] = Math.random() * gl.canvas.height;
        velocities[i * 2] = (Math.random() - 0.5) * 0.1;
        velocities[i * 2 + 1] = (Math.random() - 0.5) * 0.1;
        densities[i] = 0;
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.DYNAMIC_DRAW);

    gl.bindBuffer(gl.ARRAY_BUFFER, densityBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, densities, gl.DYNAMIC_DRAW);

    time = 0;
    isRunning = false;
    document.getElementById('startPauseBtn').textContent = 'Start';
    updateTimeDisplay();
}

function toggleSimulation() {
    isRunning = !isRunning;
    document.getElementById('startPauseBtn').textContent = isRunning ? 'Pause' : 'Start';
    if (isRunning) {
        requestAnimationFrame(updateSimulation);
    }
}

function updateSimulation() {
    if (!isRunning) return;

    updateParticles();
    drawParticles();
    time++;
    updateTimeDisplay();
    requestAnimationFrame(updateSimulation);
}

function updateParticles() {
    const width = gl.canvas.width;
    const height = gl.canvas.height;
    const halfWidth = width / 2;
    const halfHeight = height / 2;

    for (let i = 0; i < particleCount * 2; i += 2) {
        let fx = 0, fy = 0;

        for (let j = 0; j < particleCount * 2; j += 2) {
            if (i !== j) {
                let dx = positions[j] - positions[i];
                let dy = positions[j + 1] - positions[i + 1];

                // Apply periodic boundary conditions
                if (dx > halfWidth) dx -= width;
                else if (dx < -halfWidth) dx += width;
                if (dy > halfHeight) dy -= height;
                else if (dy < -halfHeight) dy += height;

                const distSq = dx * dx + dy * dy;
                if (distSq > 0 && distSq < 10000) {  // Limit interaction range
                    const force = gravityStrength / distSq;
                    fx += force * dx;
                    fy += force * dy;
                }
            }
        }

        // Apply Hubble-like expansion
        const dx = positions[i] - halfWidth;
        const dy = positions[i + 1] - halfHeight;
        fx += hubbleConstant * dx;
        fy += hubbleConstant * dy;

        // Add small random motion
        fx += (Math.random() - 0.5) * 0.001;
        fy += (Math.random() - 0.5) * 0.001;

        velocities[i] += fx;
        velocities[i + 1] += fy;

        // Apply some damping to prevent excessive speeds
        velocities[i] *= 0.99;
        velocities[i + 1] *= 0.99;

        positions[i] += velocities[i];
        positions[i + 1] += velocities[i + 1];

        // Apply periodic boundary conditions
        positions[i] = (positions[i] + width) % width;
        positions[i + 1] = (positions[i + 1] + height) % height;

        densities[i / 2] = 0;
    }

    // Calculate densities
    for (let i = 0; i < particleCount * 2; i += 2) {
        for (let j = 0; j < particleCount * 2; j += 2) {
            let dx = positions[j] - positions[i];
            let dy = positions[j + 1] - positions[i + 1];

            // Apply periodic boundary conditions for density calculation
            if (dx > halfWidth) dx -= width;
            else if (dx < -halfWidth) dx += width;
            if (dy > halfHeight) dy -= height;
            else if (dy < -halfHeight) dy += height;

            if (dx * dx + dy * dy < 25) {
                densities[i / 2]++;
            }
        }
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.DYNAMIC_DRAW);

    gl.bindBuffer(gl.ARRAY_BUFFER, densityBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, densities, gl.DYNAMIC_DRAW);
}

function drawParticles() {
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);

    const positionAttributeLocation = gl.getAttribLocation(program, 'a_position');
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.enableVertexAttribArray(positionAttributeLocation);
    gl.vertexAttribPointer(positionAttributeLocation, 2, gl.FLOAT, false, 0, 0);

    const densityAttributeLocation = gl.getAttribLocation(program, 'a_density');
    gl.bindBuffer(gl.ARRAY_BUFFER, densityBuffer);
    gl.enableVertexAttribArray(densityAttributeLocation);
    gl.vertexAttribPointer(densityAttributeLocation, 1, gl.FLOAT, false, 0, 0);

    const resolutionUniformLocation = gl.getUniformLocation(program, 'u_resolution');
    gl.uniform2f(resolutionUniformLocation, gl.canvas.width, gl.canvas.height);

    gl.drawArrays(gl.POINTS, 0, particleCount);
}

function updateTimeDisplay() {
    document.getElementById('timeValue').textContent = (10 - time / 1000).toFixed(2);
}

window.addEventListener('load', initializeSimulation);