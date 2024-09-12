// WebGL Ray Tracer for 3D Donut

// Vertex shader source code
const vsSource = `
    attribute vec4 aVertexPosition;
    void main() {
        gl_Position = aVertexPosition;
    }
`;

// Fragment shader source code
const fsSource = `
    precision highp float;

    uniform vec2 uResolution;
    uniform float uTime;
    uniform float uZoom;
    uniform mat4 uRotation;

    #define NUM_LIGHTS 3
    uniform vec3 uLightPositions[NUM_LIGHTS];
    uniform vec3 uLightColors[NUM_LIGHTS];

    const int MAX_STEPS = 100;
    const float MAX_DIST = 100.0;
    const float EPSILON = 0.001;

    // Torus (donut) parameters
    const float torusRadius = 1.2;
    const float tubeRadius = 0.4;

    // Signed distance function for a torus (donut)
    float sdTorus(vec3 p, float R, float r) {
        vec2 q = vec2(length(p.xz) - R, p.y);
        return length(q) - r;
    }

    // Scene SDF
    float sceneSDF(vec3 p) {
        p = (uRotation * vec4(p, 1.0)).xyz; // Apply rotation matrix
        return sdTorus(p, torusRadius, tubeRadius);
    }

    // Calculate normal
    vec3 calcNormal(vec3 p) {
        const vec2 e = vec2(EPSILON, 0);
        return normalize(vec3(
            sceneSDF(p + e.xyy) - sceneSDF(p - e.xyy),
            sceneSDF(p + e.yxy) - sceneSDF(p - e.yxy),
            sceneSDF(p + e.yyx) - sceneSDF(p - e.yyx)
        ));
    }

    // Ray marching
    float rayMarch(vec3 ro, vec3 rd) {
        float depth = 0.0;
        for (int i = 0; i < MAX_STEPS; i++) {
            vec3 p = ro + depth * rd;
            float dist = sceneSDF(p);
            if (dist < EPSILON) return depth;
            depth += dist;
            if (depth >= MAX_DIST) return MAX_DIST;
        }
        return MAX_DIST;
    }

    // Soft shadow
    float softShadow(vec3 ro, vec3 rd, float mint, float maxt, float k) {
        float res = 1.0;
        float t = mint;
        for(int i = 0; i < 16; i++) {
            if(t < maxt) {
                float h = sceneSDF(ro + rd * t);
                if(h < 0.001) return 0.0;
                res = min(res, k * h / t);
                t += h;
            }
        }
        return res;
    }

    void main() {
        vec2 uv = (gl_FragCoord.xy - 0.5 * uResolution.xy) / uResolution.y;
        
        // Camera setup
        vec3 ro = vec3(0.0, 0.0, uZoom);
        vec3 rd = normalize(vec3(uv, -1.0));

        float d = rayMarch(ro, rd);
        
        if (d < MAX_DIST) {
            vec3 p = ro + rd * d;
            vec3 normal = calcNormal(p);
            
            vec3 color = vec3(0.0);
            for(int i = 0; i < NUM_LIGHTS; i++) {
                vec3 lightDir = normalize(uLightPositions[i] - p);
                float diff = max(dot(normal, lightDir), 0.0);
                float shadow = softShadow(p, lightDir, 0.01, 10.0, 32.0);
                color += uLightColors[i] * diff * shadow;
            }
            
            // Add ambient lighting
            color += vec3(0.1);
            
            gl_FragColor = vec4(color, 1.0);
        } else {
            gl_FragColor = vec4(0.1, 0.1, 0.1, 1.0);
        }
    }
`;

let gl;
let program;
let positionBuffer;
let positionAttributeLocation;
let resolutionUniformLocation;
let timeUniformLocation;
let zoomUniformLocation;
let rotationUniformLocation;
let lightPositionsUniformLocation;
let lightColorsUniformLocation;
let startTime;

let isDragging = false;
let previousMousePosition = { x: 0, y: 0 };
let rotation = { x: 0, y: 0 };
let zoom = 3.0;
let autoRotation = { x: 0, y: 0 };

function initWebGL() {
    const canvas = document.getElementById('glCanvas');
    gl = canvas.getContext('webgl');

    if (!gl) {
        alert('Unable to initialize WebGL. Your browser may not support it.');
        return;
    }

    // Create shader program
    const vertexShader = createShader(gl, gl.VERTEX_SHADER, vsSource);
    const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fsSource);
    program = createProgram(gl, vertexShader, fragmentShader);

    // Look up attribute and uniform locations
    positionAttributeLocation = gl.getAttribLocation(program, 'aVertexPosition');
    resolutionUniformLocation = gl.getUniformLocation(program, 'uResolution');
    timeUniformLocation = gl.getUniformLocation(program, 'uTime');
    zoomUniformLocation = gl.getUniformLocation(program, 'uZoom');
    rotationUniformLocation = gl.getUniformLocation(program, 'uRotation');
    lightPositionsUniformLocation = gl.getUniformLocation(program, 'uLightPositions');
    lightColorsUniformLocation = gl.getUniformLocation(program, 'uLightColors');

    // Create position buffer
    positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    const positions = [
        -1, -1,
         1, -1,
        -1,  1,
        -1,  1,
         1, -1,
         1,  1,
    ];
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);

    startTime = Date.now();

    // Set up mouse event listeners
    canvas.addEventListener('mousedown', onMouseDown);
    canvas.addEventListener('mousemove', onMouseMove);
    canvas.addEventListener('mouseup', onMouseUp);
    canvas.addEventListener('wheel', onMouseWheel);

    render();
}

function createShader(gl, type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        console.error('An error occurred compiling the shaders: ' + gl.getShaderInfoLog(shader));
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
        console.error('Unable to initialize the shader program: ' + gl.getProgramInfoLog(program));
        return null;
    }
    return program;
}

function onMouseDown(e) {
    isDragging = true;
    previousMousePosition = { x: e.clientX, y: e.clientY };
    autoRotation = { x: 0, y: 0 };
}

function onMouseMove(e) {
    if (!isDragging) return;
    const deltaMove = {
        x: e.clientX - previousMousePosition.x,
        y: e.clientY - previousMousePosition.y
    };
    rotation.x += deltaMove.y * 0.005;
    rotation.y += deltaMove.x * 0.005;
    previousMousePosition = { x: e.clientX, y: e.clientY };
}

function onMouseUp() {
    isDragging = false;
}

function onMouseWheel(e) {
    zoom += e.deltaY * -0.001;
    zoom = Math.min(Math.max(1, zoom), 5);
    e.preventDefault();
}

function getRotationMatrix(rotationX, rotationY) {
    const rotationMatrix = mat4.create();
    mat4.rotateX(rotationMatrix, rotationMatrix, rotationX);
    mat4.rotateY(rotationMatrix, rotationMatrix, rotationY);
    return rotationMatrix;
}

function render() {
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
    gl.clearColor(0.0, 0.0, 0.0, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(program);

    gl.enableVertexAttribArray(positionAttributeLocation);
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.vertexAttribPointer(positionAttributeLocation, 2, gl.FLOAT, false, 0, 0);

    gl.uniform2f(resolutionUniformLocation, gl.canvas.width, gl.canvas.height);
    gl.uniform1f(timeUniformLocation, (Date.now() - startTime) * 0.001);
    gl.uniform1f(zoomUniformLocation, zoom);

    if (!isDragging) {
        autoRotation.x += 0.005;
        autoRotation.y += 0.01;
    }

    gl.uniformMatrix4fv(rotationUniformLocation, false, getRotationMatrix(
        rotation.x + autoRotation.x,
        rotation.y + autoRotation.y
    ));

    // Set light positions and colors
    const lightPositions = [
        2.0, 2.0, 2.0,
        -2.0, 2.0, -2.0,
        0.0, -2.0, 0.0
    ];
    const lightColors = [
        1.0, 0.8, 0.8,
        0.8, 1.0, 0.8,
        0.8, 0.8, 1.0
    ];
    gl.uniform3fv(lightPositionsUniformLocation, lightPositions);
    gl.uniform3fv(lightColorsUniformLocation, lightColors);

    gl.drawArrays(gl.TRIANGLES, 0, 6);

    requestAnimationFrame(render);
}

window.onload = initWebGL;
