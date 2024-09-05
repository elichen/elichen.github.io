const canvas = document.getElementById('glCanvas');
const gl = canvas.getContext('webgl');

const shaders = [
    {
        name: 'Ripple',
        vertex: `
            attribute vec4 a_position;
            void main() {
                gl_Position = a_position;
            }
        `,
        fragment: `
            precision mediump float;
            uniform vec2 u_resolution;
            uniform float u_time;

            void main() {
                vec2 st = gl_FragCoord.xy / u_resolution;
                float d = distance(st, vec2(0.5));
                float r = sin(d * 50.0 - u_time * 2.0) * 0.5 + 0.5;
                gl_FragColor = vec4(r, st.x, st.y, 1.0);
            }
        `,
        explanation: 'This shader creates a ripple effect by calculating the distance from each pixel to the center and using sine waves to create oscillating colors.',
        sourceLink: 'shaders.js#L5-L21'
    },
    {
        name: 'Plasma',
        vertex: `
            attribute vec4 a_position;
            void main() {
                gl_Position = a_position;
            }
        `,
        fragment: `
            precision mediump float;
            uniform vec2 u_resolution;
            uniform float u_time;

            void main() {
                vec2 st = gl_FragCoord.xy / u_resolution;
                float r = sin(st.x * 10.0 + u_time) * 0.5 + 0.5;
                float g = sin(st.y * 10.0 + u_time * 0.5) * 0.5 + 0.5;
                float b = sin((st.x + st.y) * 10.0 + u_time * 1.5) * 0.5 + 0.5;
                gl_FragColor = vec4(r, g, b, 1.0);
            }
        `,
        explanation: 'The plasma shader uses sine waves on different color channels to create a psychedelic, flowing effect that changes over time.',
        sourceLink: 'shaders.js#L22-L38'
    },
    {
        name: 'Fractal',
        vertex: `
            attribute vec4 a_position;
            void main() {
                gl_Position = a_position;
            }
        `,
        fragment: `
            precision mediump float;
            uniform vec2 u_resolution;
            uniform float u_time;

            void main() {
                vec2 st = (gl_FragCoord.xy * 2.0 - u_resolution) / min(u_resolution.x, u_resolution.y);
                vec2 z = st;
                vec2 c = vec2(sin(u_time * 0.3) * 0.4, cos(u_time * 0.2) * 0.4);
                float i = 0.0;
                for (int j = 0; j < 100; j++) {
                    if (length(z) > 2.0) break;
                    z = vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
                    i += 1.0;
                }
                float h = i / 100.0;
                vec3 color = 0.5 + 0.5 * cos(3.0 + h * 6.28 + vec3(0.0, 0.6, 1.0));
                gl_FragColor = vec4(color, 1.0);
            }
        `,
        explanation: 'This shader generates a fractal pattern using the Julia set. It iterates through a mathematical formula to create complex, self-similar shapes that evolve over time.',
        sourceLink: 'shaders.js#L39-L62'
    }
];

let currentShaderIndex = 0;

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

function setupShader(gl, shaderInfo) {
    const vertexShader = createShader(gl, gl.VERTEX_SHADER, shaderInfo.vertex);
    const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, shaderInfo.fragment);
    const program = createProgram(gl, vertexShader, fragmentShader);

    const positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);

    const positionAttributeLocation = gl.getAttribLocation(program, 'a_position');
    gl.enableVertexAttribArray(positionAttributeLocation);
    gl.vertexAttribPointer(positionAttributeLocation, 2, gl.FLOAT, false, 0, 0);

    return program;
}

function render(gl, program, time) {
    gl.useProgram(program);

    const resolutionUniformLocation = gl.getUniformLocation(program, 'u_resolution');
    gl.uniform2f(resolutionUniformLocation, gl.canvas.width, gl.canvas.height);

    const timeUniformLocation = gl.getUniformLocation(program, 'u_time');
    gl.uniform1f(timeUniformLocation, time * 0.001);

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
}

function resizeCanvas() {
    const containerWidth = canvas.parentElement.clientWidth;
    const size = Math.min(containerWidth, 800); // Increased max size to 800px
    canvas.width = size;
    canvas.height = size; // Make it square
    gl.viewport(0, 0, canvas.width, canvas.height);
}

function updateShaderName() {
    const currentShader = shaders[currentShaderIndex];
    document.getElementById('shaderName').textContent = currentShader.name;
    document.getElementById('shaderExplanation').innerHTML = `
        <p>${currentShader.explanation}</p>
        <h4>Vertex Shader:</h4>
        <pre><code>${currentShader.vertex}</code></pre>
        <h4>Fragment Shader:</h4>
        <pre><code>${currentShader.fragment}</code></pre>
    `;
}

let currentProgram = setupShader(gl, shaders[currentShaderIndex]);
updateShaderName();

function animate(time) {
    resizeCanvas();
    render(gl, currentProgram, time);
    requestAnimationFrame(animate);
}

requestAnimationFrame(animate);

document.getElementById('prevShader').addEventListener('click', () => {
    currentShaderIndex = (currentShaderIndex - 1 + shaders.length) % shaders.length;
    currentProgram = setupShader(gl, shaders[currentShaderIndex]);
    updateShaderName();
});

document.getElementById('nextShader').addEventListener('click', () => {
    currentShaderIndex = (currentShaderIndex + 1) % shaders.length;
    currentProgram = setupShader(gl, shaders[currentShaderIndex]);
    updateShaderName();
});

window.addEventListener('resize', resizeCanvas);