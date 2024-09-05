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
    },
    {
        name: 'Voronoi',
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

            vec2 random2(vec2 p) {
                return fract(sin(vec2(dot(p,vec2(127.1,311.7)),dot(p,vec2(269.5,183.3))))*43758.5453);
            }

            void main() {
                vec2 st = gl_FragCoord.xy/u_resolution.xy;
                st.x *= u_resolution.x/u_resolution.y;
                vec3 color = vec3(.0);

                // Scale
                st *= 5.;

                // Tile the space
                vec2 i_st = floor(st);
                vec2 f_st = fract(st);

                float m_dist = 1.;  // minimum distance

                for (int y= -1; y <= 1; y++) {
                    for (int x= -1; x <= 1; x++) {
                        // Neighbor place in the grid
                        vec2 neighbor = vec2(float(x),float(y));

                        // Random position from current + neighbor place in the grid
                        vec2 point = random2(i_st + neighbor);

                        // Animate the point
                        point = 0.5 + 0.5*sin(u_time + 6.2831*point);

                        // Vector between the pixel and the point
                        vec2 diff = neighbor + point - f_st;

                        // Distance to the point
                        float dist = length(diff);

                        // Keep the closer distance
                        m_dist = min(m_dist, dist);
                    }
                }

                // Draw the min distance (distance field)
                color += m_dist;

                // Draw cell center
                color += 1.-step(.02, m_dist);

                // Draw grid
                color.r += step(.98, f_st.x) + step(.98, f_st.y);

                gl_FragColor = vec4(color,1.0);
            }
        `,
        explanation: 'This shader creates a Voronoi diagram, which divides the space into cells based on the distance to a set of points. The points move over time, creating an animated cellular pattern.'
    },
    {
        name: 'Noise',
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

            float random (in vec2 st) {
                return fract(sin(dot(st.xy,
                                     vec2(12.9898,78.233)))
                             * 43758.5453123);
            }

            float noise (in vec2 st) {
                vec2 i = floor(st);
                vec2 f = fract(st);

                float a = random(i);
                float b = random(i + vec2(1.0, 0.0));
                float c = random(i + vec2(0.0, 1.0));
                float d = random(i + vec2(1.0, 1.0));

                vec2 u = f * f * (3.0 - 2.0 * f);

                return mix(a, b, u.x) +
                        (c - a)* u.y * (1.0 - u.x) +
                        (d - b) * u.x * u.y;
            }

            void main() {
                vec2 st = gl_FragCoord.xy/u_resolution.xy;
                st.x *= u_resolution.x/u_resolution.y;

                vec3 color = vec3(0.0);

                vec2 pos = vec2(st*10.0);

                color = vec3(noise(pos + u_time));

                gl_FragColor = vec4(color,1.0);
            }
        `,
        explanation: 'This shader demonstrates Perlin noise, a type of gradient noise used to create natural-looking textures and animations. The noise pattern evolves over time.'
    },
    {
        name: 'Wave',
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
                vec2 st = gl_FragCoord.xy/u_resolution.xy;
                st.x *= u_resolution.x/u_resolution.y;
                
                vec3 color = vec3(0.0);
                float d = 0.0;

                // Generate multiple waves
                for(float i = 1.0; i < 6.0; i++){
                    d += sin(st.x*10.0*i + u_time + i*1.5) * 0.1 / i;
                }
                
                // Create wave effect
                float wave = smoothstep(0.5 + d, 0.5 + d + 0.01, st.y);

                // Color the wave
                color = mix(
                    vec3(0.1, 0.3, 0.5),  // Deep water color
                    vec3(0.2, 0.7, 0.9),  // Shallow water color
                    wave
                );

                // Add some highlights
                color += vec3(1.0, 1.0, 0.8) * smoothstep(0.49, 0.5, st.y + d);

                gl_FragColor = vec4(color, 1.0);
            }
        `,
        explanation: 'This shader creates an animated wave pattern. It uses multiple sine waves to generate a complex wave shape, and then applies color gradients to create a water-like effect with highlights.'
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
    const size = Math.min(containerWidth, 600); // Increased max size to 600px
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