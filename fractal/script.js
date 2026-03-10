let canvas;
let gl;
let quadBuffer;
let fractalType = 'mandelbrot';
let mousePosition = { x: 0, y: 0 };
let deepPrecisionWarningShown = false;

const ZOOM_SPEED = 0.994;
const MIN_DECIMAL_DIGITS = 80;
const EXTRA_DECIMAL_DIGITS = 28;
const MAX_GPU_ITERATIONS = 1536;
const ORBIT_RECENTER_PIXELS = 16;
const MIN_GPU_SCALE = 1e-38;

let decimalDigits = MIN_DECIMAL_DIGITS;
let decimalScale = 10n ** BigInt(decimalDigits);

const vertexShaderSource = `#version 300 es
in vec2 a_position;

void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

const simpleFragmentShaderSource = `#version 300 es
precision highp float;

uniform vec2 u_resolution;
uniform vec2 u_center;
uniform float u_pixelScale;
uniform int u_fractalType;
uniform int u_maxIterations;

out vec4 outColor;

vec3 getPalette(float t) {
    vec3 c1 = vec3(0.08, 0.10, 0.30);
    vec3 c2 = vec3(0.02, 0.48, 0.65);
    vec3 c3 = vec3(0.95, 0.92, 0.24);
    vec3 c4 = vec3(0.98, 0.45, 0.05);
    vec3 c5 = vec3(0.72, 0.06, 0.42);

    float s = mod(t * 5.0, 5.0);
    if (s < 1.0) return mix(c1, c2, s);
    if (s < 2.0) return mix(c2, c3, s - 1.0);
    if (s < 3.0) return mix(c3, c4, s - 2.0);
    if (s < 4.0) return mix(c4, c5, s - 3.0);
    return mix(c5, c1, s - 4.0);
}

vec2 complexMul(vec2 a, vec2 b) {
    return vec2(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x
    );
}

float mandelbrot(vec2 c) {
    vec2 z = vec2(0.0);
    for (int i = 0; i < ${MAX_GPU_ITERATIONS}; ++i) {
        if (i >= u_maxIterations) break;
        z = complexMul(z, z) + c;
        float mag2 = dot(z, z);
        if (mag2 > 4.0) {
            float smoothValue = float(i + 1) - log2(log2(max(length(z), 1.0001)));
            return smoothValue / float(u_maxIterations);
        }
    }
    return 0.0;
}

float julia(vec2 z) {
    vec2 c = vec2(-0.8, 0.156);
    for (int i = 0; i < ${MAX_GPU_ITERATIONS}; ++i) {
        if (i >= u_maxIterations) break;
        z = complexMul(z, z) + c;
        float mag2 = dot(z, z);
        if (mag2 > 4.0) {
            float smoothValue = float(i + 1) - log2(log2(max(length(z), 1.0001)));
            return smoothValue / float(u_maxIterations);
        }
    }
    return 0.0;
}

float sierpinski(vec2 p) {
    p = (p + vec2(1.0, 0.7)) / 2.2;
    for (int i = 0; i < 64; ++i) {
        if (p.x < 0.0 || p.y < 0.0 || p.x + p.y > 1.0) {
            return float(i) / 64.0;
        }
        p *= 2.0;
        if (p.x + p.y > 1.0) {
            p = vec2(1.0 - p.y, 1.0 - p.x);
        }
        p = fract(p);
    }
    return 1.0;
}

void main() {
    vec2 plane = vec2(
        (gl_FragCoord.x - 0.5 * u_resolution.x) * u_pixelScale + u_center.x,
        (gl_FragCoord.y - 0.5 * u_resolution.y) * u_pixelScale + u_center.y
    );

    float value = 0.0;
    if (u_fractalType == 0) {
        value = mandelbrot(plane);
    } else if (u_fractalType == 1) {
        value = julia(plane);
    } else {
        value = sierpinski(plane);
    }

    vec3 color = value == 0.0 ? vec3(0.0) : getPalette(value);
    outColor = vec4(color, 1.0);
}
`;

const deepFragmentShaderSource = `#version 300 es
precision highp float;

uniform vec2 u_resolution;
uniform float u_pixelScale;
uniform vec2 u_referenceDelta;
uniform sampler2D u_referenceOrbit;
uniform int u_maxIterations;

out vec4 outColor;

vec3 getPalette(float t) {
    vec3 c1 = vec3(0.08, 0.10, 0.30);
    vec3 c2 = vec3(0.02, 0.48, 0.65);
    vec3 c3 = vec3(0.95, 0.92, 0.24);
    vec3 c4 = vec3(0.98, 0.45, 0.05);
    vec3 c5 = vec3(0.72, 0.06, 0.42);

    float s = mod(t * 5.0, 5.0);
    if (s < 1.0) return mix(c1, c2, s);
    if (s < 2.0) return mix(c2, c3, s - 1.0);
    if (s < 3.0) return mix(c3, c4, s - 2.0);
    if (s < 4.0) return mix(c4, c5, s - 3.0);
    return mix(c5, c1, s - 4.0);
}

vec2 complexMul(vec2 a, vec2 b) {
    return vec2(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x
    );
}

vec2 getOrbit(int index) {
    return texelFetch(u_referenceOrbit, ivec2(index, 0), 0).xy;
}

float mandelbrotPerturbation(vec2 deltaC) {
    vec2 delta = vec2(0.0);
    for (int i = 0; i < ${MAX_GPU_ITERATIONS}; ++i) {
        if (i >= u_maxIterations) break;

        vec2 reference = getOrbit(i);
        vec2 nextReference = getOrbit(i + 1);
        vec2 nextDelta = 2.0 * complexMul(reference, delta) + complexMul(delta, delta) + deltaC;
        vec2 z = nextReference + nextDelta;
        float mag2 = dot(z, z);
        if (mag2 > 4.0) {
            float smoothValue = float(i + 1) - log2(log2(max(length(z), 1.0001)));
            return smoothValue / float(u_maxIterations);
        }
        delta = nextDelta;
    }
    return 0.0;
}

void main() {
    vec2 pixelOffset = vec2(
        gl_FragCoord.x - 0.5 * u_resolution.x,
        gl_FragCoord.y - 0.5 * u_resolution.y
    );
    vec2 deltaC = u_referenceDelta + pixelOffset * u_pixelScale;
    float value = mandelbrotPerturbation(deltaC);
    vec3 color = value == 0.0 ? vec3(0.0) : getPalette(value);
    outColor = vec4(color, 1.0);
}
`;

let simpleProgramInfo;
let deepProgramInfo;
let orbitTexture;

let simpleCamera = createSimpleCamera(-0.745, 0.1, 1.6);
let mandelbrotCamera = null;
let mandelbrotReference = createEmptyReference();

function createSimpleCamera(centerX, centerY, viewWidth) {
    return {
        centerX,
        centerY,
        viewWidth,
        pixelScale: 0,
        maxIterations: 220,
    };
}

function createEmptyReference() {
    return {
        centerX: null,
        centerY: null,
        orbitData: null,
        orbitLength: 0,
        maxIterations: 0,
        useSimpleFallback: false,
    };
}

function decimalFromString(value) {
    const input = value.trim().toLowerCase();
    if (!input) {
        return 0n;
    }

    let sign = 1n;
    let mantissa = input;
    if (mantissa.startsWith('-')) {
        sign = -1n;
        mantissa = mantissa.slice(1);
    } else if (mantissa.startsWith('+')) {
        mantissa = mantissa.slice(1);
    }

    let exponent = 0;
    const exponentIndex = mantissa.indexOf('e');
    if (exponentIndex !== -1) {
        exponent = Number(mantissa.slice(exponentIndex + 1));
        mantissa = mantissa.slice(0, exponentIndex);
    }

    let whole = mantissa;
    let fraction = '';
    const dotIndex = mantissa.indexOf('.');
    if (dotIndex !== -1) {
        whole = mantissa.slice(0, dotIndex);
        fraction = mantissa.slice(dotIndex + 1);
    }

    const digits = `${whole}${fraction}`.replace(/^0+(?=\d)/, '');
    if (!digits || /^0+$/.test(digits)) {
        return 0n;
    }

    const scaleOffset = decimalDigits - fraction.length + exponent;
    let raw = BigInt(digits);
    if (scaleOffset >= 0) {
        raw *= 10n ** BigInt(scaleOffset);
    } else {
        raw /= 10n ** BigInt(-scaleOffset);
    }
    return raw * sign;
}

function decimalFromNumber(value) {
    if (!Number.isFinite(value) || value === 0) {
        return 0n;
    }
    return decimalFromString(value.toExponential(20));
}

function decimalToNumber(value) {
    if (value === 0n) {
        return 0;
    }

    const sign = value < 0n ? '-' : '';
    const digits = (value < 0n ? -value : value).toString();
    const exponent = digits.length - decimalDigits - 1;
    const head = digits[0];
    const tail = digits.slice(1, 18);
    const mantissa = tail ? `${head}.${tail}` : head;
    return Number(`${sign}${mantissa}e${exponent}`);
}

function addDecimal(a, b) {
    return a + b;
}

function subDecimal(a, b) {
    return a - b;
}

function absDecimal(value) {
    return value < 0n ? -value : value;
}

function mulDecimal(a, b) {
    return (a * b) / decimalScale;
}

function mulDecimalInt(value, multiplier) {
    return value * BigInt(multiplier);
}

function ensureDecimalDigits(requiredDigits) {
    if (requiredDigits <= decimalDigits) {
        return;
    }

    const increase = requiredDigits - decimalDigits;
    const factor = 10n ** BigInt(increase);

    if (mandelbrotCamera) {
        mandelbrotCamera.centerX *= factor;
        mandelbrotCamera.centerY *= factor;
        mandelbrotCamera.pixelScale *= factor;
    }

    if (mandelbrotReference.centerX !== null) {
        mandelbrotReference.centerX *= factor;
        mandelbrotReference.centerY *= factor;
    }

    decimalDigits = requiredDigits;
    decimalScale *= factor;
}

function requiredDecimalDigits(pixelScaleApprox) {
    if (!Number.isFinite(pixelScaleApprox) || pixelScaleApprox <= 0) {
        return decimalDigits;
    }
    return Math.max(
        MIN_DECIMAL_DIGITS,
        Math.ceil(-Math.log10(pixelScaleApprox)) + EXTRA_DECIMAL_DIGITS
    );
}

function computeIterationBudget(pixelScale) {
    const depth = Math.max(0, -Math.log10(Math.max(pixelScale, Number.MIN_VALUE)));
    return Math.min(
        MAX_GPU_ITERATIONS,
        Math.max(220, Math.floor(170 + depth * 34))
    );
}

function computeSimpleIterationBudget(pixelScale) {
    const depth = Math.max(0, -Math.log10(Math.max(pixelScale, Number.MIN_VALUE)));
    return Math.min(640, Math.max(140, Math.floor(120 + depth * 16)));
}

function updateSimpleCameraScale(camera) {
    if (!gl) {
        return;
    }
    const minDimension = Math.max(1, Math.min(gl.canvas.width, gl.canvas.height));
    camera.pixelScale = camera.viewWidth / minDimension;
    camera.maxIterations = computeSimpleIterationBudget(camera.pixelScale);
}

function createMandelbrotCamera() {
    const minDimension = Math.max(1, Math.min(gl.canvas.width, gl.canvas.height));
    const viewWidth = 1.6;
    const pixelScaleApprox = viewWidth / minDimension;
    ensureDecimalDigits(requiredDecimalDigits(pixelScaleApprox));

    return {
        centerX: decimalFromString('-0.745'),
        centerY: decimalFromString('0.1'),
        pixelScale: decimalFromNumber(pixelScaleApprox),
        pixelScaleApprox,
        maxIterations: computeIterationBudget(pixelScaleApprox),
        viewWidth,
    };
}

function getCanvasPixelOffset(pointer = mousePosition) {
    return {
        x: pointer.x - (0.5 * gl.canvas.width),
        y: (0.5 * gl.canvas.height) - pointer.y,
    };
}

function mulDecimalNumber(value, factor) {
    if (factor === 0) {
        return 0n;
    }
    return mulDecimal(value, decimalFromNumber(factor));
}

function screenToPlaneSimple(camera, pointer = mousePosition) {
    const offset = getCanvasPixelOffset(pointer);
    return {
        x: camera.centerX + (offset.x * camera.pixelScale),
        y: camera.centerY + (offset.y * camera.pixelScale),
    };
}

function screenToPlaneDeep(camera, pointer = mousePosition) {
    const offset = getCanvasPixelOffset(pointer);
    return {
        x: addDecimal(camera.centerX, mulDecimalNumber(camera.pixelScale, offset.x)),
        y: addDecimal(camera.centerY, mulDecimalNumber(camera.pixelScale, offset.y)),
    };
}

function planeToScreenSimple(camera, plane) {
    return {
        x: ((plane.x - camera.centerX) / camera.pixelScale) + (0.5 * gl.canvas.width),
        y: (0.5 * gl.canvas.height) - ((plane.y - camera.centerY) / camera.pixelScale),
    };
}

function planeToScreenDeep(camera, plane) {
    return {
        x: (decimalToNumber(subDecimal(plane.x, camera.centerX)) / camera.pixelScaleApprox) + (0.5 * gl.canvas.width),
        y: (0.5 * gl.canvas.height) - (decimalToNumber(subDecimal(plane.y, camera.centerY)) / camera.pixelScaleApprox),
    };
}

function getActivePointerPlane(pointer = mousePosition) {
    if (fractalType === 'mandelbrot') {
        const plane = screenToPlaneDeep(mandelbrotCamera, pointer);
        return {
            x: decimalToNumber(plane.x),
            y: decimalToNumber(plane.y),
        };
    }

    return screenToPlaneSimple(simpleCamera, pointer);
}

function stepActiveZoomOnce() {
    if (fractalType === 'mandelbrot') {
        updateMandelbrotCamera();
        return;
    }

    updateSimpleCamera(simpleCamera);
}

function updateSimpleCamera(camera) {
    const anchor = screenToPlaneSimple(camera, mousePosition);
    const offset = getCanvasPixelOffset(mousePosition);
    const nextPixelScale = camera.pixelScale * ZOOM_SPEED;

    camera.centerX = anchor.x - (offset.x * nextPixelScale);
    camera.centerY = anchor.y - (offset.y * nextPixelScale);
    camera.pixelScale = nextPixelScale;
    camera.viewWidth = camera.pixelScale * Math.min(gl.canvas.width, gl.canvas.height);
    camera.maxIterations = computeSimpleIterationBudget(camera.pixelScale);
}

function updateMandelbrotCamera() {
    const nextPixelScaleApprox = mandelbrotCamera.pixelScaleApprox * ZOOM_SPEED;
    ensureDecimalDigits(requiredDecimalDigits(nextPixelScaleApprox));

    const anchor = screenToPlaneDeep(mandelbrotCamera, mousePosition);
    const offset = getCanvasPixelOffset(mousePosition);
    const nextPixelScale = mulDecimal(mandelbrotCamera.pixelScale, decimalFromString(String(ZOOM_SPEED)));

    mandelbrotCamera.centerX = subDecimal(anchor.x, mulDecimalNumber(nextPixelScale, offset.x));
    mandelbrotCamera.centerY = subDecimal(anchor.y, mulDecimalNumber(nextPixelScale, offset.y));
    mandelbrotCamera.pixelScale = nextPixelScale;
    mandelbrotCamera.pixelScaleApprox = nextPixelScaleApprox;
    mandelbrotCamera.viewWidth = nextPixelScaleApprox * Math.min(gl.canvas.width, gl.canvas.height);
    mandelbrotCamera.maxIterations = computeIterationBudget(nextPixelScaleApprox);
}

function createShader(shaderType, source) {
    const shader = gl.createShader(shaderType);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        const message = gl.getShaderInfoLog(shader);
        gl.deleteShader(shader);
        throw new Error(message || 'Shader compilation failed');
    }
    return shader;
}

function createProgram(vertexSource, fragmentSource) {
    const vertexShader = createShader(gl.VERTEX_SHADER, vertexSource);
    const fragmentShader = createShader(gl.FRAGMENT_SHADER, fragmentSource);

    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);

    gl.deleteShader(vertexShader);
    gl.deleteShader(fragmentShader);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        const message = gl.getProgramInfoLog(program);
        gl.deleteProgram(program);
        throw new Error(message || 'Program link failed');
    }
    return program;
}

function createProgramInfo(program, uniforms) {
    const info = {
        program,
        position: gl.getAttribLocation(program, 'a_position'),
        uniforms: {},
    };

    for (const uniform of uniforms) {
        info.uniforms[uniform] = gl.getUniformLocation(program, uniform);
    }

    return info;
}

function bindQuad(programInfo) {
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
    gl.enableVertexAttribArray(programInfo.position);
    gl.vertexAttribPointer(programInfo.position, 2, gl.FLOAT, false, 0, 0);
}

function initWebGL() {
    canvas = document.getElementById('fractalCanvas');
    gl = canvas.getContext('webgl2', { antialias: false, alpha: false });

    if (!gl) {
        alert('WebGL2 is required for deep Mandelbrot zoom.');
        return false;
    }

    quadBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
        -1, -1,
        1, -1,
        -1, 1,
        1, 1,
    ]), gl.STATIC_DRAW);

    simpleProgramInfo = createProgramInfo(
        createProgram(vertexShaderSource, simpleFragmentShaderSource),
        ['u_resolution', 'u_center', 'u_pixelScale', 'u_fractalType', 'u_maxIterations']
    );

    deepProgramInfo = createProgramInfo(
        createProgram(vertexShaderSource, deepFragmentShaderSource),
        ['u_resolution', 'u_pixelScale', 'u_referenceDelta', 'u_referenceOrbit', 'u_maxIterations']
    );

    orbitTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, orbitTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    gl.clearColor(0, 0, 0, 1);
    mandelbrotCamera = createMandelbrotCamera();
    resetSimpleCamera('mandelbrot');
    return true;
}

function uploadReferenceOrbit(orbitData, orbitLength) {
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, orbitTexture);
    gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.RGBA32F,
        orbitLength,
        1,
        0,
        gl.RGBA,
        gl.FLOAT,
        orbitData
    );
}

function computeReferenceOrbit(centerX, centerY, iterations) {
    const orbitData = new Float32Array((iterations + 1) * 4);
    let zr = 0n;
    let zi = 0n;

    orbitData[0] = 0;
    orbitData[1] = 0;

    for (let i = 0; i < iterations; i += 1) {
        const zr2 = mulDecimal(zr, zr);
        const zi2 = mulDecimal(zi, zi);
        const zrzi = mulDecimal(zr, zi);

        const nextZr = addDecimal(subDecimal(zr2, zi2), centerX);
        const nextZi = addDecimal(mulDecimalInt(zrzi, 2), centerY);

        zr = nextZr;
        zi = nextZi;

        const magnitude = addDecimal(mulDecimal(zr, zr), mulDecimal(zi, zi));
        if (magnitude > mulDecimalInt(decimalScale, 4)) {
            return {
                escapedEarly: true,
                orbitData,
                orbitLength: i + 2,
            };
        }

        const baseIndex = (i + 1) * 4;
        orbitData[baseIndex] = decimalToNumber(zr);
        orbitData[baseIndex + 1] = decimalToNumber(zi);
    }

    return {
        escapedEarly: false,
        orbitData,
        orbitLength: iterations + 1,
    };
}

function shouldRecomputeReference() {
    if (mandelbrotReference.centerX === null) {
        return true;
    }

    if (mandelbrotReference.maxIterations < mandelbrotCamera.maxIterations) {
        return true;
    }

    const threshold = mulDecimalInt(mandelbrotCamera.pixelScale, ORBIT_RECENTER_PIXELS);
    const deltaX = absDecimal(subDecimal(mandelbrotCamera.centerX, mandelbrotReference.centerX));
    const deltaY = absDecimal(subDecimal(mandelbrotCamera.centerY, mandelbrotReference.centerY));
    return deltaX > threshold || deltaY > threshold;
}

function syncMandelbrotReference() {
    if (!shouldRecomputeReference()) {
        return;
    }

    const referenceOrbit = computeReferenceOrbit(
        mandelbrotCamera.centerX,
        mandelbrotCamera.centerY,
        mandelbrotCamera.maxIterations
    );

    mandelbrotReference.centerX = mandelbrotCamera.centerX;
    mandelbrotReference.centerY = mandelbrotCamera.centerY;
    mandelbrotReference.orbitData = referenceOrbit.orbitData;
    mandelbrotReference.orbitLength = referenceOrbit.orbitLength;
    mandelbrotReference.maxIterations = mandelbrotCamera.maxIterations;
    mandelbrotReference.useSimpleFallback = referenceOrbit.escapedEarly;

    if (!referenceOrbit.escapedEarly) {
        uploadReferenceOrbit(referenceOrbit.orbitData, referenceOrbit.orbitLength);
    }
}

function drawSimpleFractal(activeType, camera) {
    const fractalIndex = ['mandelbrot', 'julia', 'sierpinski'].indexOf(activeType);

    gl.useProgram(simpleProgramInfo.program);
    bindQuad(simpleProgramInfo);
    gl.uniform2f(simpleProgramInfo.uniforms.u_resolution, gl.canvas.width, gl.canvas.height);
    gl.uniform2f(simpleProgramInfo.uniforms.u_center, camera.centerX, camera.centerY);
    gl.uniform1f(simpleProgramInfo.uniforms.u_pixelScale, camera.pixelScale);
    gl.uniform1i(simpleProgramInfo.uniforms.u_fractalType, fractalIndex);
    gl.uniform1i(simpleProgramInfo.uniforms.u_maxIterations, camera.maxIterations);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
}

function drawDeepMandelbrot() {
    syncMandelbrotReference();

    if (mandelbrotCamera.pixelScaleApprox < MIN_GPU_SCALE && !deepPrecisionWarningShown) {
        console.warn('Mandelbrot zoom reached the GPU delta precision floor. The CPU camera stays precise, but visual detail will eventually plateau.');
        deepPrecisionWarningShown = true;
    }

    if (mandelbrotReference.useSimpleFallback) {
        drawSimpleFractal('mandelbrot', {
            centerX: decimalToNumber(mandelbrotCamera.centerX),
            centerY: decimalToNumber(mandelbrotCamera.centerY),
            pixelScale: mandelbrotCamera.pixelScaleApprox,
            maxIterations: mandelbrotCamera.maxIterations,
        });
        return;
    }

    gl.useProgram(deepProgramInfo.program);
    bindQuad(deepProgramInfo);
    gl.uniform2f(deepProgramInfo.uniforms.u_resolution, gl.canvas.width, gl.canvas.height);
    gl.uniform1f(deepProgramInfo.uniforms.u_pixelScale, mandelbrotCamera.pixelScaleApprox);
    gl.uniform2f(
        deepProgramInfo.uniforms.u_referenceDelta,
        decimalToNumber(subDecimal(mandelbrotCamera.centerX, mandelbrotReference.centerX)),
        decimalToNumber(subDecimal(mandelbrotCamera.centerY, mandelbrotReference.centerY))
    );
    gl.uniform1i(deepProgramInfo.uniforms.u_referenceOrbit, 0);
    gl.uniform1i(deepProgramInfo.uniforms.u_maxIterations, mandelbrotCamera.maxIterations);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, orbitTexture);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
}

function draw() {
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
    gl.clear(gl.COLOR_BUFFER_BIT);

    if (fractalType === 'mandelbrot') {
        drawDeepMandelbrot();
        return;
    }

    drawSimpleFractal(fractalType, simpleCamera);
}

function animate() {
    if (!gl) {
        return;
    }

    if (fractalType === 'mandelbrot') {
        updateMandelbrotCamera();
    } else {
        updateSimpleCamera(simpleCamera);
    }

    draw();
    requestAnimationFrame(animate);
}

function clampMousePosition() {
    mousePosition.x = Math.min(gl.canvas.width, Math.max(0, mousePosition.x));
    mousePosition.y = Math.min(gl.canvas.height, Math.max(0, mousePosition.y));
}

function setMouseToCenter() {
    mousePosition = {
        x: gl.canvas.width * 0.5,
        y: gl.canvas.height * 0.5,
    };
}

function resizeCanvas() {
    if (!gl) {
        return;
    }

    const previousWidth = gl.canvas.width || 1;
    const previousHeight = gl.canvas.height || 1;
    const dpr = window.devicePixelRatio || 1;
    gl.canvas.width = Math.max(1, Math.round(window.innerWidth * dpr));
    gl.canvas.height = Math.max(1, Math.round(window.innerHeight * dpr));
    gl.canvas.style.width = `${window.innerWidth}px`;
    gl.canvas.style.height = `${window.innerHeight}px`;

    mousePosition = {
        x: (mousePosition.x / previousWidth) * gl.canvas.width,
        y: (mousePosition.y / previousHeight) * gl.canvas.height,
    };

    clampMousePosition();

    if (fractalType !== 'mandelbrot') {
        updateSimpleCameraScale(simpleCamera);
    }
}

function resetSimpleCamera(type) {
    if (type === 'mandelbrot') {
        simpleCamera = createSimpleCamera(-0.745, 0.1, 1.6);
    } else if (type === 'julia') {
        simpleCamera = createSimpleCamera(0, 0, 3.0);
    } else {
        simpleCamera = createSimpleCamera(0, 0.2, 2.4);
    }
    updateSimpleCameraScale(simpleCamera);
}

function resetMandelbrotState() {
    mandelbrotCamera = createMandelbrotCamera();
    mandelbrotReference = createEmptyReference();
    deepPrecisionWarningShown = false;
}

function handleMouseMove(event) {
    const rect = gl.canvas.getBoundingClientRect();
    const scaleX = gl.canvas.width / rect.width;
    const scaleY = gl.canvas.height / rect.height;

    mousePosition = {
        x: (event.clientX - rect.left) * scaleX,
        y: (event.clientY - rect.top) * scaleY,
    };
    clampMousePosition();
}

function handleFractalTypeChange(event) {
    fractalType = event.target.value;
    if (fractalType === 'mandelbrot') {
        resetMandelbrotState();
    } else {
        resetSimpleCamera(fractalType);
    }
    setMouseToCenter();
}

function preventZoom(event) {
    event.preventDefault();
    event.stopPropagation();
}

window.addEventListener('load', () => {
    if (!initWebGL()) {
        return;
    }

    document.getElementById('fractalType').addEventListener('change', handleFractalTypeChange);
    gl.canvas.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('wheel', preventZoom, { passive: false });
    document.addEventListener('touchmove', preventZoom, { passive: false });

    resizeCanvas();
    resetMandelbrotState();
    setMouseToCenter();
    draw();
    requestAnimationFrame(animate);
});

window.addEventListener('resize', () => {
    resizeCanvas();
});
