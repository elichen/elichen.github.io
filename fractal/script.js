let canvas;
let gl;
let quadBuffer;
let fractalType = 'mandelbrot';
let mousePosition = { x: 0, y: 0 };
let deepPrecisionWarningShown = false;
let mandelbrotQualityHold = false;
let mandelbrotQualityHoldWarningShown = false;

const ZOOM_SPEED = 0.994;
const MIN_DECIMAL_DIGITS = 80;
const EXTRA_DECIMAL_DIGITS = 28;
const MAX_GPU_ITERATIONS = 1536;
const MIN_GPU_SCALE = 1e-45;
const GLITCH_THRESHOLD = 1e-5;
const MANDELBROT_LOG_INTERVAL = 500;
const MAX_REPAIR_REFERENCES = 192;
const MAX_REPAIR_DEPTH = 10;
const MIN_REPAIR_TILE_SIZE = 4;
const MAX_REFERENCE_CACHE_ENTRIES = 384;
const BASE_REFERENCE_SEARCH_RINGS = [0, 4, 12, 32, 96];
const DENSE_REFERENCE_SEARCH_RINGS = [0, 2, 4, 8, 16, 32, 64, 128];
const MIN_REFERENCE_RING_SAMPLES = 24;
const COARSE_REFERENCE_RING_SAMPLE_SPACING = 28;
const DENSE_REFERENCE_RING_SAMPLE_SPACING = 12;
const REFERENCE_REFINEMENT_ESCAPE_MARGIN = 8;
const REFERENCE_REUSE_RADIUS_PIXELS = 192;
const STABLE_REFERENCE_REUSE_RADIUS_PIXELS = 512;
const REFERENCE_REUSE_ESCAPE_RATIO = 0.9;

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
uniform vec2 u_referenceDeltaPixels;
uniform sampler2D u_referenceOrbit;
uniform int u_maxIterations;
uniform int u_referenceOrbitLength;
uniform float u_glitchThreshold;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outMask;

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

float mandelbrotPerturbation(vec2 deltaPixels) {
    vec2 scaledDelta = vec2(0.0);
    for (int i = 0; i < ${MAX_GPU_ITERATIONS}; ++i) {
        if (i >= u_maxIterations) break;
        if (i + 1 >= u_referenceOrbitLength) {
            outMask = vec4(1.0, 0.0, 0.0, 1.0);
            return -1.0;
        }

        vec2 reference = getOrbit(i);
        vec2 nextReference = getOrbit(i + 1);
        vec2 nextScaledDelta = (
            2.0 * complexMul(reference, scaledDelta)
        ) + (
            u_pixelScale * complexMul(scaledDelta, scaledDelta)
        ) + deltaPixels;
        vec2 z = nextReference + (u_pixelScale * nextScaledDelta);
        float mag2 = dot(z, z);
        if (mag2 > 4.0) {
            float smoothValue = float(i + 1) - log2(log2(max(length(z), 1.0001)));
            outMask = vec4(0.0, 0.0, 0.0, 1.0);
            return smoothValue / float(u_maxIterations);
        }
        float referenceMag2 = dot(nextReference, nextReference);
        if (referenceMag2 > 4.0 && mag2 < (u_glitchThreshold * referenceMag2)) {
            outMask = vec4(1.0, 0.0, 0.0, 1.0);
            return -1.0;
        }
        scaledDelta = nextScaledDelta;
    }
    outMask = vec4(0.0, 0.0, 0.0, 1.0);
    return 0.0;
}

void main() {
    vec2 pixelOffset = vec2(
        gl_FragCoord.x - 0.5 * u_resolution.x,
        gl_FragCoord.y - 0.5 * u_resolution.y
    );
    vec2 deltaPixels = u_referenceDeltaPixels + pixelOffset;
    float value = mandelbrotPerturbation(deltaPixels);
    vec3 color = value <= 0.0 ? vec3(0.0) : getPalette(value);
    outColor = vec4(color, 1.0);
}
`;

const blitFragmentShaderSource = `#version 300 es
precision highp float;

uniform sampler2D u_texture;
uniform vec2 u_resolution;

out vec4 outColor;

void main() {
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    outColor = texelFetch(u_texture, pixel, 0);
}
`;

let simpleProgramInfo;
let deepProgramInfo;
let blitProgramInfo;
let orbitTexture;
let mandelbrotWorkingFramebuffer;
let mandelbrotCommittedFramebuffer;
let mandelbrotWorkingColorTexture;
let mandelbrotWorkingMaskTexture;
let mandelbrotCommittedColorTexture;
let mandelbrotFrameReady = false;
let mandelbrotReferenceCache = new Map();
let mandelbrotRenderTargetWidth = 0;
let mandelbrotRenderTargetHeight = 0;
let mandelbrotZoomStepCount = 0;
let mandelbrotLastFrameStats = null;

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
        escapeIteration: 0,
        useSimpleFallback: false,
    };
}

function cloneMandelbrotCamera(camera) {
    return {
        centerX: camera.centerX,
        centerY: camera.centerY,
        pixelScale: camera.pixelScale,
        pixelScaleApprox: camera.pixelScaleApprox,
        maxIterations: camera.maxIterations,
        viewWidth: camera.viewWidth,
    };
}

function cloneMandelbrotReference(reference) {
    return {
        centerX: reference.centerX,
        centerY: reference.centerY,
        orbitData: reference.orbitData,
        orbitLength: reference.orbitLength,
        maxIterations: reference.maxIterations,
        escapeIteration: reference.escapeIteration,
        useSimpleFallback: reference.useSimpleFallback,
    };
}

function createEmptyMandelbrotFrameStats() {
    return {
        status: 'idle',
        reason: null,
        attemptedPixelScaleApprox: null,
        attemptedMaxIterations: null,
        initialEscapeIteration: null,
        initialOrbitLength: null,
        initialEscapedEarly: null,
        initialReferenceMode: null,
        referencesUsed: 0,
        repairPasses: 0,
        repairQueuePeak: 0,
        deepestTileDepth: 0,
        lastTileWidth: null,
        lastTileHeight: null,
        lastTileDepth: null,
        lastRepairEscapeIteration: null,
        queuedTilesRemaining: 0,
        cpuResolvedTiles: 0,
        cpuResolvedPixels: 0,
    };
}

function roundDebugNumber(value) {
    if (!Number.isFinite(value)) {
        return value;
    }
    if (value === 0) {
        return 0;
    }
    return Number(value.toPrecision(6));
}

function mixColorChannel(a, b, t) {
    return a + ((b - a) * t);
}

function getPaletteColor(t) {
    const c1 = [0.08, 0.10, 0.30];
    const c2 = [0.02, 0.48, 0.65];
    const c3 = [0.95, 0.92, 0.24];
    const c4 = [0.98, 0.45, 0.05];
    const c5 = [0.72, 0.06, 0.42];
    const clampedT = ((t % 1) + 1) % 1;
    const s = (clampedT * 5) % 5;

    if (s < 1) {
        return [
            mixColorChannel(c1[0], c2[0], s),
            mixColorChannel(c1[1], c2[1], s),
            mixColorChannel(c1[2], c2[2], s),
        ];
    }
    if (s < 2) {
        return [
            mixColorChannel(c2[0], c3[0], s - 1),
            mixColorChannel(c2[1], c3[1], s - 1),
            mixColorChannel(c2[2], c3[2], s - 1),
        ];
    }
    if (s < 3) {
        return [
            mixColorChannel(c3[0], c4[0], s - 2),
            mixColorChannel(c3[1], c4[1], s - 2),
            mixColorChannel(c3[2], c4[2], s - 2),
        ];
    }
    if (s < 4) {
        return [
            mixColorChannel(c4[0], c5[0], s - 3),
            mixColorChannel(c4[1], c5[1], s - 3),
            mixColorChannel(c4[2], c5[2], s - 3),
        ];
    }
    return [
        mixColorChannel(c5[0], c1[0], s - 4),
        mixColorChannel(c5[1], c1[1], s - 4),
        mixColorChannel(c5[2], c1[2], s - 4),
    ];
}

function finalizeMandelbrotFrameStats(frameStats, status, reason = null, extra = {}) {
    const finalized = {
        ...frameStats,
        ...extra,
        status,
        reason,
        repairPasses: Math.max(0, frameStats.referencesUsed - 1),
    };
    mandelbrotLastFrameStats = finalized;
    return finalized;
}

function getMandelbrotDebugSnapshot() {
    return {
        step: mandelbrotZoomStepCount,
        pixelScaleApprox: mandelbrotCamera ? roundDebugNumber(mandelbrotCamera.pixelScaleApprox) : null,
        maxIterations: mandelbrotCamera ? mandelbrotCamera.maxIterations : null,
        hold: mandelbrotQualityHold,
        frameReady: mandelbrotFrameReady,
        mouseX: roundDebugNumber(mousePosition.x),
        mouseY: roundDebugNumber(mousePosition.y),
        lastFrame: mandelbrotLastFrameStats || createEmptyMandelbrotFrameStats(),
    };
}

function logMandelbrotProgress() {
    if (
        fractalType !== 'mandelbrot'
        || !MANDELBROT_LOG_INTERVAL
        || mandelbrotZoomStepCount === 0
        || (mandelbrotZoomStepCount % MANDELBROT_LOG_INTERVAL) !== 0
    ) {
        return;
    }

    console.info(`Mandelbrot zoom progress ${JSON.stringify(getMandelbrotDebugSnapshot())}`);
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
        stepMandelbrotCameraWithQualityPriority();
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

function createTexture(width, height, internalFormat, format, type) {
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        internalFormat,
        width,
        height,
        0,
        format,
        type,
        null
    );
    return texture;
}

function ensureMandelbrotRenderTargets() {
    if (!gl) {
        return;
    }

    const width = gl.canvas.width;
    const height = gl.canvas.height;
    if (
        mandelbrotWorkingColorTexture
        && mandelbrotWorkingMaskTexture
        && mandelbrotCommittedColorTexture
        && mandelbrotRenderTargetWidth === width
        && mandelbrotRenderTargetHeight === height
    ) {
        return;
    }

    mandelbrotRenderTargetWidth = width;
    mandelbrotRenderTargetHeight = height;
    mandelbrotFrameReady = false;

    if (mandelbrotWorkingFramebuffer) {
        gl.deleteFramebuffer(mandelbrotWorkingFramebuffer);
        gl.deleteFramebuffer(mandelbrotCommittedFramebuffer);
        gl.deleteTexture(mandelbrotWorkingColorTexture);
        gl.deleteTexture(mandelbrotWorkingMaskTexture);
        gl.deleteTexture(mandelbrotCommittedColorTexture);
    }

    mandelbrotWorkingColorTexture = createTexture(width, height, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE);
    mandelbrotWorkingMaskTexture = createTexture(width, height, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE);
    mandelbrotCommittedColorTexture = createTexture(width, height, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE);

    mandelbrotWorkingFramebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, mandelbrotWorkingFramebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, mandelbrotWorkingColorTexture, 0);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, mandelbrotWorkingMaskTexture, 0);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
        throw new Error('Mandelbrot working framebuffer is incomplete.');
    }

    mandelbrotCommittedFramebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, mandelbrotCommittedFramebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, mandelbrotCommittedColorTexture, 0);
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
        throw new Error('Mandelbrot committed framebuffer is incomplete.');
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
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
        [
            'u_resolution',
            'u_pixelScale',
            'u_referenceDeltaPixels',
            'u_referenceOrbit',
            'u_maxIterations',
            'u_referenceOrbitLength',
            'u_glitchThreshold',
        ]
    );

    blitProgramInfo = createProgramInfo(
        createProgram(vertexShaderSource, blitFragmentShaderSource),
        ['u_texture', 'u_resolution']
    );

    orbitTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, orbitTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    ensureMandelbrotRenderTargets();

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

function createDeepPoint(x, y) {
    return { x, y };
}

function offsetDeepPoint(camera, point, offsetX, offsetY) {
    return {
        x: addDecimal(point.x, mulDecimalNumber(camera.pixelScale, offsetX)),
        y: addDecimal(point.y, mulDecimalNumber(camera.pixelScale, offsetY)),
    };
}

function getDeepPointScreenDistanceSquared(camera, point, pointer = mousePosition) {
    const screenPoint = planeToScreenDeep(camera, point);
    const dx = screenPoint.x - pointer.x;
    const dy = screenPoint.y - pointer.y;
    return (dx * dx) + (dy * dy);
}

function getReferenceRingOffsets(radius, sampleSpacing) {
    if (radius === 0) {
        return [{ x: 0, y: 0 }];
    }

    const offsets = [];
    const seen = new Set();

    const sampleCount = Math.max(
        MIN_REFERENCE_RING_SAMPLES,
        Math.ceil((2 * Math.PI * radius) / sampleSpacing)
    );

    for (let index = 0; index < sampleCount; index += 1) {
        const angle = (2 * Math.PI * index) / sampleCount;
        const x = Math.round(Math.cos(angle) * radius);
        const y = Math.round(Math.sin(angle) * radius);
        const key = `${x},${y}`;
        if (seen.has(key)) {
            continue;
        }
        seen.add(key);
        offsets.push({ x, y });
    }

    return offsets;
}

function getReferenceSearchRings(baseRings) {
    const maxViewportRadius = Math.max(
        128,
        Math.floor(Math.min(gl.canvas.width, gl.canvas.height) * 0.6)
    );
    const rings = baseRings.filter((radius) => radius <= maxViewportRadius);
    if (!rings.includes(maxViewportRadius)) {
        rings.push(maxViewportRadius);
    }
    return rings;
}

function computeEscapeIteration(centerX, centerY, iterations) {
    let zr = 0n;
    let zi = 0n;

    for (let i = 0; i < iterations; i += 1) {
        const zr2 = mulDecimal(zr, zr);
        const zi2 = mulDecimal(zi, zi);
        const zrzi = mulDecimal(zr, zi);

        zr = addDecimal(subDecimal(zr2, zi2), centerX);
        zi = addDecimal(mulDecimalInt(zrzi, 2), centerY);

        const magnitude = addDecimal(mulDecimal(zr, zr), mulDecimal(zi, zi));
        if (magnitude > mulDecimalInt(decimalScale, 4)) {
            return i + 1;
        }
    }

    return iterations;
}

function isReferenceCandidateBetter(candidate, bestCandidate) {
    if (!candidate) {
        return false;
    }
    if (!bestCandidate) {
        return true;
    }
    if (candidate.escapeIteration !== bestCandidate.escapeIteration) {
        return candidate.escapeIteration > bestCandidate.escapeIteration;
    }
    return candidate.distanceSquared < bestCandidate.distanceSquared;
}

function getReferenceSearchAnchors(anchorPoint) {
    const anchors = [anchorPoint];

    if (mandelbrotReference.centerX !== null && mandelbrotReference.centerY !== null) {
        const previousReferenceAnchor = createDeepPoint(
            mandelbrotReference.centerX,
            mandelbrotReference.centerY
        );
        const key = `${previousReferenceAnchor.x.toString()}:${previousReferenceAnchor.y.toString()}`;
        if (key !== `${anchorPoint.x.toString()}:${anchorPoint.y.toString()}`) {
            anchors.push(previousReferenceAnchor);
        }
    }

    return anchors;
}

function searchReferenceCandidates(anchorPoint, iterations, rings, sampleSpacing) {
    let bestCandidate = null;

    for (const searchAnchor of getReferenceSearchAnchors(anchorPoint)) {
        const isPrimaryAnchor = searchAnchor === anchorPoint;
        for (const radius of getReferenceSearchRings(rings)) {
            for (const offset of getReferenceRingOffsets(radius, sampleSpacing)) {
                const candidatePoint = offsetDeepPoint(mandelbrotCamera, searchAnchor, offset.x, offset.y);
                const escapeIteration = computeEscapeIteration(candidatePoint.x, candidatePoint.y, iterations);
                const distanceSquared = isPrimaryAnchor
                    ? ((offset.x * offset.x) + (offset.y * offset.y))
                    : getDeepPointScreenDistanceSquared(mandelbrotCamera, candidatePoint);
                const candidate = { point: candidatePoint, distanceSquared, escapeIteration };

                if (isReferenceCandidateBetter(candidate, bestCandidate)) {
                    bestCandidate = candidate;
                }
            }
        }
    }

    return bestCandidate;
}

function findBestReferencePoint(anchorPoint, iterations) {
    const coarseCandidate = searchReferenceCandidates(
        anchorPoint,
        iterations,
        BASE_REFERENCE_SEARCH_RINGS,
        COARSE_REFERENCE_RING_SAMPLE_SPACING
    );

    if (
        !coarseCandidate
        || coarseCandidate.escapeIteration >= (iterations - REFERENCE_REFINEMENT_ESCAPE_MARGIN)
    ) {
        return coarseCandidate;
    }

    const denseCandidate = searchReferenceCandidates(
        anchorPoint,
        iterations,
        DENSE_REFERENCE_SEARCH_RINGS,
        DENSE_REFERENCE_RING_SAMPLE_SPACING
    );

    return isReferenceCandidateBetter(denseCandidate, coarseCandidate)
        ? denseCandidate
        : coarseCandidate;
}

function isReusableReferenceStrong(reference, iterations) {
    if (!reference) {
        return false;
    }

    if (!reference.escapedEarly) {
        return true;
    }

    const minimumEscapeIteration = Math.max(
        iterations - REFERENCE_REFINEMENT_ESCAPE_MARGIN,
        Math.floor(iterations * REFERENCE_REUSE_ESCAPE_RATIO)
    );
    return reference.escapeIteration >= minimumEscapeIteration;
}

function getReferenceReuseRadiusPixels() {
    if (
        mandelbrotLastFrameStats
        && mandelbrotLastFrameStats.status === 'success'
        && mandelbrotLastFrameStats.referencesUsed === 1
        && mandelbrotLastFrameStats.cpuResolvedTiles === 0
    ) {
        return STABLE_REFERENCE_REUSE_RADIUS_PIXELS;
    }

    return REFERENCE_REUSE_RADIUS_PIXELS;
}

function getReusableReferenceSelection(anchorPoint, iterations) {
    if (mandelbrotReference.centerX === null || mandelbrotReference.centerY === null) {
        return null;
    }

    const point = createDeepPoint(mandelbrotReference.centerX, mandelbrotReference.centerY);
    const distanceSquared = getDeepPointScreenDistanceSquared(mandelbrotCamera, point);
    const reuseRadiusPixels = getReferenceReuseRadiusPixels();
    if (distanceSquared > (reuseRadiusPixels * reuseRadiusPixels)) {
        return null;
    }

    const reference = getReferenceOrbit(point, iterations);
    if (!isReusableReferenceStrong(reference, iterations)) {
        return null;
    }

    return {
        candidate: {
            point,
            distanceSquared,
            escapeIteration: reference.escapeIteration,
        },
        reference,
        mode: 'reuse',
    };
}

function selectInitialReference(anchorPoint, iterations) {
    const reusableSelection = getReusableReferenceSelection(anchorPoint, iterations);
    if (reusableSelection) {
        return reusableSelection;
    }

    const candidate = findBestReferencePoint(anchorPoint, iterations);
    if (!candidate) {
        return null;
    }

    return {
        candidate,
        reference: getReferenceOrbit(candidate.point, iterations),
        mode: 'search',
    };
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

        const baseIndex = (i + 1) * 4;
        orbitData[baseIndex] = decimalToNumber(zr);
        orbitData[baseIndex + 1] = decimalToNumber(zi);

        const magnitude = addDecimal(mulDecimal(zr, zr), mulDecimal(zi, zi));
        if (magnitude > mulDecimalInt(decimalScale, 4)) {
            return {
                escapedEarly: true,
                escapeIteration: i + 1,
                orbitData,
                orbitLength: i + 2,
            };
        }
    }

    return {
        escapedEarly: false,
        escapeIteration: iterations,
        orbitData,
        orbitLength: iterations + 1,
    };
}

function getReferenceCacheKey(point, iterations) {
    return `${point.x.toString()}:${point.y.toString()}:${iterations}`;
}

function getReferenceOrbit(point, iterations) {
    const key = getReferenceCacheKey(point, iterations);
    if (mandelbrotReferenceCache.has(key)) {
        return mandelbrotReferenceCache.get(key);
    }

    const referenceOrbit = computeReferenceOrbit(point.x, point.y, iterations);
    const reference = {
        point,
        orbitData: referenceOrbit.orbitData,
        orbitLength: referenceOrbit.orbitLength,
        escapeIteration: referenceOrbit.escapeIteration,
        escapedEarly: referenceOrbit.escapedEarly,
    };
    mandelbrotReferenceCache.set(key, reference);
    while (mandelbrotReferenceCache.size > MAX_REFERENCE_CACHE_ENTRIES) {
        const oldestKey = mandelbrotReferenceCache.keys().next().value;
        mandelbrotReferenceCache.delete(oldestKey);
    }
    return reference;
}

function commitMandelbrotReference(reference) {
    mandelbrotReference.centerX = reference.point.x;
    mandelbrotReference.centerY = reference.point.y;
    mandelbrotReference.orbitData = reference.orbitData;
    mandelbrotReference.orbitLength = reference.orbitLength;
    mandelbrotReference.maxIterations = mandelbrotCamera.maxIterations;
    mandelbrotReference.escapeIteration = reference.escapeIteration;
    mandelbrotReference.useSimpleFallback = false;
}

function clearWorkingFramebuffer() {
    gl.bindFramebuffer(gl.FRAMEBUFFER, mandelbrotWorkingFramebuffer);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
    gl.clearBufferfv(gl.COLOR, 0, new Float32Array([0, 0, 0, 1]));
    gl.clearBufferfv(gl.COLOR, 1, new Float32Array([0, 0, 0, 1]));
}

function createRepairTile(x, y, width, height, depth, maskData) {
    return { x, y, width, height, depth, maskData };
}

function readGlitchMask(tile) {
    const maskData = new Uint8Array(tile.width * tile.height * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, mandelbrotWorkingFramebuffer);
    gl.readBuffer(gl.COLOR_ATTACHMENT1);
    gl.readPixels(tile.x, tile.y, tile.width, tile.height, gl.RGBA, gl.UNSIGNED_BYTE, maskData);
    return maskData;
}

function maskHasGlitches(maskData) {
    for (let index = 0; index < maskData.length; index += 4) {
        if (maskData[index] !== 0) {
            return true;
        }
    }
    return false;
}

function extractTileMask(parentTile, childTile) {
    const maskData = new Uint8Array(childTile.width * childTile.height * 4);
    const offsetX = childTile.x - parentTile.x;
    const offsetY = childTile.y - parentTile.y;

    for (let row = 0; row < childTile.height; row += 1) {
        const parentStart = ((offsetY + row) * parentTile.width + offsetX) * 4;
        const childStart = row * childTile.width * 4;
        maskData.set(
            parentTile.maskData.subarray(parentStart, parentStart + (childTile.width * 4)),
            childStart
        );
    }

    return maskData;
}

function subdivideRepairTile(tile) {
    const halfWidth = Math.floor(tile.width / 2);
    const halfHeight = Math.floor(tile.height / 2);

    const children = [
        createRepairTile(tile.x, tile.y, halfWidth, halfHeight, tile.depth + 1, null),
        createRepairTile(tile.x + halfWidth, tile.y, tile.width - halfWidth, halfHeight, tile.depth + 1, null),
        createRepairTile(tile.x, tile.y + halfHeight, halfWidth, tile.height - halfHeight, tile.depth + 1, null),
        createRepairTile(tile.x + halfWidth, tile.y + halfHeight, tile.width - halfWidth, tile.height - halfHeight, tile.depth + 1, null),
    ];

    return children.filter((child) => child.width > 0 && child.height > 0);
}

function queueChildTilesWithGlitches(queue, tile) {
    for (const child of subdivideRepairTile(tile)) {
        child.maskData = extractTileMask(tile, child);
        if (maskHasGlitches(child.maskData)) {
            queue.push(child);
        }
    }
}

function sortRepairQueue(queue) {
    queue.sort((a, b) => {
        const areaDelta = (b.width * b.height) - (a.width * a.height);
        if (areaDelta !== 0) {
            return areaDelta;
        }
        return a.depth - b.depth;
    });
}

function findGlitchedPixelNearTileCenter(tile) {
    const centerX = (tile.width - 1) * 0.5;
    const centerY = (tile.height - 1) * 0.5;
    let bestPixel = null;
    let bestDistanceSquared = Number.POSITIVE_INFINITY;

    for (let row = 0; row < tile.height; row += 1) {
        for (let column = 0; column < tile.width; column += 1) {
            const baseIndex = ((row * tile.width) + column) * 4;
            if (tile.maskData[baseIndex] === 0) {
                continue;
            }

            const dx = column - centerX;
            const dy = row - centerY;
            const distanceSquared = (dx * dx) + (dy * dy);
            if (distanceSquared < bestDistanceSquared) {
                bestDistanceSquared = distanceSquared;
                bestPixel = {
                    x: tile.x + column,
                    y: tile.y + row,
                };
            }
        }
    }

    return bestPixel;
}

function framebufferPixelToPointer(pixel) {
    return {
        x: pixel.x + 0.5,
        y: gl.canvas.height - pixel.y - 0.5,
    };
}

function computeCpuMandelbrotColor(centerX, centerY, iterations) {
    let zr = 0n;
    let zi = 0n;

    for (let i = 0; i < iterations; i += 1) {
        const zr2 = mulDecimal(zr, zr);
        const zi2 = mulDecimal(zi, zi);
        const zrzi = mulDecimal(zr, zi);

        zr = addDecimal(subDecimal(zr2, zi2), centerX);
        zi = addDecimal(mulDecimalInt(zrzi, 2), centerY);

        const magnitude = addDecimal(mulDecimal(zr, zr), mulDecimal(zi, zi));
        if (magnitude > mulDecimalInt(decimalScale, 4)) {
            const zMagnitude = Math.max(
                1.0001,
                Math.hypot(decimalToNumber(zr), decimalToNumber(zi))
            );
            const smoothValue = (i + 1) - Math.log2(Math.log2(zMagnitude));
            const normalized = Math.max(0, Math.min(1, smoothValue / iterations));
            const [r, g, b] = getPaletteColor(normalized);
            return [
                Math.round(r * 255),
                Math.round(g * 255),
                Math.round(b * 255),
                255,
            ];
        }
    }

    return [0, 0, 0, 255];
}

function resolveMandelbrotTileOnCPU(tile) {
    const colorData = new Uint8Array(tile.width * tile.height * 4);
    const maskData = new Uint8Array(tile.width * tile.height * 4);

    for (let row = 0; row < tile.height; row += 1) {
        for (let column = 0; column < tile.width; column += 1) {
            const pointer = framebufferPixelToPointer({
                x: tile.x + column,
                y: tile.y + row,
            });
            const point = screenToPlaneDeep(mandelbrotCamera, pointer);
            const color = computeCpuMandelbrotColor(point.x, point.y, mandelbrotCamera.maxIterations);
            const baseIndex = ((row * tile.width) + column) * 4;

            colorData[baseIndex] = color[0];
            colorData[baseIndex + 1] = color[1];
            colorData[baseIndex + 2] = color[2];
            colorData[baseIndex + 3] = color[3];

            maskData[baseIndex] = 0;
            maskData[baseIndex + 1] = 0;
            maskData[baseIndex + 2] = 0;
            maskData[baseIndex + 3] = 255;
        }
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    gl.bindTexture(gl.TEXTURE_2D, mandelbrotWorkingColorTexture);
    gl.texSubImage2D(
        gl.TEXTURE_2D,
        0,
        tile.x,
        tile.y,
        tile.width,
        tile.height,
        gl.RGBA,
        gl.UNSIGNED_BYTE,
        colorData
    );

    gl.bindTexture(gl.TEXTURE_2D, mandelbrotWorkingMaskTexture);
    gl.texSubImage2D(
        gl.TEXTURE_2D,
        0,
        tile.x,
        tile.y,
        tile.width,
        tile.height,
        gl.RGBA,
        gl.UNSIGNED_BYTE,
        maskData
    );
}

function renderMandelbrotPass(reference, tile) {
    uploadReferenceOrbit(reference.orbitData, reference.orbitLength);

    gl.bindFramebuffer(gl.FRAMEBUFFER, mandelbrotWorkingFramebuffer);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
    gl.enable(gl.SCISSOR_TEST);
    gl.scissor(tile.x, tile.y, tile.width, tile.height);

    gl.useProgram(deepProgramInfo.program);
    bindQuad(deepProgramInfo);
    gl.uniform2f(deepProgramInfo.uniforms.u_resolution, gl.canvas.width, gl.canvas.height);
    gl.uniform1f(deepProgramInfo.uniforms.u_pixelScale, mandelbrotCamera.pixelScaleApprox);
    gl.uniform1f(deepProgramInfo.uniforms.u_glitchThreshold, GLITCH_THRESHOLD);
    gl.uniform1i(deepProgramInfo.uniforms.u_referenceOrbitLength, reference.orbitLength);
    gl.uniform1i(deepProgramInfo.uniforms.u_referenceOrbit, 0);
    gl.uniform1i(deepProgramInfo.uniforms.u_maxIterations, mandelbrotCamera.maxIterations);

    const referenceDeltaX = decimalToNumber(subDecimal(mandelbrotCamera.centerX, reference.point.x));
    const referenceDeltaY = decimalToNumber(subDecimal(mandelbrotCamera.centerY, reference.point.y));
    gl.uniform2f(
        deepProgramInfo.uniforms.u_referenceDeltaPixels,
        referenceDeltaX / mandelbrotCamera.pixelScaleApprox,
        referenceDeltaY / mandelbrotCamera.pixelScaleApprox
    );

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, orbitTexture);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.disable(gl.SCISSOR_TEST);
}

function copyWorkingFrameToCommitted() {
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, mandelbrotWorkingFramebuffer);
    gl.readBuffer(gl.COLOR_ATTACHMENT0);
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, mandelbrotCommittedFramebuffer);
    gl.blitFramebuffer(
        0,
        0,
        gl.canvas.width,
        gl.canvas.height,
        0,
        0,
        gl.canvas.width,
        gl.canvas.height,
        gl.COLOR_BUFFER_BIT,
        gl.NEAREST
    );
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
}

function drawCommittedMandelbrotFrame() {
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);

    gl.useProgram(blitProgramInfo.program);
    bindQuad(blitProgramInfo);
    gl.uniform2f(blitProgramInfo.uniforms.u_resolution, gl.canvas.width, gl.canvas.height);
    gl.uniform1i(blitProgramInfo.uniforms.u_texture, 0);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, mandelbrotCommittedColorTexture);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
}

function renderSharpMandelbrotFrame() {
    const frameStats = createEmptyMandelbrotFrameStats();
    frameStats.attemptedPixelScaleApprox = roundDebugNumber(mandelbrotCamera.pixelScaleApprox);
    frameStats.attemptedMaxIterations = mandelbrotCamera.maxIterations;

    if (mandelbrotCamera.pixelScaleApprox < MIN_GPU_SCALE) {
        finalizeMandelbrotFrameStats(frameStats, 'failed', 'gpu_delta_precision_floor');
        return false;
    }

    ensureMandelbrotRenderTargets();
    clearWorkingFramebuffer();

    const fullFrameTile = createRepairTile(0, 0, gl.canvas.width, gl.canvas.height, 0, null);
    const anchorPoint = screenToPlaneDeep(mandelbrotCamera, mousePosition);
    const initialSelection = selectInitialReference(anchorPoint, mandelbrotCamera.maxIterations);
    if (!initialSelection) {
        finalizeMandelbrotFrameStats(frameStats, 'failed', 'no_initial_reference');
        return false;
    }

    const { candidate: initialCandidate, reference: initialReference, mode: initialReferenceMode } = initialSelection;
    frameStats.initialEscapeIteration = initialCandidate.escapeIteration;
    frameStats.initialOrbitLength = initialReference.orbitLength;
    frameStats.initialEscapedEarly = initialReference.escapedEarly;
    frameStats.initialReferenceMode = initialReferenceMode;
    frameStats.referencesUsed = 1;
    renderMandelbrotPass(initialReference, fullFrameTile);
    fullFrameTile.maskData = readGlitchMask(fullFrameTile);

    const repairQueue = [];
    if (maskHasGlitches(fullFrameTile.maskData)) {
        if (
            fullFrameTile.depth >= MAX_REPAIR_DEPTH
            || fullFrameTile.width <= MIN_REPAIR_TILE_SIZE
            || fullFrameTile.height <= MIN_REPAIR_TILE_SIZE
        ) {
            finalizeMandelbrotFrameStats(frameStats, 'failed', 'full_frame_tile_limit', {
                queuedTilesRemaining: repairQueue.length,
            });
            return false;
        }
        queueChildTilesWithGlitches(repairQueue, fullFrameTile);
        sortRepairQueue(repairQueue);
        frameStats.repairQueuePeak = Math.max(frameStats.repairQueuePeak, repairQueue.length);
    }

    let referencesUsed = 1;

    while (repairQueue.length > 0) {
        if (referencesUsed >= MAX_REPAIR_REFERENCES) {
            finalizeMandelbrotFrameStats(frameStats, 'failed', 'repair_reference_budget_exhausted', {
                queuedTilesRemaining: repairQueue.length,
            });
            return false;
        }

        const tile = repairQueue.shift();
        frameStats.lastTileWidth = tile.width;
        frameStats.lastTileHeight = tile.height;
        frameStats.lastTileDepth = tile.depth;
        frameStats.deepestTileDepth = Math.max(frameStats.deepestTileDepth, tile.depth);
        frameStats.repairQueuePeak = Math.max(frameStats.repairQueuePeak, repairQueue.length);
        const glitchedPixel = findGlitchedPixelNearTileCenter(tile);
        if (!glitchedPixel) {
            continue;
        }

        const repairPointer = framebufferPixelToPointer(glitchedPixel);
        const repairAnchor = screenToPlaneDeep(mandelbrotCamera, repairPointer);
        const repairCandidate = findBestReferencePoint(repairAnchor, mandelbrotCamera.maxIterations);
        if (!repairCandidate) {
            finalizeMandelbrotFrameStats(frameStats, 'failed', 'no_repair_reference', {
                queuedTilesRemaining: repairQueue.length,
                lastTileWidth: tile.width,
                lastTileHeight: tile.height,
                lastTileDepth: tile.depth,
            });
            return false;
        }

        frameStats.lastRepairEscapeIteration = repairCandidate.escapeIteration;
        const repairReference = getReferenceOrbit(repairCandidate.point, mandelbrotCamera.maxIterations);
        renderMandelbrotPass(repairReference, tile);
        tile.maskData = readGlitchMask(tile);
        referencesUsed += 1;
        frameStats.referencesUsed = referencesUsed;

        if (!maskHasGlitches(tile.maskData)) {
            continue;
        }

        if (
            tile.depth >= MAX_REPAIR_DEPTH
            || tile.width <= MIN_REPAIR_TILE_SIZE
            || tile.height <= MIN_REPAIR_TILE_SIZE
        ) {
            resolveMandelbrotTileOnCPU(tile);
            frameStats.cpuResolvedTiles += 1;
            frameStats.cpuResolvedPixels += tile.width * tile.height;
            continue;
        }

        queueChildTilesWithGlitches(repairQueue, tile);
        sortRepairQueue(repairQueue);
        frameStats.repairQueuePeak = Math.max(frameStats.repairQueuePeak, repairQueue.length);
    }

    copyWorkingFrameToCommitted();
    commitMandelbrotReference(initialReference);
    mandelbrotFrameReady = true;
    finalizeMandelbrotFrameStats(frameStats, 'success');
    return true;
}

function stepMandelbrotCameraWithQualityPriority() {
    if (mandelbrotQualityHold) {
        return false;
    }

    const previousCamera = cloneMandelbrotCamera(mandelbrotCamera);
    const previousReference = cloneMandelbrotReference(mandelbrotReference);
    const previousWarningShown = deepPrecisionWarningShown;

    updateMandelbrotCamera();

    if (!renderSharpMandelbrotFrame()) {
        mandelbrotCamera = previousCamera;
        mandelbrotReference = previousReference;
        deepPrecisionWarningShown = previousWarningShown;
        mandelbrotQualityHold = true;
        if (!mandelbrotQualityHoldWarningShown) {
            console.warn(`Paused Mandelbrot zoom ${JSON.stringify(getMandelbrotDebugSnapshot())}`);
            mandelbrotQualityHoldWarningShown = true;
        }
        return false;
    }

    mandelbrotZoomStepCount += 1;
    logMandelbrotProgress();
    mandelbrotQualityHold = false;
    mandelbrotQualityHoldWarningShown = false;
    return true;
}

function drawSimpleFractal(activeType, camera) {
    const fractalIndex = ['mandelbrot', 'julia', 'sierpinski'].indexOf(activeType);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
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
    if (mandelbrotCamera.pixelScaleApprox < MIN_GPU_SCALE && !deepPrecisionWarningShown) {
        console.warn('Mandelbrot zoom reached the GPU delta precision floor. The CPU camera stays precise, but visual detail will eventually plateau.');
        deepPrecisionWarningShown = true;
    }

    if (!mandelbrotFrameReady && !renderSharpMandelbrotFrame()) {
        drawSimpleFractal('mandelbrot', {
            centerX: decimalToNumber(mandelbrotCamera.centerX),
            centerY: decimalToNumber(mandelbrotCamera.centerY),
            pixelScale: mandelbrotCamera.pixelScaleApprox,
            maxIterations: mandelbrotCamera.maxIterations,
        });
        return;
    }

    drawCommittedMandelbrotFrame();
}

function draw() {
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
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
        stepMandelbrotCameraWithQualityPriority();
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

    ensureMandelbrotRenderTargets();
    mandelbrotFrameReady = false;

    if (fractalType === 'mandelbrot' && mandelbrotCamera) {
        mandelbrotCamera.viewWidth = mandelbrotCamera.pixelScaleApprox * Math.min(gl.canvas.width, gl.canvas.height);
        mandelbrotQualityHold = false;
        mandelbrotQualityHoldWarningShown = false;
    } else {
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
    mandelbrotReferenceCache = new Map();
    mandelbrotFrameReady = false;
    deepPrecisionWarningShown = false;
    mandelbrotQualityHold = false;
    mandelbrotQualityHoldWarningShown = false;
    mandelbrotZoomStepCount = 0;
    mandelbrotLastFrameStats = createEmptyMandelbrotFrameStats();
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
    mandelbrotQualityHold = false;
    mandelbrotQualityHoldWarningShown = false;
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
