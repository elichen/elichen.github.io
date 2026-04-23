let canvas;
let gl;
let quadBuffer;
let fractalType = 'mandelbrot';
let mousePosition = { x: 0, y: 0 };
let deepPrecisionWarningShown = false;
let juliaDeepPrecisionWarningShown = false;
let newtonDeepPrecisionWarningShown = false;
let mandelbrotQualityHold = false;
let mandelbrotQualityHoldWarningShown = false;
let juliaQualityHold = false;
let juliaQualityHoldWarningShown = false;
let newtonQualityHold = false;
let newtonQualityHoldWarningShown = false;

const ZOOM_SPEED = 0.994;
const MIN_DECIMAL_DIGITS = 80;
const EXTRA_DECIMAL_DIGITS = 28;
const MAX_GPU_ITERATIONS = 1536;
const MAX_ORBIT_TEXTURE_LENGTH = MAX_GPU_ITERATIONS + 1;
const MASK_REDUCTION_FACTOR = 2;
const STABLE_MASK_VERIFY_SKIP_FRAMES = 1;
const STABLE_MASK_VERIFY_GROWTH_INTERVAL = 6;
const MAX_STABLE_MASK_VERIFY_SKIP_FRAMES = 3;
const DEEP_RENDER_IDLE_TIMEOUT_MS = 48;
const DEEP_RENDER_FALLBACK_BUDGET_MS = 4;
const DEEP_RENDER_MAX_BUDGET_MS = 6;
const DEEP_RENDER_BOOTSTRAP_BUDGET_MS = 10;
const DEEP_RENDER_MIN_BUDGET_MS = 1;
const DEEP_RENDER_EXPENSIVE_STAGE_MIN_BUDGET_MS = 3.5;
const DEEP_RENDER_STALE_SCALE_RATIO = 1.12;
const DEEP_RENDER_STALE_TRANSLATION_PIXELS = 96;
const MIN_GPU_SCALE = 1e-45;
const NEWTON_DEEP_RENDER_SCALE = 1e-6;
const NEWTON_DEEP_RETRY_SCALE_FACTOR = 0.25;
const NEWTON_PATHOLOGICAL_INITIAL_REPAIR_QUEUE_LENGTH = 4;
const NEWTON_PATHOLOGICAL_INITIAL_GLITCH_RATIO = 0.005;
const NEWTON_PATHOLOGICAL_REPAIR_QUEUE_LENGTH = 12;
const NEWTON_PATHOLOGICAL_REPAIR_QUEUE_MIN = 6;
const NEWTON_PATHOLOGICAL_REPAIR_ELAPSED_MS = 300;
const NEWTON_PATHOLOGICAL_REPAIR_REFERENCE_LIMIT = 12;
const GLITCH_THRESHOLD = 1e-5;
const MANDELBROT_LOG_INTERVAL = 500;
const DEBUG_HEARTBEAT_INTERVAL_MS = 1000;
const MIN_DEBUG_HEARTBEAT_INTERVAL_MS = 250;
const MAX_RENDER_DEVICE_PIXEL_RATIO = 1.25;
const MAX_REPAIR_REFERENCES = 192;
const MAX_REPAIR_DEPTH = 10;
const MIN_REPAIR_TILE_SIZE = 4;
const MAX_REFERENCE_CACHE_ENTRIES = 384;
const BASE_REFERENCE_SEARCH_RINGS = [0, 4, 12, 32, 96];
const DENSE_REFERENCE_SEARCH_RINGS = [0, 2, 4, 8, 16, 32, 64, 128];
const NEWTON_INITIAL_REFERENCE_SEARCH_RINGS = [0, 2, 6, 12, 24, 48];
const MIN_REFERENCE_RING_SAMPLES = 24;
const COARSE_REFERENCE_RING_SAMPLE_SPACING = 28;
const DENSE_REFERENCE_RING_SAMPLE_SPACING = 12;
const NEWTON_INITIAL_REFERENCE_RING_SAMPLE_SPACING = 18;
const REFERENCE_REFINEMENT_ESCAPE_MARGIN = 8;
const REFERENCE_REUSE_RADIUS_PIXELS = 192;
const STABLE_REFERENCE_REUSE_RADIUS_PIXELS = 512;
const REFERENCE_REUSE_ESCAPE_RATIO = 0.9;
const FLOAT32_EPSILON = 2 ** -23;
const NEWTON_DIRECT_REFERENCE_MIN_ITERATIONS = 24;
const JULIA_CONSTANT_REAL = '-0.8';
const JULIA_CONSTANT_IMAGINARY = '0.156';
const NEWTON_ROOT_IMAGINARY = '0.866025403784438646763723170753';
const NEWTON_CONVERGENCE_DISTANCE_SQUARED = '1e-12';

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

vec2 complexDiv(vec2 a, vec2 b) {
    float denominator = dot(b, b);
    return vec2(
        (a.x * b.x + a.y * b.y) / denominator,
        (a.y * b.x - a.x * b.y) / denominator
    );
}

vec2 getNewtonRoot(int index) {
    if (index == 0) {
        return vec2(1.0, 0.0);
    }
    if (index == 1) {
        return vec2(-0.5, 0.8660254);
    }
    return vec2(-0.5, -0.8660254);
}

vec3 getNewtonRootColor(int rootIndex) {
    if (rootIndex == 0) {
        return vec3(0.15, 0.78, 0.93);
    }
    if (rootIndex == 1) {
        return vec3(0.99, 0.72, 0.16);
    }
    return vec3(0.93, 0.21, 0.58);
}

int getNearestNewtonRootIndex(vec2 z) {
    int bestIndex = 0;
    float bestDistanceSquared = dot(z - getNewtonRoot(0), z - getNewtonRoot(0));
    for (int index = 1; index < 3; ++index) {
        float distanceSquared = dot(z - getNewtonRoot(index), z - getNewtonRoot(index));
        if (distanceSquared < bestDistanceSquared) {
            bestDistanceSquared = distanceSquared;
            bestIndex = index;
        }
    }
    return bestIndex;
}

vec2 newtonStep(vec2 z) {
    vec2 z2 = complexMul(z, z);
    float denominator = dot(z2, z2);
    if (denominator < 1e-12) {
        return vec2(1e20, 1e20);
    }
    vec2 z3 = complexMul(z2, z);
    return z - complexDiv(z3 - vec2(1.0, 0.0), 3.0 * z2);
}

vec4 newtonColor(vec2 z) {
    for (int i = 0; i < ${MAX_GPU_ITERATIONS}; ++i) {
        if (i >= u_maxIterations) break;
        z = newtonStep(z);
        if (dot(z, z) > 1e30) {
            return vec4(0.0, 0.0, 0.0, 1.0);
        }
        int rootIndex = getNearestNewtonRootIndex(z);
        float distanceSquared = dot(z - getNewtonRoot(rootIndex), z - getNewtonRoot(rootIndex));
        if (distanceSquared < 1e-12) {
            float normalized = float(i + 1) / float(u_maxIterations);
            float fade = 0.25 + (0.75 * pow(1.0 - normalized, 0.35));
            float band = 0.9 + (0.1 * cos((18.0 * normalized) + (float(rootIndex) * 2.0943951)));
            return vec4(getNewtonRootColor(rootIndex) * fade * band, 1.0);
        }
    }
    return vec4(0.0, 0.0, 0.0, 1.0);
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
        outColor = newtonColor(plane);
        return;
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
uniform int u_deepFractalType;
uniform vec2 u_juliaConstant;

layout(location = 0) out vec4 outColor;
layout(location = 1) out float outMask;

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

vec2 complexDiv(vec2 a, vec2 b) {
    float denominator = dot(b, b);
    return vec2(
        (a.x * b.x + a.y * b.y) / denominator,
        (a.y * b.x - a.x * b.y) / denominator
    );
}

vec2 getNewtonRoot(int index) {
    if (index == 0) {
        return vec2(1.0, 0.0);
    }
    if (index == 1) {
        return vec2(-0.5, 0.8660254);
    }
    return vec2(-0.5, -0.8660254);
}

vec3 getNewtonRootColor(int rootIndex) {
    if (rootIndex == 0) {
        return vec3(0.15, 0.78, 0.93);
    }
    if (rootIndex == 1) {
        return vec3(0.99, 0.72, 0.16);
    }
    return vec3(0.93, 0.21, 0.58);
}

int getNearestNewtonRootIndex(vec2 z) {
    int bestIndex = 0;
    float bestDistanceSquared = dot(z - getNewtonRoot(0), z - getNewtonRoot(0));
    for (int index = 1; index < 3; ++index) {
        float distanceSquared = dot(z - getNewtonRoot(index), z - getNewtonRoot(index));
        if (distanceSquared < bestDistanceSquared) {
            bestDistanceSquared = distanceSquared;
            bestIndex = index;
        }
    }
    return bestIndex;
}

vec2 getOrbit(int index) {
    return texelFetch(u_referenceOrbit, ivec2(index, 0), 0).xy;
}

float mandelbrotPerturbation(vec2 deltaPixels) {
    vec2 scaledDelta = vec2(0.0);
    for (int i = 0; i < ${MAX_GPU_ITERATIONS}; ++i) {
        if (i >= u_maxIterations) break;
        if (i + 1 >= u_referenceOrbitLength) {
            outMask = 1.0;
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
            outMask = 0.0;
            return smoothValue / float(u_maxIterations);
        }
        float referenceMag2 = dot(nextReference, nextReference);
        if (referenceMag2 > 4.0 && mag2 < (u_glitchThreshold * referenceMag2)) {
            outMask = 1.0;
            return -1.0;
        }
        scaledDelta = nextScaledDelta;
    }
    outMask = 0.0;
    return 0.0;
}

float juliaPerturbation(vec2 deltaPixels) {
    vec2 scaledDelta = deltaPixels;

    for (int i = 0; i < ${MAX_GPU_ITERATIONS}; ++i) {
        if (i >= u_maxIterations) break;
        if (i + 1 >= u_referenceOrbitLength) {
            outMask = 1.0;
            return -1.0;
        }

        vec2 reference = getOrbit(i);
        vec2 nextReference = getOrbit(i + 1);
        vec2 nextScaledDelta = (
            2.0 * complexMul(reference, scaledDelta)
        ) + (
            u_pixelScale * complexMul(scaledDelta, scaledDelta)
        );
        vec2 z = nextReference + (u_pixelScale * nextScaledDelta);
        float mag2 = dot(z, z);
        if (mag2 > 4.0) {
            float smoothValue = float(i + 1) - log2(log2(max(length(z), 1.0001)));
            outMask = 0.0;
            return smoothValue / float(u_maxIterations);
        }
        float referenceMag2 = dot(nextReference, nextReference);
        if (referenceMag2 > 4.0 && mag2 < (u_glitchThreshold * referenceMag2)) {
            outMask = 1.0;
            return -1.0;
        }
        scaledDelta = nextScaledDelta;
    }

    outMask = 0.0;
    return 0.0;
}

vec4 newtonPerturbationColor(vec2 deltaPixels) {
    vec2 scaledDelta = deltaPixels;

    for (int i = 0; i < ${MAX_GPU_ITERATIONS}; ++i) {
        if (i >= u_maxIterations) break;
        if (i + 1 >= u_referenceOrbitLength) {
            outMask = 1.0;
            return vec4(0.0, 0.0, 0.0, 1.0);
        }

        vec2 reference = getOrbit(i);
        vec2 nextReference = getOrbit(i + 1);
        vec2 referenceSquared = complexMul(reference, reference);
        vec2 referenceCubed = complexMul(referenceSquared, reference);
        vec2 referenceFourth = complexMul(referenceSquared, referenceSquared);
        vec2 scaledDeltaSquared = complexMul(scaledDelta, scaledDelta);
        vec2 scaledDeltaCubed = complexMul(scaledDeltaSquared, scaledDelta);
        vec2 shifted = reference + (u_pixelScale * scaledDelta);
        vec2 denominator = 3.0 * complexMul(referenceSquared, complexMul(shifted, shifted));
        if (dot(denominator, denominator) < 1e-18) {
            outMask = 1.0;
            return vec4(0.0, 0.0, 0.0, 1.0);
        }

        vec2 numerator = complexMul((2.0 * referenceFourth) - (2.0 * reference), scaledDelta)
            + complexMul((u_pixelScale * ((4.0 * referenceCubed) - vec2(1.0, 0.0))), scaledDeltaSquared)
            + complexMul(((2.0 * u_pixelScale) * u_pixelScale) * referenceSquared, scaledDeltaCubed);
        vec2 nextScaledDelta = complexDiv(numerator, denominator);
        if (any(greaterThan(abs(nextScaledDelta), vec2(1e12)))) {
            outMask = 1.0;
            return vec4(0.0, 0.0, 0.0, 1.0);
        }

        vec2 z = nextReference + (u_pixelScale * nextScaledDelta);
        if (dot(z, z) > 1e30) {
            outMask = 1.0;
            return vec4(0.0, 0.0, 0.0, 1.0);
        }

        int rootIndex = getNearestNewtonRootIndex(z);
        float distanceSquared = dot(z - getNewtonRoot(rootIndex), z - getNewtonRoot(rootIndex));
        if (distanceSquared < 1e-12) {
            float normalized = float(i + 1) / float(u_maxIterations);
            float fade = 0.25 + (0.75 * pow(1.0 - normalized, 0.35));
            float band = 0.9 + (0.1 * cos((18.0 * normalized) + (float(rootIndex) * 2.0943951)));
            outMask = 0.0;
            return vec4(getNewtonRootColor(rootIndex) * fade * band, 1.0);
        }

        scaledDelta = nextScaledDelta;
    }

    outMask = 0.0;
    return vec4(0.0, 0.0, 0.0, 1.0);
}

void main() {
    vec2 pixelOffset = vec2(
        gl_FragCoord.x - 0.5 * u_resolution.x,
        gl_FragCoord.y - 0.5 * u_resolution.y
    );
    vec2 deltaPixels = u_referenceDeltaPixels + pixelOffset;
    if (u_deepFractalType == 2) {
        outColor = newtonPerturbationColor(deltaPixels);
        return;
    }

    float value = u_deepFractalType == 1
        ? juliaPerturbation(deltaPixels)
        : mandelbrotPerturbation(deltaPixels);
    vec3 color = value <= 0.0 ? vec3(0.0) : getPalette(value);
    outColor = vec4(color, 1.0);
}
`;

const blitFragmentShaderSource = `#version 300 es
precision highp float;

uniform sampler2D u_texture;
uniform vec2 u_resolution;
uniform vec2 u_translationPixels;
uniform float u_scale;

out vec4 outColor;

void main() {
    vec2 halfResolution = 0.5 * u_resolution;
    vec2 sourcePixel = halfResolution
        + u_translationPixels
        + (u_scale * (gl_FragCoord.xy - halfResolution));
    vec2 uv = sourcePixel / u_resolution;
    outColor = texture(u_texture, uv);
}
`;

const maskReduceFragmentShaderSource = `#version 300 es
precision highp float;

uniform sampler2D u_texture;
uniform ivec2 u_sourceOffset;
uniform ivec2 u_sourceSize;

layout(location = 0) out float outMask;

void main() {
    ivec2 outputPixel = ivec2(gl_FragCoord.xy);
    ivec2 maxSource = u_sourceOffset + u_sourceSize - ivec2(1);
    ivec2 base = u_sourceOffset + (outputPixel * ${MASK_REDUCTION_FACTOR});
    float reduced = 0.0;

    for (int offsetY = 0; offsetY < ${MASK_REDUCTION_FACTOR}; ++offsetY) {
        for (int offsetX = 0; offsetX < ${MASK_REDUCTION_FACTOR}; ++offsetX) {
            ivec2 sourcePixel = min(base + ivec2(offsetX, offsetY), maxSource);
            reduced = max(reduced, texelFetch(u_texture, sourcePixel, 0).r);
        }
    }

    outMask = reduced;
}
`;

let simpleProgramInfo;
let deepProgramInfo;
let blitProgramInfo;
let maskReduceProgramInfo;
let orbitTexture;
let orbitTextureCapacity = 0;
let mandelbrotWorkingFramebuffer;
let mandelbrotWorkingColorTexture;
let mandelbrotWorkingMaskTexture;
let mandelbrotCommittedFramebuffer;
let mandelbrotCommittedColorTexture;
let mandelbrotCommittedCamera = null;
let juliaCommittedFramebuffer;
let juliaCommittedColorTexture;
let juliaCommittedCamera = null;
let newtonCommittedFramebuffer;
let newtonCommittedColorTexture;
let newtonCommittedCamera = null;
let maskReduceFramebufferA;
let maskReduceFramebufferB;
let maskReduceTextureA;
let maskReduceTextureB;
let mandelbrotFrameReady = false;
let mandelbrotCommittedFrameAvailable = false;
let mandelbrotReferenceCache = new Map();
let mandelbrotRenderTargetWidth = 0;
let mandelbrotRenderTargetHeight = 0;
let mandelbrotZoomStepCount = 0;
let mandelbrotLastFrameStats = null;
let mandelbrotMaskVerificationFramesRemaining = 0;
let mandelbrotStableReuseFrames = 0;
let juliaFrameReady = false;
let juliaCommittedFrameAvailable = false;
let juliaReferenceCache = new Map();
let juliaZoomStepCount = 0;
let juliaLastFrameStats = null;
let juliaMaskVerificationFramesRemaining = 0;
let juliaStableReuseFrames = 0;
let newtonFrameReady = false;
let newtonCommittedFrameAvailable = false;
let newtonReferenceCache = new Map();
let newtonZoomStepCount = 0;
let newtonLastFrameStats = null;
let newtonMaskVerificationFramesRemaining = 0;
let newtonStableReuseFrames = 0;
let newtonDeepRenderActivationScale = NEWTON_DEEP_RENDER_SCALE;
let juliaConstantCache = null;
let juliaConstantCacheDigits = 0;
let newtonRootsCache = null;
let newtonRootsCacheDigits = 0;
let newtonConvergenceDistanceSquared = 0n;
let newtonConvergenceDigits = 0;
let zoomSpeedDecimalCache = null;
let zoomSpeedDecimalCacheDigits = 0;

let simpleCamera = createSimpleCamera(-0.745, 0.1, 1.6);
let mandelbrotCamera = null;
let mandelbrotReference = createEmptyReference();
let juliaCamera = null;
let juliaReference = createEmptyReference();
let newtonCamera = null;
let newtonReference = createEmptyReference();
let mandelbrotDeepWorkState = null;
let juliaDeepWorkState = null;
let newtonDeepWorkState = null;
let debugHeartbeatEnabled = false;
let debugHeartbeatIntervalMs = DEBUG_HEARTBEAT_INTERVAL_MS;
let debugHeartbeatTimer = null;
let debugAnimationFrameCount = 0;
let debugLastAnimationFrameAt = 0;
let debugLastAnimationStage = 'not-started';
let debugLastObservedRenderPath = null;
let debugLastRenderedPath = null;
let newtonDeferredCommittedFramePending = false;
let activeDeepRenderTask = null;
let deepRenderWorkHandle = null;
let deepRenderWorkHandleIsIdleCallback = false;
const referenceRingOffsetsCache = new Map();

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

function cloneDeepCamera(camera) {
    return cloneMandelbrotCamera(camera);
}

function cloneDeepReference(reference) {
    return cloneMandelbrotReference(reference);
}

function isDeepFractalType(type = fractalType) {
    return type === 'mandelbrot' || type === 'julia' || type === 'newton';
}

function getDeepLabel(type) {
    if (type === 'julia') {
        return 'Julia';
    }
    if (type === 'newton') {
        return 'Newton';
    }
    return 'Mandelbrot';
}

function getDeepConstant(type) {
    if (type === 'julia') {
        if (!juliaConstantCache || juliaConstantCacheDigits !== decimalDigits) {
            juliaConstantCache = {
                x: decimalFromString(JULIA_CONSTANT_REAL),
                y: decimalFromString(JULIA_CONSTANT_IMAGINARY),
            };
            juliaConstantCacheDigits = decimalDigits;
        }
        return juliaConstantCache;
    }
    return null;
}

function getNewtonRoots() {
    if (!newtonRootsCache || newtonRootsCacheDigits !== decimalDigits) {
        newtonRootsCache = [
            { x: decimalFromString('1'), y: 0n },
            { x: decimalFromString('-0.5'), y: decimalFromString(NEWTON_ROOT_IMAGINARY) },
            { x: decimalFromString('-0.5'), y: decimalFromString(`-${NEWTON_ROOT_IMAGINARY}`) },
        ];
        newtonRootsCacheDigits = decimalDigits;
    }
    return newtonRootsCache;
}

function getNewtonConvergenceDistanceSquared() {
    if (newtonConvergenceDigits !== decimalDigits) {
        newtonConvergenceDistanceSquared = decimalFromString(NEWTON_CONVERGENCE_DISTANCE_SQUARED);
        newtonConvergenceDigits = decimalDigits;
    }
    return newtonConvergenceDistanceSquared;
}

function getZoomSpeedDecimal() {
    if (zoomSpeedDecimalCache === null || zoomSpeedDecimalCacheDigits !== decimalDigits) {
        zoomSpeedDecimalCache = decimalFromString(String(ZOOM_SPEED));
        zoomSpeedDecimalCacheDigits = decimalDigits;
    }
    return zoomSpeedDecimalCache;
}

function getDeepCamera(type = fractalType) {
    if (type === 'julia') {
        return juliaCamera;
    }
    if (type === 'newton') {
        return newtonCamera;
    }
    return mandelbrotCamera;
}

function setDeepCamera(type, camera) {
    if (type === 'julia') {
        juliaCamera = camera;
        return;
    }
    if (type === 'newton') {
        newtonCamera = camera;
        return;
    }
    mandelbrotCamera = camera;
}

function getDeepReference(type = fractalType) {
    if (type === 'julia') {
        return juliaReference;
    }
    if (type === 'newton') {
        return newtonReference;
    }
    return mandelbrotReference;
}

function setDeepReference(type, reference) {
    if (type === 'julia') {
        juliaReference = reference;
        return;
    }
    if (type === 'newton') {
        newtonReference = reference;
        return;
    }
    mandelbrotReference = reference;
}

function getDeepReferenceCache(type = fractalType) {
    if (type === 'julia') {
        return juliaReferenceCache;
    }
    if (type === 'newton') {
        return newtonReferenceCache;
    }
    return mandelbrotReferenceCache;
}

function setDeepReferenceCache(type, cache) {
    if (type === 'julia') {
        juliaReferenceCache = cache;
        return;
    }
    if (type === 'newton') {
        newtonReferenceCache = cache;
        return;
    }
    mandelbrotReferenceCache = cache;
}

function getDeepCommittedFramebuffer(type = fractalType) {
    if (type === 'julia') {
        return juliaCommittedFramebuffer;
    }
    if (type === 'newton') {
        return newtonCommittedFramebuffer;
    }
    return mandelbrotCommittedFramebuffer;
}

function setDeepCommittedFramebuffer(type, framebuffer) {
    if (type === 'julia') {
        juliaCommittedFramebuffer = framebuffer;
        return;
    }
    if (type === 'newton') {
        newtonCommittedFramebuffer = framebuffer;
        return;
    }
    mandelbrotCommittedFramebuffer = framebuffer;
}

function getDeepCommittedColorTexture(type = fractalType) {
    if (type === 'julia') {
        return juliaCommittedColorTexture;
    }
    if (type === 'newton') {
        return newtonCommittedColorTexture;
    }
    return mandelbrotCommittedColorTexture;
}

function setDeepCommittedColorTexture(type, texture) {
    if (type === 'julia') {
        juliaCommittedColorTexture = texture;
        return;
    }
    if (type === 'newton') {
        newtonCommittedColorTexture = texture;
        return;
    }
    mandelbrotCommittedColorTexture = texture;
}

function getDeepCommittedCamera(type = fractalType) {
    if (type === 'julia') {
        return juliaCommittedCamera;
    }
    if (type === 'newton') {
        return newtonCommittedCamera;
    }
    return mandelbrotCommittedCamera;
}

function setDeepCommittedCamera(type, camera) {
    if (type === 'julia') {
        juliaCommittedCamera = camera;
        return;
    }
    if (type === 'newton') {
        newtonCommittedCamera = camera;
        return;
    }
    mandelbrotCommittedCamera = camera;
}

function getDeepCommittedFrameAvailable(type = fractalType) {
    if (type === 'julia') {
        return juliaCommittedFrameAvailable;
    }
    if (type === 'newton') {
        return newtonCommittedFrameAvailable;
    }
    return mandelbrotCommittedFrameAvailable;
}

function setDeepCommittedFrameAvailable(type, value) {
    if (type === 'julia') {
        juliaCommittedFrameAvailable = value;
        return;
    }
    if (type === 'newton') {
        newtonCommittedFrameAvailable = value;
        return;
    }
    mandelbrotCommittedFrameAvailable = value;
}

function getDeepFrameReady(type = fractalType) {
    if (type === 'julia') {
        return juliaFrameReady;
    }
    if (type === 'newton') {
        return newtonFrameReady;
    }
    return mandelbrotFrameReady;
}

function setDeepFrameReady(type, value) {
    if (type === 'julia') {
        juliaFrameReady = value;
        return;
    }
    if (type === 'newton') {
        newtonFrameReady = value;
        return;
    }
    mandelbrotFrameReady = value;
}

function getDeepQualityHold(type = fractalType) {
    if (type === 'julia') {
        return juliaQualityHold;
    }
    if (type === 'newton') {
        return newtonQualityHold;
    }
    return mandelbrotQualityHold;
}

function setDeepQualityHold(type, value) {
    if (type === 'julia') {
        juliaQualityHold = value;
        return;
    }
    if (type === 'newton') {
        newtonQualityHold = value;
        return;
    }
    mandelbrotQualityHold = value;
}

function getDeepQualityHoldWarningShown(type = fractalType) {
    if (type === 'julia') {
        return juliaQualityHoldWarningShown;
    }
    if (type === 'newton') {
        return newtonQualityHoldWarningShown;
    }
    return mandelbrotQualityHoldWarningShown;
}

function setDeepQualityHoldWarningShown(type, value) {
    if (type === 'julia') {
        juliaQualityHoldWarningShown = value;
        return;
    }
    if (type === 'newton') {
        newtonQualityHoldWarningShown = value;
        return;
    }
    mandelbrotQualityHoldWarningShown = value;
}

function getDeepPrecisionWarningShown(type = fractalType) {
    if (type === 'julia') {
        return juliaDeepPrecisionWarningShown;
    }
    if (type === 'newton') {
        return newtonDeepPrecisionWarningShown;
    }
    return deepPrecisionWarningShown;
}

function setDeepPrecisionWarningShown(type, value) {
    if (type === 'julia') {
        juliaDeepPrecisionWarningShown = value;
        return;
    }
    if (type === 'newton') {
        newtonDeepPrecisionWarningShown = value;
        return;
    }
    deepPrecisionWarningShown = value;
}

function getDeepZoomStepCount(type = fractalType) {
    if (type === 'julia') {
        return juliaZoomStepCount;
    }
    if (type === 'newton') {
        return newtonZoomStepCount;
    }
    return mandelbrotZoomStepCount;
}

function setDeepZoomStepCount(type, value) {
    if (type === 'julia') {
        juliaZoomStepCount = value;
        return;
    }
    if (type === 'newton') {
        newtonZoomStepCount = value;
        return;
    }
    mandelbrotZoomStepCount = value;
}

function incrementDeepZoomStepCount(type) {
    setDeepZoomStepCount(type, getDeepZoomStepCount(type) + 1);
}

function getDeepLastFrameStats(type = fractalType) {
    if (type === 'julia') {
        return juliaLastFrameStats;
    }
    if (type === 'newton') {
        return newtonLastFrameStats;
    }
    return mandelbrotLastFrameStats;
}

function setDeepLastFrameStats(type, frameStats) {
    if (type === 'julia') {
        juliaLastFrameStats = frameStats;
        return;
    }
    if (type === 'newton') {
        newtonLastFrameStats = frameStats;
        return;
    }
    mandelbrotLastFrameStats = frameStats;
}

function getDeepMaskVerificationFramesRemaining(type = fractalType) {
    if (type === 'julia') {
        return juliaMaskVerificationFramesRemaining;
    }
    if (type === 'newton') {
        return newtonMaskVerificationFramesRemaining;
    }
    return mandelbrotMaskVerificationFramesRemaining;
}

function setDeepMaskVerificationFramesRemaining(type, value) {
    if (type === 'julia') {
        juliaMaskVerificationFramesRemaining = value;
        return;
    }
    if (type === 'newton') {
        newtonMaskVerificationFramesRemaining = value;
        return;
    }
    mandelbrotMaskVerificationFramesRemaining = value;
}

function getDeepStableReuseFrames(type = fractalType) {
    if (type === 'julia') {
        return juliaStableReuseFrames;
    }
    if (type === 'newton') {
        return newtonStableReuseFrames;
    }
    return mandelbrotStableReuseFrames;
}

function setDeepStableReuseFrames(type, value) {
    if (type === 'julia') {
        juliaStableReuseFrames = value;
        return;
    }
    if (type === 'newton') {
        newtonStableReuseFrames = value;
        return;
    }
    mandelbrotStableReuseFrames = value;
}

function incrementDeepStableReuseFrames(type) {
    const nextValue = getDeepStableReuseFrames(type) + 1;
    setDeepStableReuseFrames(type, nextValue);
    return nextValue;
}

function getAdaptiveMaskVerifySkipFrames(type = fractalType) {
    return Math.min(
        MAX_STABLE_MASK_VERIFY_SKIP_FRAMES,
        STABLE_MASK_VERIFY_SKIP_FRAMES
        + Math.floor(getDeepStableReuseFrames(type) / STABLE_MASK_VERIFY_GROWTH_INTERVAL)
    );
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
        schedulerYields: 0,
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

function logInfo(message) {
    if (typeof console.info === 'function') {
        console.info(message);
    } else if (typeof console.log === 'function') {
        console.log(message);
    }
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

function finalizeDeepFrameStats(type, frameStats, status, reason = null, extra = {}) {
    const finalized = {
        ...frameStats,
        ...extra,
        status,
        reason,
        repairPasses: Math.max(0, frameStats.referencesUsed - 1),
    };
    setDeepLastFrameStats(type, finalized);
    return finalized;
}

function finalizeMandelbrotFrameStats(frameStats, status, reason = null, extra = {}) {
    return finalizeDeepFrameStats('mandelbrot', frameStats, status, reason, extra);
}

function getDeepWorkState(type) {
    if (type === 'julia') {
        return juliaDeepWorkState;
    }
    if (type === 'newton') {
        return newtonDeepWorkState;
    }
    return mandelbrotDeepWorkState;
}

function setDeepWorkState(type, workState) {
    if (type === 'julia') {
        juliaDeepWorkState = workState;
        return;
    }
    if (type === 'newton') {
        newtonDeepWorkState = workState;
        return;
    }
    mandelbrotDeepWorkState = workState;
}

function sanitizeDeepWorkState(workState) {
    if (!workState) {
        return null;
    }

    return {
        stage: workState.stage || null,
        initialReferenceMode: workState.initialReferenceMode || null,
        referencesUsed: workState.referencesUsed ?? 0,
        repairQueueLength: workState.repairQueueLength ?? 0,
        repairTilesProcessed: workState.repairTilesProcessed ?? 0,
        cpuTilesResolved: workState.cpuTilesResolved ?? 0,
        elapsedMs: workState.startedAtMs
            ? Math.max(0, Date.now() - workState.startedAtMs)
            : null,
        currentTile: workState.currentTile
            ? {
                width: workState.currentTile.width,
                height: workState.currentTile.height,
                depth: workState.currentTile.depth,
            }
            : null,
        note: workState.note || null,
    };
}

function updateDeepWorkState(type, patch, reason = null) {
    const current = getDeepWorkState(type) || {};
    const next = {
        ...current,
        ...patch,
        startedAtMs: current.startedAtMs || Date.now(),
        currentTile: patch.currentTile === undefined ? current.currentTile : patch.currentTile,
    };
    setDeepWorkState(type, next);

    if (debugHeartbeatEnabled && reason) {
        logInfo(`Fractal deep work ${JSON.stringify({ reason, mode: type, ...sanitizeDeepWorkState(next) })}`);
    }

    return next;
}

function clearDeepWorkState(type, reason = null) {
    const previous = getDeepWorkState(type);
    if (debugHeartbeatEnabled && reason && previous) {
        logInfo(`Fractal deep work ${JSON.stringify({ reason, mode: type, ...sanitizeDeepWorkState(previous) })}`);
    }
    setDeepWorkState(type, null);
}

function deferNewtonDeepRender(scaleApprox) {
    if (!Number.isFinite(scaleApprox) || scaleApprox <= 0) {
        return newtonDeepRenderActivationScale;
    }

    const precisionFloor = getNewtonSimpleProxyPrecisionFloor(getDeepCamera('newton'));
    const deferredScale = Math.max(
        precisionFloor,
        Math.min(
            NEWTON_DEEP_RENDER_SCALE,
            scaleApprox * NEWTON_DEEP_RETRY_SCALE_FACTOR
        )
    );
    newtonDeepRenderActivationScale = Math.max(
        precisionFloor,
        Math.min(newtonDeepRenderActivationScale, deferredScale)
    );
    return newtonDeepRenderActivationScale;
}

function resetNewtonDeepRenderActivationScale() {
    newtonDeepRenderActivationScale = NEWTON_DEEP_RENDER_SCALE;
}

function getNewtonSimpleProxyPrecisionFloor(camera) {
    if (!camera) {
        return NEWTON_DEEP_RENDER_SCALE;
    }
    const centerMagnitude = Math.max(
        1,
        Math.abs(decimalToNumber(camera.centerX)),
        Math.abs(decimalToNumber(camera.centerY))
    );
    return centerMagnitude * FLOAT32_EPSILON * 4;
}

function canUseAccurateNewtonSimpleProxy(camera) {
    return Boolean(camera) && camera.pixelScaleApprox >= getNewtonSimpleProxyPrecisionFloor(camera);
}

function shouldDeferNewtonRepairWork(referencesUsed, repairQueueLength, elapsedMs) {
    if (repairQueueLength >= NEWTON_PATHOLOGICAL_REPAIR_QUEUE_LENGTH) {
        return true;
    }
    if (
        repairQueueLength >= NEWTON_PATHOLOGICAL_REPAIR_QUEUE_MIN
        && elapsedMs >= NEWTON_PATHOLOGICAL_REPAIR_ELAPSED_MS
    ) {
        return true;
    }
    if (
        referencesUsed >= NEWTON_PATHOLOGICAL_REPAIR_REFERENCE_LIMIT
        && repairQueueLength >= NEWTON_PATHOLOGICAL_REPAIR_QUEUE_MIN
    ) {
        return true;
    }
    return false;
}

function shouldDeferNewtonDeepFailure(reason) {
    return reason === 'pathological_initial_repair_flood'
        || reason === 'pathological_repair_growth';
}

function getNewtonSimpleProxyMaxIterations(camera) {
    if (newtonDeepRenderActivationScale >= NEWTON_DEEP_RENDER_SCALE) {
        return camera.maxIterations;
    }
    return Math.min(
        camera.maxIterations,
        computeNewtonSimplePreviewIterationBudget(camera.pixelScaleApprox)
    );
}

function getDeepDebugSnapshot(type) {
    const camera = getDeepCamera(type);
    return {
        step: getDeepZoomStepCount(type),
        pixelScaleApprox: camera ? roundDebugNumber(camera.pixelScaleApprox) : null,
        maxIterations: camera ? camera.maxIterations : null,
        hold: getDeepQualityHold(type),
        frameReady: getDeepFrameReady(type),
        maskVerificationFramesRemaining: getDeepMaskVerificationFramesRemaining(type),
        stableReuseFrames: getDeepStableReuseFrames(type),
        mouseX: roundDebugNumber(mousePosition.x),
        mouseY: roundDebugNumber(mousePosition.y),
        lastFrame: getDeepLastFrameStats(type) || createEmptyMandelbrotFrameStats(),
        activeWork: sanitizeDeepWorkState(getDeepWorkState(type)),
    };
}

function getMandelbrotDebugSnapshot() {
    return getDeepDebugSnapshot('mandelbrot');
}

function predictActiveRenderPath(type = fractalType) {
    if (type === 'newton' && newtonDeferredCommittedFramePending) {
        return getDeepCommittedFrameAvailable('newton') ? 'deep' : 'blank';
    }
    if (isDeepFractalType(type) && shouldUseDeepRender(type)) {
        return 'deep';
    }
    if (type === 'newton') {
        return 'simple-proxy';
    }
    return 'simple';
}

function getActiveRenderPath(type = fractalType) {
    if (type === fractalType && debugLastRenderedPath !== null) {
        return debugLastRenderedPath;
    }
    return predictActiveRenderPath(type);
}

function noteRenderedPath(path) {
    debugLastRenderedPath = path;
}

function getActiveDebugSnapshot(type = fractalType) {
    const renderPath = getActiveRenderPath(type);
    const deepCamera = isDeepFractalType(type) ? getDeepCamera(type) : null;
    const activeCamera = deepCamera || simpleCamera;
    const snapshot = {
        timestamp: new Date().toISOString(),
        mode: type,
        renderPath,
        rafTick: debugAnimationFrameCount,
        lastAnimationStage: debugLastAnimationStage,
        msSinceLastFrame: debugLastAnimationFrameAt
            ? Math.max(0, Date.now() - debugLastAnimationFrameAt)
            : null,
        mouseX: roundDebugNumber(mousePosition.x),
        mouseY: roundDebugNumber(mousePosition.y),
        devicePixelRatio: typeof window !== 'undefined' ? roundDebugNumber(Number(window.devicePixelRatio) || 1) : 1,
        renderDevicePixelRatio: roundDebugNumber(getRenderDevicePixelRatio()),
        canvasWidth: gl?.canvas?.width ?? null,
        canvasHeight: gl?.canvas?.height ?? null,
        pixelScaleApprox: activeCamera
            ? roundDebugNumber(
                typeof activeCamera.pixelScaleApprox === 'number'
                    ? activeCamera.pixelScaleApprox
                    : activeCamera.pixelScale
            )
            : null,
        maxIterations: activeCamera ? activeCamera.maxIterations : null,
    };

    if (isDeepFractalType(type)) {
        const lastFrame = getDeepLastFrameStats(type) || createEmptyMandelbrotFrameStats();
        snapshot.step = getDeepZoomStepCount(type);
        snapshot.hold = getDeepQualityHold(type);
        snapshot.frameReady = getDeepFrameReady(type);
        snapshot.maskVerificationFramesRemaining = getDeepMaskVerificationFramesRemaining(type);
        snapshot.stableReuseFrames = getDeepStableReuseFrames(type);
        snapshot.lastFrame = {
            status: lastFrame.status,
            reason: lastFrame.reason,
            referencesUsed: lastFrame.referencesUsed,
            repairPasses: lastFrame.repairPasses,
            schedulerYields: lastFrame.schedulerYields,
            cpuResolvedTiles: lastFrame.cpuResolvedTiles,
            cpuResolvedPixels: lastFrame.cpuResolvedPixels,
        };
        snapshot.activeWork = sanitizeDeepWorkState(getDeepWorkState(type));
    }

    if (type === 'newton' && deepCamera) {
        snapshot.deepEligible = shouldUseDeepRender(type);
        snapshot.deepRenderScale = NEWTON_DEEP_RENDER_SCALE;
        snapshot.deepActivationScale = roundDebugNumber(newtonDeepRenderActivationScale);
        snapshot.previewMaxIterations = renderPath === 'simple-proxy'
            ? getNewtonSimpleProxyMaxIterations(deepCamera)
            : null;
    }

    return snapshot;
}

function clampDebugHeartbeatInterval(intervalMs) {
    if (!Number.isFinite(intervalMs)) {
        return DEBUG_HEARTBEAT_INTERVAL_MS;
    }
    return Math.max(MIN_DEBUG_HEARTBEAT_INTERVAL_MS, Math.floor(intervalMs));
}

function stopDebugHeartbeatLogging() {
    if (debugHeartbeatTimer !== null && typeof clearInterval === 'function') {
        clearInterval(debugHeartbeatTimer);
    }
    debugHeartbeatTimer = null;
}

function emitDebugHeartbeat(reason = 'tick') {
    if (!debugHeartbeatEnabled) {
        return;
    }
    logInfo(`Fractal heartbeat ${JSON.stringify({ reason, ...getActiveDebugSnapshot() })}`);
}

function noteRenderPathChange(reason = 'render-path-change') {
    const renderPath = getActiveRenderPath();
    if (renderPath === debugLastObservedRenderPath) {
        return;
    }
    debugLastObservedRenderPath = renderPath;
    emitDebugHeartbeat(reason);
}

function setDebugHeartbeatEnabled(enabled, intervalMs = debugHeartbeatIntervalMs) {
    debugHeartbeatIntervalMs = clampDebugHeartbeatInterval(intervalMs);
    stopDebugHeartbeatLogging();
    debugHeartbeatEnabled = Boolean(enabled);
    debugLastObservedRenderPath = null;

    if (!debugHeartbeatEnabled) {
        return false;
    }

    if (typeof setInterval === 'function') {
        debugHeartbeatTimer = setInterval(() => {
            emitDebugHeartbeat();
        }, debugHeartbeatIntervalMs);
    }

    emitDebugHeartbeat('enabled');
    noteRenderPathChange('initial-render-path');
    return true;
}

function parseDebugHeartbeatBoolean(value) {
    if (value == null) {
        return null;
    }
    const normalized = String(value).trim().toLowerCase();
    return !['0', 'false', 'off', 'no'].includes(normalized);
}

function configureDebugHeartbeatFromQuery() {
    if (typeof window === 'undefined' || !window.location || typeof URLSearchParams !== 'function') {
        return;
    }

    const params = new URLSearchParams(window.location.search || '');
    const enabled = parseDebugHeartbeatBoolean(params.get('debugHeartbeat'));
    const intervalParam = params.get('debugHeartbeatMs');
    const intervalMs = intervalParam === null ? Number.NaN : Number(intervalParam);

    if (enabled === null && !Number.isFinite(intervalMs)) {
        return;
    }

    setDebugHeartbeatEnabled(enabled !== null ? enabled : true, intervalMs);
}

function installDebugControls() {
    if (typeof window === 'undefined') {
        return;
    }

    window.__fractalDebug = {
        enableHeartbeatLogs(intervalMs = debugHeartbeatIntervalMs) {
            setDebugHeartbeatEnabled(true, intervalMs);
            return getActiveDebugSnapshot();
        },
        disableHeartbeatLogs() {
            setDebugHeartbeatEnabled(false);
        },
        getSnapshot() {
            return getActiveDebugSnapshot();
        },
    };
}

function logDeepProgress(type) {
    if (
        fractalType !== type
        || !MANDELBROT_LOG_INTERVAL
        || getDeepZoomStepCount(type) === 0
        || (getDeepZoomStepCount(type) % MANDELBROT_LOG_INTERVAL) !== 0
    ) {
        return;
    }

    logInfo(`${getDeepLabel(type)} zoom progress ${JSON.stringify(getDeepDebugSnapshot(type))}`);
}

function logMandelbrotProgress() {
    logDeepProgress('mandelbrot');
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

function nowMs() {
    if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
        return performance.now();
    }
    return Date.now();
}

function getRenderDevicePixelRatio() {
    const devicePixelRatio = typeof window !== 'undefined' ? Number(window.devicePixelRatio) : 1;
    if (!Number.isFinite(devicePixelRatio) || devicePixelRatio <= 0) {
        return 1;
    }
    return Math.min(MAX_RENDER_DEVICE_PIXEL_RATIO, Math.max(1, devicePixelRatio));
}

function areDeepCamerasEquivalent(a, b) {
    return Boolean(a && b)
        && a.centerX === b.centerX
        && a.centerY === b.centerY
        && a.pixelScale === b.pixelScale
        && a.pixelScaleApprox === b.pixelScaleApprox
        && a.maxIterations === b.maxIterations
        && a.viewWidth === b.viewWidth;
}

function getDeepCameraScaleRatio(a, b) {
    if (!a || !b || !Number.isFinite(a.pixelScaleApprox) || !Number.isFinite(b.pixelScaleApprox)) {
        return Number.POSITIVE_INFINITY;
    }
    const ratio = a.pixelScaleApprox / b.pixelScaleApprox;
    if (!Number.isFinite(ratio) || ratio <= 0) {
        return Number.POSITIVE_INFINITY;
    }
    return Math.max(ratio, 1 / ratio);
}

function getDeepCameraTranslationPixels(a, b) {
    if (!a || !b || !Number.isFinite(a.pixelScaleApprox) || a.pixelScaleApprox <= 0) {
        return Number.POSITIVE_INFINITY;
    }

    const dx = decimalToNumber(subDecimal(b.centerX, a.centerX)) / a.pixelScaleApprox;
    const dy = decimalToNumber(subDecimal(b.centerY, a.centerY)) / a.pixelScaleApprox;
    return Math.hypot(dx, dy);
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

function divDecimal(a, b) {
    if (b === 0n) {
        return 0n;
    }
    return (a * decimalScale) / b;
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

    if (juliaCamera) {
        juliaCamera.centerX *= factor;
        juliaCamera.centerY *= factor;
        juliaCamera.pixelScale *= factor;
    }

    if (newtonCamera) {
        newtonCamera.centerX *= factor;
        newtonCamera.centerY *= factor;
        newtonCamera.pixelScale *= factor;
    }

    if (mandelbrotReference.centerX !== null) {
        mandelbrotReference.centerX *= factor;
        mandelbrotReference.centerY *= factor;
    }

    if (juliaReference.centerX !== null) {
        juliaReference.centerX *= factor;
        juliaReference.centerY *= factor;
    }

    if (newtonReference.centerX !== null) {
        newtonReference.centerX *= factor;
        newtonReference.centerY *= factor;
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

function computeNewtonIterationBudget(pixelScale) {
    const depth = Math.max(0, -Math.log10(Math.max(pixelScale, Number.MIN_VALUE)));
    return Math.min(384, Math.max(64, Math.floor(64 + depth * 8)));
}

function computeNewtonSimplePreviewIterationBudget(pixelScale) {
    const depth = Math.max(0, -Math.log10(Math.max(pixelScale, Number.MIN_VALUE)));
    return Math.min(48, Math.max(24, Math.floor(24 + depth * 2)));
}

function computeDeepIterationBudget(type, pixelScale) {
    if (type === 'newton') {
        return computeNewtonIterationBudget(pixelScale);
    }
    return computeIterationBudget(pixelScale);
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

function createDeepCamera(centerX, centerY, viewWidth, type = 'mandelbrot') {
    const minDimension = Math.max(1, Math.min(gl.canvas.width, gl.canvas.height));
    const pixelScaleApprox = viewWidth / minDimension;
    ensureDecimalDigits(requiredDecimalDigits(pixelScaleApprox));

    return {
        centerX: decimalFromString(centerX),
        centerY: decimalFromString(centerY),
        pixelScale: decimalFromNumber(pixelScaleApprox),
        pixelScaleApprox,
        maxIterations: computeDeepIterationBudget(type, pixelScaleApprox),
        viewWidth,
    };
}

function createMandelbrotCamera() {
    return createDeepCamera('-0.745', '0.1', 1.6, 'mandelbrot');
}

function createJuliaCamera() {
    return createDeepCamera('0', '0', 3.0, 'julia');
}

function createNewtonCamera() {
    return createDeepCamera('-1.2', '0.1', 2.4, 'newton');
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
    if (isDeepFractalType(fractalType)) {
        const plane = screenToPlaneDeep(getDeepCamera(fractalType), pointer);
        return {
            x: decimalToNumber(plane.x),
            y: decimalToNumber(plane.y),
        };
    }

    return screenToPlaneSimple(simpleCamera, pointer);
}

function shouldUseDeepRender(type = fractalType) {
    if (!isDeepFractalType(type)) {
        return false;
    }
    if (type === 'newton') {
        const camera = getDeepCamera(type);
        return camera && camera.pixelScaleApprox <= newtonDeepRenderActivationScale;
    }
    return true;
}

function stepActiveZoomOnce() {
    if (isDeepFractalType(fractalType) && shouldUseDeepRender(fractalType)) {
        stepDeepCameraWithQualityPriority(fractalType);
        return;
    }

    if (fractalType === 'newton') {
        updateNewtonCamera();
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

function updateDeepCamera(camera, type = 'mandelbrot') {
    const nextPixelScaleApprox = camera.pixelScaleApprox * ZOOM_SPEED;
    ensureDecimalDigits(requiredDecimalDigits(nextPixelScaleApprox));

    const anchor = screenToPlaneDeep(camera, mousePosition);
    const offset = getCanvasPixelOffset(mousePosition);
    const nextPixelScale = mulDecimal(camera.pixelScale, getZoomSpeedDecimal());

    camera.centerX = subDecimal(anchor.x, mulDecimalNumber(nextPixelScale, offset.x));
    camera.centerY = subDecimal(anchor.y, mulDecimalNumber(nextPixelScale, offset.y));
    camera.pixelScale = nextPixelScale;
    camera.pixelScaleApprox = nextPixelScaleApprox;
    camera.viewWidth = nextPixelScaleApprox * Math.min(gl.canvas.width, gl.canvas.height);
    camera.maxIterations = computeDeepIterationBudget(type, nextPixelScaleApprox);
}

function updateMandelbrotCamera() {
    updateDeepCamera(mandelbrotCamera, 'mandelbrot');
}

function updateJuliaCamera() {
    updateDeepCamera(juliaCamera, 'julia');
}

function updateNewtonCamera() {
    updateDeepCamera(newtonCamera, 'newton');
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

function createTexture(
    width,
    height,
    internalFormat,
    format,
    type,
    minFilter = gl.NEAREST,
    magFilter = minFilter
) {
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, minFilter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, magFilter);
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

function destroyDeepCommittedRenderTarget(type) {
    const framebuffer = getDeepCommittedFramebuffer(type);
    const colorTexture = getDeepCommittedColorTexture(type);
    if (framebuffer) {
        gl.deleteFramebuffer(framebuffer);
    }
    if (colorTexture) {
        gl.deleteTexture(colorTexture);
    }
    setDeepCommittedFramebuffer(type, null);
    setDeepCommittedColorTexture(type, null);
    setDeepCommittedCamera(type, null);
    setDeepCommittedFrameAvailable(type, false);
}

function blitColorFramebuffer(
    readFramebuffer,
    drawFramebuffer,
    sourceWidth,
    sourceHeight,
    targetWidth = gl.canvas.width,
    targetHeight = gl.canvas.height
) {
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, readFramebuffer);
    gl.readBuffer(gl.COLOR_ATTACHMENT0);
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, drawFramebuffer);
    gl.blitFramebuffer(
        0,
        0,
        sourceWidth,
        sourceHeight,
        0,
        0,
        targetWidth,
        targetHeight,
        gl.COLOR_BUFFER_BIT,
        gl.NEAREST
    );
}

function ensureDeepCommittedRenderTarget(type, width, height) {
    if (getDeepCommittedFramebuffer(type) && getDeepCommittedColorTexture(type)) {
        return;
    }

    const committedColorTexture = createTexture(
        width,
        height,
        gl.RGBA8,
        gl.RGBA,
        gl.UNSIGNED_BYTE,
        gl.LINEAR,
        gl.LINEAR
    );
    const committedFramebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, committedFramebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, committedColorTexture, 0);
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
        throw new Error(`${getDeepLabel(type)} committed framebuffer is incomplete.`);
    }

    setDeepCommittedColorTexture(type, committedColorTexture);
    setDeepCommittedFramebuffer(type, committedFramebuffer);
    setDeepCommittedFrameAvailable(type, false);
}

function destroyAllDeepCommittedRenderTargets() {
    destroyDeepCommittedRenderTarget('mandelbrot');
    destroyDeepCommittedRenderTarget('julia');
    destroyDeepCommittedRenderTarget('newton');
}

function ensureMandelbrotRenderTargets(type = fractalType) {
    if (!gl) {
        return;
    }

    const width = gl.canvas.width;
    const height = gl.canvas.height;
    if (
        mandelbrotWorkingColorTexture
        && mandelbrotWorkingMaskTexture
        && mandelbrotRenderTargetWidth === width
        && mandelbrotRenderTargetHeight === height
    ) {
        ensureDeepCommittedRenderTarget('mandelbrot', width, height);
        ensureDeepCommittedRenderTarget('julia', width, height);
        ensureDeepCommittedRenderTarget('newton', width, height);
        return;
    }

    const previousWidth = mandelbrotRenderTargetWidth;
    const previousHeight = mandelbrotRenderTargetHeight;
    const previousCommittedTargets = {
        mandelbrot: {
            framebuffer: getDeepCommittedFramebuffer('mandelbrot'),
            colorTexture: getDeepCommittedColorTexture('mandelbrot'),
            available: getDeepCommittedFrameAvailable('mandelbrot'),
        },
        julia: {
            framebuffer: getDeepCommittedFramebuffer('julia'),
            colorTexture: getDeepCommittedColorTexture('julia'),
            available: getDeepCommittedFrameAvailable('julia'),
        },
        newton: {
            framebuffer: getDeepCommittedFramebuffer('newton'),
            colorTexture: getDeepCommittedColorTexture('newton'),
            available: getDeepCommittedFrameAvailable('newton'),
        },
    };

    mandelbrotRenderTargetWidth = width;
    mandelbrotRenderTargetHeight = height;
    mandelbrotFrameReady = false;
    juliaFrameReady = false;
    newtonFrameReady = false;
    newtonDeferredCommittedFramePending = false;

    if (mandelbrotWorkingFramebuffer) {
        gl.deleteFramebuffer(mandelbrotWorkingFramebuffer);
        gl.deleteFramebuffer(maskReduceFramebufferA);
        gl.deleteFramebuffer(maskReduceFramebufferB);
        gl.deleteTexture(mandelbrotWorkingColorTexture);
        gl.deleteTexture(mandelbrotWorkingMaskTexture);
        gl.deleteTexture(maskReduceTextureA);
        gl.deleteTexture(maskReduceTextureB);
    }
    setDeepCommittedFramebuffer('mandelbrot', null);
    setDeepCommittedColorTexture('mandelbrot', null);
    setDeepCommittedFramebuffer('julia', null);
    setDeepCommittedColorTexture('julia', null);
    setDeepCommittedFramebuffer('newton', null);
    setDeepCommittedColorTexture('newton', null);
    setDeepCommittedFrameAvailable('mandelbrot', false);
    setDeepCommittedFrameAvailable('julia', false);
    setDeepCommittedFrameAvailable('newton', false);

    mandelbrotWorkingColorTexture = createTexture(width, height, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE);
    mandelbrotWorkingMaskTexture = createTexture(width, height, gl.R8, gl.RED, gl.UNSIGNED_BYTE);
    maskReduceTextureA = createTexture(width, height, gl.R8, gl.RED, gl.UNSIGNED_BYTE);
    maskReduceTextureB = createTexture(width, height, gl.R8, gl.RED, gl.UNSIGNED_BYTE);

    mandelbrotWorkingFramebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, mandelbrotWorkingFramebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, mandelbrotWorkingColorTexture, 0);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, mandelbrotWorkingMaskTexture, 0);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
        throw new Error('Mandelbrot working framebuffer is incomplete.');
    }

    maskReduceFramebufferA = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, maskReduceFramebufferA);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, maskReduceTextureA, 0);
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
        throw new Error('Mask reduction framebuffer A is incomplete.');
    }

    maskReduceFramebufferB = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, maskReduceFramebufferB);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, maskReduceTextureB, 0);
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
        throw new Error('Mask reduction framebuffer B is incomplete.');
    }

    for (const deepType of ['mandelbrot', 'julia', 'newton']) {
        ensureDeepCommittedRenderTarget(deepType, width, height);
        const previousTarget = previousCommittedTargets[deepType];
        if (previousTarget.available && previousTarget.framebuffer) {
            blitColorFramebuffer(
                previousTarget.framebuffer,
                getDeepCommittedFramebuffer(deepType),
                previousWidth,
                previousHeight,
                width,
                height
            );
            setDeepCommittedFrameAvailable(deepType, true);
        }
    }

    for (const deepType of ['mandelbrot', 'julia', 'newton']) {
        const previousTarget = previousCommittedTargets[deepType];
        if (previousTarget.framebuffer) {
            gl.deleteFramebuffer(previousTarget.framebuffer);
        }
        if (previousTarget.colorTexture) {
            gl.deleteTexture(previousTarget.colorTexture);
        }
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
}

function initWebGL() {
    canvas = document.getElementById('fractalCanvas');
    gl = canvas.getContext('webgl2', { antialias: false, alpha: false });

    if (!gl) {
        alert('WebGL2 is required for deep fractal zoom.');
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
            'u_deepFractalType',
        ]
    );

    blitProgramInfo = createProgramInfo(
        createProgram(vertexShaderSource, blitFragmentShaderSource),
        ['u_texture', 'u_resolution', 'u_translationPixels', 'u_scale']
    );

    maskReduceProgramInfo = createProgramInfo(
        createProgram(vertexShaderSource, maskReduceFragmentShaderSource),
        ['u_texture', 'u_sourceOffset', 'u_sourceSize']
    );

    orbitTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, orbitTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    orbitTextureCapacity = MAX_ORBIT_TEXTURE_LENGTH;
    gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.RGBA32F,
        orbitTextureCapacity,
        1,
        0,
        gl.RGBA,
        gl.FLOAT,
        null
    );

    ensureMandelbrotRenderTargets();

    gl.clearColor(0, 0, 0, 1);
    mandelbrotCamera = createMandelbrotCamera();
    juliaCamera = createJuliaCamera();
    newtonCamera = createNewtonCamera();
    resetSimpleCamera('mandelbrot');
    return true;
}

function uploadReferenceOrbit(orbitData, orbitLength) {
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, orbitTexture);
    if (orbitTextureCapacity < orbitLength) {
        orbitTextureCapacity = orbitLength;
        gl.texImage2D(
            gl.TEXTURE_2D,
            0,
            gl.RGBA32F,
            orbitTextureCapacity,
            1,
            0,
            gl.RGBA,
            gl.FLOAT,
            null
        );
    }
    gl.texSubImage2D(
        gl.TEXTURE_2D,
        0,
        0,
        0,
        orbitLength,
        1,
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
    const cacheKey = `${radius}:${sampleSpacing}`;
    if (referenceRingOffsetsCache.has(cacheKey)) {
        return referenceRingOffsetsCache.get(cacheKey);
    }

    if (radius === 0) {
        const offsets = [{ x: 0, y: 0 }];
        referenceRingOffsetsCache.set(cacheKey, offsets);
        return offsets;
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

    referenceRingOffsetsCache.set(cacheKey, offsets);
    return offsets;
}

function getReferenceSearchRings(baseRings, includeViewportRadius = true) {
    const maxViewportRadius = Math.max(
        128,
        Math.floor(Math.min(gl.canvas.width, gl.canvas.height) * 0.6)
    );
    const rings = baseRings.filter((radius) => radius <= maxViewportRadius);
    if (includeViewportRadius && !rings.includes(maxViewportRadius)) {
        rings.push(maxViewportRadius);
    }
    return rings;
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

function getReferenceSearchAnchors(type, anchorPoint, includeCommittedReferenceAnchor = true) {
    const anchors = [anchorPoint];
    if (!includeCommittedReferenceAnchor) {
        return anchors;
    }
    const reference = getDeepReference(type);

    if (reference.centerX !== null && reference.centerY !== null) {
        const previousReferenceAnchor = createDeepPoint(
            reference.centerX,
            reference.centerY
        );
        const key = `${previousReferenceAnchor.x.toString()}:${previousReferenceAnchor.y.toString()}`;
        if (key !== `${anchorPoint.x.toString()}:${anchorPoint.y.toString()}`) {
            anchors.push(previousReferenceAnchor);
        }
    }

    return anchors;
}

function searchReferenceCandidates(
    type,
    anchorPoint,
    iterations,
    rings,
    sampleSpacing,
    includeViewportRadius = true,
    includeCommittedReferenceAnchor = true
) {
    let bestCandidate = null;
    const camera = getDeepCamera(type);

    for (const searchAnchor of getReferenceSearchAnchors(
        type,
        anchorPoint,
        includeCommittedReferenceAnchor
    )) {
        const isPrimaryAnchor = searchAnchor === anchorPoint;
        for (const radius of getReferenceSearchRings(rings, includeViewportRadius)) {
            for (const offset of getReferenceRingOffsets(radius, sampleSpacing)) {
                const candidatePoint = offsetDeepPoint(camera, searchAnchor, offset.x, offset.y);
                const escapeIteration = computeEscapeIteration(candidatePoint.x, candidatePoint.y, iterations, type);
                const distanceSquared = isPrimaryAnchor
                    ? ((offset.x * offset.x) + (offset.y * offset.y))
                    : getDeepPointScreenDistanceSquared(camera, candidatePoint);
                const candidate = { point: candidatePoint, distanceSquared, escapeIteration };

                if (isReferenceCandidateBetter(candidate, bestCandidate)) {
                    bestCandidate = candidate;
                }
            }
        }
    }

    return bestCandidate;
}

function findBestReferencePoint(anchorPoint, iterations, type = fractalType) {
    const coarseCandidate = searchReferenceCandidates(
        type,
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
        type,
        anchorPoint,
        iterations,
        DENSE_REFERENCE_SEARCH_RINGS,
        DENSE_REFERENCE_RING_SAMPLE_SPACING
    );

    return isReferenceCandidateBetter(denseCandidate, coarseCandidate)
        ? denseCandidate
        : coarseCandidate;
}

function isReusableReferenceStrong(reference, iterations, type = fractalType) {
    if (!reference) {
        return false;
    }

    if (!reference.escapedEarly) {
        return true;
    }

    const lastFrame = getDeepLastFrameStats(type);
    if (
        lastFrame
        && lastFrame.status === 'success'
        && lastFrame.referencesUsed === 1
        && lastFrame.repairPasses === 0
        && lastFrame.cpuResolvedTiles === 0
    ) {
        return true;
    }

    const minimumEscapeIteration = Math.max(
        iterations - REFERENCE_REFINEMENT_ESCAPE_MARGIN,
        Math.floor(iterations * REFERENCE_REUSE_ESCAPE_RATIO)
    );
    return reference.escapeIteration >= minimumEscapeIteration;
}

function getReferenceReuseRadiusPixels(type) {
    if (
        getDeepLastFrameStats(type)
        && getDeepLastFrameStats(type).status === 'success'
        && getDeepLastFrameStats(type).referencesUsed === 1
        && getDeepLastFrameStats(type).cpuResolvedTiles === 0
    ) {
        return STABLE_REFERENCE_REUSE_RADIUS_PIXELS;
    }

    return REFERENCE_REUSE_RADIUS_PIXELS;
}

function canUseDeferredMaskVerification(type, initialReferenceMode) {
    const previousFrameStats = getDeepLastFrameStats(type);
    return (
        initialReferenceMode === 'reuse'
        && getDeepMaskVerificationFramesRemaining(type) > 0
        && previousFrameStats
        && previousFrameStats.status === 'success'
        && previousFrameStats.initialReferenceMode === 'reuse'
        && previousFrameStats.referencesUsed === 1
        && previousFrameStats.repairPasses === 0
        && previousFrameStats.cpuResolvedTiles === 0
    );
}

function getReusableReferenceSelection(anchorPoint, iterations, type = fractalType) {
    const reference = getDeepReference(type);
    const camera = getDeepCamera(type);

    if (reference.centerX === null || reference.centerY === null) {
        return null;
    }

    const point = createDeepPoint(reference.centerX, reference.centerY);
    const distanceSquared = getDeepPointScreenDistanceSquared(camera, point);
    const reuseRadiusPixels = getReferenceReuseRadiusPixels(type);
    if (distanceSquared > (reuseRadiusPixels * reuseRadiusPixels)) {
        return null;
    }

    const cachedReference = getReferenceOrbit(point, iterations, type);
    if (!isReusableReferenceStrong(cachedReference, iterations, type)) {
        return null;
    }

    return {
        candidate: {
            point,
            distanceSquared,
            escapeIteration: cachedReference.escapeIteration,
        },
        reference: cachedReference,
        mode: 'reuse',
    };
}

function selectInitialReference(anchorPoint, iterations, type = fractalType) {
    const reusableSelection = getReusableReferenceSelection(anchorPoint, iterations, type);
    if (reusableSelection) {
        return reusableSelection;
    }

    if (type === 'newton') {
        const directEscapeIteration = computeEscapeIteration(
            anchorPoint.x,
            anchorPoint.y,
            iterations,
            type
        );
        const directReferenceThreshold = Math.max(
            NEWTON_DIRECT_REFERENCE_MIN_ITERATIONS,
            iterations - REFERENCE_REFINEMENT_ESCAPE_MARGIN
        );

        if (directEscapeIteration >= directReferenceThreshold) {
            return {
                candidate: {
                    point: anchorPoint,
                    distanceSquared: 0,
                    escapeIteration: directEscapeIteration,
                },
                reference: getReferenceOrbit(anchorPoint, iterations, type),
                mode: 'search',
            };
        }

        const candidate = searchReferenceCandidates(
            type,
            anchorPoint,
            iterations,
            NEWTON_INITIAL_REFERENCE_SEARCH_RINGS,
            NEWTON_INITIAL_REFERENCE_RING_SAMPLE_SPACING,
            false,
            false
        );
        if (!candidate) {
            return null;
        }

        return {
            candidate,
            reference: getReferenceOrbit(candidate.point, iterations, type),
            mode: 'search',
        };
    }

    const candidate = findBestReferencePoint(anchorPoint, iterations, type);
    if (!candidate) {
        return null;
    }

    return {
        candidate,
        reference: getReferenceOrbit(candidate.point, iterations, type),
        mode: 'search',
    };
}

function mulDecimalComplex(ax, ay, bx, by) {
    return {
        x: subDecimal(mulDecimal(ax, bx), mulDecimal(ay, by)),
        y: addDecimal(mulDecimal(ax, by), mulDecimal(ay, bx)),
    };
}

function divDecimalComplex(ax, ay, bx, by) {
    const denominator = addDecimal(mulDecimal(bx, bx), mulDecimal(by, by));
    if (denominator === 0n) {
        return null;
    }
    return {
        x: divDecimal(addDecimal(mulDecimal(ax, bx), mulDecimal(ay, by)), denominator),
        y: divDecimal(subDecimal(mulDecimal(ay, bx), mulDecimal(ax, by)), denominator),
    };
}

function computeNewtonStepDecimal(zr, zi) {
    const z2 = mulDecimalComplex(zr, zi, zr, zi);
    const z3 = mulDecimalComplex(z2.x, z2.y, zr, zi);
    const fraction = divDecimalComplex(
        subDecimal(z3.x, decimalScale),
        z3.y,
        mulDecimalInt(z2.x, 3),
        mulDecimalInt(z2.y, 3)
    );
    if (!fraction) {
        return null;
    }
    return {
        x: subDecimal(zr, fraction.x),
        y: subDecimal(zi, fraction.y),
    };
}

function getNewtonRootDistanceSquared(zr, zi, root) {
    const dx = subDecimal(zr, root.x);
    const dy = subDecimal(zi, root.y);
    return addDecimal(mulDecimal(dx, dx), mulDecimal(dy, dy));
}

function getNearestNewtonRootIndex(zr, zi) {
    const roots = getNewtonRoots();
    let bestIndex = 0;
    let bestDistanceSquared = getNewtonRootDistanceSquared(zr, zi, roots[0]);

    for (let index = 1; index < roots.length; index += 1) {
        const distanceSquared = getNewtonRootDistanceSquared(zr, zi, roots[index]);
        if (distanceSquared < bestDistanceSquared) {
            bestDistanceSquared = distanceSquared;
            bestIndex = index;
        }
    }

    return bestIndex;
}

function getNewtonColorComponents(rootIndex, normalized) {
    const baseColors = [
        [0.15, 0.78, 0.93],
        [0.99, 0.72, 0.16],
        [0.93, 0.21, 0.58],
    ];
    const safeNormalized = Math.max(0, Math.min(1, normalized));
    const fade = 0.25 + (0.75 * Math.pow(1 - safeNormalized, 0.35));
    const band = 0.9 + (0.1 * Math.cos((18 * safeNormalized) + (rootIndex * 2.09439510239)));
    return baseColors[rootIndex].map((channel) => channel * fade * band);
}

function computeEscapeIteration(centerX, centerY, iterations, type = fractalType) {
    if (type === 'newton') {
        let zr = centerX;
        let zi = centerY;
        const roots = getNewtonRoots();
        const convergenceDistanceSquared = getNewtonConvergenceDistanceSquared();

        for (let i = 0; i < iterations; i += 1) {
            const next = computeNewtonStepDecimal(zr, zi);
            if (!next) {
                return 0;
            }
            zr = next.x;
            zi = next.y;

            const rootIndex = getNearestNewtonRootIndex(zr, zi);
            if (getNewtonRootDistanceSquared(zr, zi, roots[rootIndex]) <= convergenceDistanceSquared) {
                return i + 1;
            }
        }

        return iterations;
    }

    const juliaConstant = type === 'julia' ? getDeepConstant(type) : null;
    let zr = type === 'julia' ? centerX : 0n;
    let zi = type === 'julia' ? centerY : 0n;

    for (let i = 0; i < iterations; i += 1) {
        const zr2 = mulDecimal(zr, zr);
        const zi2 = mulDecimal(zi, zi);
        const zrzi = mulDecimal(zr, zi);

        zr = addDecimal(
            subDecimal(zr2, zi2),
            type === 'julia' ? juliaConstant.x : centerX
        );
        zi = addDecimal(
            mulDecimalInt(zrzi, 2),
            type === 'julia' ? juliaConstant.y : centerY
        );

        const magnitude = addDecimal(mulDecimal(zr, zr), mulDecimal(zi, zi));
        if (magnitude > mulDecimalInt(decimalScale, 4)) {
            return i + 1;
        }
    }

    return iterations;
}

function computeReferenceOrbit(centerX, centerY, iterations, type = fractalType) {
    if (type === 'newton') {
        const orbitData = new Float32Array(MAX_ORBIT_TEXTURE_LENGTH * 4);
        const roots = getNewtonRoots();
        const convergenceDistanceSquared = getNewtonConvergenceDistanceSquared();
        let zr = centerX;
        let zi = centerY;

        orbitData[0] = decimalToNumber(zr);
        orbitData[1] = decimalToNumber(zi);

        for (let i = 0; i < iterations; i += 1) {
            const next = computeNewtonStepDecimal(zr, zi);
            if (!next) {
                return {
                    escapedEarly: true,
                    escapeIteration: 0,
                    computedIterations: i,
                    exactZr: zr,
                    exactZi: zi,
                    orbitData,
                    orbitLength: i + 1,
                };
            }

            zr = next.x;
            zi = next.y;

            const baseIndex = (i + 1) * 4;
            orbitData[baseIndex] = decimalToNumber(zr);
            orbitData[baseIndex + 1] = decimalToNumber(zi);

            const rootIndex = getNearestNewtonRootIndex(zr, zi);
            if (getNewtonRootDistanceSquared(zr, zi, roots[rootIndex]) <= convergenceDistanceSquared) {
                return {
                    escapedEarly: true,
                    escapeIteration: i + 1,
                    computedIterations: i + 1,
                    exactZr: zr,
                    exactZi: zi,
                    orbitData,
                    orbitLength: i + 2,
                };
            }
        }

        return {
            escapedEarly: false,
            escapeIteration: iterations,
            computedIterations: iterations,
            exactZr: zr,
            exactZi: zi,
            orbitData,
            orbitLength: iterations + 1,
        };
    }

    const orbitData = new Float32Array(MAX_ORBIT_TEXTURE_LENGTH * 4);
    const juliaConstant = type === 'julia' ? getDeepConstant(type) : null;
    let zr = type === 'julia' ? centerX : 0n;
    let zi = type === 'julia' ? centerY : 0n;

    orbitData[0] = decimalToNumber(zr);
    orbitData[1] = decimalToNumber(zi);

    for (let i = 0; i < iterations; i += 1) {
        const zr2 = mulDecimal(zr, zr);
        const zi2 = mulDecimal(zi, zi);
        const zrzi = mulDecimal(zr, zi);

        const nextZr = addDecimal(
            subDecimal(zr2, zi2),
            type === 'julia' ? juliaConstant.x : centerX
        );
        const nextZi = addDecimal(
            mulDecimalInt(zrzi, 2),
            type === 'julia' ? juliaConstant.y : centerY
        );

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
                computedIterations: i + 1,
                exactZr: zr,
                exactZi: zi,
                orbitData,
                orbitLength: i + 2,
            };
        }
    }

    return {
        escapedEarly: false,
        escapeIteration: iterations,
        computedIterations: iterations,
        exactZr: zr,
        exactZi: zi,
        orbitData,
        orbitLength: iterations + 1,
    };
}

function getReferenceCacheKey(point) {
    return `${point.x.toString()}:${point.y.toString()}`;
}

function extendReferenceOrbit(reference, iterations, type = fractalType) {
    if (reference.escapedEarly || reference.computedIterations >= iterations) {
        return reference;
    }

    if (type === 'newton') {
        const roots = getNewtonRoots();
        const convergenceDistanceSquared = getNewtonConvergenceDistanceSquared();
        let zr = reference.exactZr;
        let zi = reference.exactZi;

        for (let i = reference.computedIterations; i < iterations; i += 1) {
            const next = computeNewtonStepDecimal(zr, zi);
            if (!next) {
                reference.escapedEarly = true;
                reference.escapeIteration = 0;
                reference.computedIterations = i;
                reference.exactZr = zr;
                reference.exactZi = zi;
                reference.orbitLength = i + 1;
                return reference;
            }

            zr = next.x;
            zi = next.y;

            const baseIndex = (i + 1) * 4;
            reference.orbitData[baseIndex] = decimalToNumber(zr);
            reference.orbitData[baseIndex + 1] = decimalToNumber(zi);

            const rootIndex = getNearestNewtonRootIndex(zr, zi);
            if (getNewtonRootDistanceSquared(zr, zi, roots[rootIndex]) <= convergenceDistanceSquared) {
                reference.escapedEarly = true;
                reference.escapeIteration = i + 1;
                reference.computedIterations = i + 1;
                reference.exactZr = zr;
                reference.exactZi = zi;
                reference.orbitLength = i + 2;
                return reference;
            }
        }

        reference.escapedEarly = false;
        reference.escapeIteration = iterations;
        reference.computedIterations = iterations;
        reference.exactZr = zr;
        reference.exactZi = zi;
        reference.orbitLength = iterations + 1;
        return reference;
    }

    const juliaConstant = type === 'julia' ? getDeepConstant(type) : null;
    let zr = reference.exactZr;
    let zi = reference.exactZi;

    for (let i = reference.computedIterations; i < iterations; i += 1) {
        const zr2 = mulDecimal(zr, zr);
        const zi2 = mulDecimal(zi, zi);
        const zrzi = mulDecimal(zr, zi);

        zr = addDecimal(
            subDecimal(zr2, zi2),
            type === 'julia' ? juliaConstant.x : reference.point.x
        );
        zi = addDecimal(
            mulDecimalInt(zrzi, 2),
            type === 'julia' ? juliaConstant.y : reference.point.y
        );

        const baseIndex = (i + 1) * 4;
        reference.orbitData[baseIndex] = decimalToNumber(zr);
        reference.orbitData[baseIndex + 1] = decimalToNumber(zi);

        const magnitude = addDecimal(mulDecimal(zr, zr), mulDecimal(zi, zi));
        if (magnitude > mulDecimalInt(decimalScale, 4)) {
            reference.escapedEarly = true;
            reference.escapeIteration = i + 1;
            reference.computedIterations = i + 1;
            reference.exactZr = zr;
            reference.exactZi = zi;
            reference.orbitLength = i + 2;
            return reference;
        }
    }

    reference.escapedEarly = false;
    reference.escapeIteration = iterations;
    reference.computedIterations = iterations;
    reference.exactZr = zr;
    reference.exactZi = zi;
    reference.orbitLength = iterations + 1;
    return reference;
}

function getReferenceOrbit(point, iterations, type = fractalType) {
    const referenceCache = getDeepReferenceCache(type);
    const key = getReferenceCacheKey(point);
    if (referenceCache.has(key)) {
        const cachedReference = referenceCache.get(key);
        extendReferenceOrbit(cachedReference, iterations, type);
        referenceCache.delete(key);
        referenceCache.set(key, cachedReference);
        return cachedReference;
    }

    const referenceOrbit = computeReferenceOrbit(point.x, point.y, iterations, type);
    const reference = {
        point,
        orbitData: referenceOrbit.orbitData,
        orbitLength: referenceOrbit.orbitLength,
        escapeIteration: referenceOrbit.escapeIteration,
        escapedEarly: referenceOrbit.escapedEarly,
        computedIterations: referenceOrbit.computedIterations,
        exactZr: referenceOrbit.exactZr,
        exactZi: referenceOrbit.exactZi,
    };
    referenceCache.set(key, reference);
    while (referenceCache.size > MAX_REFERENCE_CACHE_ENTRIES) {
        const oldestKey = referenceCache.keys().next().value;
        referenceCache.delete(oldestKey);
    }
    return reference;
}

function commitDeepReference(type, reference, cameraOverride = null) {
    const deepReference = getDeepReference(type);
    const camera = cameraOverride || getDeepCamera(type);
    deepReference.centerX = reference.point.x;
    deepReference.centerY = reference.point.y;
    deepReference.orbitData = reference.orbitData;
    deepReference.orbitLength = reference.orbitLength;
    deepReference.maxIterations = camera.maxIterations;
    deepReference.escapeIteration = reference.escapeIteration;
    deepReference.useSimpleFallback = false;
}

function commitMandelbrotReference(reference) {
    commitDeepReference('mandelbrot', reference);
}

function getCommittedReferenceAnchorDistanceSquared(type, selection) {
    if (!selection || !selection.candidate || !selection.candidate.point) {
        return Number.POSITIVE_INFINITY;
    }
    return getDeepPointScreenDistanceSquared(getDeepCamera(type), selection.candidate.point);
}

function chooseCommittedReferenceSelection(type, initialSelection, bestReferenceSelection, referencesUsed) {
    if (referencesUsed <= 1) {
        return bestReferenceSelection;
    }

    const reuseRadiusSquared = REFERENCE_REUSE_RADIUS_PIXELS ** 2;
    const initialDistanceSquared = getCommittedReferenceAnchorDistanceSquared(type, initialSelection);
    const bestDistanceSquared = getCommittedReferenceAnchorDistanceSquared(type, bestReferenceSelection);
    const initialIsReusable = initialDistanceSquared <= reuseRadiusSquared;
    const bestIsReusable = bestDistanceSquared <= reuseRadiusSquared;
    const initialEscapeIteration = initialSelection?.candidate?.escapeIteration ?? Number.NEGATIVE_INFINITY;
    const bestEscapeIteration = bestReferenceSelection?.candidate?.escapeIteration ?? Number.NEGATIVE_INFINITY;

    if (initialIsReusable && bestIsReusable) {
        if (initialEscapeIteration !== bestEscapeIteration) {
            return initialEscapeIteration > bestEscapeIteration
                ? initialSelection
                : bestReferenceSelection;
        }
        if (initialDistanceSquared !== bestDistanceSquared) {
            return initialDistanceSquared < bestDistanceSquared
                ? initialSelection
                : bestReferenceSelection;
        }
        if (initialSelection && !bestReferenceSelection) {
            return initialSelection;
        }
        if (bestReferenceSelection) {
            return bestReferenceSelection;
        }
    }

    if (initialIsReusable !== bestIsReusable) {
        return initialIsReusable ? initialSelection : bestReferenceSelection;
    }

    if (initialDistanceSquared !== bestDistanceSquared) {
        return initialDistanceSquared < bestDistanceSquared
            ? initialSelection
            : bestReferenceSelection;
    }

    return bestReferenceSelection;
}

function createRepairTile(x, y, width, height, depth, maskData) {
    return { x, y, width, height, depth, maskData };
}

function prepareTightMaskReadback() {
    if (typeof gl.pixelStorei === 'function' && gl.PACK_ALIGNMENT !== undefined) {
        gl.pixelStorei(gl.PACK_ALIGNMENT, 1);
    }
}

function prepareTightMaskUpload() {
    if (typeof gl.pixelStorei === 'function' && gl.UNPACK_ALIGNMENT !== undefined) {
        gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    }
}

function readGlitchMask(tile) {
    const maskData = new Uint8Array(tile.width * tile.height);
    prepareTightMaskReadback();
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, mandelbrotWorkingFramebuffer);
    gl.readBuffer(gl.COLOR_ATTACHMENT1);
    gl.readPixels(tile.x, tile.y, tile.width, tile.height, gl.RED, gl.UNSIGNED_BYTE, maskData);
    return maskData;
}

function readGlitchMaskAndCheck(tile) {
    if (!tileHasGlitches(tile)) {
        return { hasGlitches: false, maskData: null };
    }
    return { hasGlitches: true, maskData: readGlitchMask(tile) };
}

const reducedMaskSample = new Uint8Array(1);

function tileHasGlitches(tile) {
    let inputTexture = mandelbrotWorkingMaskTexture;
    let sourceOffsetX = tile.x;
    let sourceOffsetY = tile.y;
    let sourceWidth = tile.width;
    let sourceHeight = tile.height;
    let outputTexture = maskReduceTextureA;
    let outputFramebuffer = maskReduceFramebufferA;

    while (true) {
        const outputWidth = Math.max(1, Math.ceil(sourceWidth / MASK_REDUCTION_FACTOR));
        const outputHeight = Math.max(1, Math.ceil(sourceHeight / MASK_REDUCTION_FACTOR));

        gl.bindFramebuffer(gl.FRAMEBUFFER, outputFramebuffer);
        gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
        gl.viewport(0, 0, outputWidth, outputHeight);

        gl.useProgram(maskReduceProgramInfo.program);
        bindQuad(maskReduceProgramInfo);
        gl.uniform1i(maskReduceProgramInfo.uniforms.u_texture, 0);
        gl.uniform2i(maskReduceProgramInfo.uniforms.u_sourceOffset, sourceOffsetX, sourceOffsetY);
        gl.uniform2i(maskReduceProgramInfo.uniforms.u_sourceSize, sourceWidth, sourceHeight);

        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, inputTexture);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

        if (outputWidth === 1 && outputHeight === 1) {
            prepareTightMaskReadback();
            gl.bindFramebuffer(gl.READ_FRAMEBUFFER, outputFramebuffer);
            gl.readBuffer(gl.COLOR_ATTACHMENT0);
            gl.readPixels(0, 0, 1, 1, gl.RED, gl.UNSIGNED_BYTE, reducedMaskSample);
            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
            return reducedMaskSample[0] !== 0;
        }

        inputTexture = outputTexture;
        sourceOffsetX = 0;
        sourceOffsetY = 0;
        sourceWidth = outputWidth;
        sourceHeight = outputHeight;

        if (outputTexture === maskReduceTextureA) {
            outputTexture = maskReduceTextureB;
            outputFramebuffer = maskReduceFramebufferB;
        } else {
            outputTexture = maskReduceTextureA;
            outputFramebuffer = maskReduceFramebufferA;
        }
    }
}

function maskHasGlitches(maskData) {
    for (let index = 0; index < maskData.length; index += 1) {
        if (maskData[index] !== 0) {
            return true;
        }
    }
    return false;
}

function countMaskGlitches(maskData) {
    let glitchCount = 0;
    for (let index = 0; index < maskData.length; index += 1) {
        if (maskData[index] !== 0) {
            glitchCount += 1;
        }
    }
    return glitchCount;
}

function assessNewtonInitialRepairPressure(repairQueueLength, maskData) {
    const glitchCount = countMaskGlitches(maskData);
    const glitchRatio = maskData.length > 0 ? glitchCount / maskData.length : 0;
    return {
        glitchCount,
        glitchRatio,
        shouldDefer: (
            repairQueueLength >= NEWTON_PATHOLOGICAL_INITIAL_REPAIR_QUEUE_LENGTH
            && glitchRatio >= NEWTON_PATHOLOGICAL_INITIAL_GLITCH_RATIO
        ),
    };
}

function extractTileMask(parentTile, childTile) {
    const maskData = new Uint8Array(childTile.width * childTile.height);
    const offsetX = childTile.x - parentTile.x;
    const offsetY = childTile.y - parentTile.y;

    for (let row = 0; row < childTile.height; row += 1) {
        const parentStart = ((offsetY + row) * parentTile.width) + offsetX;
        const childStart = row * childTile.width;
        maskData.set(
            parentTile.maskData.subarray(parentStart, parentStart + childTile.width),
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
            const baseIndex = (row * tile.width) + column;
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

function computeCpuDeepColor(type, centerX, centerY, iterations) {
    if (type === 'newton') {
        let zr = centerX;
        let zi = centerY;
        const roots = getNewtonRoots();
        const convergenceDistanceSquared = getNewtonConvergenceDistanceSquared();

        for (let i = 0; i < iterations; i += 1) {
            const next = computeNewtonStepDecimal(zr, zi);
            if (!next) {
                return [0, 0, 0, 255];
            }
            zr = next.x;
            zi = next.y;

            const rootIndex = getNearestNewtonRootIndex(zr, zi);
            if (getNewtonRootDistanceSquared(zr, zi, roots[rootIndex]) <= convergenceDistanceSquared) {
                const normalized = (i + 1) / Math.max(1, iterations);
                const [r, g, b] = getNewtonColorComponents(rootIndex, normalized);
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

    const juliaConstant = type === 'julia' ? getDeepConstant(type) : null;
    let zr = type === 'julia' ? centerX : 0n;
    let zi = type === 'julia' ? centerY : 0n;

    for (let i = 0; i < iterations; i += 1) {
        const zr2 = mulDecimal(zr, zr);
        const zi2 = mulDecimal(zi, zi);
        const zrzi = mulDecimal(zr, zi);

        zr = addDecimal(
            subDecimal(zr2, zi2),
            type === 'julia' ? juliaConstant.x : centerX
        );
        zi = addDecimal(
            mulDecimalInt(zrzi, 2),
            type === 'julia' ? juliaConstant.y : centerY
        );

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

function computeCpuMandelbrotColor(centerX, centerY, iterations) {
    return computeCpuDeepColor('mandelbrot', centerX, centerY, iterations);
}

function resolveDeepTileOnCPU(type, tile, cameraOverride = null) {
    const camera = cameraOverride || getDeepCamera(type);
    const colorData = new Uint8Array(tile.width * tile.height * 4);
    const maskData = new Uint8Array(tile.width * tile.height);

    for (let row = 0; row < tile.height; row += 1) {
        for (let column = 0; column < tile.width; column += 1) {
            const pointer = framebufferPixelToPointer({
                x: tile.x + column,
                y: tile.y + row,
            });
            const point = screenToPlaneDeep(camera, pointer);
            const color = computeCpuDeepColor(type, point.x, point.y, camera.maxIterations);
            const baseIndex = ((row * tile.width) + column) * 4;

            colorData[baseIndex] = color[0];
            colorData[baseIndex + 1] = color[1];
            colorData[baseIndex + 2] = color[2];
            colorData[baseIndex + 3] = color[3];
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
    prepareTightMaskUpload();
    gl.texSubImage2D(
        gl.TEXTURE_2D,
        0,
        tile.x,
        tile.y,
        tile.width,
        tile.height,
        gl.RED,
        gl.UNSIGNED_BYTE,
        maskData
    );
}

function resolveMandelbrotTileOnCPU(tile) {
    resolveDeepTileOnCPU('mandelbrot', tile);
}

function resolveRepairQueueOnCPU(type, repairQueue, frameStats) {
    let processedTiles = 0;
    while (repairQueue.length > 0) {
        const tile = repairQueue.shift();
        processedTiles += 1;
        frameStats.lastTileWidth = tile.width;
        frameStats.lastTileHeight = tile.height;
        frameStats.lastTileDepth = tile.depth;
        frameStats.deepestTileDepth = Math.max(frameStats.deepestTileDepth, tile.depth);
        updateDeepWorkState(type, {
            stage: 'cpu-repair-drain',
            repairQueueLength: repairQueue.length,
            repairTilesProcessed: processedTiles,
            cpuTilesResolved: frameStats.cpuResolvedTiles,
            currentTile: {
                width: tile.width,
                height: tile.height,
                depth: tile.depth,
            },
            note: 'resolving-repair-queue-on-cpu',
        }, processedTiles <= 4 || processedTiles % 16 === 0 ? 'cpu-repair-progress' : null);
        resolveDeepTileOnCPU(type, tile);
        frameStats.cpuResolvedTiles += 1;
        frameStats.cpuResolvedPixels += tile.width * tile.height;
    }
}

function setWorkingFramebufferDrawBuffers(writeMask = true) {
    if (writeMask) {
        gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
        return;
    }
    gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
}

function renderDeepPass(type, reference, tile, writeMask = true, cameraOverride = null) {
    const camera = cameraOverride || getDeepCamera(type);
    uploadReferenceOrbit(reference.orbitData, reference.orbitLength);

    gl.bindFramebuffer(gl.FRAMEBUFFER, mandelbrotWorkingFramebuffer);
    setWorkingFramebufferDrawBuffers(writeMask);
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
    gl.enable(gl.SCISSOR_TEST);
    gl.scissor(tile.x, tile.y, tile.width, tile.height);

    gl.useProgram(deepProgramInfo.program);
    bindQuad(deepProgramInfo);
    gl.uniform2f(deepProgramInfo.uniforms.u_resolution, gl.canvas.width, gl.canvas.height);
    gl.uniform1f(deepProgramInfo.uniforms.u_pixelScale, camera.pixelScaleApprox);
    gl.uniform1f(deepProgramInfo.uniforms.u_glitchThreshold, GLITCH_THRESHOLD);
    gl.uniform1i(
        deepProgramInfo.uniforms.u_deepFractalType,
        type === 'julia' ? 1 : type === 'newton' ? 2 : 0
    );
    gl.uniform1i(deepProgramInfo.uniforms.u_referenceOrbitLength, reference.orbitLength);
    gl.uniform1i(deepProgramInfo.uniforms.u_referenceOrbit, 0);
    gl.uniform1i(deepProgramInfo.uniforms.u_maxIterations, camera.maxIterations);

    const referenceDeltaX = decimalToNumber(subDecimal(camera.centerX, reference.point.x));
    const referenceDeltaY = decimalToNumber(subDecimal(camera.centerY, reference.point.y));
    gl.uniform2f(
        deepProgramInfo.uniforms.u_referenceDeltaPixels,
        referenceDeltaX / camera.pixelScaleApprox,
        referenceDeltaY / camera.pixelScaleApprox
    );

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, orbitTexture);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.disable(gl.SCISSOR_TEST);
}

function renderMandelbrotPass(reference, tile) {
    renderDeepPass('mandelbrot', reference, tile);
}

function copyWorkingFrameToCommitted(type = fractalType) {
    if (!mandelbrotWorkingFramebuffer || !getDeepCommittedFramebuffer(type)) {
        ensureMandelbrotRenderTargets(type);
    }
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, mandelbrotWorkingFramebuffer);
    gl.readBuffer(gl.COLOR_ATTACHMENT0);
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, getDeepCommittedFramebuffer(type));
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
    setDeepCommittedFrameAvailable(type, true);
}

function drawCommittedDeepFrame(type = fractalType, targetCamera = null) {
    const committedFramebuffer = getDeepCommittedFramebuffer(type);
    const committedTexture = getDeepCommittedColorTexture(type);
    const committedCamera = getDeepCommittedCamera(type);
    if (!committedFramebuffer || !committedTexture || !getDeepCommittedFrameAvailable(type)) {
        return false;
    }

    if (!targetCamera || !committedCamera || areDeepCamerasEquivalent(targetCamera, committedCamera)) {
        gl.bindFramebuffer(gl.READ_FRAMEBUFFER, committedFramebuffer);
        gl.readBuffer(gl.COLOR_ATTACHMENT0);
        gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, null);
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
        return true;
    }

    const translationX = decimalToNumber(
        subDecimal(targetCamera.centerX, committedCamera.centerX)
    ) / committedCamera.pixelScaleApprox;
    const translationY = decimalToNumber(
        subDecimal(targetCamera.centerY, committedCamera.centerY)
    ) / committedCamera.pixelScaleApprox;
    const scale = targetCamera.pixelScaleApprox / committedCamera.pixelScaleApprox;

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
    gl.useProgram(blitProgramInfo.program);
    bindQuad(blitProgramInfo);
    gl.uniform1i(blitProgramInfo.uniforms.u_texture, 0);
    gl.uniform2f(blitProgramInfo.uniforms.u_resolution, gl.canvas.width, gl.canvas.height);
    gl.uniform2f(blitProgramInfo.uniforms.u_translationPixels, translationX, translationY);
    gl.uniform1f(blitProgramInfo.uniforms.u_scale, scale);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, committedTexture);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    return true;
}

function drawCommittedMandelbrotFrame() {
    drawCommittedDeepFrame('mandelbrot');
}

function renderSharpDeepFrame(type) {
    const camera = getDeepCamera(type);
    const frameStats = createEmptyMandelbrotFrameStats();
    const frameStartMs = Date.now();
    frameStats.attemptedPixelScaleApprox = roundDebugNumber(camera.pixelScaleApprox);
    frameStats.attemptedMaxIterations = camera.maxIterations;
    updateDeepWorkState(type, {
        stage: 'frame-start',
        referencesUsed: 0,
        repairQueueLength: 0,
        repairTilesProcessed: 0,
        cpuTilesResolved: 0,
        currentTile: null,
        note: null,
    }, 'frame-start');

    if (camera.pixelScaleApprox < MIN_GPU_SCALE) {
        finalizeDeepFrameStats(type, frameStats, 'failed', 'gpu_delta_precision_floor');
        clearDeepWorkState(type, 'frame-failed');
        return false;
    }

    ensureMandelbrotRenderTargets();

    const fullFrameTile = createRepairTile(0, 0, gl.canvas.width, gl.canvas.height, 0, null);
    const anchorPoint = screenToPlaneDeep(camera, mousePosition);
    updateDeepWorkState(type, {
        stage: 'initial-reference-search',
        note: 'selecting-initial-reference',
        currentTile: {
            width: fullFrameTile.width,
            height: fullFrameTile.height,
            depth: fullFrameTile.depth,
        },
    }, 'initial-reference-search');
    const initialSelection = selectInitialReference(anchorPoint, camera.maxIterations, type);
    if (!initialSelection) {
        finalizeDeepFrameStats(type, frameStats, 'failed', 'no_initial_reference');
        clearDeepWorkState(type, 'frame-failed');
        return false;
    }

    const { candidate: initialCandidate, reference: initialReference, mode: initialReferenceMode } = initialSelection;
    frameStats.initialEscapeIteration = initialCandidate.escapeIteration;
    frameStats.initialOrbitLength = initialReference.orbitLength;
    frameStats.initialEscapedEarly = initialReference.escapedEarly;
    frameStats.initialReferenceMode = initialReferenceMode;
    frameStats.referencesUsed = 1;
    let bestReferenceSelection = {
        candidate: initialCandidate,
        reference: initialReference,
    };
    const deferredMaskVerification = canUseDeferredMaskVerification(type, initialReferenceMode);
    updateDeepWorkState(type, {
        stage: 'full-frame-render',
        initialReferenceMode,
        referencesUsed: 1,
        note: deferredMaskVerification ? 'deferred-mask-verification' : 'exact-mask-verification',
    }, 'full-frame-render');
    renderDeepPass(type, initialReference, fullFrameTile, !deferredMaskVerification);
    const repairQueue = [];
    let fullFrameHasGlitches = false;
    if (deferredMaskVerification) {
        setDeepMaskVerificationFramesRemaining(
            type,
            Math.max(0, getDeepMaskVerificationFramesRemaining(type) - 1)
        );
    } else {
        updateDeepWorkState(type, {
            stage: 'full-frame-verify',
            note: 'reading-full-frame-mask',
        }, 'full-frame-verify');
        const fullFrameVerification = readGlitchMaskAndCheck(fullFrameTile);
        fullFrameHasGlitches = fullFrameVerification.hasGlitches;
        if (fullFrameHasGlitches) {
            fullFrameTile.maskData = fullFrameVerification.maskData;
        }
        if (!fullFrameHasGlitches && initialReferenceMode === 'reuse') {
            incrementDeepStableReuseFrames(type);
            setDeepMaskVerificationFramesRemaining(type, getAdaptiveMaskVerifySkipFrames(type));
        } else {
            setDeepStableReuseFrames(type, 0);
            setDeepMaskVerificationFramesRemaining(type, 0);
        }
    }

    if (fullFrameHasGlitches) {
        if (!fullFrameTile.maskData) {
            fullFrameTile.maskData = readGlitchMask(fullFrameTile);
        }
        if (
            fullFrameTile.depth >= MAX_REPAIR_DEPTH
            || fullFrameTile.width <= MIN_REPAIR_TILE_SIZE
            || fullFrameTile.height <= MIN_REPAIR_TILE_SIZE
        ) {
            finalizeDeepFrameStats(type, frameStats, 'failed', 'full_frame_tile_limit', {
                queuedTilesRemaining: repairQueue.length,
            });
            clearDeepWorkState(type, 'frame-failed');
            return false;
        }
        queueChildTilesWithGlitches(repairQueue, fullFrameTile);
        sortRepairQueue(repairQueue);
        frameStats.repairQueuePeak = Math.max(frameStats.repairQueuePeak, repairQueue.length);
        updateDeepWorkState(type, {
            stage: 'repair-queue-seeded',
            repairQueueLength: repairQueue.length,
            note: 'queued-initial-repair-tiles',
        }, 'repair-queue-seeded');
        if (type === 'newton') {
            const initialRepairPressure = assessNewtonInitialRepairPressure(
                repairQueue.length,
                fullFrameTile.maskData
            );
            if (initialRepairPressure.shouldDefer) {
                finalizeDeepFrameStats(type, frameStats, 'failed', 'pathological_initial_repair_flood', {
                    queuedTilesRemaining: repairQueue.length,
                    fullFrameGlitchRatio: roundDebugNumber(initialRepairPressure.glitchRatio),
                    fullFrameGlitchedPixels: initialRepairPressure.glitchCount,
                });
                updateDeepWorkState(type, {
                    stage: 'repair-queue-seeded',
                    referencesUsed: 1,
                    repairQueueLength: repairQueue.length,
                    repairTilesProcessed: 0,
                    cpuTilesResolved: frameStats.cpuResolvedTiles,
                    note: 'deferring-deep-render-due-to-initial-glitch-flood',
                }, 'pathological-initial-repair-flood');
                clearDeepWorkState(type, 'frame-failed');
                return false;
            }
        }
    }

    let referencesUsed = 1;
    let processedRepairTiles = 0;
    const repairWorkStartMs = Date.now();

    while (repairQueue.length > 0) {
        const elapsedMs = Date.now() - repairWorkStartMs;
        if (
            type === 'newton'
            && shouldDeferNewtonRepairWork(referencesUsed, repairQueue.length, elapsedMs)
        ) {
            finalizeDeepFrameStats(type, frameStats, 'failed', 'pathological_repair_growth', {
                queuedTilesRemaining: repairQueue.length,
                elapsedMs,
                referencesUsed,
            });
            updateDeepWorkState(type, {
                stage: 'repair-reference-search',
                referencesUsed,
                repairQueueLength: repairQueue.length,
                repairTilesProcessed: processedRepairTiles,
                cpuTilesResolved: frameStats.cpuResolvedTiles,
                note: 'deferring-deep-render-due-to-repair-growth',
            }, 'pathological-repair-growth');
            clearDeepWorkState(type, 'frame-failed');
            return false;
        }

        if (referencesUsed >= MAX_REPAIR_REFERENCES) {
            finalizeDeepFrameStats(type, frameStats, 'failed', 'repair_reference_budget_exhausted', {
                queuedTilesRemaining: repairQueue.length,
            });
            clearDeepWorkState(type, 'frame-failed');
            return false;
        }

        const tile = repairQueue.shift();
        processedRepairTiles += 1;
        frameStats.lastTileWidth = tile.width;
        frameStats.lastTileHeight = tile.height;
        frameStats.lastTileDepth = tile.depth;
        frameStats.deepestTileDepth = Math.max(frameStats.deepestTileDepth, tile.depth);
        frameStats.repairQueuePeak = Math.max(frameStats.repairQueuePeak, repairQueue.length);
        updateDeepWorkState(type, {
            stage: 'repair-reference-search',
            referencesUsed,
            repairQueueLength: repairQueue.length,
            repairTilesProcessed: processedRepairTiles,
            cpuTilesResolved: frameStats.cpuResolvedTiles,
            currentTile: {
                width: tile.width,
                height: tile.height,
                depth: tile.depth,
            },
            note: 'searching-repair-reference',
        }, processedRepairTiles <= 4 || processedRepairTiles % 16 === 0 ? 'repair-progress' : null);
        const glitchedPixel = findGlitchedPixelNearTileCenter(tile);
        if (!glitchedPixel) {
            continue;
        }

        const repairPointer = framebufferPixelToPointer(glitchedPixel);
        const repairAnchor = screenToPlaneDeep(camera, repairPointer);
        const repairCandidate = findBestReferencePoint(repairAnchor, camera.maxIterations, type);
        if (!repairCandidate) {
            finalizeDeepFrameStats(type, frameStats, 'failed', 'no_repair_reference', {
                queuedTilesRemaining: repairQueue.length,
                lastTileWidth: tile.width,
                lastTileHeight: tile.height,
                lastTileDepth: tile.depth,
            });
            clearDeepWorkState(type, 'frame-failed');
            return false;
        }

        frameStats.lastRepairEscapeIteration = repairCandidate.escapeIteration;
        const repairReference = getReferenceOrbit(repairCandidate.point, camera.maxIterations, type);
        const repairCandidateForCommit = {
            ...repairCandidate,
            distanceSquared: getDeepPointScreenDistanceSquared(camera, repairCandidate.point),
        };
        if (isReferenceCandidateBetter(repairCandidateForCommit, bestReferenceSelection.candidate)) {
            bestReferenceSelection = {
                candidate: repairCandidateForCommit,
                reference: repairReference,
            };
        }
        updateDeepWorkState(type, {
            stage: 'repair-render',
            referencesUsed,
            repairQueueLength: repairQueue.length,
            repairTilesProcessed: processedRepairTiles,
            cpuTilesResolved: frameStats.cpuResolvedTiles,
            note: 'rendering-repair-tile',
        }, null);
        renderDeepPass(type, repairReference, tile);
        referencesUsed += 1;
        frameStats.referencesUsed = referencesUsed;

        updateDeepWorkState(type, {
            stage: 'repair-verify',
            referencesUsed,
            repairQueueLength: repairQueue.length,
            repairTilesProcessed: processedRepairTiles,
            cpuTilesResolved: frameStats.cpuResolvedTiles,
            note: 'checking-repair-mask',
        }, null);
        if (!tileHasGlitches(tile)) {
            continue;
        }

        tile.maskData = readGlitchMask(tile);

        if (
            tile.depth >= MAX_REPAIR_DEPTH
            || tile.width <= MIN_REPAIR_TILE_SIZE
            || tile.height <= MIN_REPAIR_TILE_SIZE
        ) {
            updateDeepWorkState(type, {
                stage: 'cpu-tile-resolve',
                referencesUsed,
                repairQueueLength: repairQueue.length,
                repairTilesProcessed: processedRepairTiles,
                cpuTilesResolved: frameStats.cpuResolvedTiles,
                note: 'resolving-minimum-tile-on-cpu',
            }, 'cpu-tile-resolve');
            resolveDeepTileOnCPU(type, tile);
            frameStats.cpuResolvedTiles += 1;
            frameStats.cpuResolvedPixels += tile.width * tile.height;
            continue;
        }

        queueChildTilesWithGlitches(repairQueue, tile);
        sortRepairQueue(repairQueue);
        frameStats.repairQueuePeak = Math.max(frameStats.repairQueuePeak, repairQueue.length);
    }

    updateDeepWorkState(type, {
        stage: 'commit-frame',
        referencesUsed,
        repairQueueLength: repairQueue.length,
        repairTilesProcessed: processedRepairTiles,
        cpuTilesResolved: frameStats.cpuResolvedTiles,
        currentTile: null,
        note: 'committing-rendered-frame',
    }, 'commit-frame');
    const committedReferenceSelection = chooseCommittedReferenceSelection(
        type,
        initialSelection,
        bestReferenceSelection,
        referencesUsed
    );
    copyWorkingFrameToCommitted(type);
    commitDeepReference(type, committedReferenceSelection.reference);
    setDeepCommittedCamera(type, cloneDeepCamera(camera));
    setDeepFrameReady(type, true);
    if (type === 'newton') {
        newtonDeferredCommittedFramePending = false;
    }
    finalizeDeepFrameStats(type, frameStats, 'success');
    clearDeepWorkState(type, 'frame-complete');
    return true;
}

function renderSharpMandelbrotFrame() {
    return renderSharpDeepFrame('mandelbrot');
}

function renderSharpJuliaFrame() {
    return renderSharpDeepFrame('julia');
}

function renderSharpNewtonFrame() {
    return renderSharpDeepFrame('newton');
}

function createDeepRenderTask(type) {
    const cameraSnapshot = cloneDeepCamera(getDeepCamera(type));
    const frameStats = createEmptyMandelbrotFrameStats();
    frameStats.attemptedPixelScaleApprox = roundDebugNumber(cameraSnapshot.pixelScaleApprox);
    frameStats.attemptedMaxIterations = cameraSnapshot.maxIterations;

    return {
        type,
        camera: cameraSnapshot,
        pointer: { ...mousePosition },
        frameStats,
        stage: 'init',
        fullFrameTile: createRepairTile(0, 0, gl.canvas.width, gl.canvas.height, 0, null),
        initialSelection: null,
        bestReferenceSelection: null,
        deferredMaskVerification: false,
        repairQueue: [],
        referencesUsed: 0,
        processedRepairTiles: 0,
        startedAtMs: nowMs(),
        repairWorkStartMs: 0,
    };
}

function cancelScheduledDeepRenderWork() {
    if (deepRenderWorkHandle === null) {
        return;
    }

    if (
        deepRenderWorkHandleIsIdleCallback
        && typeof window !== 'undefined'
        && typeof window.cancelIdleCallback === 'function'
    ) {
        window.cancelIdleCallback(deepRenderWorkHandle);
    } else if (typeof clearTimeout === 'function') {
        clearTimeout(deepRenderWorkHandle);
    }

    deepRenderWorkHandle = null;
    deepRenderWorkHandleIsIdleCallback = false;
}

function invalidateActiveDeepRenderTask(reason = null) {
    if (!activeDeepRenderTask) {
        return;
    }

    clearDeepWorkState(activeDeepRenderTask.type, reason);
    activeDeepRenderTask = null;
}

function isDeepRenderTaskStale(task) {
    if (!task) {
        return false;
    }

    const currentCamera = getDeepCamera(task.type);
    if (!currentCamera) {
        return true;
    }

    return (
        getDeepCameraScaleRatio(task.camera, currentCamera) > DEEP_RENDER_STALE_SCALE_RATIO
        || getDeepCameraTranslationPixels(task.camera, currentCamera) > DEEP_RENDER_STALE_TRANSLATION_PIXELS
    );
}

function shouldTrackDeepRenderTask(type) {
    return Boolean(gl)
        && fractalType === type
        && isDeepFractalType(type)
        && shouldUseDeepRender(type)
        && !getDeepQualityHold(type);
}

function ensureDeepRenderTask(type) {
    if (!shouldTrackDeepRenderTask(type)) {
        invalidateActiveDeepRenderTask('task-disabled');
        return null;
    }

    if (activeDeepRenderTask && activeDeepRenderTask.type !== type) {
        invalidateActiveDeepRenderTask('mode-switch');
    }

    if (activeDeepRenderTask && isDeepRenderTaskStale(activeDeepRenderTask)) {
        invalidateActiveDeepRenderTask('task-stale');
    }

    if (!activeDeepRenderTask) {
        const committedCamera = getDeepCommittedCamera(type);
        const currentCamera = getDeepCamera(type);
        if (areDeepCamerasEquivalent(committedCamera, currentCamera) && getDeepCommittedFrameAvailable(type)) {
            setDeepFrameReady(type, true);
            return null;
        }
        activeDeepRenderTask = createDeepRenderTask(type);
    }

    return activeDeepRenderTask;
}

function getDeepRenderWorkBudgetMs(type, idleDeadline = null) {
    const maxBudget = getDeepCommittedFrameAvailable(type)
        ? DEEP_RENDER_MAX_BUDGET_MS
        : DEEP_RENDER_BOOTSTRAP_BUDGET_MS;

    if (idleDeadline && idleDeadline.didTimeout) {
        return maxBudget;
    }

    if (!idleDeadline || typeof idleDeadline.timeRemaining !== 'function') {
        return Math.min(maxBudget, DEEP_RENDER_FALLBACK_BUDGET_MS);
    }

    return Math.max(
        DEEP_RENDER_MIN_BUDGET_MS,
        Math.min(maxBudget, idleDeadline.timeRemaining())
    );
}

function isExpensiveDeepRenderStage(stage) {
    return [
        'initial-reference-search',
        'full-frame-render',
        'full-frame-verify',
        'full-frame-read-mask',
        'seed-repair-queue',
        'repair-tile',
        'commit',
    ].includes(stage);
}

function getMinimumBudgetForDeepRenderStage(type, stage) {
    if (!isExpensiveDeepRenderStage(stage)) {
        return 0;
    }

    return DEEP_RENDER_EXPENSIVE_STAGE_MIN_BUDGET_MS;
}

function shouldYieldBeforeDeepRenderStage(task, remainingBudgetMs) {
    if (!task || !Number.isFinite(remainingBudgetMs)) {
        return false;
    }

    const requiredBudgetMs = getMinimumBudgetForDeepRenderStage(task.type, task.stage);
    return requiredBudgetMs > 0 && remainingBudgetMs < requiredBudgetMs;
}

function noteDeepRenderSchedulerYield(task, remainingBudgetMs) {
    if (!task) {
        return;
    }

    task.frameStats.schedulerYields += 1;
    updateDeepWorkState(task.type, {
        stage: task.stage,
        referencesUsed: task.referencesUsed,
        repairQueueLength: task.repairQueue.length,
        repairTilesProcessed: task.processedRepairTiles,
        cpuTilesResolved: task.frameStats.cpuResolvedTiles,
        note: `yielding-before-${task.stage}`,
        remainingBudgetMs: roundDebugNumber(remainingBudgetMs),
    }, task.frameStats.schedulerYields <= 3 ? 'scheduler-yield' : null);
}

function restoreCommittedDeepCamera(type) {
    const committedCamera = getDeepCommittedCamera(type);
    if (!committedCamera) {
        setDeepFrameReady(type, false);
        return;
    }

    setDeepCamera(type, cloneDeepCamera(committedCamera));
    setDeepFrameReady(type, true);
}

function handleDeepRenderTaskFailure(task) {
    const type = task.type;
    const failedFrameStats = getDeepLastFrameStats(type) || task.frameStats;
    invalidateActiveDeepRenderTask('frame-failed');
    setDeepMaskVerificationFramesRemaining(type, 0);
    setDeepStableReuseFrames(type, 0);

    if (
        type === 'newton'
        && shouldDeferNewtonDeepFailure(failedFrameStats.reason)
        && canUseAccurateNewtonSimpleProxy(getDeepCamera(type))
    ) {
        setDeepFrameReady(type, false);
        setDeepReference(type, createEmptyReference());
        deferNewtonDeepRender(getDeepCamera(type).pixelScaleApprox);
        setDeepQualityHold(type, false);
        setDeepQualityHoldWarningShown(type, false);
        newtonDeferredCommittedFramePending = false;
        scheduleDeepRenderWork();
        return;
    }

    restoreCommittedDeepCamera(type);
    setDeepQualityHold(type, true);
    if (!getDeepQualityHoldWarningShown(type)) {
        console.warn(`Paused ${getDeepLabel(type)} zoom ${JSON.stringify(getDeepDebugSnapshot(type))}`);
        setDeepQualityHoldWarningShown(type, true);
    }
}

function finalizeDeepRenderTaskSuccess(task) {
    const type = task.type;
    const committedReferenceSelection = chooseCommittedReferenceSelection(
        type,
        task.initialSelection,
        task.bestReferenceSelection,
        task.referencesUsed
    );

    copyWorkingFrameToCommitted(type);
    commitDeepReference(type, committedReferenceSelection.reference, task.camera);
    setDeepCommittedCamera(type, cloneDeepCamera(task.camera));
    setDeepFrameReady(type, areDeepCamerasEquivalent(task.camera, getDeepCamera(type)));
    if (type === 'newton') {
        newtonDeferredCommittedFramePending = false;
        resetNewtonDeepRenderActivationScale();
    }
    finalizeDeepFrameStats(type, task.frameStats, 'success');
    setDeepQualityHold(type, false);
    setDeepQualityHoldWarningShown(type, false);
    invalidateActiveDeepRenderTask('frame-complete');
}

function advanceSingleDeepRenderTask(task) {
    const type = task.type;
    const camera = task.camera;
    const frameStats = task.frameStats;

    if (task.stage === 'init') {
        updateDeepWorkState(type, {
            stage: 'frame-start',
            referencesUsed: 0,
            repairQueueLength: 0,
            repairTilesProcessed: 0,
            cpuTilesResolved: 0,
            currentTile: null,
            note: null,
        }, 'frame-start');

        if (camera.pixelScaleApprox < MIN_GPU_SCALE) {
            finalizeDeepFrameStats(type, frameStats, 'failed', 'gpu_delta_precision_floor');
            handleDeepRenderTaskFailure(task);
            return false;
        }

        ensureMandelbrotRenderTargets();
        task.stage = 'initial-reference-search';
        return true;
    }

    if (task.stage === 'initial-reference-search') {
        updateDeepWorkState(type, {
            stage: 'initial-reference-search',
            note: 'selecting-initial-reference',
            currentTile: {
                width: task.fullFrameTile.width,
                height: task.fullFrameTile.height,
                depth: task.fullFrameTile.depth,
            },
        }, 'initial-reference-search');

        const anchorPoint = screenToPlaneDeep(camera, task.pointer);
        const initialSelection = selectInitialReference(anchorPoint, camera.maxIterations, type);
        if (!initialSelection) {
            finalizeDeepFrameStats(type, frameStats, 'failed', 'no_initial_reference');
            handleDeepRenderTaskFailure(task);
            return false;
        }

        task.initialSelection = initialSelection;
        task.bestReferenceSelection = {
            candidate: initialSelection.candidate,
            reference: initialSelection.reference,
        };
        task.deferredMaskVerification = canUseDeferredMaskVerification(type, initialSelection.mode);
        task.referencesUsed = 1;
        frameStats.initialEscapeIteration = initialSelection.candidate.escapeIteration;
        frameStats.initialOrbitLength = initialSelection.reference.orbitLength;
        frameStats.initialEscapedEarly = initialSelection.reference.escapedEarly;
        frameStats.initialReferenceMode = initialSelection.mode;
        frameStats.referencesUsed = 1;
        task.stage = 'full-frame-render';
        return true;
    }

    if (task.stage === 'full-frame-render') {
        updateDeepWorkState(type, {
            stage: 'full-frame-render',
            initialReferenceMode: task.initialSelection.mode,
            referencesUsed: 1,
            note: task.deferredMaskVerification ? 'deferred-mask-verification' : 'exact-mask-verification',
        }, 'full-frame-render');

        renderDeepPass(
            type,
            task.initialSelection.reference,
            task.fullFrameTile,
            !task.deferredMaskVerification,
            camera
        );

        if (task.deferredMaskVerification) {
            setDeepMaskVerificationFramesRemaining(
                type,
                Math.max(0, getDeepMaskVerificationFramesRemaining(type) - 1)
            );
            task.stage = 'commit';
            return true;
        }

        task.stage = 'full-frame-verify';
        return true;
    }

    if (task.stage === 'full-frame-verify') {
        updateDeepWorkState(type, {
            stage: 'full-frame-verify',
            note: 'checking-full-frame-mask',
        }, 'full-frame-verify');

        if (!tileHasGlitches(task.fullFrameTile)) {
            if (task.initialSelection.mode === 'reuse') {
                incrementDeepStableReuseFrames(type);
                setDeepMaskVerificationFramesRemaining(type, getAdaptiveMaskVerifySkipFrames(type));
            } else {
                setDeepStableReuseFrames(type, 0);
                setDeepMaskVerificationFramesRemaining(type, 0);
            }
            task.stage = 'commit';
            return true;
        }

        setDeepStableReuseFrames(type, 0);
        setDeepMaskVerificationFramesRemaining(type, 0);
        task.stage = 'full-frame-read-mask';
        return true;
    }

    if (task.stage === 'full-frame-read-mask') {
        updateDeepWorkState(type, {
            stage: 'full-frame-verify',
            note: 'reading-full-frame-mask',
        }, 'full-frame-read-mask');

        task.fullFrameTile.maskData = readGlitchMask(task.fullFrameTile);
        if (
            task.fullFrameTile.depth >= MAX_REPAIR_DEPTH
            || task.fullFrameTile.width <= MIN_REPAIR_TILE_SIZE
            || task.fullFrameTile.height <= MIN_REPAIR_TILE_SIZE
        ) {
            finalizeDeepFrameStats(type, frameStats, 'failed', 'full_frame_tile_limit', {
                queuedTilesRemaining: task.repairQueue.length,
            });
            handleDeepRenderTaskFailure(task);
            return false;
        }

        task.stage = 'seed-repair-queue';
        return true;
    }

    if (task.stage === 'seed-repair-queue') {
        queueChildTilesWithGlitches(task.repairQueue, task.fullFrameTile);
        sortRepairQueue(task.repairQueue);
        frameStats.repairQueuePeak = Math.max(frameStats.repairQueuePeak, task.repairQueue.length);
        updateDeepWorkState(type, {
            stage: 'repair-queue-seeded',
            repairQueueLength: task.repairQueue.length,
            note: 'queued-initial-repair-tiles',
        }, 'repair-queue-seeded');

        if (type === 'newton') {
            const initialRepairPressure = assessNewtonInitialRepairPressure(
                task.repairQueue.length,
                task.fullFrameTile.maskData
            );
            if (initialRepairPressure.shouldDefer) {
                finalizeDeepFrameStats(type, frameStats, 'failed', 'pathological_initial_repair_flood', {
                    queuedTilesRemaining: task.repairQueue.length,
                    fullFrameGlitchRatio: roundDebugNumber(initialRepairPressure.glitchRatio),
                    fullFrameGlitchedPixels: initialRepairPressure.glitchCount,
                });
                updateDeepWorkState(type, {
                    stage: 'repair-queue-seeded',
                    referencesUsed: task.referencesUsed,
                    repairQueueLength: task.repairQueue.length,
                    repairTilesProcessed: task.processedRepairTiles,
                    cpuTilesResolved: frameStats.cpuResolvedTiles,
                    note: 'deferring-deep-render-due-to-initial-glitch-flood',
                }, 'pathological-initial-repair-flood');
                handleDeepRenderTaskFailure(task);
                return false;
            }
        }

        task.repairWorkStartMs = nowMs();
        task.stage = task.repairQueue.length > 0 ? 'repair-tile' : 'commit';
        return true;
    }

    if (task.stage === 'repair-tile') {
        if (task.repairQueue.length === 0) {
            task.stage = 'commit';
            return true;
        }

        const elapsedMs = nowMs() - task.repairWorkStartMs;
        if (
            type === 'newton'
            && shouldDeferNewtonRepairWork(task.referencesUsed, task.repairQueue.length, elapsedMs)
        ) {
            finalizeDeepFrameStats(type, frameStats, 'failed', 'pathological_repair_growth', {
                queuedTilesRemaining: task.repairQueue.length,
                elapsedMs: roundDebugNumber(elapsedMs),
                referencesUsed: task.referencesUsed,
            });
            updateDeepWorkState(type, {
                stage: 'repair-reference-search',
                referencesUsed: task.referencesUsed,
                repairQueueLength: task.repairQueue.length,
                repairTilesProcessed: task.processedRepairTiles,
                cpuTilesResolved: frameStats.cpuResolvedTiles,
                note: 'deferring-deep-render-due-to-repair-growth',
            }, 'pathological-repair-growth');
            handleDeepRenderTaskFailure(task);
            return false;
        }

        if (task.referencesUsed >= MAX_REPAIR_REFERENCES) {
            finalizeDeepFrameStats(type, frameStats, 'failed', 'repair_reference_budget_exhausted', {
                queuedTilesRemaining: task.repairQueue.length,
            });
            handleDeepRenderTaskFailure(task);
            return false;
        }

        const tile = task.repairQueue.shift();
        task.processedRepairTiles += 1;
        frameStats.lastTileWidth = tile.width;
        frameStats.lastTileHeight = tile.height;
        frameStats.lastTileDepth = tile.depth;
        frameStats.deepestTileDepth = Math.max(frameStats.deepestTileDepth, tile.depth);
        frameStats.repairQueuePeak = Math.max(frameStats.repairQueuePeak, task.repairQueue.length);

        updateDeepWorkState(type, {
            stage: 'repair-reference-search',
            referencesUsed: task.referencesUsed,
            repairQueueLength: task.repairQueue.length,
            repairTilesProcessed: task.processedRepairTiles,
            cpuTilesResolved: frameStats.cpuResolvedTiles,
            currentTile: {
                width: tile.width,
                height: tile.height,
                depth: tile.depth,
            },
            note: 'searching-repair-reference',
        }, task.processedRepairTiles <= 4 || task.processedRepairTiles % 16 === 0 ? 'repair-progress' : null);

        const glitchedPixel = findGlitchedPixelNearTileCenter(tile);
        if (!glitchedPixel) {
            return true;
        }

        const repairPointer = framebufferPixelToPointer(glitchedPixel);
        const repairAnchor = screenToPlaneDeep(camera, repairPointer);
        const repairCandidate = findBestReferencePoint(repairAnchor, camera.maxIterations, type);
        if (!repairCandidate) {
            finalizeDeepFrameStats(type, frameStats, 'failed', 'no_repair_reference', {
                queuedTilesRemaining: task.repairQueue.length,
                lastTileWidth: tile.width,
                lastTileHeight: tile.height,
                lastTileDepth: tile.depth,
            });
            handleDeepRenderTaskFailure(task);
            return false;
        }

        frameStats.lastRepairEscapeIteration = repairCandidate.escapeIteration;
        const repairReference = getReferenceOrbit(repairCandidate.point, camera.maxIterations, type);
        const repairCandidateForCommit = {
            ...repairCandidate,
            distanceSquared: getDeepPointScreenDistanceSquared(camera, repairCandidate.point),
        };
        if (isReferenceCandidateBetter(repairCandidateForCommit, task.bestReferenceSelection.candidate)) {
            task.bestReferenceSelection = {
                candidate: repairCandidateForCommit,
                reference: repairReference,
            };
        }

        updateDeepWorkState(type, {
            stage: 'repair-render',
            referencesUsed: task.referencesUsed,
            repairQueueLength: task.repairQueue.length,
            repairTilesProcessed: task.processedRepairTiles,
            cpuTilesResolved: frameStats.cpuResolvedTiles,
            note: 'rendering-repair-tile',
        }, null);
        renderDeepPass(type, repairReference, tile, true, camera);
        task.referencesUsed += 1;
        frameStats.referencesUsed = task.referencesUsed;

        updateDeepWorkState(type, {
            stage: 'repair-verify',
            referencesUsed: task.referencesUsed,
            repairQueueLength: task.repairQueue.length,
            repairTilesProcessed: task.processedRepairTiles,
            cpuTilesResolved: frameStats.cpuResolvedTiles,
            note: 'checking-repair-mask',
        }, null);
        if (!tileHasGlitches(tile)) {
            return true;
        }

        tile.maskData = readGlitchMask(tile);
        if (
            tile.depth >= MAX_REPAIR_DEPTH
            || tile.width <= MIN_REPAIR_TILE_SIZE
            || tile.height <= MIN_REPAIR_TILE_SIZE
        ) {
            updateDeepWorkState(type, {
                stage: 'cpu-tile-resolve',
                referencesUsed: task.referencesUsed,
                repairQueueLength: task.repairQueue.length,
                repairTilesProcessed: task.processedRepairTiles,
                cpuTilesResolved: frameStats.cpuResolvedTiles,
                note: 'resolving-minimum-tile-on-cpu',
            }, 'cpu-tile-resolve');
            resolveDeepTileOnCPU(type, tile, camera);
            frameStats.cpuResolvedTiles += 1;
            frameStats.cpuResolvedPixels += tile.width * tile.height;
            return true;
        }

        queueChildTilesWithGlitches(task.repairQueue, tile);
        sortRepairQueue(task.repairQueue);
        frameStats.repairQueuePeak = Math.max(frameStats.repairQueuePeak, task.repairQueue.length);
        return true;
    }

    if (task.stage === 'commit') {
        updateDeepWorkState(type, {
            stage: 'commit-frame',
            referencesUsed: task.referencesUsed,
            repairQueueLength: task.repairQueue.length,
            repairTilesProcessed: task.processedRepairTiles,
            cpuTilesResolved: frameStats.cpuResolvedTiles,
            currentTile: null,
            note: 'committing-rendered-frame',
        }, 'commit-frame');
        finalizeDeepRenderTaskSuccess(task);
        return false;
    }

    return false;
}

function advanceDeepRenderTask(type, budgetMs) {
    const task = ensureDeepRenderTask(type);
    if (!task) {
        return false;
    }

    const clampedBudgetMs = Math.max(DEEP_RENDER_MIN_BUDGET_MS, budgetMs);
    const deadline = nowMs() + clampedBudgetMs;
    let madeProgress = false;

    while (activeDeepRenderTask && activeDeepRenderTask === task && nowMs() < deadline) {
        const remainingBudgetMs = deadline - nowMs();
        if (madeProgress && shouldYieldBeforeDeepRenderStage(task, remainingBudgetMs)) {
            noteDeepRenderSchedulerYield(task, remainingBudgetMs);
            break;
        }

        const advanced = advanceSingleDeepRenderTask(task);
        madeProgress = madeProgress || advanced;
        if (!advanced) {
            break;
        }
    }

    return madeProgress;
}

function runScheduledDeepRenderWork(idleDeadline = null) {
    deepRenderWorkHandle = null;
    deepRenderWorkHandleIsIdleCallback = false;

    if (!gl || !shouldTrackDeepRenderTask(fractalType)) {
        invalidateActiveDeepRenderTask('task-disabled');
        return;
    }

    const type = fractalType;
    const budgetMs = getDeepRenderWorkBudgetMs(type, idleDeadline);
    advanceDeepRenderTask(type, budgetMs);

    if (shouldTrackDeepRenderTask(type) && (activeDeepRenderTask || !getDeepFrameReady(type))) {
        scheduleDeepRenderWork();
    }
}

function scheduleDeepRenderWork() {
    if (deepRenderWorkHandle !== null || !gl || !shouldTrackDeepRenderTask(fractalType)) {
        return;
    }

    if (typeof window !== 'undefined' && typeof window.requestIdleCallback === 'function') {
        deepRenderWorkHandleIsIdleCallback = true;
        deepRenderWorkHandle = window.requestIdleCallback(
            (idleDeadline) => {
                runScheduledDeepRenderWork(idleDeadline);
            },
            { timeout: DEEP_RENDER_IDLE_TIMEOUT_MS }
        );
        return;
    }

    if (typeof setTimeout === 'function') {
        deepRenderWorkHandleIsIdleCallback = false;
        deepRenderWorkHandle = setTimeout(() => {
            runScheduledDeepRenderWork();
        }, 0);
        return;
    }

    runScheduledDeepRenderWork();
}

function updateDeepCameraForType(type) {
    if (type === 'julia') {
        updateJuliaCamera();
        return;
    }
    if (type === 'newton') {
        updateNewtonCamera();
        return;
    }
    updateMandelbrotCamera();
}

function stepDeepCameraWithQualityPriority(type) {
    if (getDeepQualityHold(type)) {
        return false;
    }

    updateDeepCameraForType(type);
    setDeepFrameReady(type, false);
    incrementDeepZoomStepCount(type);
    logDeepProgress(type);
    setDeepQualityHold(type, false);
    setDeepQualityHoldWarningShown(type, false);
    scheduleDeepRenderWork();
    return true;
}

function stepMandelbrotCameraWithQualityPriority() {
    return stepDeepCameraWithQualityPriority('mandelbrot');
}

function stepJuliaCameraWithQualityPriority() {
    return stepDeepCameraWithQualityPriority('julia');
}

function stepNewtonCameraWithQualityPriority() {
    return stepDeepCameraWithQualityPriority('newton');
}

function drawSimpleFractalToFramebuffer(activeType, camera, framebuffer = null) {
    const fractalIndex = ['mandelbrot', 'julia', 'newton'].indexOf(activeType);

    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
    gl.useProgram(simpleProgramInfo.program);
    bindQuad(simpleProgramInfo);
    gl.uniform2f(simpleProgramInfo.uniforms.u_resolution, gl.canvas.width, gl.canvas.height);
    gl.uniform2f(simpleProgramInfo.uniforms.u_center, camera.centerX, camera.centerY);
    gl.uniform1f(simpleProgramInfo.uniforms.u_pixelScale, camera.pixelScale);
    gl.uniform1i(simpleProgramInfo.uniforms.u_fractalType, fractalIndex);
    gl.uniform1i(simpleProgramInfo.uniforms.u_maxIterations, camera.maxIterations);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
}

function drawSimpleFractal(activeType, camera) {
    drawSimpleFractalToFramebuffer(activeType, camera, null);
}

function cacheNewtonSimpleProxyFrame(camera) {
    const committedFramebuffer = getDeepCommittedFramebuffer('newton');
    if (!committedFramebuffer) {
        return false;
    }
    drawSimpleFractalToFramebuffer('newton', camera, committedFramebuffer);
    setDeepCommittedCamera('newton', cloneDeepCamera(getDeepCamera('newton')));
    setDeepCommittedFrameAvailable('newton', true);
    return true;
}

function drawDeepFractal(type) {
    const camera = getDeepCamera(type);

    if (camera.pixelScaleApprox < MIN_GPU_SCALE && !getDeepPrecisionWarningShown(type)) {
        console.warn(`${getDeepLabel(type)} zoom reached the GPU delta precision floor. The CPU camera stays precise, but visual detail will eventually plateau.`);
        setDeepPrecisionWarningShown(type, true);
    }

    scheduleDeepRenderWork();

    if (drawCommittedDeepFrame(type, camera)) {
        noteRenderedPath('deep');
        return;
    }

    if (type === 'newton') {
        if (drawSimpleProxyFromDeepCamera(type)) {
            noteRenderedPath('simple-proxy');
        } else {
            noteRenderedPath('blank');
        }
        return;
    }

    drawSimpleFractal(type, {
        centerX: decimalToNumber(camera.centerX),
        centerY: decimalToNumber(camera.centerY),
        pixelScale: camera.pixelScaleApprox,
        maxIterations: camera.maxIterations,
    });
    noteRenderedPath('simple');
}

function drawSimpleProxyFromDeepCamera(type) {
    const camera = getDeepCamera(type);
    if (type === 'newton' && !canUseAccurateNewtonSimpleProxy(camera)) {
        return false;
    }
    let maxIterations = camera.maxIterations;
    if (type === 'newton') {
        maxIterations = getNewtonSimpleProxyMaxIterations(camera);
    }
    const simpleProxyCamera = {
        centerX: decimalToNumber(camera.centerX),
        centerY: decimalToNumber(camera.centerY),
        pixelScale: camera.pixelScaleApprox,
        maxIterations,
    };
    drawSimpleFractal(type, simpleProxyCamera);
    if (type === 'newton') {
        cacheNewtonSimpleProxyFrame(simpleProxyCamera);
    }
    return true;
}

function drawDeepMandelbrot() {
    drawDeepFractal('mandelbrot');
}

function drawDeepJulia() {
    drawDeepFractal('julia');
}

function drawDeepNewton() {
    drawDeepFractal('newton');
}

function draw() {
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
    gl.clear(gl.COLOR_BUFFER_BIT);

    if (fractalType === 'newton' && newtonDeferredCommittedFramePending) {
        if (drawCommittedDeepFrame('newton', getDeepCamera('newton'))) {
            noteRenderedPath('deep');
        } else {
            noteRenderedPath('blank');
        }
        newtonDeferredCommittedFramePending = false;
        return;
    }

    if (isDeepFractalType(fractalType) && shouldUseDeepRender(fractalType)) {
        drawDeepFractal(fractalType);
        return;
    }

    if (fractalType === 'newton') {
        if (drawSimpleProxyFromDeepCamera('newton')) {
            noteRenderedPath('simple-proxy');
        } else if (drawCommittedDeepFrame('newton', getDeepCamera('newton'))) {
            noteRenderedPath('deep');
        } else {
            noteRenderedPath('blank');
        }
        return;
    }

    drawSimpleFractal(fractalType, simpleCamera);
    noteRenderedPath('simple');
}

function animate() {
    if (!gl) {
        return;
    }

    debugAnimationFrameCount += 1;
    debugLastAnimationFrameAt = Date.now();
    debugLastAnimationStage = 'step';
    if (isDeepFractalType(fractalType) && shouldUseDeepRender(fractalType)) {
        stepDeepCameraWithQualityPriority(fractalType);
    } else if (fractalType === 'newton') {
        updateNewtonCamera();
    } else {
        updateSimpleCamera(simpleCamera);
    }

    debugLastAnimationStage = 'draw';
    draw();
    if (isDeepFractalType(fractalType) && shouldUseDeepRender(fractalType)) {
        scheduleDeepRenderWork();
    }
    noteRenderPathChange();
    debugLastAnimationStage = 'frame-complete';
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
    const dpr = getRenderDevicePixelRatio();
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
    juliaFrameReady = false;
    newtonFrameReady = false;
    newtonDeferredCommittedFramePending = false;
    invalidateActiveDeepRenderTask('resize');
    cancelScheduledDeepRenderWork();

    if (isDeepFractalType(fractalType) && getDeepCamera(fractalType)) {
        const deepCamera = getDeepCamera(fractalType);
        deepCamera.viewWidth = deepCamera.pixelScaleApprox * Math.min(gl.canvas.width, gl.canvas.height);
        setDeepQualityHold(fractalType, false);
        setDeepQualityHoldWarningShown(fractalType, false);
        setDeepMaskVerificationFramesRemaining(fractalType, 0);
        setDeepStableReuseFrames(fractalType, 0);
        clearDeepWorkState(fractalType);
        if (fractalType === 'newton') {
            resetNewtonDeepRenderActivationScale();
        }
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
        simpleCamera = createSimpleCamera(-1.2, 0.1, 2.4);
    }
    updateSimpleCameraScale(simpleCamera);
}

function resetMandelbrotState() {
    invalidateActiveDeepRenderTask('reset-mandelbrot');
    cancelScheduledDeepRenderWork();
    mandelbrotCamera = createMandelbrotCamera();
    mandelbrotReference = createEmptyReference();
    mandelbrotReferenceCache = new Map();
    mandelbrotFrameReady = false;
    mandelbrotCommittedFrameAvailable = false;
    mandelbrotCommittedCamera = null;
    deepPrecisionWarningShown = false;
    mandelbrotQualityHold = false;
    mandelbrotQualityHoldWarningShown = false;
    mandelbrotZoomStepCount = 0;
    mandelbrotLastFrameStats = createEmptyMandelbrotFrameStats();
    mandelbrotMaskVerificationFramesRemaining = 0;
    mandelbrotStableReuseFrames = 0;
    mandelbrotDeepWorkState = null;
    debugLastRenderedPath = null;
}

function resetJuliaState() {
    invalidateActiveDeepRenderTask('reset-julia');
    cancelScheduledDeepRenderWork();
    juliaCamera = createJuliaCamera();
    juliaReference = createEmptyReference();
    juliaReferenceCache = new Map();
    juliaFrameReady = false;
    juliaCommittedFrameAvailable = false;
    juliaCommittedCamera = null;
    juliaDeepPrecisionWarningShown = false;
    juliaQualityHold = false;
    juliaQualityHoldWarningShown = false;
    juliaZoomStepCount = 0;
    juliaLastFrameStats = createEmptyMandelbrotFrameStats();
    juliaMaskVerificationFramesRemaining = 0;
    juliaStableReuseFrames = 0;
    juliaDeepWorkState = null;
    debugLastRenderedPath = null;
}

function resetNewtonState() {
    invalidateActiveDeepRenderTask('reset-newton');
    cancelScheduledDeepRenderWork();
    newtonCamera = createNewtonCamera();
    newtonReference = createEmptyReference();
    newtonReferenceCache = new Map();
    newtonFrameReady = false;
    newtonCommittedFrameAvailable = false;
    newtonCommittedCamera = null;
    newtonDeepPrecisionWarningShown = false;
    newtonQualityHold = false;
    newtonQualityHoldWarningShown = false;
    newtonZoomStepCount = 0;
    newtonLastFrameStats = createEmptyMandelbrotFrameStats();
    newtonMaskVerificationFramesRemaining = 0;
    newtonStableReuseFrames = 0;
    newtonDeepWorkState = null;
    newtonDeferredCommittedFramePending = false;
    debugLastRenderedPath = null;
    resetNewtonDeepRenderActivationScale();
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
    if (isDeepFractalType(fractalType)) {
        setDeepQualityHold(fractalType, false);
        setDeepQualityHoldWarningShown(fractalType, false);
        setDeepMaskVerificationFramesRemaining(fractalType, 0);
        setDeepStableReuseFrames(fractalType, 0);
        clearDeepWorkState(fractalType);
        scheduleDeepRenderWork();
    }
}

function handleFractalTypeChange(event) {
    fractalType = event.target.value;
    if (fractalType === 'mandelbrot') {
        resetMandelbrotState();
    } else if (fractalType === 'julia') {
        resetJuliaState();
    } else if (fractalType === 'newton') {
        resetNewtonState();
    } else {
        resetSimpleCamera(fractalType);
    }
    setMouseToCenter();
    draw();
    if (isDeepFractalType(fractalType) && shouldUseDeepRender(fractalType)) {
        scheduleDeepRenderWork();
    }
    noteRenderPathChange('mode-change');
}

function preventZoom(event) {
    event.preventDefault();
    event.stopPropagation();
}

window.addEventListener('load', () => {
    if (!initWebGL()) {
        return;
    }

    installDebugControls();
    document.getElementById('fractalType').addEventListener('change', handleFractalTypeChange);
    gl.canvas.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('wheel', preventZoom, { passive: false });
    document.addEventListener('touchmove', preventZoom, { passive: false });

    resizeCanvas();
    resetMandelbrotState();
    resetJuliaState();
    resetNewtonState();
    setMouseToCenter();
    configureDebugHeartbeatFromQuery();
    draw();
    if (isDeepFractalType(fractalType) && shouldUseDeepRender(fractalType)) {
        scheduleDeepRenderWork();
    }
    noteRenderPathChange('load');
    requestAnimationFrame(animate);
});

window.addEventListener('resize', () => {
    resizeCanvas();
});
