const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function loadHarness() {
    const scriptPath = path.join(__dirname, 'script.js');
    const source = fs.readFileSync(scriptPath, 'utf8');
    const harnessSource = `${source}
;globalThis.__testHarness = {
    setGL(value) { gl = value; },
    setMouse(value) { mousePosition = value; },
    initMandelbrot() {
        fractalType = 'mandelbrot';
        mandelbrotCamera = createMandelbrotCamera();
        mandelbrotReference = createEmptyReference();
        mandelbrotReferenceCache = new Map();
        mandelbrotQualityHold = false;
        mandelbrotQualityHoldWarningShown = false;
        mandelbrotFrameReady = false;
        deepPrecisionWarningShown = false;
        mandelbrotMaskVerificationFramesRemaining = 0;
        mandelbrotStableReuseFrames = 0;
    },
    initJulia() {
        fractalType = 'julia';
        juliaCamera = createJuliaCamera();
        juliaReference = createEmptyReference();
        juliaReferenceCache = new Map();
        juliaQualityHold = false;
        juliaQualityHoldWarningShown = false;
        juliaFrameReady = false;
        juliaDeepPrecisionWarningShown = false;
        juliaMaskVerificationFramesRemaining = 0;
        juliaStableReuseFrames = 0;
    },
    initNewton() {
        fractalType = 'newton';
        newtonCamera = createNewtonCamera();
        newtonReference = createEmptyReference();
        newtonReferenceCache = new Map();
        newtonQualityHold = false;
        newtonQualityHoldWarningShown = false;
        newtonFrameReady = false;
        newtonDeepPrecisionWarningShown = false;
        newtonMaskVerificationFramesRemaining = 0;
        newtonStableReuseFrames = 0;
    },
    setRenderSharpImpl(fn) { renderSharpDeepFrame = fn; },
    setEscapeIterationImpl(fn) { computeEscapeIteration = fn; },
    setOrbitTexture(value) { orbitTexture = value; },
    setOrbitTextureCapacity(value) { orbitTextureCapacity = value; },
    getOrbitTextureCapacity() { return orbitTextureCapacity; },
    uploadOrbit(orbitData, orbitLength) { uploadReferenceOrbit(orbitData, orbitLength); },
    setRenderTargets(targets) {
        mandelbrotWorkingFramebuffer = targets.workingFramebuffer;
        mandelbrotCommittedFramebuffer = targets.committedFramebuffer;
        mandelbrotWorkingColorTexture = targets.workingColorTexture;
        mandelbrotWorkingMaskTexture = targets.workingMaskTexture;
        mandelbrotCommittedColorTexture = targets.committedColorTexture;
    },
    getRenderTargets() {
        return {
            workingColorTexture: mandelbrotWorkingColorTexture,
            workingMaskTexture: mandelbrotWorkingMaskTexture,
            committedColorTexture: mandelbrotCommittedColorTexture,
        };
    },
    commitWorkingFrame() { copyWorkingFrameToCommitted(); },
    createPoint(x, y) {
        return {
            x: decimalFromString(x),
            y: decimalFromString(y),
        };
    },
    getOrbit(point, iterations, type) { return getReferenceOrbit(point, iterations, type); },
    findReferenceAtMouse() {
        return findBestReferencePoint(
            screenToPlaneDeep(getDeepCamera(fractalType), mousePosition),
            getDeepCamera(fractalType).maxIterations
        );
    },
    queueChildren(tile) {
        const queue = [];
        queueChildTilesWithGlitches(queue, tile);
        return queue;
    },
    createTile(x, y, width, height, depth, maskData) {
        return createRepairTile(x, y, width, height, depth, maskData);
    },
    readMaskAndCheck(tile) { return readGlitchMaskAndCheck(tile); },
    setWorkingFramebufferDrawBuffers(writeMask) { setWorkingFramebufferDrawBuffers(writeMask); },
    stepQuality() { return stepMandelbrotCameraWithQualityPriority(); },
    stepJuliaQuality() { return stepJuliaCameraWithQualityPriority(); },
    stepNewtonQuality() { return stepNewtonCameraWithQualityPriority(); },
    setStableReuseFrames(type, value) { setDeepStableReuseFrames(type, value); },
    getStableReuseFrames(type) { return getDeepStableReuseFrames(type); },
    getAdaptiveMaskVerifySkipFrames(type) { return getAdaptiveMaskVerifySkipFrames(type); },
    computeIterationBudget(pixelScale) { return computeIterationBudget(pixelScale); },
    computeNewtonIterationBudget(pixelScale) { return computeNewtonIterationBudget(pixelScale); },
    shouldUseDeepRender(type) { return shouldUseDeepRender(type); },
    setNewtonPixelScaleApprox(value) { newtonCamera.pixelScaleApprox = value; },
    computeEscape(point, iterations, type) {
        return computeEscapeIteration(point.x, point.y, iterations, type);
    },
    computeCpuColor(point, iterations, type) {
        return computeCpuDeepColor(type, point.x, point.y, iterations);
    },
    selectInitialReference(point, iterations, type) {
        return selectInitialReference(point, iterations, type);
    },
    getState() {
        const camera = getDeepCamera(fractalType);
        return {
            pixelScaleApprox: camera.pixelScaleApprox,
            maxIterations: camera.maxIterations,
            hold: getDeepQualityHold(fractalType),
        };
    },
};
`;

    const context = {
        console: { warn() {}, log() {}, error() {} },
        Float32Array,
        Uint8Array,
        BigInt,
        Math,
        Number,
        Map,
        Set,
        window: {
            addEventListener() {},
            devicePixelRatio: 1,
            innerWidth: 1600,
            innerHeight: 900,
        },
        document: {
            getElementById() {
                return { addEventListener() {} };
            },
            addEventListener() {},
        },
        requestAnimationFrame() {},
        alert() {},
    };

    vm.createContext(context);
    vm.runInContext(harnessSource, context);

    context.__testHarness.setGL({
        canvas: { width: 1600, height: 900 },
    });

    return context.__testHarness;
}

test('findBestReferencePoint prefers longer dwell over nearer distance', () => {
    const harness = loadHarness();
    harness.initMandelbrot();
    harness.setMouse({ x: 800, y: 450 });

    let callCount = 0;
    harness.setEscapeIterationImpl(() => {
        callCount += 1;
        if (callCount === 1) {
            return 40;
        }
        if (callCount === 2) {
            return 120;
        }
        return 80;
    });

    const candidate = harness.findReferenceAtMouse();
    assert.ok(candidate);
    assert.equal(candidate.escapeIteration, 120);
});

test('queueChildTilesWithGlitches only enqueues children that contain glitches', () => {
    const harness = loadHarness();
    const maskData = new Uint8Array(4 * 4);

    for (let row = 0; row < 2; row += 1) {
        for (let column = 0; column < 2; column += 1) {
            const baseIndex = (row * 4) + column;
            maskData[baseIndex] = 255;
        }
    }

    const tile = harness.createTile(0, 0, 4, 4, 0, maskData);
    const children = harness.queueChildren(tile);

    assert.equal(children.length, 1);
    assert.deepEqual(
        { x: children[0].x, y: children[0].y, width: children[0].width, height: children[0].height },
        { x: 0, y: 0, width: 2, height: 2 }
    );
});

test('readGlitchMaskAndCheck returns false for a clean mask', () => {
    const harness = loadHarness();
    const tile = harness.createTile(0, 0, 4, 2, 0, null);
    let readPixelsCount = 0;

    harness.setGL({
        canvas: { width: 1600, height: 900 },
        READ_FRAMEBUFFER: 'READ_FRAMEBUFFER',
        COLOR_ATTACHMENT1: 'COLOR_ATTACHMENT1',
        RED: 'RED',
        UNSIGNED_BYTE: 'UNSIGNED_BYTE',
        bindFramebuffer() {},
        readBuffer() {},
        readPixels(x, y, width, height, format, type, destination) {
            readPixelsCount += 1;
            destination.fill(0);
        },
    });

    const result = harness.readMaskAndCheck(tile);

    assert.equal(readPixelsCount, 1);
    assert.equal(result.hasGlitches, false);
    assert.deepEqual(Array.from(result.maskData), [0, 0, 0, 0, 0, 0, 0, 0]);
});

test('readGlitchMaskAndCheck returns true and preserves the mask bytes', () => {
    const harness = loadHarness();
    const tile = harness.createTile(2, 3, 4, 2, 0, null);

    harness.setGL({
        canvas: { width: 1600, height: 900 },
        READ_FRAMEBUFFER: 'READ_FRAMEBUFFER',
        COLOR_ATTACHMENT1: 'COLOR_ATTACHMENT1',
        RED: 'RED',
        UNSIGNED_BYTE: 'UNSIGNED_BYTE',
        bindFramebuffer() {},
        readBuffer() {},
        readPixels(x, y, width, height, format, type, destination) {
            destination.set([0, 0, 255, 0, 0, 0, 0, 0]);
        },
    });

    const result = harness.readMaskAndCheck(tile);

    assert.equal(result.hasGlitches, true);
    assert.deepEqual(Array.from(result.maskData), [0, 0, 255, 0, 0, 0, 0, 0]);
});

test('stepMandelbrotCameraWithQualityPriority restores the previous camera when repair fails', () => {
    const harness = loadHarness();
    harness.initMandelbrot();
    harness.setMouse({ x: 1320, y: 180 });

    const before = harness.getState();
    harness.setRenderSharpImpl(() => false);

    const advanced = harness.stepQuality();
    const after = harness.getState();

    assert.equal(advanced, false);
    assert.equal(after.hold, true);
    assert.equal(after.pixelScaleApprox, before.pixelScaleApprox);
    assert.equal(after.maxIterations, before.maxIterations);
});

test('stepMandelbrotCameraWithQualityPriority advances when sharp-frame render succeeds', () => {
    const harness = loadHarness();
    harness.initMandelbrot();
    harness.setMouse({ x: 1320, y: 180 });

    const before = harness.getState();
    harness.setRenderSharpImpl(() => true);

    const advanced = harness.stepQuality();
    const after = harness.getState();

    assert.equal(advanced, true);
    assert.equal(after.hold, false);
    assert.ok(after.pixelScaleApprox < before.pixelScaleApprox);
});

test('uploadReferenceOrbit reuses existing texture storage when capacity is sufficient', () => {
    const harness = loadHarness();
    const calls = [];
    harness.setGL({
        canvas: { width: 1600, height: 900 },
        TEXTURE0: 'TEXTURE0',
        TEXTURE_2D: 'TEXTURE_2D',
        RGBA32F: 'RGBA32F',
        RGBA: 'RGBA',
        FLOAT: 'FLOAT',
        activeTexture(value) {
            calls.push(['activeTexture', value]);
        },
        bindTexture(target, texture) {
            calls.push(['bindTexture', target, texture]);
        },
        texImage2D(...args) {
            calls.push(['texImage2D', ...args]);
        },
        texSubImage2D(...args) {
            calls.push(['texSubImage2D', ...args]);
        },
    });
    harness.setOrbitTexture({ id: 'orbit' });
    harness.setOrbitTextureCapacity(16);

    harness.uploadOrbit(new Float32Array(8 * 4), 8);

    assert.equal(calls.filter(([name]) => name === 'texImage2D').length, 0);
    assert.equal(calls.filter(([name]) => name === 'texSubImage2D').length, 1);
    assert.equal(harness.getOrbitTextureCapacity(), 16);
});

test('getReferenceOrbit extends a cached reference instead of recomputing from scratch', () => {
    const harness = loadHarness();
    const point = harness.createPoint('-0.745', '0.1');

    const initialReference = harness.getOrbit(point, 24, 'mandelbrot');
    const initialOrbitLength = initialReference.orbitLength;
    const initialComputedIterations = initialReference.computedIterations;

    const extendedReference = harness.getOrbit(point, 40, 'mandelbrot');

    assert.equal(extendedReference, initialReference);
    assert.ok(extendedReference.computedIterations > initialComputedIterations);
    assert.ok(extendedReference.orbitLength > initialOrbitLength);
});

test('copyWorkingFrameToCommitted swaps color textures instead of blitting', () => {
    const harness = loadHarness();
    const calls = [];
    harness.setGL({
        FRAMEBUFFER: 'FRAMEBUFFER',
        COLOR_ATTACHMENT0: 'COLOR_ATTACHMENT0',
        COLOR_ATTACHMENT1: 'COLOR_ATTACHMENT1',
        TEXTURE_2D: 'TEXTURE_2D',
        bindFramebuffer(target, framebuffer) {
            calls.push(['bindFramebuffer', target, framebuffer]);
        },
        framebufferTexture2D(...args) {
            calls.push(['framebufferTexture2D', ...args]);
        },
        drawBuffers(buffers) {
            calls.push(['drawBuffers', ...buffers]);
        },
    });
    harness.setRenderTargets({
        workingFramebuffer: { id: 'working-fbo' },
        committedFramebuffer: { id: 'committed-fbo' },
        workingColorTexture: { id: 'working-color' },
        workingMaskTexture: { id: 'working-mask' },
        committedColorTexture: { id: 'committed-color' },
    });

    harness.commitWorkingFrame();

    const targets = harness.getRenderTargets();
    assert.equal(targets.workingColorTexture.id, 'committed-color');
    assert.equal(targets.committedColorTexture.id, 'working-color');
    assert.ok(calls.some(([name]) => name === 'framebufferTexture2D'));
});

test('setWorkingFramebufferDrawBuffers can disable mask writes on deferred frames', () => {
    const harness = loadHarness();
    const calls = [];
    harness.setGL({
        COLOR_ATTACHMENT0: 'COLOR_ATTACHMENT0',
        COLOR_ATTACHMENT1: 'COLOR_ATTACHMENT1',
        drawBuffers(buffers) {
            calls.push(buffers.slice());
        },
    });

    harness.setWorkingFramebufferDrawBuffers(false);
    harness.setWorkingFramebufferDrawBuffers(true);

    assert.equal(calls[0].join(','), 'COLOR_ATTACHMENT0');
    assert.equal(calls[1].join(','), 'COLOR_ATTACHMENT0,COLOR_ATTACHMENT1');
});

test('adaptive mask verification skip grows only after repeated clean reuse frames', () => {
    const harness = loadHarness();
    harness.initMandelbrot();

    harness.setStableReuseFrames('mandelbrot', 0);
    assert.equal(harness.getAdaptiveMaskVerifySkipFrames('mandelbrot'), 1);

    harness.setStableReuseFrames('mandelbrot', 6);
    assert.equal(harness.getAdaptiveMaskVerifySkipFrames('mandelbrot'), 2);

    harness.setStableReuseFrames('mandelbrot', 12);
    assert.equal(harness.getAdaptiveMaskVerifySkipFrames('mandelbrot'), 3);

    harness.setStableReuseFrames('mandelbrot', 16);
    assert.equal(harness.getAdaptiveMaskVerifySkipFrames('mandelbrot'), 3);

    harness.setStableReuseFrames('mandelbrot', 48);
    assert.equal(harness.getAdaptiveMaskVerifySkipFrames('mandelbrot'), 3);
});

test('deep iteration budget keeps its floor while growing with depth', () => {
    const harness = loadHarness();

    assert.equal(harness.computeIterationBudget(1), 220);
    assert.equal(harness.computeIterationBudget(1e-6), 374);
    assert.ok(harness.computeIterationBudget(1e-12) > harness.computeIterationBudget(1e-6));
});

test('Newton iteration budget starts lower while still growing with depth', () => {
    const harness = loadHarness();

    assert.equal(harness.computeNewtonIterationBudget(1), 64);
    assert.equal(harness.computeNewtonIterationBudget(1e-6), 112);
    assert.ok(harness.computeNewtonIterationBudget(1e-12) > harness.computeNewtonIterationBudget(1e-6));
});

test('initJulia resets stable reuse cadence state', () => {
    const harness = loadHarness();
    harness.setStableReuseFrames('julia', 9);

    harness.initJulia();

    assert.equal(harness.getStableReuseFrames('julia'), 0);
    assert.equal(harness.getAdaptiveMaskVerifySkipFrames('julia'), 1);
});

test('initNewton resets stable reuse cadence state', () => {
    const harness = loadHarness();
    harness.setStableReuseFrames('newton', 9);

    harness.initNewton();

    assert.equal(harness.getStableReuseFrames('newton'), 0);
    assert.equal(harness.getAdaptiveMaskVerifySkipFrames('newton'), 1);
});

test('stepJuliaCameraWithQualityPriority restores the previous camera when repair fails', () => {
    const harness = loadHarness();
    harness.initJulia();
    harness.setMouse({ x: 1320, y: 180 });

    const before = harness.getState();
    harness.setRenderSharpImpl(() => false);

    const advanced = harness.stepJuliaQuality();
    const after = harness.getState();

    assert.equal(advanced, false);
    assert.equal(after.hold, true);
    assert.equal(after.pixelScaleApprox, before.pixelScaleApprox);
    assert.equal(after.maxIterations, before.maxIterations);
});

test('stepJuliaCameraWithQualityPriority advances when sharp-frame render succeeds', () => {
    const harness = loadHarness();
    harness.initJulia();
    harness.setMouse({ x: 1320, y: 180 });

    const before = harness.getState();
    harness.setRenderSharpImpl(() => true);

    const advanced = harness.stepJuliaQuality();
    const after = harness.getState();

    assert.equal(advanced, true);
    assert.equal(after.hold, false);
    assert.ok(after.pixelScaleApprox < before.pixelScaleApprox);
});

test('stepNewtonCameraWithQualityPriority restores the previous camera when repair fails', () => {
    const harness = loadHarness();
    harness.initNewton();
    harness.setMouse({ x: 1320, y: 180 });

    const before = harness.getState();
    harness.setRenderSharpImpl(() => false);

    const advanced = harness.stepNewtonQuality();
    const after = harness.getState();

    assert.equal(advanced, false);
    assert.equal(after.hold, true);
    assert.equal(after.pixelScaleApprox, before.pixelScaleApprox);
    assert.equal(after.maxIterations, before.maxIterations);
});

test('stepNewtonCameraWithQualityPriority advances when sharp-frame render succeeds', () => {
    const harness = loadHarness();
    harness.initNewton();
    harness.setMouse({ x: 1320, y: 180 });

    const before = harness.getState();
    harness.setRenderSharpImpl(() => true);

    const advanced = harness.stepNewtonQuality();
    const after = harness.getState();

    assert.equal(advanced, true);
    assert.equal(after.hold, false);
    assert.ok(after.pixelScaleApprox < before.pixelScaleApprox);
});

test('computeEscapeIteration converges immediately for an exact Newton root', () => {
    const harness = loadHarness();
    const point = harness.createPoint('1', '0');

    assert.equal(harness.computeEscape(point, 12, 'newton'), 1);
});

test('Newton selects the anchor directly when it is already a strong reference', () => {
    const harness = loadHarness();
    harness.initNewton();
    const point = harness.createPoint('-1.2', '0.1');
    let callCount = 0;

    harness.setEscapeIterationImpl(() => {
        callCount += 1;
        return 48;
    });

    const selection = harness.selectInitialReference(point, 220, 'newton');

    assert.ok(selection);
    assert.equal(callCount, 1);
    assert.equal(selection.candidate.distanceSquared, 0);
    assert.equal(selection.candidate.escapeIteration, 48);
    assert.equal(selection.candidate.point, point);
});

test('Newton stays on the simple render path until the zoom is genuinely deep', () => {
    const harness = loadHarness();
    harness.initNewton();

    assert.equal(harness.shouldUseDeepRender('newton'), false);
    harness.setNewtonPixelScaleApprox(1e-7);

    assert.equal(harness.shouldUseDeepRender('newton'), true);
});

test('computeCpuDeepColor returns a colored Newton basin for an exact root', () => {
    const harness = loadHarness();
    const point = harness.createPoint('1', '0');
    const color = harness.computeCpuColor(point, 12, 'newton');

    assert.equal(color[3], 255);
    assert.ok(color[0] > 0 || color[1] > 0 || color[2] > 0);
});
