const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function loadHarness() {
    const scriptPath = path.join(__dirname, 'script.js');
    const source = fs.readFileSync(scriptPath, 'utf8');
    let nextTimerId = 1;
    const scheduledTimeouts = new Map();
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
        mandelbrotCommittedFrameAvailable = false;
        deepPrecisionWarningShown = false;
        mandelbrotMaskVerificationFramesRemaining = 0;
        mandelbrotStableReuseFrames = 0;
        mandelbrotDeepWorkState = null;
    },
    initJulia() {
        fractalType = 'julia';
        juliaCamera = createJuliaCamera();
        juliaReference = createEmptyReference();
        juliaReferenceCache = new Map();
        juliaQualityHold = false;
        juliaQualityHoldWarningShown = false;
        juliaFrameReady = false;
        juliaCommittedFrameAvailable = false;
        juliaDeepPrecisionWarningShown = false;
        juliaMaskVerificationFramesRemaining = 0;
        juliaStableReuseFrames = 0;
        juliaDeepWorkState = null;
    },
    initNewton() {
        fractalType = 'newton';
        newtonCamera = createNewtonCamera();
        newtonReference = createEmptyReference();
        newtonReferenceCache = new Map();
        newtonQualityHold = false;
        newtonQualityHoldWarningShown = false;
        newtonFrameReady = false;
        newtonCommittedFrameAvailable = false;
        newtonDeepPrecisionWarningShown = false;
        newtonMaskVerificationFramesRemaining = 0;
        newtonStableReuseFrames = 0;
        newtonDeepWorkState = null;
        newtonDeepRenderActivationScale = NEWTON_DEEP_RENDER_SCALE;
        newtonDeferredCommittedFramePending = false;
    },
    setRenderSharpImpl(fn) { renderSharpDeepFrame = fn; },
    setEscapeIterationImpl(fn) { computeEscapeIteration = fn; },
    setCpuResolveImpl(fn) { resolveDeepTileOnCPU = fn; },
    setOrbitTexture(value) { orbitTexture = value; },
    setOrbitTextureCapacity(value) { orbitTextureCapacity = value; },
    getOrbitTextureCapacity() { return orbitTextureCapacity; },
    uploadOrbit(orbitData, orbitLength) { uploadReferenceOrbit(orbitData, orbitLength); },
    setRenderTargets(targets, type = fractalType) {
        mandelbrotWorkingFramebuffer = targets.workingFramebuffer;
        mandelbrotWorkingColorTexture = targets.workingColorTexture;
        mandelbrotWorkingMaskTexture = targets.workingMaskTexture;
        setDeepCommittedFramebuffer(type, targets.committedFramebuffer);
        setDeepCommittedColorTexture(type, targets.committedColorTexture);
        if (targets.committedCamera) {
            setDeepCommittedCamera(type, cloneDeepCamera(targets.committedCamera));
        }
        setDeepCommittedFrameAvailable(type, targets.committedFrameAvailable ?? true);
        mandelbrotRenderTargetWidth = gl?.canvas?.width ?? mandelbrotRenderTargetWidth;
        mandelbrotRenderTargetHeight = gl?.canvas?.height ?? mandelbrotRenderTargetHeight;
    },
    getRenderTargets(type = fractalType) {
        return {
            workingColorTexture: mandelbrotWorkingColorTexture,
            workingMaskTexture: mandelbrotWorkingMaskTexture,
            committedFramebuffer: getDeepCommittedFramebuffer(type),
            committedColorTexture: getDeepCommittedColorTexture(type),
            committedFrameAvailable: getDeepCommittedFrameAvailable(type),
        };
    },
    commitWorkingFrame() { copyWorkingFrameToCommitted(); },
    drawCommittedFrame(type) { return drawCommittedDeepFrame(type); },
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
    initWebGL() {
        return initWebGL();
    },
    queueChildren(tile) {
        const queue = [];
        queueChildTilesWithGlitches(queue, tile);
        return queue;
    },
    createTile(x, y, width, height, depth, maskData) {
        return createRepairTile(x, y, width, height, depth, maskData);
    },
    renderSharp(type) {
        return renderSharpDeepFrame(type);
    },
    resolveRepairQueueOnCPU(type, tiles, frameStats) {
        resolveRepairQueueOnCPU(type, tiles, frameStats);
    },
    resolveTileOnCPU(type, tile) {
        resolveDeepTileOnCPU(type, tile);
    },
    createFrameStats() {
        return createEmptyMandelbrotFrameStats();
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
    computeNewtonSimplePreviewIterationBudget(pixelScale) {
        return computeNewtonSimplePreviewIterationBudget(pixelScale);
    },
    setDeepCameraState(type, state) {
        const camera = getDeepCamera(type);
        if (typeof state.pixelScaleApprox === 'number') {
            camera.pixelScaleApprox = state.pixelScaleApprox;
        }
        if (typeof state.maxIterations === 'number') {
            camera.maxIterations = state.maxIterations;
        }
    },
    captureSimpleProxy(type) {
        const previousDrawSimpleFractal = drawSimpleFractal;
        let captured = null;
        drawSimpleFractal = (activeType, camera) => {
            captured = {
                activeType,
                camera: { ...camera },
            };
        };
        try {
            drawSimpleProxyFromDeepCamera(type);
        } finally {
            drawSimpleFractal = previousDrawSimpleFractal;
        }
        return captured;
    },
    drawCurrentFrame() {
        draw();
    },
    getActiveDebugSnapshot() {
        return getActiveDebugSnapshot();
    },
    configureDebugHeartbeatFromQuery() {
        return configureDebugHeartbeatFromQuery();
    },
    setDocumentCanvas(value) {
        document.getElementById = function (id) {
            if (id === 'fractalCanvas') {
                return value;
            }
            return { addEventListener() {} };
        };
    },
    setWindowSearch(value) {
        window.location = { search: value };
    },
    getDebugHeartbeatState() {
        return {
            enabled: debugHeartbeatEnabled,
            intervalMs: debugHeartbeatIntervalMs,
        };
    },
    getDeepDebugSnapshot(type) {
        return getDeepDebugSnapshot(type);
    },
    getLastFrameStats(type) {
        return getDeepLastFrameStats(type);
    },
    setDeepWorkState(type, value) {
        setDeepWorkState(type, value);
    },
    setEnsureRenderTargetsImpl(fn) { ensureMandelbrotRenderTargets = fn; },
    setCreateProgramImpl(fn) { createProgram = fn; },
    setCreateProgramInfoImpl(fn) { createProgramInfo = fn; },
    setSimpleProgramInfo(value) { simpleProgramInfo = value; },
    setQuadBuffer(value) { quadBuffer = value; },
    setCommitWorkingFrameImpl(fn) { copyWorkingFrameToCommitted = fn; },
    setSelectInitialReferenceImpl(fn) { selectInitialReference = fn; },
    setRenderDeepPassImpl(fn) { renderDeepPass = fn; },
    setReadGlitchMaskImpl(fn) { readGlitchMask = fn; },
    setDrawSimpleFractalImpl(fn) { drawSimpleFractal = fn; },
    setDrawCommittedDeepFrameImpl(fn) { drawCommittedDeepFrame = fn; },
    setDrawSimpleProxyFromDeepCameraImpl(fn) { drawSimpleProxyFromDeepCamera = fn; },
    setReadMaskAndCheckImpl(fn) { readGlitchMaskAndCheck = fn; },
    setQueueChildrenImpl(fn) { queueChildTilesWithGlitches = fn; },
    setSortRepairQueueImpl(fn) { sortRepairQueue = fn; },
    setFindBestReferencePointImpl(fn) { findBestReferencePoint = fn; },
    setGetReferenceOrbitImpl(fn) { getReferenceOrbit = fn; },
    setTileHasGlitchesImpl(fn) { tileHasGlitches = fn; },
    setShouldDeferNewtonRepairWorkImpl(fn) { shouldDeferNewtonRepairWork = fn; },
    getNewtonDeepRenderActivationScale() {
        return newtonDeepRenderActivationScale;
    },
    getNewtonSimpleProxyPrecisionFloor() {
        return getNewtonSimpleProxyPrecisionFloor(getDeepCamera('newton'));
    },
    deferNewtonDeepRender(scaleApprox) {
        return deferNewtonDeepRender(scaleApprox);
    },
    handleMouseMove(event) {
        handleMouseMove(event);
    },
    resizeCanvas() {
        resizeCanvas();
    },
    resetMandelbrotState() {
        resetMandelbrotState();
    },
    resetJuliaState() {
        resetJuliaState();
    },
    resetNewtonState() {
        resetNewtonState();
    },
    setLastFrameStats(type, value) {
        setDeepLastFrameStats(type, value);
    },
    setReferenceCenter(type, x, y) {
        const reference = getDeepReference(type);
        reference.centerX = decimalFromString(x);
        reference.centerY = decimalFromString(y);
    },
    getReferenceCenter(type) {
        const reference = getDeepReference(type);
        return {
            x: reference.centerX !== null ? reference.centerX.toString() : null,
            y: reference.centerY !== null ? reference.centerY.toString() : null,
        };
    },
    hasReference(type) {
        const reference = getDeepReference(type);
        return reference.centerX !== null && reference.centerY !== null;
    },
    getFrameReady(type) {
        return getDeepFrameReady(type);
    },
    setFrameReady(type, value) {
        setDeepFrameReady(type, value);
    },
    setCommittedFrameAvailable(type, value) {
        setDeepCommittedFrameAvailable(type, value);
    },
    getCommittedFrameAvailable(type) {
        return getDeepCommittedFrameAvailable(type);
    },
    syncCommittedCamera(type = fractalType) {
        setDeepCommittedCamera(type, cloneDeepCamera(getDeepCamera(type)));
    },
    setNewtonDeferredCommittedFramePending(value) {
        newtonDeferredCommittedFramePending = value;
    },
    getNewtonDeferredCommittedFramePending() {
        return newtonDeferredCommittedFramePending;
    },
    setWindowMetrics({ innerWidth, innerHeight, devicePixelRatio }) {
        if (typeof innerWidth === 'number') {
            window.innerWidth = innerWidth;
        }
        if (typeof innerHeight === 'number') {
            window.innerHeight = innerHeight;
        }
        if (typeof devicePixelRatio === 'number') {
            window.devicePixelRatio = devicePixelRatio;
        }
    },
    assessNewtonInitialRepairPressure(repairQueueLength, maskData) {
        return assessNewtonInitialRepairPressure(repairQueueLength, maskData);
    },
    chooseCommittedReferenceSelection(type, initialSelection, bestReferenceSelection, referencesUsed) {
        return chooseCommittedReferenceSelection(type, initialSelection, bestReferenceSelection, referencesUsed);
    },
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
        console: { warn() {}, info() {}, log() {}, error() {} },
        Float32Array,
        Uint8Array,
        BigInt,
        Math,
        Number,
        Map,
        Set,
        Date,
        URLSearchParams,
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
        setTimeout(fn) {
            const id = nextTimerId++;
            scheduledTimeouts.set(id, fn);
            return id;
        },
        clearTimeout(id) {
            scheduledTimeouts.delete(id);
        },
        setInterval() { return 1; },
        clearInterval() {},
        alert() {},
    };

    vm.createContext(context);
    vm.runInContext(harnessSource, context);

    context.__testHarness.setGL({
        canvas: { width: 1600, height: 900 },
    });
    context.__testHarness.flushScheduledWork = function (limit = 32) {
        let remaining = limit;
        while (scheduledTimeouts.size > 0 && remaining > 0) {
            const callbacks = Array.from(scheduledTimeouts.values());
            scheduledTimeouts.clear();
            for (const callback of callbacks) {
                callback();
            }
            remaining -= 1;
        }
        return scheduledTimeouts.size;
    };

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

test('readGlitchMaskAndCheck skips the full mask readback when the GPU reduction reports a clean tile', () => {
    const harness = loadHarness();
    const tile = harness.createTile(0, 0, 4, 2, 0, null);
    let readPixelsCount = 0;

    harness.setGL({
        canvas: { width: 1600, height: 900 },
        bindFramebuffer() {},
        readBuffer() {},
        readPixels() {
            readPixelsCount += 1;
        },
    });
    harness.setTileHasGlitchesImpl(() => false);

    const result = harness.readMaskAndCheck(tile);

    assert.equal(readPixelsCount, 0);
    assert.equal(result.hasGlitches, false);
    assert.equal(result.maskData, null);
});

test('readGlitchMaskAndCheck returns true and preserves the mask bytes when the reduction signals a glitch', () => {
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
    harness.setTileHasGlitchesImpl(() => true);

    const result = harness.readMaskAndCheck(tile);

    assert.equal(result.hasGlitches, true);
    assert.deepEqual(Array.from(result.maskData), [0, 0, 255, 0, 0, 0, 0, 0]);
});

test('readGlitchMaskAndCheck forces tight packing for odd-width mask reads when a glitch is present', () => {
    const harness = loadHarness();
    const tile = harness.createTile(1, 2, 3, 2, 0, null);
    const pixelStoreCalls = [];

    harness.setGL({
        canvas: { width: 1600, height: 900 },
        READ_FRAMEBUFFER: 'READ_FRAMEBUFFER',
        COLOR_ATTACHMENT1: 'COLOR_ATTACHMENT1',
        RED: 'RED',
        UNSIGNED_BYTE: 'UNSIGNED_BYTE',
        PACK_ALIGNMENT: 'PACK_ALIGNMENT',
        bindFramebuffer() {},
        readBuffer() {},
        pixelStorei(pname, value) {
            pixelStoreCalls.push([pname, value]);
        },
        readPixels(x, y, width, height, format, type, destination) {
            destination.set([0, 0, 0, 0, 0, 0]);
        },
    });
    harness.setTileHasGlitchesImpl(() => true);

    const result = harness.readMaskAndCheck(tile);

    assert.deepEqual(pixelStoreCalls, [['PACK_ALIGNMENT', 1]]);
    assert.equal(result.maskData.length, 6);
    assert.equal(result.hasGlitches, true);
});

test('resolveRepairQueueOnCPU drains the remaining repair queue into CPU work', () => {
    const harness = loadHarness();
    harness.initNewton();
    const tiles = [
        harness.createTile(0, 0, 4, 4, 1, null),
        harness.createTile(4, 0, 2, 3, 2, null),
    ];
    const frameStats = harness.createFrameStats();
    frameStats.referencesUsed = 9;
    const resolvedTiles = [];

    harness.setCpuResolveImpl((type, tile) => {
        resolvedTiles.push({ type, tile });
    });

    harness.resolveRepairQueueOnCPU('newton', tiles, frameStats);

    assert.equal(tiles.length, 0);
    assert.deepEqual(
        resolvedTiles.map(({ type, tile }) => ({ type, width: tile.width, height: tile.height, depth: tile.depth })),
        [
            { type: 'newton', width: 4, height: 4, depth: 1 },
            { type: 'newton', width: 2, height: 3, depth: 2 },
        ]
    );
    assert.equal(frameStats.cpuResolvedTiles, 2);
    assert.equal(frameStats.cpuResolvedPixels, 22);
    assert.equal(frameStats.lastTileWidth, 2);
    assert.equal(frameStats.lastTileHeight, 3);
    assert.equal(frameStats.lastTileDepth, 2);
    assert.equal(frameStats.deepestTileDepth, 2);
    assert.equal(harness.getDeepDebugSnapshot('newton').activeWork.repairTilesProcessed, 2);
});

test('resolveDeepTileOnCPU forces tight packing for odd-width mask uploads', () => {
    const harness = loadHarness();
    harness.initNewton();
    const tile = harness.createTile(0, 0, 3, 1, 0, null);
    const pixelStoreCalls = [];
    const uploadCalls = [];

    harness.setGL({
        canvas: { width: 1600, height: 900 },
        FRAMEBUFFER: 'FRAMEBUFFER',
        TEXTURE_2D: 'TEXTURE_2D',
        RGBA: 'RGBA',
        RED: 'RED',
        UNSIGNED_BYTE: 'UNSIGNED_BYTE',
        UNPACK_ALIGNMENT: 'UNPACK_ALIGNMENT',
        bindFramebuffer() {},
        bindTexture() {},
        pixelStorei(pname, value) {
            pixelStoreCalls.push([pname, value]);
        },
        texSubImage2D(target, level, x, y, width, height, format, type, data) {
            uploadCalls.push({ format, width, height, byteLength: data.length });
        },
    });

    harness.resolveTileOnCPU('newton', tile);

    assert.deepEqual(pixelStoreCalls, [['UNPACK_ALIGNMENT', 1]]);
    assert.deepEqual(
        uploadCalls.map((call) => ({
            format: call.format,
            width: call.width,
            height: call.height,
            byteLength: call.byteLength,
        })),
        [
            { format: 'RGBA', width: 3, height: 1, byteLength: 12 },
            { format: 'RED', width: 3, height: 1, byteLength: 3 },
        ]
    );
});

test('stepMandelbrotCameraWithQualityPriority restores the previous camera when repair fails', () => {
    const harness = loadHarness();
    harness.initMandelbrot();
    harness.setMouse({ x: 1320, y: 180 });
    harness.setCommittedFrameAvailable('mandelbrot', true);
    harness.syncCommittedCamera('mandelbrot');
    harness.setEnsureRenderTargetsImpl(() => {});
    harness.setSelectInitialReferenceImpl(() => null);

    const before = harness.getState();
    const advanced = harness.stepQuality();
    harness.flushScheduledWork();
    const after = harness.getState();

    assert.equal(advanced, true);
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

test('copyWorkingFrameToCommitted blits the shared working color into the active mode committed target', () => {
    const harness = loadHarness();
    const calls = [];
    harness.setGL({
        canvas: { width: 1600, height: 900 },
        FRAMEBUFFER: 'FRAMEBUFFER',
        READ_FRAMEBUFFER: 'READ_FRAMEBUFFER',
        DRAW_FRAMEBUFFER: 'DRAW_FRAMEBUFFER',
        COLOR_ATTACHMENT0: 'COLOR_ATTACHMENT0',
        COLOR_BUFFER_BIT: 'COLOR_BUFFER_BIT',
        NEAREST: 'NEAREST',
        bindFramebuffer(target, framebuffer) {
            calls.push(['bindFramebuffer', target, framebuffer]);
        },
        readBuffer(attachment) {
            calls.push(['readBuffer', attachment]);
        },
        blitFramebuffer(...args) {
            calls.push(['blitFramebuffer', ...args]);
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
    assert.equal(targets.workingColorTexture.id, 'working-color');
    assert.equal(targets.committedColorTexture.id, 'committed-color');
    assert.equal(targets.committedFrameAvailable, true);
    assert.ok(calls.some(([name, target, framebuffer]) => (
        name === 'bindFramebuffer'
        && target === 'READ_FRAMEBUFFER'
        && framebuffer.id === 'working-fbo'
    )));
    assert.ok(calls.some(([name, target, framebuffer]) => (
        name === 'bindFramebuffer'
        && target === 'DRAW_FRAMEBUFFER'
        && framebuffer.id === 'committed-fbo'
    )));
    assert.ok(calls.some(([name]) => name === 'blitFramebuffer'));
});

test('drawCommittedDeepFrame helper rejects a stale Mandelbrot committed frame after a mode switch', () => {
    const harness = loadHarness();
    const calls = [];
    harness.setGL({
        READ_FRAMEBUFFER: 'READ_FRAMEBUFFER',
        DRAW_FRAMEBUFFER: 'DRAW_FRAMEBUFFER',
        COLOR_ATTACHMENT0: 'COLOR_ATTACHMENT0',
        COLOR_BUFFER_BIT: 'COLOR_BUFFER_BIT',
        NEAREST: 'NEAREST',
        canvas: { width: 1600, height: 900 },
        bindFramebuffer(target, framebuffer) {
            calls.push(['bindFramebuffer', target, framebuffer]);
        },
        readBuffer(attachment) {
            calls.push(['readBuffer', attachment]);
        },
        blitFramebuffer(...args) {
            calls.push(['blitFramebuffer', ...args]);
        },
    });
    harness.setRenderTargets({
        workingFramebuffer: { id: 'shared-working-fbo' },
        committedFramebuffer: { id: 'mandelbrot-committed-fbo' },
        workingColorTexture: { id: 'shared-working-color' },
        workingMaskTexture: { id: 'shared-working-mask' },
        committedColorTexture: { id: 'mandelbrot-committed-color' },
        committedFrameAvailable: true,
    }, 'mandelbrot');

    harness.initNewton();
    const drew = harness.drawCommittedFrame('newton');

    assert.equal(drew, false);
    assert.ok(!calls.some(([name]) => name === 'blitFramebuffer'));
});

test('drawCurrentFrame does not reuse a stale Mandelbrot committed frame after switching to Newton', () => {
    const harness = loadHarness();
    const drawCalls = [];
    harness.setGL({
        canvas: { width: 1600, height: 900 },
        FRAMEBUFFER: 'FRAMEBUFFER',
        READ_FRAMEBUFFER: 'READ_FRAMEBUFFER',
        DRAW_FRAMEBUFFER: 'DRAW_FRAMEBUFFER',
        COLOR_BUFFER_BIT: 1,
        COLOR_ATTACHMENT0: 'COLOR_ATTACHMENT0',
        NEAREST: 'NEAREST',
        bindFramebuffer() {},
        readBuffer() {},
        blitFramebuffer() {
            drawCalls.push('deep');
        },
        viewport() {},
        clear() {},
    });
    harness.setRenderTargets({
        workingFramebuffer: { id: 'shared-working-fbo' },
        committedFramebuffer: { id: 'mandelbrot-committed-fbo' },
        workingColorTexture: { id: 'shared-working-color' },
        workingMaskTexture: { id: 'shared-working-mask' },
        committedColorTexture: { id: 'mandelbrot-committed-color' },
        committedFrameAvailable: true,
    }, 'mandelbrot');
    harness.initNewton();
    harness.setDeepCameraState('newton', { pixelScaleApprox: 1e-7 });
    harness.setDrawSimpleProxyFromDeepCameraImpl(() => {
        drawCalls.push('simple-proxy');
        return true;
    });

    harness.drawCurrentFrame();
    const snapshot = harness.getActiveDebugSnapshot();

    assert.deepEqual(drawCalls, ['simple-proxy']);
    assert.equal(snapshot.renderPath, 'simple-proxy');
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

test('Newton simple preview budget stays capped well below the deep Newton budget', () => {
    const harness = loadHarness();

    assert.equal(harness.computeNewtonSimplePreviewIterationBudget(1), 24);
    assert.equal(harness.computeNewtonSimplePreviewIterationBudget(1e-6), 36);
    assert.equal(harness.computeNewtonSimplePreviewIterationBudget(1e-12), 48);
    assert.ok(
        harness.computeNewtonSimplePreviewIterationBudget(1e-6)
        < harness.computeNewtonIterationBudget(1e-6)
    );
});

test('simple proxy preview cap only applies to Newton', () => {
    const harness = loadHarness();

    harness.initMandelbrot();
    harness.setDeepCameraState('mandelbrot', { maxIterations: 222 });
    assert.equal(harness.captureSimpleProxy('mandelbrot').camera.maxIterations, 222);

    harness.initJulia();
    harness.setDeepCameraState('julia', { maxIterations: 333 });
    assert.equal(harness.captureSimpleProxy('julia').camera.maxIterations, 333);

    harness.initNewton();
    harness.setDeepCameraState('newton', { pixelScaleApprox: 1e-6, maxIterations: 105 });
    assert.equal(harness.captureSimpleProxy('newton').camera.maxIterations, 105);
});

test('deferred Newton simple proxy uses the capped preview budget', () => {
    const harness = loadHarness();

    harness.initNewton();
    harness.setDeepCameraState('newton', { pixelScaleApprox: 1e-6, maxIterations: 105 });
    harness.deferNewtonDeepRender(1e-6);

    assert.equal(harness.captureSimpleProxy('newton').camera.maxIterations, 36);
});

test('active debug snapshot reports the Newton preview path and deep handoff', () => {
    const harness = loadHarness();

    harness.initNewton();
    let snapshot = harness.getActiveDebugSnapshot();
    assert.equal(snapshot.mode, 'newton');
    assert.equal(snapshot.renderPath, 'simple-proxy');
    assert.equal(snapshot.deepEligible, false);
    assert.equal(snapshot.previewMaxIterations, snapshot.maxIterations);

    harness.setNewtonPixelScaleApprox(1e-7);
    snapshot = harness.getActiveDebugSnapshot();
    assert.equal(snapshot.renderPath, 'deep');
    assert.equal(snapshot.deepEligible, true);
    assert.equal(snapshot.previewMaxIterations, null);
    assert.equal(snapshot.deepActivationScale, 1e-6);
});

test('Newton repair deferral timer starts after the full-frame pass', () => {
    const harness = loadHarness();
    harness.initNewton();
    harness.setGL({
        canvas: { width: 16, height: 16 },
    });
    harness.setMouse({ x: 8, y: 8 });

    const point = harness.createPoint('-1.2', '0.1');
    const selection = {
        candidate: {
            point,
            escapeIteration: 80,
        },
        reference: {
            point,
            orbitLength: 1,
            orbitData: new Float32Array(2),
            escapedEarly: false,
        },
        mode: 'search',
    };

    harness.setEnsureRenderTargetsImpl(() => {});
    harness.setCommitWorkingFrameImpl(() => {});
    harness.setSelectInitialReferenceImpl(() => selection);
    harness.setRenderDeepPassImpl(() => {});
    harness.setReadMaskAndCheckImpl((tile) => {
        const maskData = new Uint8Array(tile.width * tile.height);
        maskData[0] = 255;
        const deadline = Date.now() + 320;
        while (Date.now() < deadline) {
            // Force the full-frame verification path to be slow enough to trip
            // the old frame-start-based repair timer.
        }
        return {
            hasGlitches: true,
            maskData,
        };
    });
    harness.setQueueChildrenImpl((queue, tile) => {
        if (tile.depth !== 0) {
            return;
        }
        for (let i = 0; i < 6; i += 1) {
            const maskData = new Uint8Array(16);
            maskData[0] = 255;
            queue.push(harness.createTile(0, 0, 4, 4, 1, maskData));
        }
    });
    harness.setSortRepairQueueImpl(() => {});
    harness.setFindBestReferencePointImpl(() => ({
        point,
        escapeIteration: 80,
    }));
    harness.setGetReferenceOrbitImpl(() => ({
        point,
        orbitLength: 1,
        orbitData: new Float32Array(2),
        escapedEarly: false,
    }));
    harness.setTileHasGlitchesImpl(() => false);

    const rendered = harness.renderSharp('newton');
    const frameStats = harness.getLastFrameStats('newton');

    assert.equal(rendered, true);
    assert.equal(frameStats.status, 'success');
    assert.equal(frameStats.reason, null);
    assert.equal(frameStats.referencesUsed, 7);
});

test('configureDebugHeartbeatFromQuery ignores a missing interval parameter', () => {
    const harness = loadHarness();

    harness.setWindowSearch('');
    harness.configureDebugHeartbeatFromQuery();
    let heartbeatState = harness.getDebugHeartbeatState();
    assert.equal(heartbeatState.enabled, false);

    harness.setWindowSearch('?debugHeartbeat=1');
    harness.configureDebugHeartbeatFromQuery();
    heartbeatState = harness.getDebugHeartbeatState();
    assert.equal(heartbeatState.enabled, true);
    assert.equal(heartbeatState.intervalMs, 1000);
});

test('initWebGL initializes render targets without referencing an undefined mode variable', () => {
    const harness = loadHarness();
    const fakeGL = {
        canvas: { width: 1600, height: 900, style: {} },
        ARRAY_BUFFER: 'ARRAY_BUFFER',
        STATIC_DRAW: 'STATIC_DRAW',
        TEXTURE_2D: 'TEXTURE_2D',
        TEXTURE_MIN_FILTER: 'TEXTURE_MIN_FILTER',
        TEXTURE_MAG_FILTER: 'TEXTURE_MAG_FILTER',
        NEAREST: 'NEAREST',
        TEXTURE_WRAP_S: 'TEXTURE_WRAP_S',
        TEXTURE_WRAP_T: 'TEXTURE_WRAP_T',
        CLAMP_TO_EDGE: 'CLAMP_TO_EDGE',
        RGBA32F: 'RGBA32F',
        RGBA8: 'RGBA8',
        RGBA: 'RGBA',
        FLOAT: 'FLOAT',
        UNSIGNED_BYTE: 'UNSIGNED_BYTE',
        R8: 'R8',
        RED: 'RED',
        FRAMEBUFFER: 'FRAMEBUFFER',
        COLOR_ATTACHMENT0: 'COLOR_ATTACHMENT0',
        COLOR_ATTACHMENT1: 'COLOR_ATTACHMENT1',
        FRAMEBUFFER_COMPLETE: 'FRAMEBUFFER_COMPLETE',
        createBuffer() { return {}; },
        bindBuffer() {},
        bufferData() {},
        createTexture() { return {}; },
        bindTexture() {},
        texParameteri() {},
        texImage2D() {},
        createFramebuffer() { return {}; },
        bindFramebuffer() {},
        framebufferTexture2D() {},
        drawBuffers() {},
        checkFramebufferStatus() { return 'FRAMEBUFFER_COMPLETE'; },
        clearColor() {},
    };
    const fakeCanvas = {
        addEventListener() {},
        style: {},
        getContext() {
            return fakeGL;
        },
    };

    harness.setDocumentCanvas(fakeCanvas);
    harness.setCreateProgramImpl(() => ({}));
    harness.setCreateProgramInfoImpl(() => ({}));

    const initialized = harness.initWebGL();

    assert.equal(initialized, true);
});

test('deep debug snapshot includes active work state', () => {
    const harness = loadHarness();

    harness.initNewton();
    harness.setDeepWorkState('newton', {
        startedAtMs: Date.now() - 25,
        stage: 'repair-reference-search',
        referencesUsed: 7,
        repairQueueLength: 12,
        repairTilesProcessed: 5,
        cpuTilesResolved: 1,
        currentTile: {
            width: 32,
            height: 24,
            depth: 3,
        },
        note: 'searching-repair-reference',
    });

    const snapshot = harness.getDeepDebugSnapshot('newton');
    assert.equal(snapshot.activeWork.stage, 'repair-reference-search');
    assert.equal(snapshot.activeWork.referencesUsed, 7);
    assert.equal(snapshot.activeWork.repairQueueLength, 12);
    assert.equal(snapshot.activeWork.repairTilesProcessed, 5);
    assert.equal(snapshot.activeWork.cpuTilesResolved, 1);
    assert.equal(snapshot.activeWork.currentTile.width, 32);
    assert.equal(snapshot.activeWork.currentTile.height, 24);
    assert.equal(snapshot.activeWork.currentTile.depth, 3);
    assert.equal(snapshot.activeWork.note, 'searching-repair-reference');
    assert.ok(snapshot.activeWork.elapsedMs >= 0);
});

test('Newton deep render deferral lowers the activation scale until reset', () => {
    const harness = loadHarness();

    harness.initNewton();
    assert.equal(harness.getNewtonDeepRenderActivationScale(), 1e-6);
    const precisionFloor = harness.getNewtonSimpleProxyPrecisionFloor();

    harness.deferNewtonDeepRender(8e-7);
    assert.equal(harness.getNewtonDeepRenderActivationScale(), precisionFloor);

    harness.setNewtonPixelScaleApprox(8e-7);
    assert.equal(harness.shouldUseDeepRender('newton'), false);

    harness.setNewtonPixelScaleApprox(1e-7);
    assert.equal(harness.shouldUseDeepRender('newton'), true);

    harness.initNewton();
    assert.equal(harness.getNewtonDeepRenderActivationScale(), 1e-6);
});

test('Newton deep render deferral does not drop below the simple-proxy precision floor', () => {
    const harness = loadHarness();
    harness.initNewton();

    const precisionFloor = harness.getNewtonSimpleProxyPrecisionFloor();
    const deferredScale = harness.deferNewtonDeepRender(precisionFloor * 1.1);

    assert.equal(deferredScale, precisionFloor);
    harness.setNewtonPixelScaleApprox(precisionFloor * 0.9);
    assert.equal(harness.shouldUseDeepRender('newton'), true);
});

test('Newton mouse moves preserve an active deep-render deferral', () => {
    const harness = loadHarness();
    harness.initNewton();
    harness.setGL({
        canvas: {
            width: 1600,
            height: 900,
            getBoundingClientRect() {
                return {
                    left: 0,
                    top: 0,
                    width: 800,
                    height: 450,
                };
            },
        },
    });

    const precisionFloor = harness.getNewtonSimpleProxyPrecisionFloor();
    harness.setNewtonPixelScaleApprox(8e-7);
    harness.deferNewtonDeepRender(8e-7);
    assert.equal(harness.getNewtonDeepRenderActivationScale(), precisionFloor);

    harness.handleMouseMove({ clientX: 400, clientY: 225 });

    assert.equal(harness.getNewtonDeepRenderActivationScale(), precisionFloor);
    assert.equal(harness.shouldUseDeepRender('newton'), false);
});

test('repaired frames commit the most reusable successful reference near the anchor', () => {
    const harness = loadHarness();
    harness.initNewton();
    harness.setMouse({ x: 800, y: 450 });

    const anchorLocalSelection = {
        candidate: {
            point: harness.createPoint('-1.2', '0.1'),
            escapeIteration: 80,
        },
        reference: { label: 'initial' },
    };
    const farRepairSelection = {
        candidate: {
            point: harness.createPoint('0.4', '0.1'),
            escapeIteration: 120,
        },
        reference: { label: 'repair-far' },
    };
    const nearbyRepairSelection = {
        candidate: {
            point: harness.createPoint('-1.16', '0.1'),
            escapeIteration: 120,
        },
        reference: { label: 'repair-near' },
    };

    assert.equal(
        harness.chooseCommittedReferenceSelection('newton', anchorLocalSelection, farRepairSelection, 1).reference.label,
        'repair-far'
    );
    assert.equal(
        harness.chooseCommittedReferenceSelection('newton', anchorLocalSelection, farRepairSelection, 2).reference.label,
        'initial'
    );
    assert.equal(
        harness.chooseCommittedReferenceSelection('newton', anchorLocalSelection, nearbyRepairSelection, 2).reference.label,
        'repair-near'
    );
    assert.equal(
        harness.chooseCommittedReferenceSelection('newton', farRepairSelection, nearbyRepairSelection, 2).reference.label,
        'repair-near'
    );
});

test('repaired frames compare reusable tie distances from the shared screen anchor', () => {
    const harness = loadHarness();
    harness.initNewton();
    harness.setMouse({ x: 800, y: 450 });

    const closerInitialSelection = {
        candidate: {
            point: harness.createPoint('-1.18', '0.1'),
            distanceSquared: 400,
            escapeIteration: 120,
        },
        reference: { label: 'initial-near-anchor' },
    };
    const fartherRepairSelection = {
        candidate: {
            point: harness.createPoint('-1.05', '0.1'),
            distanceSquared: 0,
            escapeIteration: 120,
        },
        reference: { label: 'repair-farther-from-anchor' },
    };

    assert.equal(
        harness.chooseCommittedReferenceSelection('newton', closerInitialSelection, fartherRepairSelection, 2).reference.label,
        'initial-near-anchor'
    );
});

test('renderSharpDeepFrame keeps the nearest equal-strength repair reference for commit', () => {
    const harness = loadHarness();
    harness.initNewton();
    harness.setMouse({ x: 800, y: 450 });
    harness.setGL({
        canvas: { width: 1600, height: 900 },
    });

    const initialPoint = harness.createPoint('-1.2', '0.1');
    const nearbyPoint = harness.createPoint('-1.18', '0.1');
    const fartherPoint = harness.createPoint('-1.05', '0.1');
    let repairCandidateCalls = 0;

    harness.setEnsureRenderTargetsImpl(() => {});
    harness.setCommitWorkingFrameImpl(() => {});
    harness.setSelectInitialReferenceImpl(() => ({
        candidate: {
            point: initialPoint,
            distanceSquared: 0,
            escapeIteration: 80,
        },
        reference: {
            point: initialPoint,
            orbitLength: 1,
            orbitData: new Float32Array(2),
            escapedEarly: false,
            escapeIteration: 80,
        },
        mode: 'search',
    }));
    harness.setRenderDeepPassImpl(() => {});
    harness.setReadMaskAndCheckImpl((tile) => {
        const maskData = new Uint8Array(tile.width * tile.height);
        maskData[0] = 255;
        return {
            hasGlitches: true,
            maskData,
        };
    });
    harness.setQueueChildrenImpl((queue, tile) => {
        if (tile.depth !== 0) {
            return;
        }
        const firstMask = new Uint8Array(16);
        firstMask[0] = 255;
        const secondMask = new Uint8Array(16);
        secondMask[0] = 255;
        queue.push(harness.createTile(0, 0, 4, 4, 1, firstMask));
        queue.push(harness.createTile(4, 0, 4, 4, 1, secondMask));
    });
    harness.setSortRepairQueueImpl(() => {});
    harness.setFindBestReferencePointImpl(() => {
        repairCandidateCalls += 1;
        if (repairCandidateCalls === 1) {
            return {
                point: nearbyPoint,
                distanceSquared: 400,
                escapeIteration: 120,
            };
        }
        return {
            point: fartherPoint,
            distanceSquared: 0,
            escapeIteration: 120,
        };
    });
    harness.setGetReferenceOrbitImpl((point) => ({
        point,
        orbitLength: 1,
        orbitData: new Float32Array(2),
        escapedEarly: false,
        escapeIteration: point === initialPoint ? 80 : 120,
    }));
    harness.setTileHasGlitchesImpl(() => false);

    const rendered = harness.renderSharp('newton');
    const committedReference = harness.getReferenceCenter('newton');

    assert.equal(rendered, true);
    assert.equal(committedReference.x, nearbyPoint.x.toString());
    assert.equal(committedReference.y, nearbyPoint.y.toString());
});

test('assessNewtonInitialRepairPressure only defers on broad initial glitch coverage', () => {
    const harness = loadHarness();
    const sparseMask = new Uint8Array(1000);
    sparseMask[0] = 255;
    sparseMask[250] = 255;
    sparseMask[500] = 255;
    sparseMask[750] = 255;

    const sparseAssessment = harness.assessNewtonInitialRepairPressure(4, sparseMask);
    assert.equal(sparseAssessment.shouldDefer, false);
    assert.equal(sparseAssessment.glitchCount, 4);

    const broadMask = new Uint8Array(1000);
    broadMask.fill(255, 0, 8);

    const broadAssessment = harness.assessNewtonInitialRepairPressure(4, broadMask);
    assert.equal(broadAssessment.shouldDefer, true);
    assert.equal(broadAssessment.glitchCount, 8);
    assert.equal(broadAssessment.glitchRatio, 0.008);
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

test('production reset paths clear committed frame availability and deferred Newton pending state', () => {
    const harness = loadHarness();

    harness.setCommittedFrameAvailable('mandelbrot', true);
    harness.setCommittedFrameAvailable('julia', true);
    harness.setCommittedFrameAvailable('newton', true);
    harness.setNewtonDeferredCommittedFramePending(true);

    harness.resetMandelbrotState();
    harness.resetJuliaState();
    harness.resetNewtonState();

    assert.equal(harness.getCommittedFrameAvailable('mandelbrot'), false);
    assert.equal(harness.getCommittedFrameAvailable('julia'), false);
    assert.equal(harness.getCommittedFrameAvailable('newton'), false);
    assert.equal(harness.getNewtonDeferredCommittedFramePending(), false);
});

test('resizeCanvas preserves committed deep frames while clearing pending Newton deferred frame reuse', () => {
    const harness = loadHarness();
    harness.initNewton();
    harness.setGL({
        canvas: {
            width: 1600,
            height: 900,
            style: {},
        },
    });
    harness.setWindowMetrics({
        innerWidth: 800,
        innerHeight: 450,
        devicePixelRatio: 2,
    });
    harness.setEnsureRenderTargetsImpl(() => {});
    harness.setCommittedFrameAvailable('mandelbrot', true);
    harness.setCommittedFrameAvailable('julia', true);
    harness.setCommittedFrameAvailable('newton', true);
    harness.setNewtonDeferredCommittedFramePending(true);

    harness.resizeCanvas();

    assert.equal(harness.getCommittedFrameAvailable('mandelbrot'), true);
    assert.equal(harness.getCommittedFrameAvailable('julia'), true);
    assert.equal(harness.getCommittedFrameAvailable('newton'), true);
    assert.equal(harness.getNewtonDeferredCommittedFramePending(), false);
});

test('stepJuliaCameraWithQualityPriority restores the previous camera when repair fails', () => {
    const harness = loadHarness();
    harness.initJulia();
    harness.setMouse({ x: 1320, y: 180 });
    harness.setCommittedFrameAvailable('julia', true);
    harness.syncCommittedCamera('julia');
    harness.setEnsureRenderTargetsImpl(() => {});
    harness.setSelectInitialReferenceImpl(() => null);

    const before = harness.getState();
    const advanced = harness.stepJuliaQuality();
    harness.flushScheduledWork();
    const after = harness.getState();

    assert.equal(advanced, true);
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
    harness.setDeepCameraState('newton', { pixelScaleApprox: 8e-7 });
    harness.setMouse({ x: 1320, y: 180 });
    harness.setCommittedFrameAvailable('newton', true);
    harness.syncCommittedCamera('newton');
    harness.setEnsureRenderTargetsImpl(() => {});
    harness.setSelectInitialReferenceImpl(() => null);

    const before = harness.getState();
    const advanced = harness.stepNewtonQuality();
    harness.flushScheduledWork();
    const after = harness.getState();

    assert.equal(advanced, true);
    assert.equal(after.hold, true);
    assert.equal(after.pixelScaleApprox, before.pixelScaleApprox);
    assert.equal(after.maxIterations, before.maxIterations);
});

test('stepNewtonCameraWithQualityPriority defers deep render on pathological repair growth without holding', () => {
    const harness = loadHarness();
    harness.initNewton();
    const precisionFloor = harness.getNewtonSimpleProxyPrecisionFloor();
    harness.setGL({
        canvas: { width: 16, height: 16 },
    });
    harness.setDeepCameraState('newton', { pixelScaleApprox: 8e-7 });
    harness.setMouse({ x: 8, y: 8 });
    harness.setFrameReady('newton', true);
    harness.setReferenceCenter('newton', '-1.2', '0.1');
    const point = harness.createPoint('-1.2', '0.1');
    harness.setEnsureRenderTargetsImpl(() => {});
    harness.setSelectInitialReferenceImpl(() => ({
        candidate: {
            point,
            escapeIteration: 80,
        },
        reference: {
            point,
            orbitLength: 1,
            orbitData: new Float32Array(2),
            escapedEarly: false,
        },
        mode: 'search',
    }));
    harness.setRenderDeepPassImpl(() => {});
    harness.setTileHasGlitchesImpl((tile) => tile.depth === 0);
    harness.setReadGlitchMaskImpl((tile) => {
        const maskData = new Uint8Array(tile.width * tile.height);
        maskData[0] = 255;
        return maskData;
    });
    harness.setQueueChildrenImpl((queue, tile) => {
        if (tile.depth !== 0) {
            return;
        }
        const maskData = new Uint8Array(16);
        maskData[0] = 255;
        queue.push(harness.createTile(0, 0, 4, 4, 1, maskData));
    });
    harness.setSortRepairQueueImpl(() => {});
    harness.setShouldDeferNewtonRepairWorkImpl(() => true);

    const advanced = harness.stepNewtonQuality();
    harness.flushScheduledWork();
    const after = harness.getState();

    assert.equal(advanced, true);
    assert.equal(after.hold, false);
    assert.ok(after.pixelScaleApprox < 8e-7);
    assert.equal(harness.shouldUseDeepRender('newton'), false);
    assert.equal(harness.getFrameReady('newton'), false);
    assert.equal(harness.hasReference('newton'), false);
    assert.equal(harness.getNewtonDeepRenderActivationScale(), precisionFloor);
});

test('stepNewtonCameraWithQualityPriority holds instead of deferring to an inaccurate simple proxy at very deep scales', () => {
    const harness = loadHarness();
    harness.initNewton();
    harness.setGL({
        canvas: { width: 16, height: 16 },
    });
    harness.setDeepCameraState('newton', { pixelScaleApprox: 1e-18 });
    harness.setMouse({ x: 8, y: 8 });
    harness.setFrameReady('newton', true);
    harness.setCommittedFrameAvailable('newton', true);
    harness.syncCommittedCamera('newton');
    harness.setReferenceCenter('newton', '-1.2', '0.1');
    const point = harness.createPoint('-1.2', '0.1');
    harness.setEnsureRenderTargetsImpl(() => {});
    harness.setSelectInitialReferenceImpl(() => ({
        candidate: {
            point,
            escapeIteration: 80,
        },
        reference: {
            point,
            orbitLength: 1,
            orbitData: new Float32Array(2),
            escapedEarly: false,
        },
        mode: 'search',
    }));
    harness.setRenderDeepPassImpl(() => {});
    harness.setTileHasGlitchesImpl((tile) => tile.depth === 0);
    harness.setReadGlitchMaskImpl((tile) => {
        const maskData = new Uint8Array(tile.width * tile.height);
        maskData[0] = 255;
        return maskData;
    });
    harness.setQueueChildrenImpl((queue, tile) => {
        if (tile.depth !== 0) {
            return;
        }
        const maskData = new Uint8Array(16);
        maskData[0] = 255;
        queue.push(harness.createTile(0, 0, 4, 4, 1, maskData));
    });
    harness.setSortRepairQueueImpl(() => {});
    harness.setShouldDeferNewtonRepairWorkImpl(() => true);

    const advanced = harness.stepNewtonQuality();
    harness.flushScheduledWork();
    const after = harness.getState();

    assert.equal(advanced, true);
    assert.equal(after.hold, true);
    assert.equal(after.pixelScaleApprox, 1e-18);
    assert.equal(harness.shouldUseDeepRender('newton'), true);
    assert.equal(harness.getFrameReady('newton'), true);
    assert.equal(harness.hasReference('newton'), true);
    assert.equal(harness.getNewtonDeepRenderActivationScale(), 1e-6);
});

test('Newton pathological deferral reuses the committed frame for the immediate fallback draw', () => {
    const harness = loadHarness();
    const drawCalls = [];
    harness.initNewton();
    const precisionFloor = harness.getNewtonSimpleProxyPrecisionFloor();
    harness.setGL({
        canvas: { width: 1600, height: 900 },
        FRAMEBUFFER: 'FRAMEBUFFER',
        READ_FRAMEBUFFER: 'READ_FRAMEBUFFER',
        DRAW_FRAMEBUFFER: 'DRAW_FRAMEBUFFER',
        COLOR_BUFFER_BIT: 1,
        COLOR_ATTACHMENT0: 'COLOR_ATTACHMENT0',
        NEAREST: 'NEAREST',
        bindFramebuffer() {},
        readBuffer() {},
        blitFramebuffer() {
            drawCalls.push('deep');
        },
        viewport() {},
        clear() {},
    });
    harness.setRenderTargets({
        workingFramebuffer: { id: 'shared-working-fbo' },
        committedFramebuffer: { id: 'newton-committed-fbo' },
        workingColorTexture: { id: 'shared-working-color' },
        workingMaskTexture: { id: 'shared-working-mask' },
        committedColorTexture: { id: 'newton-committed-color' },
        committedFrameAvailable: true,
    }, 'newton');
    harness.setDeepCameraState('newton', { pixelScaleApprox: 8e-7 });
    harness.setMouse({ x: 1320, y: 180 });
    harness.setFrameReady('newton', true);
    harness.setReferenceCenter('newton', '-1.2', '0.1');
    const point = harness.createPoint('-1.2', '0.1');
    harness.setEnsureRenderTargetsImpl(() => {});
    harness.setSelectInitialReferenceImpl(() => ({
        candidate: {
            point,
            escapeIteration: 80,
        },
        reference: {
            point,
            orbitLength: 1,
            orbitData: new Float32Array(2),
            escapedEarly: false,
        },
        mode: 'search',
    }));
    harness.setRenderDeepPassImpl(() => {});
    harness.setTileHasGlitchesImpl((tile) => tile.depth === 0);
    harness.setReadGlitchMaskImpl((tile) => {
        const maskData = new Uint8Array(tile.width * tile.height);
        maskData[0] = 255;
        return maskData;
    });
    harness.setQueueChildrenImpl((queue, tile) => {
        if (tile.depth !== 0) {
            return;
        }
        const maskData = new Uint8Array(16);
        maskData[0] = 255;
        queue.push(harness.createTile(0, 0, 4, 4, 1, maskData));
    });
    harness.setSortRepairQueueImpl(() => {});
    harness.setShouldDeferNewtonRepairWorkImpl(() => true);
    harness.setDrawSimpleProxyFromDeepCameraImpl(() => {
        drawCalls.push('simple-proxy');
        return true;
    });

    const advanced = harness.stepNewtonQuality();
    harness.flushScheduledWork();
    harness.drawCurrentFrame();
    const snapshot = harness.getActiveDebugSnapshot();

    assert.equal(advanced, true);
    assert.deepEqual(drawCalls, ['simple-proxy']);
    assert.equal(snapshot.renderPath, 'simple-proxy');
    assert.equal(snapshot.deepEligible, false);
    assert.equal(snapshot.frameReady, false);
    assert.equal(harness.getNewtonDeepRenderActivationScale(), harness.getNewtonSimpleProxyPrecisionFloor());
});

test('stepNewtonCameraWithQualityPriority also defers on pathological initial repair flood', () => {
    const harness = loadHarness();
    harness.initNewton();
    const precisionFloor = harness.getNewtonSimpleProxyPrecisionFloor();
    harness.setGL({
        canvas: { width: 16, height: 16 },
    });
    harness.setDeepCameraState('newton', { pixelScaleApprox: 8e-7 });
    harness.setMouse({ x: 8, y: 8 });
    harness.setFrameReady('newton', true);
    harness.setReferenceCenter('newton', '-1.2', '0.1');
    const point = harness.createPoint('-1.2', '0.1');
    harness.setEnsureRenderTargetsImpl(() => {});
    harness.setSelectInitialReferenceImpl(() => ({
        candidate: {
            point,
            escapeIteration: 80,
        },
        reference: {
            point,
            orbitLength: 1,
            orbitData: new Float32Array(2),
            escapedEarly: false,
        },
        mode: 'search',
    }));
    harness.setRenderDeepPassImpl(() => {});
    harness.setTileHasGlitchesImpl((tile) => tile.depth === 0);
    harness.setReadGlitchMaskImpl((tile) => {
        const maskData = new Uint8Array(tile.width * tile.height);
        maskData.fill(255, 0, 8);
        return maskData;
    });
    harness.setQueueChildrenImpl((queue, tile) => {
        if (tile.depth !== 0) {
            return;
        }
        for (let i = 0; i < 4; i += 1) {
            const maskData = new Uint8Array(16);
            maskData[0] = 255;
            queue.push(harness.createTile(0, 0, 4, 4, 1, maskData));
        }
    });
    harness.setSortRepairQueueImpl(() => {});

    const advanced = harness.stepNewtonQuality();
    harness.flushScheduledWork();
    const after = harness.getState();

    assert.equal(advanced, true);
    assert.equal(after.hold, false);
    assert.ok(after.pixelScaleApprox < 8e-7);
    assert.equal(harness.shouldUseDeepRender('newton'), false);
    assert.equal(harness.getFrameReady('newton'), false);
    assert.equal(harness.hasReference('newton'), false);
    assert.equal(harness.getNewtonDeepRenderActivationScale(), precisionFloor);
});

test('debug snapshot reports the simple Newton fallback when deep render fails during draw', () => {
    const harness = loadHarness();
    const drawCalls = [];
    harness.initNewton();
    harness.setGL({
        canvas: { width: 1600, height: 900 },
        FRAMEBUFFER: 'FRAMEBUFFER',
        COLOR_BUFFER_BIT: 1,
        bindFramebuffer() {},
        viewport() {},
        clear() {},
    });
    harness.setDeepCameraState('newton', { pixelScaleApprox: 8e-7 });
    harness.setFrameReady('newton', false);
    harness.setDrawSimpleFractalImpl(() => {
        drawCalls.push('simple-proxy');
    });

    harness.drawCurrentFrame();
    const snapshot = harness.getActiveDebugSnapshot();

    assert.deepEqual(drawCalls, ['simple-proxy']);
    assert.equal(snapshot.renderPath, 'simple-proxy');
    assert.equal(snapshot.deepEligible, true);
});

test('draw-time Newton deep failures below the proxy precision floor reuse the committed frame', () => {
    const harness = loadHarness();
    const drawCalls = [];
    harness.initNewton();
    harness.setGL({
        canvas: { width: 1600, height: 900 },
        FRAMEBUFFER: 'FRAMEBUFFER',
        READ_FRAMEBUFFER: 'READ_FRAMEBUFFER',
        DRAW_FRAMEBUFFER: 'DRAW_FRAMEBUFFER',
        COLOR_BUFFER_BIT: 1,
        COLOR_ATTACHMENT0: 'COLOR_ATTACHMENT0',
        NEAREST: 'NEAREST',
        bindFramebuffer() {},
        readBuffer() {},
        blitFramebuffer() {
            drawCalls.push('deep');
        },
        viewport() {},
        clear() {},
    });
    harness.setRenderTargets({
        workingFramebuffer: { id: 'shared-working-fbo' },
        committedFramebuffer: { id: 'newton-committed-fbo' },
        workingColorTexture: { id: 'shared-working-color' },
        workingMaskTexture: { id: 'shared-working-mask' },
        committedColorTexture: { id: 'newton-committed-color' },
        committedFrameAvailable: true,
    }, 'newton');
    harness.setDeepCameraState('newton', { pixelScaleApprox: 1e-7 });
    harness.setFrameReady('newton', false);
    harness.setDrawSimpleFractalImpl(() => {
        drawCalls.push('simple-proxy');
    });
    harness.setRenderSharpImpl(() => {
        harness.setLastFrameStats('newton', {
            ...harness.createFrameStats(),
            status: 'failed',
            reason: 'repair_reference_budget_exhausted',
        });
        return false;
    });

    harness.drawCurrentFrame();
    const snapshot = harness.getActiveDebugSnapshot();

    assert.deepEqual(drawCalls, ['deep']);
    assert.equal(snapshot.renderPath, 'deep');
    assert.equal(snapshot.deepEligible, true);
});

test('an accurate Newton simple-proxy frame seeds the committed fallback before the first deep failure', () => {
    const harness = loadHarness();
    const drawCalls = [];
    harness.initNewton();
    harness.setGL({
        canvas: { width: 1600, height: 900 },
        FRAMEBUFFER: 'FRAMEBUFFER',
        READ_FRAMEBUFFER: 'READ_FRAMEBUFFER',
        DRAW_FRAMEBUFFER: 'DRAW_FRAMEBUFFER',
        COLOR_BUFFER_BIT: 1,
        COLOR_ATTACHMENT0: 'COLOR_ATTACHMENT0',
        NEAREST: 'NEAREST',
        TRIANGLE_STRIP: 'TRIANGLE_STRIP',
        ARRAY_BUFFER: 'ARRAY_BUFFER',
        FLOAT: 'FLOAT',
        bindFramebuffer() {},
        viewport() {},
        useProgram() {},
        bindBuffer() {},
        enableVertexAttribArray() {},
        vertexAttribPointer() {},
        uniform2f() {},
        uniform1f() {},
        uniform1i() {},
        drawArrays() {},
        readBuffer() {},
        blitFramebuffer() {
            drawCalls.push('deep');
        },
        clear() {},
    });
    harness.setSimpleProgramInfo({
        program: {},
        position: 0,
        uniforms: {
            u_resolution: {},
            u_center: {},
            u_pixelScale: {},
            u_fractalType: {},
            u_maxIterations: {},
        },
    });
    harness.setQuadBuffer({});
    harness.setRenderTargets({
        workingFramebuffer: { id: 'shared-working-fbo' },
        committedFramebuffer: { id: 'newton-committed-fbo' },
        workingColorTexture: { id: 'shared-working-color' },
        workingMaskTexture: { id: 'shared-working-mask' },
        committedColorTexture: { id: 'newton-committed-color' },
        committedFrameAvailable: false,
    }, 'newton');
    harness.setEnsureRenderTargetsImpl(() => {});
    harness.setDeepCameraState('newton', { pixelScaleApprox: 8e-7 });

    harness.captureSimpleProxy('newton');
    assert.equal(harness.getCommittedFrameAvailable('newton'), true);

    harness.setDeepCameraState('newton', { pixelScaleApprox: 1e-7 });
    harness.setFrameReady('newton', false);
    harness.setDrawCommittedDeepFrameImpl(() => {
        drawCalls.push('deep');
        return true;
    });

    harness.drawCurrentFrame();
    const snapshot = harness.getActiveDebugSnapshot();

    assert.deepEqual(drawCalls, ['deep']);
    assert.equal(snapshot.renderPath, 'deep');
    assert.equal(snapshot.deepEligible, true);
});

test('draw-time Newton deep failures below the proxy precision floor do not draw an inaccurate simple proxy when no committed frame exists', () => {
    const harness = loadHarness();
    const drawCalls = [];
    harness.initNewton();
    harness.setGL({
        canvas: { width: 1600, height: 900 },
        FRAMEBUFFER: 'FRAMEBUFFER',
        COLOR_BUFFER_BIT: 1,
        bindFramebuffer() {},
        viewport() {},
        clear() {},
    });
    harness.setDeepCameraState('newton', { pixelScaleApprox: 1e-7 });
    harness.setFrameReady('newton', false);
    harness.setDrawSimpleFractalImpl(() => {
        drawCalls.push('simple-proxy');
    });
    harness.setRenderSharpImpl(() => {
        harness.setLastFrameStats('newton', {
            ...harness.createFrameStats(),
            status: 'failed',
            reason: 'repair_reference_budget_exhausted',
        });
        return false;
    });

    harness.drawCurrentFrame();
    const snapshot = harness.getActiveDebugSnapshot();

    assert.deepEqual(drawCalls, []);
    assert.equal(snapshot.renderPath, 'blank');
    assert.equal(snapshot.deepEligible, true);
});

test('drawCurrentFrame reports blank when Newton simple proxy rendering is unavailable', () => {
    const harness = loadHarness();

    harness.initNewton();
    harness.setGL({
        canvas: { width: 1600, height: 900 },
        FRAMEBUFFER: 'FRAMEBUFFER',
        COLOR_BUFFER_BIT: 1,
        bindFramebuffer() {},
        viewport() {},
        clear() {},
    });
    harness.setDeepCameraState('newton', { pixelScaleApprox: 2e-6 });
    harness.setDrawSimpleProxyFromDeepCameraImpl(() => false);

    harness.drawCurrentFrame();
    const snapshot = harness.getActiveDebugSnapshot();

    assert.equal(snapshot.renderPath, 'blank');
    assert.equal(snapshot.deepEligible, false);
});

test('renderSharpDeepFrame fails fast instead of CPU-draining Newton repairs when repair references are exhausted', () => {
    const harness = loadHarness();
    harness.initNewton();
    harness.setGL({
        canvas: { width: 16, height: 16 },
    });
    harness.setMouse({ x: 8, y: 8 });

    const point = harness.createPoint('-1.2', '0.1');
    const selection = {
        candidate: {
            point,
            escapeIteration: 80,
        },
        reference: {
            point,
            orbitLength: 1,
            orbitData: new Float32Array(2),
            escapedEarly: false,
        },
        mode: 'search',
    };
    let cpuResolveCalls = 0;

    harness.setEnsureRenderTargetsImpl(() => {});
    harness.setSelectInitialReferenceImpl(() => selection);
    harness.setRenderDeepPassImpl(() => {});
    harness.setReadMaskAndCheckImpl((tile) => {
        const maskData = new Uint8Array(tile.width * tile.height);
        maskData[0] = 255;
        return {
            hasGlitches: true,
            maskData,
        };
    });
    harness.setQueueChildrenImpl((queue, tile) => {
        if (tile.depth !== 0) {
            return;
        }
        for (let i = 0; i < 192; i += 1) {
            const maskData = new Uint8Array(16);
            maskData[0] = 255;
            queue.push(harness.createTile(0, 0, 4, 4, 1, maskData));
        }
    });
    harness.setSortRepairQueueImpl(() => {});
    harness.setFindBestReferencePointImpl(() => ({
        point,
        escapeIteration: 80,
    }));
    harness.setGetReferenceOrbitImpl(() => ({
        point,
        orbitLength: 1,
        orbitData: new Float32Array(2),
        escapedEarly: false,
    }));
    harness.setTileHasGlitchesImpl(() => false);
    harness.setShouldDeferNewtonRepairWorkImpl(() => false);
    harness.setCpuResolveImpl(() => {
        cpuResolveCalls += 1;
    });

    const rendered = harness.renderSharp('newton');
    const frameStats = harness.getLastFrameStats('newton');

    assert.equal(rendered, false);
    assert.equal(cpuResolveCalls, 0);
    assert.equal(frameStats.status, 'failed');
    assert.equal(frameStats.reason, 'repair_reference_budget_exhausted');
    assert.equal(frameStats.referencesUsed, 192);
    assert.equal(frameStats.cpuResolvedTiles, 0);
    assert.equal(frameStats.queuedTilesRemaining, 1);
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

    harness.setEscapeIterationImpl((x, y) => {
        callCount += 1;
        if (x === point.x && y === point.y) {
            return 218;
        }
        return 120;
    });

    const selection = harness.selectInitialReference(point, 220, 'newton');

    assert.ok(selection);
    assert.equal(callCount, 1);
    assert.equal(selection.candidate.distanceSquared, 0);
    assert.equal(selection.candidate.escapeIteration, 218);
    assert.equal(selection.candidate.point, point);
});

test('Newton searches for a stronger initial reference when the anchor is weak', () => {
    const harness = loadHarness();
    harness.initNewton();
    const point = harness.createPoint('-1.2', '0.1');
    const stalePoint = harness.createPoint('0.4', '0.2');
    let callCount = 0;
    let staleAnchorCalls = 0;
    harness.setFindBestReferencePointImpl(() => {
        throw new Error('Newton weak-anchor fallback should stay on the bounded local search');
    });
    harness.setReferenceCenter('newton', '0.4', '0.2');

    harness.setEscapeIterationImpl((x, y) => {
        callCount += 1;
        if (x === point.x && y === point.y) {
            return 30;
        }
        if (x === stalePoint.x && y === stalePoint.y) {
            staleAnchorCalls += 1;
            return 200;
        }
        return 80;
    });

    const selection = harness.selectInitialReference(point, 220, 'newton');

    assert.ok(selection);
    assert.ok(callCount > 1);
    assert.ok(callCount <= 114);
    assert.equal(staleAnchorCalls, 0);
    assert.equal(selection.candidate.escapeIteration, 80);
    assert.notEqual(selection.candidate.distanceSquared, 0);
    assert.ok(selection.candidate.distanceSquared <= (48 * 48));
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
