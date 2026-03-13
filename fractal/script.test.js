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
    },
    setRenderSharpImpl(fn) { renderSharpDeepFrame = fn; },
    setEscapeIterationImpl(fn) { computeEscapeIteration = fn; },
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
    stepQuality() { return stepMandelbrotCameraWithQualityPriority(); },
    stepJuliaQuality() { return stepJuliaCameraWithQualityPriority(); },
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
    const maskData = new Uint8Array(4 * 4 * 4);

    for (let row = 0; row < 2; row += 1) {
        for (let column = 0; column < 2; column += 1) {
            const baseIndex = ((row * 4) + column) * 4;
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
