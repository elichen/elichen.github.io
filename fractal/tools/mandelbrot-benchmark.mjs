#!/usr/bin/env node

import { spawn } from 'node:child_process';
import { createReadStream } from 'node:fs';
import { copyFile, mkdir, readFile, stat, writeFile } from 'node:fs/promises';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { setTimeout as sleep } from 'node:timers/promises';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '..');

const defaults = {
    targetStep: 1000,
    timeoutMs: 180000,
    pollMs: 1000,
    settleMs: 2000,
    viewportWidth: 1600,
    viewportHeight: 900,
    outputDir: path.join(projectRoot, '.agent-browser-artifacts'),
    label: null,
    session: `mandelbrot-bench-${process.pid}`,
    baseline: null,
    repeat: 1,
    targetBaselineScreenshot: null,
};

const mimeTypes = {
    '.css': 'text/css; charset=utf-8',
    '.html': 'text/html; charset=utf-8',
    '.ico': 'image/x-icon',
    '.jpeg': 'image/jpeg',
    '.jpg': 'image/jpeg',
    '.js': 'text/javascript; charset=utf-8',
    '.json': 'application/json; charset=utf-8',
    '.mjs': 'text/javascript; charset=utf-8',
    '.png': 'image/png',
    '.svg': 'image/svg+xml; charset=utf-8',
    '.txt': 'text/plain; charset=utf-8',
};

function parseArgs(argv) {
    const options = { ...defaults };

    for (let index = 0; index < argv.length; index += 1) {
        const value = argv[index];
        const nextValue = argv[index + 1];

        if (value === '--target-step') {
            options.targetStep = Number(nextValue);
            index += 1;
        } else if (value === '--timeout-ms') {
            options.timeoutMs = Number(nextValue);
            index += 1;
        } else if (value === '--poll-ms') {
            options.pollMs = Number(nextValue);
            index += 1;
        } else if (value === '--settle-ms') {
            options.settleMs = Number(nextValue);
            index += 1;
        } else if (value === '--viewport-width') {
            options.viewportWidth = Number(nextValue);
            index += 1;
        } else if (value === '--viewport-height') {
            options.viewportHeight = Number(nextValue);
            index += 1;
        } else if (value === '--output-dir') {
            options.outputDir = path.resolve(projectRoot, nextValue);
            index += 1;
        } else if (value === '--label') {
            options.label = nextValue;
            index += 1;
        } else if (value === '--session') {
            options.session = nextValue;
            index += 1;
        } else if (value === '--baseline') {
            options.baseline = path.resolve(projectRoot, nextValue);
            index += 1;
        } else if (value === '--repeat') {
            options.repeat = Number(nextValue);
            index += 1;
        } else if (value === '--target-baseline-screenshot') {
            options.targetBaselineScreenshot = path.resolve(projectRoot, nextValue);
            index += 1;
        } else if (value === '--help' || value === '-h') {
            printHelp();
            process.exit(0);
        } else {
            throw new Error(`Unknown argument: ${value}`);
        }
    }

    for (const numericKey of [
        'targetStep',
        'timeoutMs',
        'pollMs',
        'settleMs',
        'viewportWidth',
        'viewportHeight',
        'repeat',
    ]) {
        if (!Number.isFinite(options[numericKey]) || options[numericKey] <= 0) {
            throw new Error(`Invalid numeric value for ${numericKey}`);
        }
    }

    return options;
}

function printHelp() {
    console.log(`Usage: node tools/mandelbrot-benchmark.mjs [options]

Options:
  --target-step <n>       Successful Mandelbrot zoom step target (default: 1000)
  --timeout-ms <ms>       Max wall-clock time to reach target (default: 180000)
  --poll-ms <ms>          Snapshot polling interval (default: 1000)
  --settle-ms <ms>        Extra time after target to confirm continued progress (default: 2000)
  --viewport-width <px>   Browser viewport width (default: 1600)
  --viewport-height <px>  Browser viewport height (default: 900)
  --output-dir <path>     Directory for screenshots and JSON reports
  --label <name>          Optional label used in output filenames
  --session <name>        agent-browser session name
  --baseline <file>       Compare current run against a prior JSON report
  --repeat <n>            Run multiple sequential trials and aggregate with medians/maxima
  --target-baseline-screenshot <file>
                          Compare the exact target-step screenshot against a baseline PNG
  --help                  Show this message
`);
}

function createStaticServer(rootDir) {
    return http.createServer(async (request, response) => {
        const requestUrl = new URL(request.url || '/', 'http://127.0.0.1');
        const pathname = decodeURIComponent(requestUrl.pathname === '/' ? '/index.html' : requestUrl.pathname);
        const resolvedPath = path.resolve(rootDir, `.${pathname}`);

        if (!resolvedPath.startsWith(rootDir)) {
            response.writeHead(403, { 'Content-Type': 'text/plain; charset=utf-8' });
            response.end('Forbidden');
            return;
        }

        let filePath = resolvedPath;

        try {
            const fileStats = await stat(filePath);
            if (fileStats.isDirectory()) {
                filePath = path.join(filePath, 'index.html');
            }
            await stat(filePath);
        } catch {
            response.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
            response.end('Not found');
            return;
        }

        const extension = path.extname(filePath).toLowerCase();
        response.writeHead(200, {
            'Content-Type': mimeTypes[extension] || 'application/octet-stream',
            'Cache-Control': 'no-store',
        });
        createReadStream(filePath).pipe(response);
    });
}

async function runAgentBrowser(session, args, { timeoutMs = 30000, allowFailure = false } = {}) {
    const payload = await new Promise((resolve, reject) => {
        const child = spawn(
            'npx',
            ['-y', 'agent-browser', '--session', session, '--json', ...args],
            {
                cwd: projectRoot,
                stdio: ['ignore', 'pipe', 'pipe'],
            }
        );

        let stdout = '';
        let stderr = '';
        const timer = setTimeout(() => {
            child.kill('SIGTERM');
            reject(new Error(`agent-browser timed out for args: ${args.join(' ')}`));
        }, timeoutMs);

        child.stdout.setEncoding('utf8');
        child.stderr.setEncoding('utf8');
        child.stdout.on('data', (chunk) => {
            stdout += chunk;
        });
        child.stderr.on('data', (chunk) => {
            stderr += chunk;
        });
        child.on('error', (error) => {
            clearTimeout(timer);
            reject(error);
        });
        child.on('close', (code) => {
            clearTimeout(timer);

            const trimmedStdout = stdout.trim();
            const trimmedStderr = stderr.trim();
            let parsedPayload = null;

            if (trimmedStdout) {
                try {
                    parsedPayload = JSON.parse(trimmedStdout);
                } catch {
                    parsedPayload = {
                        success: code === 0,
                        data: trimmedStdout,
                        error: trimmedStderr || null,
                    };
                }
            } else {
                parsedPayload = {
                    success: code === 0,
                    data: null,
                    error: trimmedStderr || null,
                };
            }

            if (!allowFailure && (code !== 0 || parsedPayload.success === false)) {
                reject(new Error(parsedPayload.error || trimmedStderr || `agent-browser failed for args: ${args.join(' ')}`));
                return;
            }

            resolve(parsedPayload);
        });
    });

    return payload;
}

function normalizeEvalResult(value) {
    if (typeof value !== 'string') {
        return value;
    }

    const trimmed = value.trim();
    if (!trimmed) {
        return value;
    }

    try {
        return JSON.parse(trimmed);
    } catch {
        return value;
    }
}

function getPayloadCount(payload) {
    if (!payload) {
        return 0;
    }
    if (Array.isArray(payload)) {
        return payload.length;
    }
    if (typeof payload === 'object') {
        for (const key of ['messages', 'entries', 'errors', 'logs', 'items', 'requests']) {
            if (Array.isArray(payload[key])) {
                return payload[key].length;
            }
        }
        if (Object.keys(payload).length === 0) {
            return 0;
        }
    }
    return 1;
}

function collectSampleMetrics(samples) {
    const referencesUsed = samples.map((sample) => sample.snapshot?.lastFrame?.referencesUsed ?? 0);
    const repairPasses = samples.map((sample) => sample.snapshot?.lastFrame?.repairPasses ?? 0);
    const cpuResolvedTiles = samples.map((sample) => sample.snapshot?.lastFrame?.cpuResolvedTiles ?? 0);
    const cpuResolvedPixels = samples.map((sample) => sample.snapshot?.lastFrame?.cpuResolvedPixels ?? 0);
    const queuedTilesRemaining = samples.map((sample) => sample.snapshot?.lastFrame?.queuedTilesRemaining ?? 0);
    const deepestTileDepth = samples.map((sample) => sample.snapshot?.lastFrame?.deepestTileDepth ?? 0);
    const holdCount = samples.filter((sample) => sample.snapshot?.hold).length;

    return {
        sampledMaxReferencesUsed: Math.max(0, ...referencesUsed),
        sampledMaxRepairPasses: Math.max(0, ...repairPasses),
        sampledMaxCpuResolvedTiles: Math.max(0, ...cpuResolvedTiles),
        sampledMaxCpuResolvedPixels: Math.max(0, ...cpuResolvedPixels),
        sampledMaxQueuedTilesRemaining: Math.max(0, ...queuedTilesRemaining),
        sampledMaxDeepestTileDepth: Math.max(0, ...deepestTileDepth),
        sampledHoldSnapshots: holdCount,
    };
}

function isHealthySnapshot(snapshot) {
    return Boolean(
        snapshot
        && snapshot.hold === false
        && snapshot.frameReady === true
        && snapshot.lastFrame
        && snapshot.lastFrame.status === 'success'
    );
}

function percentDelta(currentValue, baselineValue) {
    if (!Number.isFinite(currentValue) || !Number.isFinite(baselineValue) || baselineValue === 0) {
        return null;
    }
    return ((currentValue - baselineValue) / baselineValue) * 100;
}

function compareAgainstBaseline(current, baseline) {
    const currentMetrics = current.metrics;
    const baselineMetrics = baseline.metrics || {};
    const speedMetrics = {
        targetStepsPerSecondDeltaPct: percentDelta(
            currentMetrics.targetStepsPerSecond,
            baselineMetrics.targetStepsPerSecond
        ),
        settleStepsPerSecondDeltaPct: percentDelta(
            currentMetrics.settleStepsPerSecond,
            baselineMetrics.settleStepsPerSecond
        ),
    };

    const qualityMetrics = {
        sampledMaxReferencesUsedDelta: currentMetrics.sampledMaxReferencesUsed - (baselineMetrics.sampledMaxReferencesUsed ?? 0),
        sampledMaxRepairPassesDelta: currentMetrics.sampledMaxRepairPasses - (baselineMetrics.sampledMaxRepairPasses ?? 0),
        sampledMaxCpuResolvedTilesDelta: currentMetrics.sampledMaxCpuResolvedTiles - (baselineMetrics.sampledMaxCpuResolvedTiles ?? 0),
        sampledHoldSnapshotsDelta: currentMetrics.sampledHoldSnapshots - (baselineMetrics.sampledHoldSnapshots ?? 0),
        browserErrorsDelta: currentMetrics.browserErrors - (baselineMetrics.browserErrors ?? 0),
        consoleEntriesDelta: currentMetrics.consoleEntries - (baselineMetrics.consoleEntries ?? 0),
    };

    const dominates = (
        currentMetrics.targetStepsPerSecond >= (baselineMetrics.targetStepsPerSecond ?? 0)
        && currentMetrics.sampledMaxReferencesUsed <= (baselineMetrics.sampledMaxReferencesUsed ?? Number.POSITIVE_INFINITY)
        && currentMetrics.sampledMaxRepairPasses <= (baselineMetrics.sampledMaxRepairPasses ?? Number.POSITIVE_INFINITY)
        && currentMetrics.sampledMaxCpuResolvedTiles <= (baselineMetrics.sampledMaxCpuResolvedTiles ?? Number.POSITIVE_INFINITY)
        && currentMetrics.sampledHoldSnapshots <= (baselineMetrics.sampledHoldSnapshots ?? Number.POSITIVE_INFINITY)
        && currentMetrics.browserErrors <= (baselineMetrics.browserErrors ?? Number.POSITIVE_INFINITY)
    );

    return {
        baselinePath: baseline.reportPath || null,
        dominates,
        speedMetrics,
        qualityMetrics,
    };
}

function formatMetric(value, digits = 2) {
    if (!Number.isFinite(value)) {
        return 'n/a';
    }
    return value.toFixed(digits);
}

function median(values) {
    const finiteValues = values.filter(Number.isFinite).sort((a, b) => a - b);
    if (finiteValues.length === 0) {
        return null;
    }

    const midpoint = Math.floor(finiteValues.length / 2);
    if ((finiteValues.length % 2) === 1) {
        return finiteValues[midpoint];
    }

    return (finiteValues[midpoint - 1] + finiteValues[midpoint]) / 2;
}

function maxMetric(reports, key, fallback = 0) {
    const values = reports.map((report) => report.metrics?.[key]).filter(Number.isFinite);
    if (values.length === 0) {
        return fallback;
    }
    return Math.max(...values);
}

function aggregateReports(label, reportPath, options, baseUrl, reports) {
    const targetScreenshotMismatchValues = reports
        .map((report) => report.metrics?.targetScreenshotMismatchPercentage)
        .filter(Number.isFinite);

    return {
        label,
        reportPath,
        createdAt: new Date().toISOString(),
        config: {
            targetStep: options.targetStep,
            timeoutMs: options.timeoutMs,
            pollMs: options.pollMs,
            settleMs: options.settleMs,
            viewportWidth: options.viewportWidth,
            viewportHeight: options.viewportHeight,
            session: options.session,
            baseUrl,
            repeat: options.repeat,
            targetBaselineScreenshot: options.targetBaselineScreenshot,
        },
        targetElapsedMs: median(reports.map((report) => report.targetElapsedMs)),
        totalElapsedMs: median(reports.map((report) => report.totalElapsedMs)),
        metrics: {
            targetStepsPerSecond: median(reports.map((report) => report.metrics.targetStepsPerSecond)),
            settleStepsPerSecond: median(reports.map((report) => report.metrics.settleStepsPerSecond)),
            totalStepsPerSecond: median(reports.map((report) => report.metrics.totalStepsPerSecond)),
            browserErrors: maxMetric(reports, 'browserErrors'),
            consoleEntries: maxMetric(reports, 'consoleEntries'),
            hasQualityHoldWarning: reports.some((report) => report.metrics.hasQualityHoldWarning),
            hasPrecisionWarning: reports.some((report) => report.metrics.hasPrecisionWarning),
            sampledSnapshots: reports.reduce((sum, report) => sum + (report.metrics.sampledSnapshots ?? 0), 0),
            sampledMaxReferencesUsed: maxMetric(reports, 'sampledMaxReferencesUsed'),
            sampledMaxRepairPasses: maxMetric(reports, 'sampledMaxRepairPasses'),
            sampledMaxCpuResolvedTiles: maxMetric(reports, 'sampledMaxCpuResolvedTiles'),
            sampledMaxCpuResolvedPixels: maxMetric(reports, 'sampledMaxCpuResolvedPixels'),
            sampledMaxQueuedTilesRemaining: maxMetric(reports, 'sampledMaxQueuedTilesRemaining'),
            sampledMaxDeepestTileDepth: maxMetric(reports, 'sampledMaxDeepestTileDepth'),
            sampledHoldSnapshots: maxMetric(reports, 'sampledHoldSnapshots'),
            targetScreenshotMismatchPercentage: targetScreenshotMismatchValues.length > 0
                ? Math.max(...targetScreenshotMismatchValues)
                : null,
        },
        pass: {
            reachedTarget: reports.every((report) => report.pass.reachedTarget),
            targetHealthy: reports.every((report) => report.pass.targetHealthy),
            settleHealthy: reports.every((report) => report.pass.settleHealthy),
            stillAdvancing: reports.every((report) => report.pass.stillAdvancing),
            noBrowserErrors: reports.every((report) => report.pass.noBrowserErrors),
            noQualityHoldWarning: reports.every((report) => report.pass.noQualityHoldWarning),
            noPrecisionWarning: reports.every((report) => report.pass.noPrecisionWarning),
            targetScreenshotMatches: reports.every((report) => report.pass.targetScreenshotMatches !== false),
        },
        trials: reports.map((report) => ({
            label: report.label,
            reportPath: report.reportPath,
            targetElapsedMs: report.targetElapsedMs,
            totalElapsedMs: report.totalElapsedMs,
            screenshots: report.screenshots,
            metrics: report.metrics,
            pass: report.pass,
        })),
    };
}

async function installTargetPause(session, targetStep) {
    const pauseEval = `
        (function () {
            window.__benchmarkPauseEnabled = true;
            window.__benchmarkPauseReached = false;
            window.__benchmarkPauseTargetStep = ${targetStep};
            window.__benchmarkPausePerfMs = null;
            window.__benchmarkPauseStartPerfMs = performance.now();
            window.__benchmarkStartStep = window.getMandelbrotDebugSnapshot
                ? (window.getMandelbrotDebugSnapshot().step || 0)
                : 0;
            if (!window.__benchmarkPauseInstalled) {
                window.__benchmarkPauseInstalled = true;
                const originalStepMandelbrot = stepMandelbrotCameraWithQualityPriority;
                stepMandelbrotCameraWithQualityPriority = function (...args) {
                    if (window.__benchmarkPauseEnabled && window.__benchmarkPauseReached) {
                        return false;
                    }

                    const result = originalStepMandelbrot.apply(this, args);
                    if (
                        window.__benchmarkPauseEnabled
                        && !window.__benchmarkPauseReached
                        && window.getMandelbrotDebugSnapshot
                        && window.getMandelbrotDebugSnapshot().step >= window.__benchmarkPauseTargetStep
                    ) {
                        window.__benchmarkPauseReached = true;
                        window.__benchmarkPausePerfMs = performance.now() - window.__benchmarkPauseStartPerfMs;
                    }
                    return result;
                };
            }
            return true;
        })()
    `;

    await runAgentBrowser(session, ['eval', pauseEval], { timeoutMs: 60000 });
}

async function resumeAfterTargetPause(session) {
    const resumeEval = `
        (function () {
            window.__benchmarkPauseEnabled = false;
            window.__benchmarkPauseReached = false;
            return true;
        })()
    `;

    await runAgentBrowser(session, ['eval', resumeEval], { timeoutMs: 60000 });
}

async function readSnapshot(session) {
    const evalPayload = await runAgentBrowser(
        session,
        [
            'eval',
            `JSON.stringify({
                snapshot: window.getMandelbrotDebugSnapshot && window.getMandelbrotDebugSnapshot(),
                pauseReached: window.__benchmarkPauseReached === true,
                pauseElapsedMs: window.__benchmarkPausePerfMs,
                startStep: window.__benchmarkStartStep || 0
            })`,
        ]
    );

    return normalizeEvalResult(evalPayload.data?.result);
}

async function captureScreenshot(session, outputDir, destinationPath) {
    const screenshotPayload = await runAgentBrowser(
        session,
        ['screenshot', '--screenshot-dir', outputDir],
        { timeoutMs: 60000 }
    );
    const sourcePath = screenshotPayload.data?.path || screenshotPayload.data?.screenshotPath || null;
    if (!sourcePath) {
        throw new Error('agent-browser screenshot did not return a path');
    }
    await copyFile(sourcePath, destinationPath);
}

async function runSingleBenchmark(options, baseUrl, label, trialIndex) {
    const session = options.repeat > 1 ? `${options.session}-trial-${trialIndex + 1}` : options.session;
    const reportPath = path.join(options.outputDir, `${label}.json`);
    const targetScreenshotPath = path.join(options.outputDir, `${label}-target.png`);
    const settleScreenshotPath = path.join(options.outputDir, `${label}-settle.png`);
    const startTimeIso = new Date().toISOString();

    let report = null;

    try {
        await runAgentBrowser(session, ['close'], { allowFailure: true });
        await runAgentBrowser(session, ['set', 'viewport', String(options.viewportWidth), String(options.viewportHeight)]);
        await runAgentBrowser(session, ['open', baseUrl], { timeoutMs: 60000 });
        await runAgentBrowser(session, ['wait', '--load', 'networkidle'], { timeoutMs: 60000 });
        await runAgentBrowser(session, ['snapshot', '-i']);
        await runAgentBrowser(session, ['errors', '--clear']);
        await runAgentBrowser(session, ['console', '--clear']);
        await installTargetPause(session, options.targetStep);
        await runAgentBrowser(
            session,
            ['wait', '--fn', 'window.getMandelbrotDebugSnapshot && window.getMandelbrotDebugSnapshot().step >= 1'],
            { timeoutMs: 60000 }
        );
        const startState = await readSnapshot(session);
        const benchmarkStartStep = startState?.startStep ?? startState?.snapshot?.step ?? 0;

        const samples = [];
        const wallClockStart = Date.now();
        let targetSample = null;

        while ((Date.now() - wallClockStart) < options.timeoutMs) {
            const result = await readSnapshot(session);
            const snapshot = result?.snapshot || null;
            const elapsedMs = Date.now() - wallClockStart;
            samples.push({ elapsedMs, snapshot });

            if (result?.pauseReached || (snapshot && snapshot.step >= options.targetStep)) {
                targetSample = {
                    elapsedMs: result?.pauseElapsedMs ?? elapsedMs,
                    snapshot,
                };
                break;
            }

            await sleep(options.pollMs);
        }

        if (!targetSample) {
            throw new Error(`Timed out before reaching step ${options.targetStep}`);
        }

        const targetSnapshotResult = await readSnapshot(session);
        if (targetSnapshotResult?.snapshot) {
            targetSample = {
                elapsedMs: targetSnapshotResult.pauseElapsedMs ?? targetSample.elapsedMs,
                snapshot: targetSnapshotResult.snapshot,
            };
        }

        await captureScreenshot(session, options.outputDir, targetScreenshotPath);

        let targetScreenshotDiff = null;
        if (options.targetBaselineScreenshot) {
            const diffPayload = await runAgentBrowser(
                session,
                ['diff', 'screenshot', '--baseline', options.targetBaselineScreenshot],
                { timeoutMs: 60000 }
            );
            targetScreenshotDiff = diffPayload.data || null;
        }

        await resumeAfterTargetPause(session);
        await sleep(options.settleMs);

        const settleSnapshotResult = await readSnapshot(session);
        const settleSnapshot = settleSnapshotResult?.snapshot || null;
        const totalElapsedMs = (Date.now() - wallClockStart);

        await captureScreenshot(session, options.outputDir, settleScreenshotPath);

        const consolePayload = await runAgentBrowser(session, ['console']);
        const errorsPayload = await runAgentBrowser(session, ['errors']);
        const consoleText = JSON.stringify(consolePayload.data ?? '');
        const errorsText = JSON.stringify(errorsPayload.data ?? '');
        const sampleMetrics = collectSampleMetrics(samples);

        report = {
            label,
            reportPath,
            createdAt: startTimeIso,
            config: {
                targetStep: options.targetStep,
                timeoutMs: options.timeoutMs,
                pollMs: options.pollMs,
                settleMs: options.settleMs,
                viewportWidth: options.viewportWidth,
                viewportHeight: options.viewportHeight,
                session,
                baseUrl,
                targetBaselineScreenshot: options.targetBaselineScreenshot,
            },
            targetSnapshot: targetSample.snapshot,
            settleSnapshot,
            benchmarkStartStep,
            targetElapsedMs: targetSample.elapsedMs,
            totalElapsedMs,
            screenshots: {
                target: targetScreenshotPath,
                settle: settleScreenshotPath,
            },
            metrics: {
                targetStepsPerSecond: (targetSample.snapshot.step - benchmarkStartStep) / Math.max(0.001, targetSample.elapsedMs / 1000),
                settleStepsPerSecond: (settleSnapshot.step - targetSample.snapshot.step) / Math.max(0.001, options.settleMs / 1000),
                totalStepsPerSecond: (settleSnapshot.step - benchmarkStartStep) / Math.max(0.001, totalElapsedMs / 1000),
                browserErrors: getPayloadCount(errorsPayload.data),
                consoleEntries: getPayloadCount(consolePayload.data),
                hasQualityHoldWarning: /Paused Mandelbrot zoom/.test(consoleText),
                hasPrecisionWarning: /precision floor/i.test(consoleText),
                sampledSnapshots: samples.length,
                targetScreenshotMismatchPercentage: targetScreenshotDiff?.mismatchPercentage ?? null,
                ...sampleMetrics,
            },
            pass: {
                reachedTarget: targetSample.snapshot.step >= options.targetStep,
                targetHealthy: isHealthySnapshot(targetSample.snapshot),
                settleHealthy: isHealthySnapshot(settleSnapshot),
                stillAdvancing: settleSnapshot.step > targetSample.snapshot.step,
                noBrowserErrors: getPayloadCount(errorsPayload.data) === 0,
                noQualityHoldWarning: !/Paused Mandelbrot zoom/.test(consoleText),
                noPrecisionWarning: !/precision floor/i.test(consoleText),
                targetScreenshotMatches: targetScreenshotDiff ? targetScreenshotDiff.match === true : true,
            },
            samples,
            rawConsole: consolePayload.data,
            rawErrors: errorsPayload.data,
            rawConsoleText: consoleText,
            rawErrorsText: errorsText,
            targetScreenshotDiff,
        };

        await writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`, 'utf8');
        return report;
    } finally {
        await runAgentBrowser(session, ['close'], { allowFailure: true, timeoutMs: 15000 });
    }
}

async function main() {
    const options = parseArgs(process.argv.slice(2));
    const server = createStaticServer(projectRoot);
    const label = options.label || `mandelbrot-step${options.targetStep}-${Date.now()}`;
    const reportPath = path.join(options.outputDir, `${label}.json`);

    await mkdir(options.outputDir, { recursive: true });

    await new Promise((resolve) => {
        server.listen(0, '127.0.0.1', resolve);
    });

    const address = server.address();
    if (!address || typeof address === 'string') {
        throw new Error('Could not determine static server address');
    }

    const baseUrl = `http://127.0.0.1:${address.port}`;
    let report = null;

    try {
        if (options.repeat === 1) {
            report = await runSingleBenchmark(options, baseUrl, label, 0);
        } else {
            const trialReports = [];
            for (let trialIndex = 0; trialIndex < options.repeat; trialIndex += 1) {
                const trialLabel = `${label}-trial${trialIndex + 1}`;
                const trialReport = await runSingleBenchmark(options, baseUrl, trialLabel, trialIndex);
                trialReports.push(trialReport);
                console.log(
                    `Trial ${trialIndex + 1}/${options.repeat}: `
                    + `target=${formatMetric(trialReport.metrics.targetStepsPerSecond)} steps/s, `
                    + `settle=${formatMetric(trialReport.metrics.settleStepsPerSecond)} steps/s`
                );
            }
            report = aggregateReports(label, reportPath, options, baseUrl, trialReports);
            await writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`, 'utf8');
        }

        if (options.baseline) {
            const baseline = JSON.parse(await readFile(options.baseline, 'utf8'));
            report.comparison = compareAgainstBaseline(report, baseline);
            await writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`, 'utf8');
        }

        const passed = Object.values(report.pass).every(Boolean);
        console.log(`Report: ${reportPath}`);
        console.log(
            `Target step: ${report.targetSnapshot?.step ?? options.targetStep} in ${formatMetric((report.targetElapsedMs ?? 0) / 1000)}s`
        );
        console.log(
            `Settle step: ${report.settleSnapshot?.step ?? 'n/a'} after +${formatMetric(options.settleMs / 1000)}s`
        );
        console.log(`Target throughput: ${formatMetric(report.metrics.targetStepsPerSecond)} steps/s`);
        console.log(`Settle throughput: ${formatMetric(report.metrics.settleStepsPerSecond)} steps/s`);
        console.log(
            `Quality metrics: refs<=${report.metrics.sampledMaxReferencesUsed}, `
            + `repair<=${report.metrics.sampledMaxRepairPasses}, `
            + `cpuTiles<=${report.metrics.sampledMaxCpuResolvedTiles}, `
            + `holds=${report.metrics.sampledHoldSnapshots}`
        );
        if (Number.isFinite(report.metrics.targetScreenshotMismatchPercentage)) {
            console.log(`Target screenshot mismatch: ${formatMetric(report.metrics.targetScreenshotMismatchPercentage, 4)}%`);
        }
        console.log(`Pass: ${passed ? 'yes' : 'no'}`);

        if (report.comparison) {
            const comparison = report.comparison;
            console.log(
                `Baseline compare: target throughput ${formatMetric(comparison.speedMetrics.targetStepsPerSecondDeltaPct)}%`
                + `, settle throughput ${formatMetric(comparison.speedMetrics.settleStepsPerSecondDeltaPct)}%`
                + `, dominates=${comparison.dominates}`
            );
        }

        if (!passed) {
            process.exitCode = 1;
        }
    } finally {
        await runAgentBrowser(options.session, ['close'], { allowFailure: true, timeoutMs: 15000 });
        await new Promise((resolve) => server.close(resolve));
    }
}

main().catch((error) => {
    console.error(error.message || error);
    process.exitCode = 1;
});
