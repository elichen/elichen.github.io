const canvas = document.getElementById("scene");
const ctx = canvas.getContext("2d");

const assetSources = {
    sky: "assets/sky-water.png",
    midground: "assets/midground.png",
    foreground: "assets/foreground.png",
    runner: "assets/optimus-runner.png",
};

const assets = {};
const metrics = {};

const state = {
    lastTime: 0,
    distance: 0,
    runnerTime: 0,
};

const pace = 1;
const depth = 1.15;
const grade = { fill: "rgba(255, 190, 94, 0.04)", fog: "rgba(255, 222, 196, 0)", tint: null };
const SCENE = {
    pathTop: 0.82,
    pathBottom: 1,
    runnerGround: 0.94,
};
const RUN_CYCLE = [1, 2, 3, 4, 5, 6];

function loadImage(src) {
    const image = new Image();
    return new Promise((resolve, reject) => {
        image.addEventListener("load", () => resolve(image), { once: true });
        image.addEventListener("error", () => reject(new Error(`Unable to load ${src}`)), { once: true });
        image.src = src;
    });
}

function resizeCanvas() {
    const bounds = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.max(640, Math.floor(bounds.width * dpr));
    canvas.height = Math.max(360, Math.floor(bounds.height * dpr));
    ctx.imageSmoothingEnabled = false;
}

function measureAlphaBounds(image, columns = 1) {
    const scratch = document.createElement("canvas");
    scratch.width = image.width;
    scratch.height = image.height;
    const scratchCtx = scratch.getContext("2d", { willReadFrequently: true });
    scratchCtx.drawImage(image, 0, 0);

    const boxes = [];
    for (let column = 0; column < columns; column += 1) {
        const sx = Math.floor((image.width * column) / columns);
        const ex = Math.floor((image.width * (column + 1)) / columns);
        const width = ex - sx;
        const data = scratchCtx.getImageData(sx, 0, width, image.height).data;
        let minX = width;
        let minY = image.height;
        let maxX = -1;
        let maxY = -1;

        for (let y = 0; y < image.height; y += 1) {
            for (let x = 0; x < width; x += 1) {
                if (data[(y * width + x) * 4 + 3] > 12) {
                    minX = Math.min(minX, x);
                    minY = Math.min(minY, y);
                    maxX = Math.max(maxX, x);
                    maxY = Math.max(maxY, y);
                }
            }
        }

        boxes.push({
            sx: sx + Math.max(0, minX - 2),
            sy: Math.max(0, minY - 2),
            sw: Math.min(width, maxX - minX + 5),
            sh: Math.min(image.height, maxY - minY + 5),
        });
    }

    return columns === 1 ? boxes[0] : boxes;
}

function measureColorBounds(image, predicate) {
    const scratch = document.createElement("canvas");
    scratch.width = image.width;
    scratch.height = image.height;
    const scratchCtx = scratch.getContext("2d", { willReadFrequently: true });
    scratchCtx.drawImage(image, 0, 0);

    const data = scratchCtx.getImageData(0, 0, image.width, image.height).data;
    let minX = image.width;
    let minY = image.height;
    let maxX = -1;
    let maxY = -1;

    for (let y = 0; y < image.height; y += 1) {
        for (let x = 0; x < image.width; x += 1) {
            const index = (y * image.width + x) * 4;
            if (predicate(data[index], data[index + 1], data[index + 2], data[index + 3])) {
                minX = Math.min(minX, x);
                minY = Math.min(minY, y);
                maxX = Math.max(maxX, x);
                maxY = Math.max(maxY, y);
            }
        }
    }

    return { sx: minX, sy: minY, sw: maxX - minX + 1, sh: maxY - minY + 1 };
}

function measureOpaqueRow(image, minPixels) {
    const scratch = document.createElement("canvas");
    scratch.width = image.width;
    scratch.height = image.height;
    const scratchCtx = scratch.getContext("2d", { willReadFrequently: true });
    scratchCtx.drawImage(image, 0, 0);
    const data = scratchCtx.getImageData(0, 0, image.width, image.height).data;

    for (let y = 0; y < image.height; y += 1) {
        let count = 0;
        for (let x = 0; x < image.width; x += 1) {
            if (data[(y * image.width + x) * 4 + 3] > 12) {
                count += 1;
            }
        }
        if (count >= minPixels) {
            return y;
        }
    }
    return 0;
}

function yFromAnchor(targetY, sourceY, scale) {
    return targetY - sourceY * scale;
}

function measureRunnerHeadAnchors(image, frameBounds) {
    const scratch = document.createElement("canvas");
    scratch.width = image.width;
    scratch.height = image.height;
    const scratchCtx = scratch.getContext("2d", { willReadFrequently: true });
    scratchCtx.drawImage(image, 0, 0);
    const data = scratchCtx.getImageData(0, 0, image.width, image.height).data;

    return frameBounds.map((box) => {
        const top = box.sy;
        const bottom = box.sy + Math.round(box.sh * 0.28);
        let weightedX = 0;
        let weight = 0;

        for (let y = top; y < bottom; y += 1) {
            for (let x = box.sx; x < box.sx + box.sw; x += 1) {
                const alpha = data[(y * image.width + x) * 4 + 3];
                if (alpha > 12) {
                    weightedX += x * alpha;
                    weight += alpha;
                }
            }
        }

        return weight > 0 ? weightedX / weight : box.sx + box.sw / 2;
    });
}

function makeRunnerCells(image, frameBounds, headAnchors) {
    const frameWidth = image.width / 8;
    const minY = Math.min(...frameBounds.map((box) => box.sy));
    const maxY = Math.max(...frameBounds.map((box) => box.sy + box.sh));
    const maxWidth = Math.max(...frameBounds.map((box) => box.sw));

    return frameBounds.map((box, index) => {
        const cellLeft = index * frameWidth;
        const centerX = box.sx + box.sw / 2;
        const sx = Math.max(cellLeft, centerX - maxWidth / 2);
        const right = Math.min(cellLeft + frameWidth, sx + maxWidth);

        return {
            sx,
            sy: minY,
            sw: right - sx,
            sh: maxY - minY,
            anchorX: headAnchors[index] - sx,
        };
    });
}

function drawWrappedImage(image, time, options) {
    const source = options.source || { sx: 0, sy: 0, sw: image.width, sh: image.height };
    const scale = options.height / source.sh;
    const drawW = Math.ceil(source.sw * scale);
    const drawH = Math.ceil(options.height);
    const speed = options.speed * pace * (0.55 + depth * 0.45);
    const scroll = time * speed + (options.offset || 0);
    const y = Math.round(options.y);
    const originX = options.originX || 0;
    const startIndex = Math.floor((scroll - originX) / drawW) - 1;
    const endIndex = startIndex + Math.ceil(canvas.width / drawW) + 3;

    for (let i = startIndex; i < endIndex; i += 1) {
        const x = Math.round(i * drawW - scroll + originX);
        if (options.mirror && Math.abs(i) % 2 === 1) {
            ctx.save();
            ctx.translate(x + drawW, y);
            ctx.scale(-1, 1);
            ctx.drawImage(image, source.sx, source.sy, source.sw, source.sh, 0, 0, drawW, drawH);
            ctx.restore();
        } else {
            ctx.drawImage(image, source.sx, source.sy, source.sw, source.sh, x, y, drawW, drawH);
        }
    }
}

function drawSky(time) {
    drawWrappedImage(assets.sky, time, {
        y: 0,
        height: canvas.height,
        speed: 8,
        mirror: true,
    });
}

function drawMidground(time) {
    const targetPathTop = canvas.height * SCENE.pathTop;
    const scale = (canvas.height * 0.53) / metrics.midground.sh;
    const y = yFromAnchor(targetPathTop, metrics.bridgeBaseY - metrics.midground.sy, scale);

    drawWrappedImage(assets.midground, time, {
        source: metrics.midground,
        y,
        height: metrics.midground.sh * scale,
        speed: 24,
        originX: -canvas.width * 0.18,
        mirror: true,
    });
}

function drawForeground(time) {
    const targetPathTop = canvas.height * SCENE.pathTop;
    const targetPathBottom = canvas.height * SCENE.pathBottom;
    const scale = (targetPathBottom - targetPathTop) / (metrics.foreground.sy + metrics.foreground.sh - metrics.pathTopY);
    const y = yFromAnchor(targetPathTop, metrics.pathTopY - metrics.foreground.sy, scale);

    drawWrappedImage(assets.foreground, time, {
        source: metrics.foreground,
        y,
        height: metrics.foreground.sh * scale,
        speed: 126,
        mirror: true,
    });
}

function drawFog(time) {
    if (grade.fog.endsWith(", 0)")) return;

    ctx.save();
    ctx.fillStyle = grade.fog;
    for (let i = 0; i < 5; i += 1) {
        const y = Math.round(canvas.height * (0.42 + i * 0.07));
        const h = Math.max(8, Math.round(canvas.height * (0.02 + i * 0.004)));
        const x = -((time * (10 + i * 7)) % (canvas.width * 0.55));
        for (let j = 0; j < 4; j += 1) {
            const left = Math.round(x + j * canvas.width * 0.55);
            ctx.fillRect(left, y, Math.round(canvas.width * 0.34), h);
            ctx.fillRect(left + Math.round(canvas.width * 0.08), y - h, Math.round(canvas.width * 0.18), h);
        }
    }
    ctx.restore();
}

function drawSpeedLines(time) {
    if (pace < 1.15) return;

    ctx.save();
    ctx.globalAlpha = Math.min(0.38, (pace - 1) * 0.2);
    ctx.strokeStyle = "#fff1b8";
    ctx.lineWidth = Math.max(2, canvas.height * 0.004);
    for (let i = 0; i < 18; i += 1) {
        const x = canvas.width - ((time * 900 * pace + i * 173) % (canvas.width + 260));
        const y = canvas.height * (0.20 + ((i * 47) % 430) / 1000);
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x + 120 + pace * 36, y);
        ctx.stroke();
    }
    ctx.restore();
}

function drawRunner(time) {
    const runner = assets.runner;
    const cycleIndex = Math.floor(time * 9 * pace) % RUN_CYCLE.length;
    const frame = RUN_CYCLE[cycleIndex];
    const source = metrics.runnerCells[frame];
    const drawH = canvas.height * 0.31;
    const scale = drawH / source.sh;
    const drawW = source.sw * scale;
    const bob = Math.sin(time * Math.PI * 18 * pace) * canvas.height * 0.003;
    const x = canvas.width * 0.34 - source.anchorX * scale;
    const y = canvas.height * SCENE.runnerGround - drawH + bob;

    ctx.drawImage(runner, source.sx, source.sy, source.sw, source.sh, Math.round(x), Math.round(y), Math.round(drawW), Math.round(drawH));
}

function drawVignette() {
    const gradient = ctx.createRadialGradient(
        canvas.width * 0.52,
        canvas.height * 0.48,
        canvas.width * 0.18,
        canvas.width * 0.52,
        canvas.height * 0.48,
        canvas.width * 0.72
    );
    gradient.addColorStop(0, "rgba(0, 0, 0, 0)");
    gradient.addColorStop(1, "rgba(0, 0, 0, 0.34)");
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function render(timestamp) {
    const seconds = timestamp / 1000;
    const dt = Math.min(0.05, seconds - state.lastTime || 0);
    state.lastTime = seconds;

    state.distance += dt;
    state.runnerTime += dt;

    const sceneTime = state.distance;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#0f172a";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    drawSky(sceneTime);
    drawMidground(sceneTime);
    drawFog(sceneTime);
    drawForeground(sceneTime);
    drawRunner(state.runnerTime);
    drawSpeedLines(sceneTime);

    if (grade.tint) {
        ctx.fillStyle = grade.tint;
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    }

    ctx.fillStyle = grade.fill;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    drawVignette();

    requestAnimationFrame(render);
}

window.addEventListener("resize", resizeCanvas);

Promise.all(Object.entries(assetSources).map(([name, src]) => (
    loadImage(src).then((image) => {
        assets[name] = image;
    })
))).then(() => {
    metrics.midground = measureAlphaBounds(assets.midground);
    metrics.foreground = measureAlphaBounds(assets.foreground);
    metrics.runnerFrames = measureAlphaBounds(assets.runner, 8);
    metrics.runnerHeadAnchors = measureRunnerHeadAnchors(assets.runner, metrics.runnerFrames);
    metrics.runnerCells = makeRunnerCells(assets.runner, metrics.runnerFrames, metrics.runnerHeadAnchors);
    metrics.bridge = measureColorBounds(assets.midground, (r, g, b, a) => (
        a > 20 && r > 105 && g < 115 && b < 90 && r > g * 1.15
    ));
    metrics.bridgeBaseY = metrics.bridge.sy + metrics.bridge.sh;
    metrics.pathTopY = measureOpaqueRow(assets.foreground, assets.foreground.width * 0.25);
    resizeCanvas();
    requestAnimationFrame(render);
}).catch((error) => {
    ctx.fillStyle = "#0f172a";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#f8d26a";
    ctx.font = "24px monospace";
    ctx.fillText(error.message, 24, 48);
});
