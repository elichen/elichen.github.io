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

function drawWrappedImage(image, time, options) {
    const source = options.source || { sx: 0, sy: 0, sw: image.width, sh: image.height };
    const scale = options.height / source.sh;
    const drawW = Math.ceil(source.sw * scale);
    const drawH = Math.ceil(options.height);
    const speed = options.speed * pace * (0.55 + depth * 0.45);
    const scroll = (time * speed + (options.offset || 0)) % drawW;
    const y = Math.round(options.y);
    const startIndex = Math.floor(scroll / drawW) - 1;

    for (let i = startIndex; i < startIndex + Math.ceil(canvas.width / drawW) + 3; i += 1) {
        const x = Math.round(i * drawW - scroll + (options.originX || 0));
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
    const height = canvas.height * 0.54;
    drawWrappedImage(assets.midground, time, {
        source: metrics.midground,
        y: canvas.height * 0.92 - height,
        height,
        speed: 24,
        originX: -canvas.width * 0.18,
        mirror: true,
    });
}

function drawForeground(time) {
    drawWrappedImage(assets.foreground, time, {
        source: metrics.foreground,
        y: canvas.height * 0.76,
        height: canvas.height * 0.24,
        speed: 126,
        mirror: true,
    });
}

function drawRowScrolledRoad(time) {
    const top = canvas.height * 0.89;
    const bottom = canvas.height * 0.985;
    const palette = ["rgba(92, 52, 33, 0.34)", "rgba(255, 208, 119, 0.22)", "rgba(43, 27, 21, 0.28)"];

    ctx.save();
    ctx.beginPath();
    ctx.rect(0, top, canvas.width, bottom - top);
    ctx.clip();

    for (let row = 0; row < 9; row += 1) {
        const rowT = row / 8;
        const y = Math.round(top + rowT * (bottom - top));
        const h = Math.max(2, Math.round(canvas.height * (0.003 + rowT * 0.006)));
        const speed = (130 + rowT * 270) * pace * (0.45 + depth * 0.55);
        const spacing = canvas.width * (0.16 - rowT * 0.07);
        const dashW = spacing * (0.28 + rowT * 0.35);
        const offset = (time * speed + row * 91) % spacing;

        ctx.fillStyle = palette[row % palette.length];
        for (let x = -offset - spacing; x < canvas.width + spacing; x += spacing) {
            ctx.fillRect(Math.round(x), y, Math.round(dashW), h);
        }
    }
    ctx.restore();
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
    const frameCount = 8;
    const frame = Math.floor(time * 13 * pace) % frameCount;
    const source = metrics.runnerFrames[frame];
    const drawH = canvas.height * 0.34;
    const drawW = source.sw * (drawH / source.sh);
    const bob = Math.sin(time * Math.PI * 26 * pace) * canvas.height * 0.005;
    const x = canvas.width * 0.34 - drawW * 0.5;
    const y = canvas.height * 0.965 - drawH + bob;

    ctx.save();
    ctx.shadowColor = "rgba(0, 0, 0, 0.38)";
    ctx.shadowBlur = 0;
    ctx.shadowOffsetX = Math.round(canvas.width * 0.008);
    ctx.shadowOffsetY = Math.round(canvas.height * 0.012);
    ctx.drawImage(runner, source.sx, source.sy, source.sw, source.sh, Math.round(x), Math.round(y), Math.round(drawW), Math.round(drawH));
    ctx.restore();
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
    drawRowScrolledRoad(sceneTime);
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
    resizeCanvas();
    requestAnimationFrame(render);
}).catch((error) => {
    ctx.fillStyle = "#0f172a";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#f8d26a";
    ctx.font = "24px monospace";
    ctx.fillText(error.message, 24, 48);
});
