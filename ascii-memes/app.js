(() => {
    "use strict";

    const GLYPHS = " .,:;irsXA253hMHGS#9B&@|-\\";
    const clips = [
        {
            id: "nyan-cat",
            title: "NYAN CAT",
            ascii: "ascii/nyan-cat.ascv",
            accent: "#ff76bc",
            label: "Nyan Cat YouTube footage pre-transcoded into animated ASCII characters"
        },
        {
            id: "keyboard-cat",
            title: "KEYBOARD CAT",
            ascii: "ascii/keyboard-cat.ascv",
            accent: "#b6ff6a",
            label: "Keyboard Cat YouTube footage pre-transcoded into animated ASCII characters"
        },
        {
            id: "dramatic-chipmunk",
            title: "DRAMATIC CHIPMUNK",
            ascii: "ascii/dramatic-chipmunk.ascv",
            accent: "#f4c542",
            label: "Dramatic Chipmunk YouTube footage pre-transcoded into animated ASCII characters"
        },
        {
            id: "chocolate-rain",
            title: "CHOCOLATE RAIN",
            ascii: "ascii/chocolate-rain.ascv",
            accent: "#e8b768",
            label: "Chocolate Rain YouTube footage pre-transcoded into animated ASCII characters"
        },
        {
            id: "sneezing-panda",
            title: "SNEEZING PANDA",
            ascii: "ascii/sneezing-panda.ascv",
            accent: "#e8e2cf",
            label: "Sneezing Baby Panda YouTube footage pre-transcoded into animated ASCII characters"
        },
        {
            id: "rickroll",
            title: "RICKROLL",
            ascii: "ascii/rickroll.ascv",
            accent: "#ff6658",
            label: "Rickroll YouTube footage pre-transcoded into animated ASCII characters"
        }
    ];

    const canvas = document.querySelector("#ascii-canvas");
    const context = canvas.getContext("2d", { alpha: false });
    const stage = document.querySelector("#ascii-stage");
    const reelList = document.querySelector("#reel-list");

    let currentIndex = Math.max(0, clips.findIndex((clip) => `#${clip.id}` === window.location.hash));
    let currentStream = null;
    let loadGeneration = 0;
    let playbackStartedAt = performance.now();
    let lastFrame = -1;

    function decodeAsciiStream(buffer) {
        const view = new DataView(buffer);
        const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 4));
        if (magic !== "ASCV" || view.getUint8(4) !== 1) throw new Error("Unsupported ASCII stream");
        const columns = view.getUint8(5);
        const rows = view.getUint8(6);
        const fps = view.getUint8(7);
        const frameCount = view.getUint32(8, true);
        const cellsPerFrame = columns * rows;
        const frames = new Uint16Array(frameCount * cellsPerFrame);
        const previous = new Uint16Array(cellsPerFrame);
        let offset = 12;

        for (let frame = 0; frame < frameCount; frame += 1) {
            const encodedLength = view.getUint32(offset, true);
            offset += 4;
            const frameEnd = offset + encodedLength;
            let cell = 0;
            while (offset < frameEnd && cell < cellsPerFrame) {
                const command = view.getUint8(offset);
                offset += 1;
                const run = (command & 0x7f) + 1;
                if (command & 0x80) {
                    cell += run;
                } else {
                    for (let i = 0; i < run; i += 1) {
                        previous[cell] = view.getUint16(offset, true);
                        offset += 2;
                        cell += 1;
                    }
                }
            }
            if (cell !== cellsPerFrame || offset !== frameEnd) throw new Error("Corrupt ASCII frame");
            frames.set(previous, frame * cellsPerFrame);
        }

        return { columns, rows, fps, frameCount, cellsPerFrame, frames };
    }

    function fitCanvas() {
        const deviceScale = Math.min(window.devicePixelRatio || 1, 2);
        const width = Math.max(1, Math.round(stage.clientWidth));
        const height = Math.max(1, Math.round(stage.clientHeight));
        const pixelWidth = Math.round(width * deviceScale);
        const pixelHeight = Math.round(height * deviceScale);
        if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
            canvas.width = pixelWidth;
            canvas.height = pixelHeight;
        }
        canvas.style.width = `${width}px`;
        canvas.style.height = `${height}px`;
        context.setTransform(deviceScale, 0, 0, deviceScale, 0, 0);
    }

    function drawStatus(message) {
        fitCanvas();
        const width = stage.clientWidth;
        const height = stage.clientHeight;
        context.fillStyle = "#071008";
        context.fillRect(0, 0, width, height);
        context.fillStyle = clips[currentIndex].accent;
        context.font = '700 12px "Azeret Mono", monospace';
        context.textAlign = "center";
        context.textBaseline = "middle";
        context.fillText(`[ ${message} ]`, width / 2, height / 2);
        context.textAlign = "left";
        context.textBaseline = "top";
    }

    function drawFrame(frameIndex) {
        if (!currentStream) return;
        fitCanvas();
        const { columns, rows, cellsPerFrame, frames } = currentStream;
        const safeFrame = ((frameIndex % currentStream.frameCount) + currentStream.frameCount) % currentStream.frameCount;
        const start = safeFrame * cellsPerFrame;
        const width = stage.clientWidth;
        const height = stage.clientHeight;
        const cellWidth = width / columns;
        const cellHeight = height / rows;
        const fontSize = cellHeight * 0.96;
        context.fillStyle = "#071008";
        context.fillRect(0, 0, width, height);
        context.font = `700 ${fontSize}px "Azeret Mono", "Courier New", monospace`;
        context.textAlign = "left";
        context.textBaseline = "top";

        for (let cell = 0; cell < cellsPerFrame; cell += 1) {
            const packed = frames[start + cell];
            const glyph = GLYPHS[packed & 0xff] || " ";
            if (glyph === " ") continue;
            const palette = packed >> 8;
            const red = Math.floor(palette / 36) * 51;
            const green = Math.floor((palette % 36) / 6) * 51;
            const blue = (palette % 6) * 51;
            context.fillStyle = `rgb(${red} ${green} ${blue})`;
            const x = cell % columns;
            const y = Math.floor(cell / columns);
            context.fillText(glyph, x * cellWidth, y * cellHeight);
        }
    }

    function renderReel() {
        reelList.replaceChildren();
        clips.forEach((clip, index) => {
            const button = document.createElement("button");
            button.type = "button";
            button.className = `reel-item${index === currentIndex ? " is-active" : ""}`;
            if (index === currentIndex) button.setAttribute("aria-current", "true");
            button.setAttribute("aria-label", `Show ${clip.title}`);
            const title = document.createElement("span");
            title.className = "reel-title";
            title.textContent = clip.title;
            button.append(title);
            button.addEventListener("click", () => selectClip(index, true));
            reelList.append(button);
        });
    }

    async function selectClip(index, updateHash = false) {
        currentIndex = (index + clips.length) % clips.length;
        const clip = clips[currentIndex];
        const generation = ++loadGeneration;
        currentStream = null;
        lastFrame = -1;
        playbackStartedAt = performance.now();
        stage.setAttribute("aria-label", clip.label);
        if (updateHash) history.replaceState(null, "", `#${clip.id}`);
        renderReel();
        drawStatus("LOADING PRECOMPUTED GLYPHS");

        try {
            const response = await fetch(clip.ascii);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const stream = decodeAsciiStream(await response.arrayBuffer());
            if (generation !== loadGeneration) return;
            currentStream = stream;
            playbackStartedAt = performance.now();
            drawFrame(0);
        } catch (error) {
            if (generation === loadGeneration) drawStatus("ASCII STREAM UNAVAILABLE");
            console.error(error);
        }
    }

    function animationLoop(timestamp) {
        if (currentStream) {
            const elapsed = (timestamp - playbackStartedAt) / 1000;
            const frame = Math.floor(elapsed * currentStream.fps) % currentStream.frameCount;
            if (frame !== lastFrame) {
                lastFrame = frame;
                drawFrame(frame);
            }
        }
        requestAnimationFrame(animationLoop);
    }

    window.addEventListener("hashchange", () => {
        const index = clips.findIndex((clip) => `#${clip.id}` === window.location.hash);
        if (index >= 0 && index !== currentIndex) selectClip(index);
    });

    new ResizeObserver(() => {
        fitCanvas();
        lastFrame = -1;
        if (currentStream) drawFrame(0);
    }).observe(stage);

    renderReel();
    selectClip(currentIndex);
    requestAnimationFrame(animationLoop);
})();
