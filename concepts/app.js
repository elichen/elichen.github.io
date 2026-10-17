// The explorer: a photo, the concepts the network finds in it, why it gives its answer, and painting.
//
// Data: ../lucid/robust-resnet50/ (the network), model/ (the sparse autoencoder), data/concepts.json (per
// concept: how often it fires, its best activation over ImageNet's 50,000 validation images, the classes of its
// top 20 of them, and example images from the pool at full, half and a fifth of that), data/pool.json +
// pool/{i}.jpg (6,106 validation images that hold every concept's examples), renders/{i}.jpg.

const SIZE = 224, GRID = 7, CELL = SIZE / GRID, VIEW = 448;
const BRUSH_RADIUS = 22;   // image pixels
const LIST_LENGTH = 15;
const AGAINST_LENGTH = 5;
const UNDO_LIMIT = 12;

const $ = id => document.getElementById(id);
const els = {
    photo: $('photo'), status: $('status'), another: $('another'), file: $('file'), undo: $('undo'), reset: $('reset'),
    search: $('search'), prediction: $('prediction'), listTitle: $('list-title'), listNote: $('list-note'),
    list: $('concept-list'), againstHead: $('against-head'), against: $('against-list'), detail: $('detail'), detailRender: $('detail-render'), detailTitle: $('detail-title'),
    detailStats: $('detail-stats'), examples: $('examples'), pushes: $('pushes'), neighbours: $('neighbours')
};
const ctx = els.photo.getContext('2d');
const scratch = Object.assign(document.createElement('canvas'), { width: SIZE, height: SIZE });
const shade = Object.assign(document.createElement('canvas'), { width: GRID, height: GRID });

const app = {
    resnet: null, sae: null, painter: null, concepts: null, pool: null, labels: null, fc: null,
    original: null,      // the photo as loaded, [1, 224, 224, 3] in [0, 1]
    image: null,         // the photo with any paint
    pixels: null,        // app.image's values, for drawing
    undo: [],
    code: null, probs: null, top5: [], classIndex: null, contributions: [],
    selected: null,      // the picked concept
    preview: null,       // a concept hovered in a list, shown on the photo until the pointer leaves it
    hover: null,         // the photo cell under the pointer
    pointer: null, painting: false, paintLoop: null, exampleToken: 0
};
window.conceptsApp = app;

// ---------- Loading ----------

async function selectBackend() {
    for (const name of ['webgpu', 'webgl', 'cpu']) {
        try {
            if (await tf.setBackend(name)) {
                await tf.ready();
                return name;
            }
        } catch (error) {
            console.warn(`Backend ${name} unavailable:`, error);
        }
    }
    throw new Error('No TensorFlow.js backend is available');
}

async function loadImage(url) {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = url;
    await img.decode();
    return img;
}

// The image's central square, resized to 224, as [224 * 224 * 3] values in [0, 1].
function imagePixels(img) {
    const side = Math.min(img.naturalWidth || img.width, img.naturalHeight || img.height);
    const sx = ((img.naturalWidth || img.width) - side) / 2, sy = ((img.naturalHeight || img.height) - side) / 2;
    const c = scratch.getContext('2d', { willReadFrequently: true });
    c.drawImage(img, sx, sy, side, side, 0, 0, SIZE, SIZE);
    const rgba = c.getImageData(0, 0, SIZE, SIZE).data;
    const out = new Float32Array(SIZE * SIZE * 3);
    for (let i = 0; i < SIZE * SIZE; i++) {
        for (let ch = 0; ch < 3; ch++) out[i * 3 + ch] = rgba[i * 4 + ch] / 255;
    }
    return out;
}

async function setPhoto(img) {
    tf.dispose([app.original, app.image, ...app.undo]);
    app.undo = [];
    app.original = tf.tensor4d(imagePixels(img), [1, SIZE, SIZE, 3]);
    app.image = app.original.clone();
    app.classIndex = null;
    await analyze();
}

async function loadPoolPhoto(n) {
    await setPhoto(await loadImage(`pool/${n}.jpg`));
}

// ---------- Analysis ----------

async function analyze() {
    const [acts, logits] = app.resnet.execute(app.image, [app.sae.layer, 'logits']);
    const [logitValues, pixels] = await Promise.all([logits.data(), app.image.data()]);
    app.code = await app.sae.encode(acts);
    tf.dispose([acts, logits]);
    app.pixels = pixels;
    const peak = Math.max(...logitValues);
    const exps = logitValues.map(v => Math.exp(v - peak));
    const total = exps.reduce((a, b) => a + b, 0);
    app.probs = exps.map(v => v / total);
    app.top5 = [...app.probs.keys()].sort((a, b) => app.probs[b] - app.probs[a]).slice(0, 5);
    if (!app.top5.includes(app.classIndex)) app.classIndex = app.top5[0];
    computeContributions();
    els.undo.disabled = app.undo.length === 0;
    els.reset.disabled = app.undo.length === 0;
    drawPhoto();
    renderPrediction();
    renderList();
}

function computeContributions() {
    app.contributions = [...app.sae.contributions(app.code, app.fc, app.classIndex)].sort((a, b) => b[1] - a[1]);
}

// ---------- Names ----------

function className(c) {
    return app.labels[c];
}

// Names were written by Claude from each concept's render and top examples (data/names.json, with "clear",
// "mixed" or "unclear"). Without one, fall back to the ImageNet classes of the concept's top 20 images.
function conceptName(j) {
    if (app.names?.[j]) return app.names[j][0];
    const [[c0, n0], second] = app.concepts[j].classes;
    return n0 >= 10 || !second ? className(c0) : `${className(c0)} · ${className(second[0])}`;
}

function conceptQuality(j) {
    return app.names?.[j]?.[1] ?? 'clear';
}

// ---------- The photo ----------

function cellAt(event) {
    const rect = els.photo.getBoundingClientRect();
    const x = (event.clientX - rect.left) / rect.width * SIZE, y = (event.clientY - rect.top) / rect.height * SIZE;
    const col = Math.min(GRID - 1, Math.max(0, Math.floor(x / CELL)));
    const row = Math.min(GRID - 1, Math.max(0, Math.floor(y / CELL)));
    return { x, y, cell: row * GRID + col };
}

function drawPhoto(pixels = app.pixels, spotlight = true) {
    const c = scratch.getContext('2d', { willReadFrequently: true });
    const data = c.createImageData(SIZE, SIZE);
    for (let i = 0; i < SIZE * SIZE; i++) {
        for (let ch = 0; ch < 3; ch++) data.data[i * 4 + ch] = pixels[i * 3 + ch] * 255;
        data.data[i * 4 + 3] = 255;
    }
    c.putImageData(data, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(scratch, 0, 0, VIEW, VIEW);
    const concept = app.preview ?? app.selected;
    if (spotlight && concept !== null && app.code) {
        drawSpotlight(ctx, app.sae.map(app.code, concept), app.concepts[concept].max, VIEW);
    }
    if (spotlight && app.hover !== null) {
        const s = VIEW / GRID;
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
        ctx.lineWidth = 2;
        ctx.strokeRect((app.hover % GRID) * s + 1, Math.floor(app.hover / GRID) * s + 1, s - 2, s - 2);
    }
}

// Dims the image where the concept is quiet. Activations are measured against a share of the concept's best
// on ImageNet, so a weak trace stays dim instead of being stretched to full brightness.
function drawSpotlight(target, map, best, size) {
    const peak = Math.max(...map);
    const norm = Math.max(peak, 0.3 * best, 1e-6);
    const c = shade.getContext('2d');
    const data = c.createImageData(GRID, GRID);
    for (let p = 0; p < GRID * GRID; p++) {
        data.data[p * 4 + 3] = 255 * 0.72 * (1 - Math.min(1, map[p] / norm));
    }
    c.putImageData(data, 0, 0);
    target.imageSmoothingEnabled = true;
    target.drawImage(shade, 0, 0, size, size);
}

els.photo.addEventListener('pointerdown', event => {
    if (!app.code) return;
    els.photo.setPointerCapture(event.pointerId);
    app.pointer = { ...cellAt(event), startX: event.clientX, startY: event.clientY, moved: false };
});

els.photo.addEventListener('pointermove', event => {
    if (!app.code) return;
    const at = cellAt(event);
    if (app.pointer) {
        if (Math.hypot(event.clientX - app.pointer.startX, event.clientY - app.pointer.startY) > 4) app.pointer.moved = true;
        if (app.pointer.moved && !app.painting) startPainting(app.pointer);
        if (app.painting) app.painter.dab(at.x, at.y, BRUSH_RADIUS);
        return;
    }
    if (at.cell !== app.hover) {
        app.hover = at.cell;
        drawPhoto();
        renderList();
    }
});

els.photo.addEventListener('pointerup', event => {
    if (!app.pointer) return;
    const pointer = app.pointer;
    app.pointer = null;
    if (app.painting) {
        stopPainting();
    } else if (!pointer.moved) {
        // A click picks the strongest concept in that cell.
        const here = conceptsAt(pointer.cell);
        if (here.length) select(here[0][0]);
    }
});

els.photo.addEventListener('pointerleave', () => {
    if (app.pointer || app.hover === null) return;
    app.hover = null;
    drawPhoto();
    renderList();
});

function conceptsAt(cell) {
    const { idx, val } = app.code, k = app.sae.k, out = [];
    for (let q = cell * k; q < (cell + 1) * k; q++) {
        if (val[q] > 0) out.push([idx[q], val[q]]);
    }
    return out.sort((a, b) => b[1] - a[1]);
}

// ---------- Painting ----------

function startPainting(pointer) {
    if (app.selected === null) {
        els.status.textContent = 'Pick a concept first (click one in the list), then drag to paint it.';
        setTimeout(() => { if (!app.painting) els.status.textContent = ''; }, 2500);
        return;
    }
    app.painting = true;
    app.hover = null;
    app.painter.begin(app.image, app.selected);
    app.painter.dab(pointer.x, pointer.y, BRUSH_RADIUS);
    app.paintLoop = (async () => {
        const name = conceptName(app.selected);
        while (app.painting) {
            tf.dispose(app.painter.step());
            const img = app.painter.image();
            const pixels = await img.data();
            img.dispose();
            drawPhoto(pixels, false);
            els.status.textContent = `Painting ${name} · step ${app.painter.stroke.steps}`;
        }
    })();
}

async function stopPainting() {
    app.painting = false;
    await app.paintLoop;
    app.undo.push(app.image);
    if (app.undo.length > UNDO_LIMIT) app.undo.shift().dispose();
    app.image = app.painter.end();
    els.status.textContent = '';
    await analyze();
    if (app.selected !== null) renderDetail(app.selected);
}

// ---------- Lists ----------

function conceptButton(j, value, kind) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'concept' + (j === app.selected ? ' on' : '');
    const img = document.createElement('img');
    img.src = `renders/${j}.jpg`;
    img.alt = '';
    img.loading = 'lazy';
    const name = document.createElement('span');
    name.className = 'name';
    name.textContent = conceptName(j);
    b.append(img, name);
    if (value !== undefined) {
        const v = document.createElement('span');
        v.className = 'value' + (kind === 'contribution' ? (value >= 0 ? ' plus' : ' minus') : '');
        v.textContent = kind === 'contribution' ? `${value >= 0 ? '+' : '−'}${Math.abs(value).toFixed(2)}`
            : kind === 'activation' ? value.toFixed(1) : value;
        b.append(v);
    }
    b.title = `#${j} ${conceptName(j)}`;
    b.addEventListener('click', () => select(j));
    b.addEventListener('mouseenter', () => { app.preview = j; if (!app.painting) drawPhoto(); });
    b.addEventListener('mouseleave', () => { app.preview = null; if (!app.painting) drawPhoto(); });
    return b;
}

function renderList() {
    const query = els.search.value.trim();
    const items = [], against = [];
    if (query) {
        const results = search(query);
        els.listTitle.textContent = `${results.length} concepts`;
        els.listNote.textContent = /^#?\d+$/.test(query) ? 'By number.'
            : `Concepts named “${query}”, then those whose top 20 images include it (how many of the 20).`;
        results.forEach(([j, n]) => items.push(conceptButton(j, n === null ? undefined : `${n}/20`)));
    } else if (app.hover !== null) {
        const here = conceptsAt(app.hover);
        els.listTitle.textContent = 'At this spot';
        els.listNote.textContent = `The ${here.length} concepts active in this cell, strongest first.`;
        here.forEach(([j, v]) => items.push(conceptButton(j, v, 'activation')));
    } else {
        const cls = className(app.classIndex);
        els.listTitle.textContent = `Why “${cls}”`;
        els.listNote.textContent = `Concepts in this photo, by how much each adds to the network’s score for ${cls}.`;
        app.contributions.slice(0, LIST_LENGTH).filter(([, v]) => v > 0)
            .forEach(([j, v]) => items.push(conceptButton(j, v, 'contribution')));
        app.contributions.filter(([, v]) => v < 0).slice(-AGAINST_LENGTH).reverse()
            .forEach(([j, v]) => against.push(conceptButton(j, v, 'contribution')));
    }
    els.list.replaceChildren(...items);
    els.against.replaceChildren(...against);
    els.againstHead.hidden = against.length === 0;
}

function renderPrediction() {
    const top = app.top5[0];
    const pct = c => `${(app.probs[c] * 100).toFixed(app.probs[c] < 0.1 ? 1 : 0)}%`;
    const says = document.createElement('div');
    says.className = 'says';
    says.textContent = 'The network says';
    const answer = document.createElement('div');
    answer.className = 'answer';
    answer.textContent = `${className(top)} `;
    const p = document.createElement('span');
    p.textContent = pct(top);
    answer.append(p);
    const chips = document.createElement('div');
    chips.className = 'chips';
    for (const c of app.top5) {
        const chip = document.createElement('button');
        chip.type = 'button';
        chip.className = 'chip' + (c === app.classIndex ? ' on' : '');
        chip.textContent = `${c === top ? 'why' : 'why not'} ${className(c)} · ${pct(c)}`;
        chip.addEventListener('click', () => {
            app.classIndex = c;
            computeContributions();
            renderPrediction();
            renderList();
        });
        chips.append(chip);
    }
    els.prediction.replaceChildren(says, answer, chips);
}

// Concepts whose name matches come first, then those whose top images include a matching class.
function search(query) {
    const q = query.toLowerCase().replace(/^#/, '');
    if (/^\d+$/.test(q)) return +q < app.sae.size ? [[+q, null]] : [];
    const results = [];
    app.concepts.forEach((c, j) => {
        let n = 0;
        for (const [cls, count] of c.classes) if (className(cls).toLowerCase().includes(q)) n += count;
        const named = conceptName(j).toLowerCase().includes(q);
        if (named || n) results.push([j, n, named]);
    });
    return results.sort((a, b) => b[2] - a[2] || b[1] - a[1]).slice(0, 48).map(([j, n]) => [j, n || null]);
}

els.search.addEventListener('input', renderList);

// ---------- The picked concept ----------

function select(j) {
    app.selected = j;
    for (const b of document.querySelectorAll('.side-column .concept')) b.classList.toggle('on', b.title.startsWith(`#${j} `));
    drawPhoto();
    renderDetail(j);
}

function renderDetail(j) {
    const c = app.concepts[j];
    els.detail.hidden = false;
    els.detailRender.src = `renders/${j}.jpg`;
    els.detailTitle.innerHTML = '';
    els.detailTitle.append(conceptName(j) + ' ');
    const id = document.createElement('span');
    id.className = 'id';
    id.textContent = `#${j}`;
    els.detailTitle.append(id);
    const map = app.code ? app.sae.map(app.code, j) : [];
    const cells = map.filter(v => v > 0).length;
    const quality = conceptQuality(j);
    els.detailStats.textContent = (quality === 'clear' ? '' : quality === 'mixed'
        ? 'A loose name: only about half its top examples fit it. '
        : 'No clear common thread in its top examples; the name is a guess. ') + `Fires on ${(c.freq * 100).toFixed(c.freq < 0.001 ? 3 : 2)}% of image patches in ImageNet` +
        ` · in ${cells} of this photo’s 49 cells` + (cells ? ` (up to ${Math.max(...map).toFixed(1)}; its best is ${c.max.toFixed(1)})` : '');

    const weights = app.sae.classWeights(j, app.fc);
    const order = [...weights.keys()].sort((a, b) => weights[b] - weights[a]);
    els.pushes.textContent = order.slice(0, 5).map(className).join(', ');

    els.neighbours.replaceChildren(...app.sae.neighbours(j, 6).map(([n]) => conceptButton(n)));
    renderExamples(j);
}

// Real examples from the pool, strongest first, each dimmed where the concept is quiet.
async function renderExamples(j) {
    const token = ++app.exampleToken;
    const c = app.concepts[j];
    const nodes = [], canvases = [];
    let previous = null;
    for (const [n, value] of c.examples) {
        const level = value >= 0.7 * c.max ? 0 : value >= 0.35 * c.max ? 1 : 2;
        if (previous !== null && level !== previous) {
            const gap = document.createElement('div');
            gap.className = 'gap';
            nodes.push(gap);
        }
        previous = level;
        const b = document.createElement('button');
        b.type = 'button';
        b.className = 'example';
        b.title = `${className(app.pool.labels[n])} · activation ${value.toFixed(1)} (${Math.round(100 * value / c.max)}% of its best)`;
        const canvas = Object.assign(document.createElement('canvas'), { width: 152, height: 152 });
        const label = document.createElement('span');
        label.textContent = `${Math.round(100 * value / c.max)}%`;
        b.append(canvas, label);
        b.addEventListener('click', () => loadPoolPhoto(n).then(() => renderDetail(app.selected)));
        nodes.push(b);
        canvases.push([canvas, n]);
    }
    els.examples.replaceChildren(...nodes);

    const imgs = await Promise.all(canvases.map(([, n]) => loadImage(`pool/${n}.jpg`)));
    if (token !== app.exampleToken) return;
    imgs.forEach((img, i) => canvases[i][0].getContext('2d').drawImage(img, 0, 0, 152, 152));
    // Where the concept fires in each example, one image at a time: a batch of 16 would hold 16 times the
    // memory, and single images reuse the GPU buffers of the photo's own pass.
    for (const [i, img] of imgs.entries()) {
        const x = tf.tensor4d(imagePixels(img), [1, SIZE, SIZE, 3]);
        const acts = app.resnet.execute(x, app.sae.layer);
        const code = await app.sae.encode(acts);
        tf.dispose([x, acts]);
        if (token !== app.exampleToken) return;
        drawSpotlight(canvases[i][0].getContext('2d'), app.sae.map(code, j), c.max, 152);
    }
}

// ---------- Photo tools ----------

els.another.addEventListener('click', () => loadPoolPhoto(Math.floor(Math.random() * app.pool.images.length)));

els.file.addEventListener('change', async () => {
    const file = els.file.files[0];
    if (file) await loadFile(file);
    els.file.value = '';
});

async function loadFile(file) {
    const url = URL.createObjectURL(file);
    try {
        await setPhoto(await loadImage(url));
    } finally {
        URL.revokeObjectURL(url);
    }
}

document.addEventListener('paste', event => {
    const item = [...(event.clipboardData?.items || [])].find(i => i.type.startsWith('image/'));
    if (item) loadFile(item.getAsFile());
});
els.photo.addEventListener('dragover', event => event.preventDefault());
els.photo.addEventListener('drop', event => {
    event.preventDefault();
    const file = [...event.dataTransfer.files].find(f => f.type.startsWith('image/'));
    if (file) loadFile(file);
});

els.undo.addEventListener('click', async () => {
    if (!app.undo.length || app.painting) return;
    app.image.dispose();
    app.image = app.undo.pop();
    await analyze();
    if (app.selected !== null) renderDetail(app.selected);
});

els.reset.addEventListener('click', async () => {
    if (app.painting) return;
    tf.dispose([app.image, ...app.undo]);
    app.undo = [];
    app.image = app.original.clone();
    await analyze();
    if (app.selected !== null) renderDetail(app.selected);
});

// ---------- Start ----------

(async () => {
    try {
        await selectBackend();
        const json = url => fetch(url).then(r => r.json());
        const [resnet, sae, concepts, pool, labels, names] = await Promise.all([
            RobustResNet50.load('../lucid/robust-resnet50/'), ConceptDictionary.load('model/'),
            json('data/concepts.json'), json('data/pool.json'), json('data/labels.json'),
            json('data/names.json').catch(() => null)
        ]);
        Object.assign(app, { resnet, sae, concepts, pool, labels, names });
        app.fc = await resnet.weights['fc/kernel'].data();
        app.painter = new Painter(resnet, sae);
        await loadPoolPhoto(Math.max(0, pool.labels.indexOf(207)));
        select(app.contributions[0][0]);
        els.status.textContent = '';
        // Compile the painting shaders now, so the first stroke doesn't stall for a second.
        app.painter.begin(app.image, 0);
        app.painter.dab(SIZE / 2, SIZE / 2, BRUSH_RADIUS);
        tf.dispose(app.painter.step());
        const warm = app.painter.end();
        await warm.data();
        warm.dispose();
    } catch (error) {
        console.error(error);
        els.status.textContent = `Couldn’t start: ${error.message}`;
    }
})();
