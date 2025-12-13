// Minimal Stick Figure Comic Maker

class MinimalComicMaker {
    constructor() {
        this.canvas = document.getElementById('comicCanvas');
        this.ctx = this.canvas.getContext('2d');

        this.elements = [];
        this.history = [];
        this.selectedId = null;

        this.draggingId = null;
        this.dragOffset = { x: 0, y: 0 };
        this.dragStartPos = null;
        this.dragMoved = false;

        this.editingId = null;
        this.editingWasNew = false;

        this.dpr = window.devicePixelRatio || 1;
        this.width = 0;
        this.height = 0;

        this.bindUI();
        this.resize();
        this.draw();
    }

    debug(...args) {
        if (window.DEBUG_COMIC) {
            console.log('[comic]', ...args);
        }
    }

    bindUI() {
        // Toolbar - add buttons add items at random positions
        document.querySelectorAll('.tool-btn[data-add]').forEach(btn => {
            btn.addEventListener('click', () => this.addItemAtRandom(btn.dataset.add));
        });
        document.getElementById('undoBtn').addEventListener('click', () => this.undo());
        document.getElementById('clearBtn').addEventListener('click', () => this.clear());
        document.getElementById('saveBtn').addEventListener('click', () => this.save());

        // Canvas pointer events
        this.canvas.addEventListener('pointerdown', e => this.onPointerDown(e));
        this.canvas.addEventListener('pointermove', e => this.onPointerMove(e));
        this.canvas.addEventListener('pointerup', () => this.onPointerUp());
        this.canvas.addEventListener('pointerleave', () => this.onPointerUp());

        // Text editor
        document.getElementById('saveEdit').addEventListener('click', () => this.saveText());
        document.getElementById('cancelEdit').addEventListener('click', () => this.cancelEdit());

        document.getElementById('textInput').addEventListener('keydown', e => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.saveText();
            } else if (e.key === 'Escape') {
                e.preventDefault();
                this.cancelEdit();
            }
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', e => this.onKeyDown(e));

        // Resize
        window.addEventListener('resize', () => this.resize());
    }

    addItemAtRandom(type) {
        const margin = 60;
        const p = {
            x: margin + Math.random() * (this.width - margin * 2),
            y: margin + Math.random() * (this.height - margin * 2)
        };

        if (type === 'man') this.addPerson(p, 'standing', 'm');
        else if (type === 'woman') this.addPerson(p, 'standing', 'f');
        else if (type === 'chair') this.addChair(p);
        else if (type === 'desk') this.addDesk(p);
        else if (type === 'bubble') this.addBubble(p);
        else if (type === 'text') this.addText(p);
    }

    resize() {
        const rect = this.canvas.getBoundingClientRect();
        this.width = rect.width;
        this.height = rect.height;
        this.dpr = window.devicePixelRatio || 1;

        this.canvas.width = Math.round(this.width * this.dpr);
        this.canvas.height = Math.round(this.height * this.dpr);
        this.autoScalePeople();
        this.autoScaleFurniture();
        this.draw();
    }

    toCanvasPos(e) {
        const rect = this.canvas.getBoundingClientRect();
        return {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
        };
    }

    pushHistory() {
        this.history.push(JSON.parse(JSON.stringify(this.elements)));
        if (this.history.length > 50) this.history.shift();
    }

    undo() {
        if (!this.history.length) return;
        this.elements = this.history.pop();
        this.selectedId = null;
        this.draw();
    }

    clear() {
        if (!confirm('Clear everything?')) return;
        this.pushHistory();
        this.elements = [];
        this.selectedId = null;
        this.draw();
    }

    onPointerDown(e) {
        const p = this.toCanvasPos(e);
        const hit = this.hitTest(p);

        if (hit) {
            this.selectedId = hit.id;

            // Start dragging (for all element types)
            this.draggingId = hit.id;
            this.dragOffset = { x: p.x - hit.x, y: p.y - hit.y };
            this.dragStartPos = { x: p.x, y: p.y };
            this.dragMoved = false;
            this.pushHistory();
            this.canvas.setPointerCapture(e.pointerId);
        } else {
            this.selectedId = null;
        }
        this.draw();
    }

    onPointerMove(e) {
        if (!this.draggingId) return;
        const el = this.getById(this.draggingId);
        if (!el) return;

        const p = this.toCanvasPos(e);

        // Check if we've moved enough to count as a drag (5px threshold)
        if (this.dragStartPos && !this.dragMoved) {
            const dx = p.x - this.dragStartPos.x;
            const dy = p.y - this.dragStartPos.y;
            if (Math.abs(dx) > 5 || Math.abs(dy) > 5) {
                this.dragMoved = true;
            }
        }

        el.x = p.x - this.dragOffset.x;
        el.y = p.y - this.dragOffset.y;
        this.draw();
    }

    onPointerUp() {
        if (this.draggingId) {
            const el = this.getById(this.draggingId);
            if (el) {
                // Delete if dragged off canvas
                if (this.isOffCanvas(el)) {
                    this.elements = this.elements.filter(e => e.id !== el.id);
                    if (this.selectedId === el.id) this.selectedId = null;
                    this.debug('deleted off-canvas element', el.id);
                } else if (!this.dragMoved && (el.type === 'bubble' || el.type === 'text')) {
                    // Click without drag on bubble/text opens editor
                    this.openEditor(el);
                } else if (el.type === 'person') {
                    this.debug('pointer up, auto-facing from drag', el.id);
                    this.autoSetFacing();
                    this.updateChairPersonLinks();
                } else if (el.type === 'chair') {
                    this.updateChairPersonLinks();
                }
                this.draw();
            }
        }
        this.draggingId = null;
        this.dragStartPos = null;
        this.dragMoved = false;
    }

    isOffCanvas(el) {
        const b = this.getBounds(el);
        if (!b) return false;
        // Consider off-canvas if center is outside bounds
        const cx = b.x + b.w / 2;
        const cy = b.y + b.h / 2;
        return cx < 0 || cx > this.width || cy < 0 || cy > this.height;
    }

    onKeyDown(e) {
        const editorOpen = !document.getElementById('textEditor').classList.contains('hidden');
        if (editorOpen) return;

        const key = e.key.toLowerCase();

        // Add item shortcuts
        if (key === 'p' || key === 'm') return this.addItemAtRandom('man');
        if (key === 'w') return this.addItemAtRandom('woman');
        if (key === 'c') return this.addItemAtRandom('chair');
        if (key === 'e') return this.addItemAtRandom('desk');
        if (key === 'b') return this.addItemAtRandom('bubble');
        if (key === 't') return this.addItemAtRandom('text');

        // Undo / duplicate
        if ((e.metaKey || e.ctrlKey) && key === 'z') {
            e.preventDefault();
            return this.undo();
        }
        if ((e.metaKey || e.ctrlKey) && key === 'd') {
            e.preventDefault();
            return this.duplicateSelected();
        }

        // Delete
        if (e.key === 'Delete' || e.key === 'Backspace') {
            e.preventDefault();
            return this.deleteSelected();
        }

        // Person tweaks
        const sel = this.getSelected();
        if (!sel || sel.type !== 'person') return;

        if (key === 'h') {
            this.pushHistory();
            sel.hasHair = !sel.hasHair;
            return this.draw();
        }
        if (key === 'd') {
            this.pushHistory();
            sel.hasDress = !sel.hasDress;
            return this.draw();
        }
        if (key === 'f') {
            this.pushHistory();
            sel.facing *= -1;
            sel.autoFacing = false;
            return this.draw();
        }
        if (key === 'a') {
            this.pushHistory();
            sel.autoFacing = sel.autoFacing === false;
            if (sel.autoFacing !== false) {
                this.autoSetFacing();
            }
            return this.draw();
        }
        if (key === '+' || key === '=') {
            this.pushHistory();
            sel.scale = Math.min(3, (sel.scale || 1) + 0.1);
            sel.autoScale = false;
            return this.draw();
        }
        if (key === '-') {
            this.pushHistory();
            sel.scale = Math.max(0.6, (sel.scale || 1) - 0.1);
            sel.autoScale = false;
            return this.draw();
        }
    }

    addPerson(p, pose, gender = 'm') {
        const id = Date.now() + Math.random();
        const isFemale = gender === 'f';
        const el = {
            id,
            type: 'person',
            pose,
            x: p.x,
            y: p.y,
            scale: 1,
            facing: 1,
            autoFacing: true,
            autoScale: true,
            gender,
            hasHair: isFemale,
            hasDress: isFemale && pose === 'standing'
        };

        this.pushHistory();
        this.elements.push(el);
        this.selectedId = id;
        this.debug('add person', { id, pose, x: el.x, y: el.y });
        this.autoSetFacing();
        this.autoScalePeople();
        this.updateChairPersonLinks();
        this.draw();
    }

    addBubble(p) {
        const id = Date.now() + Math.random();
        const nearest = this.findNearestPerson(p);
        const el = {
            id,
            type: 'bubble',
            x: p.x,
            y: p.y,
            w: 240,
            h: 120,
            text: '',
            tailToId: nearest ? nearest.id : null
        };

        this.pushHistory();
        this.elements.push(el);
        this.selectedId = id;
        this.openEditor(el);
        this.draw();
    }

    addText(p) {
        const id = Date.now() + Math.random();
        const el = {
            id,
            type: 'text',
            x: p.x,
            y: p.y,
            text: '',
            size: 22
        };

        this.pushHistory();
        this.elements.push(el);
        this.selectedId = id;
        this.openEditor(el);
        this.draw();
    }

    addChair(p) {
        const id = Date.now() + Math.random();

        // Find nearest standing person anywhere to snap to
        const standing = this.elements.filter(e =>
            e.type === 'person' && e.pose === 'standing' &&
            !this.elements.some(c => c.type === 'chair' && c.seatedPersonId === e.id)
        );
        let target = null;
        let targetDist = Infinity;
        standing.forEach(person => {
            const dx = person.x - p.x;
            const dy = person.y - p.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < targetDist) {
                targetDist = dist;
                target = person;
            }
        });

        const el = {
            id,
            type: 'chair',
            x: target ? target.x : p.x,
            y: target ? target.y : p.y,
            scale: target ? target.scale : 1,
            seatedPersonId: null
        };

        this.pushHistory();
        this.elements.push(el);
        this.selectedId = id;
        this.updateChairPersonLinks();
        this.autoScaleFurniture();
        this.draw();
    }

    addDesk(p) {
        const id = Date.now() + Math.random();
        const el = {
            id,
            type: 'desk',
            x: p.x,
            y: p.y,
            scale: 1,
            autoScale: true
        };

        this.pushHistory();
        this.elements.push(el);
        this.selectedId = id;
        this.autoScaleFurniture();
        this.draw();
    }

    openEditor(el) {
        const editor = document.getElementById('textEditor');
        const input = document.getElementById('textInput');
        editor.classList.remove('hidden');
        input.value = el.text || '';
        input.focus();
        input.select();
        this.editingId = el.id;
        this.editingWasNew = !el.text;
    }

    saveText() {
        if (!this.editingId) return;
        const el = this.getById(this.editingId);
        const input = document.getElementById('textInput');
        const text = input.value.trim();

        if (!el) {
            this.cancelEdit();
            return;
        }

        if (!text) {
            if (this.editingWasNew) {
                this.elements = this.elements.filter(e => e.id !== el.id);
                if (this.selectedId === el.id) this.selectedId = null;
            }
            this.cancelEdit();
            return;
        }

        this.pushHistory();
        el.text = text;

        if (el.type === 'bubble') {
            const fontSize = 20;
            const lines = this.wrapText(text, 300, fontSize);
            const maxW = Math.max(...lines.map(l => this.measureTextWidth(l, fontSize)));
            el.w = Math.min(360, Math.max(160, maxW + 40));
            el.h = Math.max(80, lines.length * (fontSize + 6) + 30);
        }

        this.cancelEdit();
    }

    cancelEdit() {
        if (this.editingId && this.editingWasNew) {
            const el = this.getById(this.editingId);
            if (el && !el.text) {
                this.elements = this.elements.filter(e => e.id !== this.editingId);
                if (this.selectedId === this.editingId) this.selectedId = null;
            }
        }

        const editor = document.getElementById('textEditor');
        editor.classList.add('hidden');
        this.editingId = null;
        this.editingWasNew = false;
        this.draw();
    }

    deleteSelected() {
        if (!this.selectedId) return;
        const removed = this.getSelected();
        this.pushHistory();

        // If deleting a chair, stand up the seated person
        if (removed && removed.type === 'chair' && removed.seatedPersonId) {
            const person = this.getById(removed.seatedPersonId);
            if (person) {
                person.pose = 'standing';
            }
        }

        this.elements = this.elements.filter(e => e.id !== this.selectedId);
        this.selectedId = null;
        if (removed && removed.type === 'person') {
            this.autoSetFacing();
            this.autoScalePeople();
        }
        this.draw();
    }

    duplicateSelected() {
        const el = this.getSelected();
        if (!el) return;
        const copy = JSON.parse(JSON.stringify(el));
        copy.id = Date.now() + Math.random();
        copy.x += 20;
        copy.y += 20;
        this.pushHistory();
        this.elements.push(copy);
        this.selectedId = copy.id;
        if (copy.type === 'person') {
            this.debug('duplicate person', copy.id);
            this.autoSetFacing();
            this.autoScalePeople();
        }
        this.draw();
    }

    getById(id) {
        return this.elements.find(e => e.id === id);
    }

    getSelected() {
        return this.selectedId ? this.getById(this.selectedId) : null;
    }

    findNearestPerson(p) {
        let best = null;
        let bestDist = Infinity;
        this.elements.forEach(el => {
            if (el.type !== 'person') return;
            const dx = el.x - p.x;
            const dy = el.y - p.y;
            const d = dx * dx + dy * dy;
            if (d < bestDist) {
                bestDist = d;
                best = el;
            }
        });
        return best;
    }

    // Link/unlink chairs and people based on proximity
    updateChairPersonLinks() {
        const chairs = this.elements.filter(e => e.type === 'chair');
        const people = this.elements.filter(e => e.type === 'person');
        const linkDist = 50;  // Distance to link
        const unlinkDist = 80; // Distance to unlink (with hysteresis)

        chairs.forEach(chair => {
            if (chair.seatedPersonId) {
                // Check if seated person moved away
                const person = this.getById(chair.seatedPersonId);
                if (!person) {
                    // Person was deleted
                    chair.seatedPersonId = null;
                    return;
                }
                const dx = person.x - chair.x;
                const dy = person.y - chair.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist > unlinkDist) {
                    // Unlink - person stands up
                    person.pose = 'standing';
                    chair.seatedPersonId = null;
                    this.debug('person stood up from chair', person.id);
                }
            } else {
                // Find nearest standing person to sit down
                let nearest = null;
                let nearestDist = linkDist;
                people.forEach(person => {
                    if (person.pose !== 'standing') return;
                    // Check person isn't already seated in another chair
                    const alreadySeated = chairs.some(c => c.seatedPersonId === person.id);
                    if (alreadySeated) return;
                    const dx = person.x - chair.x;
                    const dy = person.y - chair.y;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist < nearestDist) {
                        nearestDist = dist;
                        nearest = person;
                    }
                });
                if (nearest) {
                    // Link - person sits down, chair snaps to person
                    nearest.pose = 'sitting';
                    chair.seatedPersonId = nearest.id;
                    chair.x = nearest.x;
                    chair.y = nearest.y;
                    chair.scale = nearest.scale;
                    this.debug('person sat in chair', nearest.id);
                }
            }
        });
    }

    // Auto-default: people face nearest person (for 2 people, they face each other).
    autoSetFacing() {
        const people = this.elements.filter(e => e.type === 'person');
        if (people.length < 2) {
            this.debug('autoFacing skipped (need 2+ people)');
            return;
        }

        this.debug('autoFacing run', people.map(p => p.id));
        people.forEach(p => {
            if (p.autoFacing === false) return;
            if (p.pose === 'sitting') return; // Seated people keep their facing
            let nearest = null;
            let bestDist = Infinity;
            people.forEach(o => {
                if (o === p) return;
                const dx = o.x - p.x;
                const dy = o.y - p.y;
                const d = dx * dx + dy * dy;
                if (d < bestDist) {
                    bestDist = d;
                    nearest = o;
                }
            });
            if (nearest) {
                p.facing = nearest.x >= p.x ? 1 : -1;
                this.debug('autoFacing set', { id: p.id, facing: p.facing, nearest: nearest.id });
            }
        });
    }

    // Auto-default: scale people up to fill the panel.
    // Applies only to people with autoScale !== false.
    autoScalePeople() {
        const people = this.elements.filter(e => e.type === 'person');
        if (!people.length) return;

        const autoPeople = people.filter(p => p.autoScale !== false);
        if (!autoPeople.length) return;

        const n = people.length;
        const baseW = 90;
        const baseH = 130;

        const targetHeightFrac = n === 1 ? 0.75 : n === 2 ? 0.6 : n === 3 ? 0.5 : 0.42;
        const scaleByHeight = (this.height * targetHeightFrac) / baseH;

        const marginX = this.width * 0.1;
        const availW = Math.max(1, this.width - marginX * 2);
        const perW = availW / n;
        const scaleByWidth = (perW * 0.9) / baseW;

        let scale = Math.min(scaleByHeight, scaleByWidth);
        scale = Math.max(0.7, Math.min(3, scale));

        autoPeople.forEach(p => {
            p.scale = scale;
        });

        // Sync chair scales with their seated persons
        this.elements.forEach(el => {
            if (el.type === 'chair' && el.seatedPersonId) {
                const person = this.getById(el.seatedPersonId);
                if (person) {
                    el.scale = person.scale;
                }
            }
        });
    }

    autoScaleFurniture() {
        // Scale standalone furniture (desks, unlinked chairs) based on average person scale
        const people = this.elements.filter(e => e.type === 'person');
        let avgScale;

        if (people.length > 0) {
            avgScale = people.reduce((sum, p) => sum + (p.scale || 1), 0) / people.length;
        } else {
            // No people - calculate scale as if there was 1 person
            const baseH = 130;
            const targetHeightFrac = 0.75;
            avgScale = (this.height * targetHeightFrac) / baseH;
            avgScale = Math.max(0.7, Math.min(3, avgScale));
        }

        this.elements.forEach(el => {
            if (el.type === 'desk' && el.autoScale !== false) {
                el.scale = avgScale;
            }
            if (el.type === 'chair' && !el.seatedPersonId) {
                el.scale = avgScale;
            }
        });
    }

    hitTest(p) {
        for (let i = this.elements.length - 1; i >= 0; i--) {
            const el = this.elements[i];
            const b = this.getBounds(el);
            if (!b) continue;
            if (p.x >= b.x && p.x <= b.x + b.w && p.y >= b.y && p.y <= b.y + b.h) {
                return el;
            }
        }
        return null;
    }

    getBounds(el) {
        const s = el.scale || 1;
        if (el.type === 'person') {
            const w = 90 * s;
            const h = 130 * s;
            return { x: el.x - w / 2, y: el.y - 22 * s, w, h };
        }
        if (el.type === 'chair') {
            // Chair bounds: seat at y+48, legs down 28px (scaled)
            return { x: el.x - 22 * s, y: el.y + 48 * s, w: 40 * s, h: 28 * s };
        }
        if (el.type === 'desk') {
            // Desk bounds: 120 wide, 50 tall (scaled)
            return { x: el.x - 60 * s, y: el.y, w: 120 * s, h: 50 * s };
        }
        if (el.type === 'bubble') {
            return { x: el.x - el.w / 2, y: el.y - el.h / 2, w: el.w, h: el.h + 24 };
        }
        if (el.type === 'text') {
            const fontSize = el.size || 22;
            const lines = (el.text || '').split('\n');
            const maxW = Math.max(...lines.map(l => this.measureTextWidth(l, fontSize)));
            const h = lines.length * (fontSize + 6);
            return { x: el.x, y: el.y, w: maxW, h };
        }
        return null;
    }

    measureTextWidth(text, fontSize) {
        const ctx = this.ctx;
        ctx.save();
        ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
        ctx.font = `700 ${fontSize}px system-ui, sans-serif`;
        const w = ctx.measureText(text).width;
        ctx.restore();
        return w;
    }

    wrapText(text, maxWidth, fontSize) {
        const ctx = this.ctx;
        ctx.save();
        ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
        ctx.font = `700 ${fontSize}px system-ui, sans-serif`;

        const words = text.replace(/\n/g, ' \n ').split(/\s+/);
        const lines = [];
        let line = '';

        words.forEach(word => {
            if (word === '\n') {
                if (line.trim()) lines.push(line.trim());
                line = '';
                return;
            }
            const test = line ? `${line} ${word}` : word;
            if (ctx.measureText(test).width > maxWidth && line) {
                lines.push(line);
                line = word;
            } else {
                line = test;
            }
        });

        if (line.trim()) lines.push(line.trim());
        ctx.restore();
        return lines.length ? lines : [''];
    }

    draw() {
        const ctx = this.ctx;
        ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
        ctx.clearRect(0, 0, this.width, this.height);

        // Background
        ctx.fillStyle = '#fff';
        ctx.fillRect(0, 0, this.width, this.height);

        // Frame
        const frameW = 6;
        ctx.lineWidth = frameW;
        ctx.strokeStyle = '#000';
        ctx.strokeRect(frameW / 2, frameW / 2, this.width - frameW, this.height - frameW);

        // Elements
        this.elements.forEach(el => {
            if (el.type === 'person') this.drawPerson(el);
            if (el.type === 'chair') this.drawChair(el);
            if (el.type === 'desk') this.drawDesk(el);
            if (el.type === 'bubble') this.drawBubble(el);
            if (el.type === 'text') this.drawText(el);

            if (el.id === this.selectedId) {
                this.drawSelection(el);
            }
        });
    }

    drawPerson(el) {
        const ctx = this.ctx;
        const s = el.scale || 1;
        const facing = el.facing || 1;

        ctx.save();
        ctx.translate(el.x, el.y);
        ctx.scale(facing * s, s);

        ctx.strokeStyle = '#000';
        ctx.fillStyle = '#000';
        ctx.lineWidth = 4;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        const headR = 18;

        // Head
        ctx.beginPath();
        ctx.arc(0, 0, headR, 0, Math.PI * 2);
        ctx.stroke();

        // Hair
        if (el.hasHair) {
            ctx.beginPath();
            ctx.moveTo(-headR + 2, -headR + 4);
            ctx.quadraticCurveTo(-headR - 8, 0, -headR + 2, headR + 6);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(-headR / 2, -headR + 2);
            ctx.quadraticCurveTo(0, -headR - 6, headR / 2, -headR + 2);
            ctx.stroke();
        }

        // Eye (single eye gives profile/facing direction)
        ctx.beginPath();
        ctx.arc(7, -4, 2, 0, Math.PI * 2);
        ctx.fill();

        // Smile (slightly offset to match profile)
        ctx.beginPath();
        ctx.arc(4, 2, 7, 0.1 * Math.PI, 0.9 * Math.PI);
        ctx.stroke();

        if (el.pose === 'sitting') {
            this.drawSittingBody(ctx, headR);
        } else {
            this.drawStandingBody(ctx, headR, el.hasDress);
        }

        ctx.restore();
    }

    drawStandingBody(ctx, headR, hasDress) {
        const bodyStartY = headR;
        const bodyEndY = headR + 42;

        // Body
        ctx.beginPath();
        ctx.moveTo(0, bodyStartY);
        ctx.lineTo(0, bodyEndY);
        ctx.stroke();

        // Arms
        ctx.beginPath();
        ctx.moveTo(-18, headR + 12);
        ctx.lineTo(0, headR + 18);
        ctx.lineTo(18, headR + 12);
        ctx.stroke();

        // Dress
        if (hasDress) {
            ctx.beginPath();
            ctx.moveTo(-14, bodyEndY);
            ctx.lineTo(14, bodyEndY);
            ctx.lineTo(0, headR + 18);
            ctx.closePath();
            ctx.stroke();
        }

        // Legs
        ctx.beginPath();
        ctx.moveTo(0, bodyEndY);
        ctx.lineTo(-12, bodyEndY + 28);
        ctx.moveTo(0, bodyEndY);
        ctx.lineTo(12, bodyEndY + 28);
        ctx.stroke();
    }

    drawSittingBody(ctx, headR) {
        const bodyStartY = headR;
        const bodyEndY = headR + 30;

        // Body (slight lean back)
        ctx.beginPath();
        ctx.moveTo(0, bodyStartY);
        ctx.lineTo(-2, bodyEndY);
        ctx.stroke();

        // Arms forward (resting on lap or desk)
        ctx.beginPath();
        ctx.moveTo(-8, headR + 14);
        ctx.lineTo(0, headR + 20);
        ctx.lineTo(14, headR + 22);
        ctx.stroke();

        // Seated legs - profile view (thigh forward, lower leg down)
        ctx.beginPath();
        // Thigh extends forward
        ctx.moveTo(-2, bodyEndY);
        ctx.lineTo(18, bodyEndY + 6);
        // Lower leg hangs down
        ctx.lineTo(18, bodyEndY + 28);
        ctx.stroke();
    }

    drawChair(el) {
        const ctx = this.ctx;
        const s = el.scale || 1;
        ctx.save();
        ctx.translate(el.x, el.y);
        ctx.scale(s, s);

        ctx.strokeStyle = '#000';
        ctx.lineWidth = 4;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        // Get facing from seated person, or null if standalone
        let facing = null;
        if (el.seatedPersonId) {
            const person = this.getById(el.seatedPersonId);
            if (person) facing = person.facing;
        }

        const seatY = 48; // matches bodyEndY + 6 from sitting body

        if (facing === null) {
            // Standalone chair - front view with high back
            ctx.beginPath();
            // Seat
            ctx.moveTo(-22, seatY);
            ctx.lineTo(22, seatY);
            // Back legs
            ctx.moveTo(-18, seatY);
            ctx.lineTo(-18, seatY + 28);
            ctx.moveTo(18, seatY);
            ctx.lineTo(18, seatY + 28);
            // High backrest
            ctx.moveTo(-22, seatY);
            ctx.lineTo(-22, seatY - 30);
            ctx.lineTo(22, seatY - 30);
            ctx.lineTo(22, seatY);
            ctx.stroke();
        } else {
            // Side view - backrest on opposite side of facing
            const backX = facing > 0 ? -22 : 18;
            const frontX = facing > 0 ? 18 : -22;
            ctx.beginPath();
            // Seat
            ctx.moveTo(-22, seatY);
            ctx.lineTo(18, seatY);
            // Front leg
            ctx.moveTo(frontX, seatY);
            ctx.lineTo(frontX, seatY + 28);
            // Back leg
            ctx.moveTo(backX, seatY);
            ctx.lineTo(backX, seatY + 28);
            // Backrest (on back side)
            ctx.moveTo(backX, seatY);
            ctx.lineTo(backX, seatY - 30);
            ctx.stroke();
        }

        ctx.restore();
    }

    drawDesk(el) {
        const ctx = this.ctx;
        const s = el.scale || 1;
        ctx.save();
        ctx.translate(el.x, el.y);
        ctx.scale(s, s);

        ctx.strokeStyle = '#000';
        ctx.lineWidth = 4;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        // Simple desk - surface and legs
        const deskW = 120;
        const deskH = 50;

        // Surface
        ctx.beginPath();
        ctx.moveTo(-deskW / 2, 0);
        ctx.lineTo(deskW / 2, 0);
        ctx.stroke();

        // Legs
        ctx.beginPath();
        ctx.moveTo(-deskW / 2 + 15, 0);
        ctx.lineTo(-deskW / 2 + 15, deskH);
        ctx.moveTo(deskW / 2 - 15, 0);
        ctx.lineTo(deskW / 2 - 15, deskH);
        ctx.stroke();

        ctx.restore();
    }

    drawBubble(el) {
        const ctx = this.ctx;
        const x = el.x - el.w / 2;
        const y = el.y - el.h / 2;
        const r = 18;

        ctx.save();
        ctx.lineWidth = 4;
        ctx.strokeStyle = '#000';
        ctx.fillStyle = '#fff';
        ctx.lineJoin = 'round';

        // Rounded rect
        ctx.beginPath();
        ctx.moveTo(x + r, y);
        ctx.lineTo(x + el.w - r, y);
        ctx.quadraticCurveTo(x + el.w, y, x + el.w, y + r);
        ctx.lineTo(x + el.w, y + el.h - r);
        ctx.quadraticCurveTo(x + el.w, y + el.h, x + el.w - r, y + el.h);
        ctx.lineTo(x + r, y + el.h);
        ctx.quadraticCurveTo(x, y + el.h, x, y + el.h - r);
        ctx.lineTo(x, y + r);
        ctx.quadraticCurveTo(x, y, x + r, y);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();

        // Tail
        const target = el.tailToId ? this.getById(el.tailToId) : null;
        const tx = target ? target.x : el.x;
        const ty = target ? target.y + 40 * (target.scale || 1) : el.y + el.h / 2 + 40;
        const tailBaseX = el.x + Math.max(-el.w / 4, Math.min(el.w / 4, tx - el.x));
        const tailInset = 2; // overlap a bit to avoid seam
        const tailBaseY = y + el.h - tailInset;

        // Fill tail
        ctx.beginPath();
        ctx.moveTo(tailBaseX - 10, tailBaseY);
        ctx.lineTo(tailBaseX, tailBaseY + 20);
        ctx.lineTo(tailBaseX + 10, tailBaseY);
        ctx.closePath();
        ctx.fill();

        // Stroke only the sides (no base line) to prevent double-stroke seam.
        ctx.beginPath();
        ctx.moveTo(tailBaseX - 10, tailBaseY);
        ctx.lineTo(tailBaseX, tailBaseY + 20);
        ctx.lineTo(tailBaseX + 10, tailBaseY);
        ctx.stroke();

        // Text
        if (el.text) {
            const fontSize = 20;
            ctx.font = `700 ${fontSize}px system-ui, sans-serif`;
            ctx.fillStyle = '#000';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';

            const lines = this.wrapText(el.text, el.w - 28, fontSize);
            const lineH = fontSize + 6;
            const startY = el.y - ((lines.length - 1) * lineH) / 2;
            lines.forEach((line, i) => {
                ctx.fillText(line, el.x, startY + i * lineH);
            });
        }

        ctx.restore();
    }

    drawText(el) {
        const ctx = this.ctx;
        const size = el.size || 22;
        ctx.save();
        ctx.font = `700 ${size}px system-ui, sans-serif`;
        ctx.fillStyle = '#000';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';

        const lines = (el.text || '').split('\n');
        const lineH = size + 6;
        lines.forEach((line, i) => {
            ctx.fillText(line, el.x, el.y + i * lineH);
        });
        ctx.restore();
    }

    drawSelection(el) {
        const b = this.getBounds(el);
        if (!b) return;
        const ctx = this.ctx;
        ctx.save();
        ctx.setLineDash([6, 4]);
        ctx.strokeStyle = '#111';
        ctx.lineWidth = 1.5;
        ctx.strokeRect(b.x, b.y, b.w, b.h);
        ctx.restore();
    }

    save() {
        const outScale = 2;
        const exportCanvas = document.createElement('canvas');
        exportCanvas.width = this.width * outScale;
        exportCanvas.height = this.height * outScale;
        const ectx = exportCanvas.getContext('2d');

        ectx.setTransform(outScale, 0, 0, outScale, 0, 0);

        // Background + frame
        ectx.fillStyle = '#fff';
        ectx.fillRect(0, 0, this.width, this.height);
        ectx.lineWidth = 6;
        ectx.strokeStyle = '#000';
        ectx.strokeRect(3, 3, this.width - 6, this.height - 6);

        // Draw elements using same routines
        const prevCtx = this.ctx;
        this.ctx = ectx;
        this.elements.forEach(el => {
            if (el.type === 'person') this.drawPerson(el);
            if (el.type === 'chair') this.drawChair(el);
            if (el.type === 'desk') this.drawDesk(el);
            if (el.type === 'bubble') this.drawBubble(el);
            if (el.type === 'text') this.drawText(el);
        });
        this.ctx = prevCtx;

        exportCanvas.toBlob(blob => {
            if (!blob) return;
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `comic-${Date.now()}.png`;
            a.click();
            URL.revokeObjectURL(url);
        });
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.comic = new MinimalComicMaker();
});
