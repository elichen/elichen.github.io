// Minimal Stick Figure Comic Maker

class MinimalComicMaker {
    constructor() {
        this.canvas = document.getElementById('comicCanvas');
        this.ctx = this.canvas.getContext('2d');

        this.elements = [];
        this.history = [];
        this.tool = 'select';
        this.selectedId = null;

        this.draggingId = null;
        this.dragOffset = { x: 0, y: 0 };

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
        // Toolbar
        document.querySelectorAll('.tool-btn').forEach(btn => {
            btn.addEventListener('click', () => this.setTool(btn.dataset.tool));
        });
        document.getElementById('undoBtn').addEventListener('click', () => this.undo());
        document.getElementById('clearBtn').addEventListener('click', () => this.clear());
        document.getElementById('saveBtn').addEventListener('click', () => this.save());

        // Canvas pointer events
        this.canvas.addEventListener('pointerdown', e => this.onPointerDown(e));
        this.canvas.addEventListener('pointermove', e => this.onPointerMove(e));
        this.canvas.addEventListener('pointerup', () => this.onPointerUp());
        this.canvas.addEventListener('pointerleave', () => this.onPointerUp());
        this.canvas.addEventListener('dblclick', e => this.onDoubleClick(e));

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

    setTool(tool) {
        this.tool = tool;
        document.querySelectorAll('.tool-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tool === tool);
        });
    }

    resize() {
        const rect = this.canvas.getBoundingClientRect();
        this.width = rect.width;
        this.height = rect.height;
        this.dpr = window.devicePixelRatio || 1;

        this.canvas.width = Math.round(this.width * this.dpr);
        this.canvas.height = Math.round(this.height * this.dpr);
        this.autoScalePeople();
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

        if (this.tool === 'man') {
            this.addPerson(p, 'standing', 'm');
            return;
        }
        if (this.tool === 'woman') {
            this.addPerson(p, 'standing', 'f');
            return;
        }
        if (this.tool === 'seated-man') {
            this.addPerson(p, 'sitting', 'm');
            return;
        }
        if (this.tool === 'seated-woman') {
            this.addPerson(p, 'sitting', 'f');
            return;
        }
        if (this.tool === 'bubble') {
            this.addBubble(p);
            return;
        }
        if (this.tool === 'text') {
            this.addText(p);
            return;
        }

        const hit = this.hitTest(p);
        if (hit) {
            this.selectedId = hit.id;
            this.draggingId = hit.id;
            this.dragOffset = { x: p.x - hit.x, y: p.y - hit.y };
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
        el.x = p.x - this.dragOffset.x;
        el.y = p.y - this.dragOffset.y;
        this.draw();
    }

    onPointerUp() {
        if (this.draggingId) {
            const el = this.getById(this.draggingId);
            if (el && el.type === 'person') {
                this.debug('pointer up, auto-facing from drag', el.id);
                this.autoSetFacing();
                this.draw();
            }
        }
        this.draggingId = null;
    }

    onDoubleClick(e) {
        const p = this.toCanvasPos(e);
        const hit = this.hitTest(p);
        if (hit && (hit.type === 'bubble' || hit.type === 'text')) {
            this.selectedId = hit.id;
            this.openEditor(hit);
            this.draw();
        }
    }

    onKeyDown(e) {
        const editorOpen = !document.getElementById('textEditor').classList.contains('hidden');
        if (editorOpen) return;

        const key = e.key.toLowerCase();

        // Tool shortcuts
        if (key === 'v' || e.key === 'Escape') return this.setTool('select');
        if (key === 'p' || key === 'm') return this.setTool('man');
        if (key === 'w') return this.setTool('woman');
        if (key === 's') return this.setTool(e.shiftKey ? 'seated-woman' : 'seated-man');
        if (key === 'b') return this.setTool('bubble');
        if (key === 't') return this.setTool('text');

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
            this.drawDesk(ctx);
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

        // Body
        ctx.beginPath();
        ctx.moveTo(0, bodyStartY);
        ctx.lineTo(0, bodyEndY);
        ctx.stroke();

        // Arms forward
        ctx.beginPath();
        ctx.moveTo(-14, headR + 12);
        ctx.lineTo(0, headR + 18);
        ctx.lineTo(16, headR + 16);
        ctx.stroke();

        // Seated legs
        ctx.beginPath();
        ctx.moveTo(0, bodyEndY);
        ctx.lineTo(-14, bodyEndY + 8);
        ctx.lineTo(-14, bodyEndY + 26);
        ctx.moveTo(0, bodyEndY);
        ctx.lineTo(10, bodyEndY + 10);
        ctx.lineTo(10, bodyEndY + 26);
        ctx.stroke();

        // Chair
        ctx.beginPath();
        ctx.moveTo(-22, bodyEndY + 6);
        ctx.lineTo(18, bodyEndY + 6);
        ctx.moveTo(-18, bodyEndY + 6);
        ctx.lineTo(-18, bodyEndY + 34);
        ctx.moveTo(14, bodyEndY + 6);
        ctx.lineTo(14, bodyEndY + 34);
        ctx.stroke();
    }

    drawDesk(ctx) {
        // Desk surface
        const deskY = 62;
        const deskX = 20;
        const deskW = 140;

        ctx.beginPath();
        ctx.moveTo(deskX, deskY);
        ctx.lineTo(deskX + deskW, deskY);
        ctx.stroke();

        // Desk legs
        ctx.beginPath();
        ctx.moveTo(deskX + 18, deskY);
        ctx.lineTo(deskX + 18, deskY + 48);
        ctx.moveTo(deskX + deskW - 18, deskY);
        ctx.lineTo(deskX + deskW - 18, deskY + 48);
        ctx.stroke();

        // Laptop
        ctx.beginPath();
        ctx.moveTo(deskX + 68, deskY - 2);
        ctx.lineTo(deskX + 96, deskY - 20);
        ctx.lineTo(deskX + 126, deskY - 20);
        ctx.lineTo(deskX + 98, deskY - 2);
        ctx.closePath();
        ctx.stroke();
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
