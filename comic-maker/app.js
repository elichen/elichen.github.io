// Comic Maker - Simple & Intuitive
class ComicMaker {
    constructor() {
        this.panels = [];
        this.currentPanel = 0;
        this.elements = new Map();
        this.history = [];
        this.bubbleMode = null;
        this.isDragging = false;
        this.dragTarget = null;
        this.dragOffset = { x: 0, y: 0 };
        this.editingBubble = null;

        this.init();
    }

    init() {
        this.setupPanels();
        this.setupPreviews();
        this.setupEvents();
    }

    // Setup panels from DOM
    setupPanels() {
        const container = document.getElementById('comic');
        container.querySelectorAll('.panel').forEach((el, i) => {
            const canvas = el.querySelector('canvas');
            const rect = el.getBoundingClientRect();
            canvas.width = rect.width;
            canvas.height = rect.height;

            this.panels.push({
                el,
                canvas,
                ctx: canvas.getContext('2d'),
                index: i
            });
            this.elements.set(i, []);

            // Panel events
            canvas.addEventListener('click', e => this.onPanelClick(e, i));
            canvas.addEventListener('mousedown', e => this.onMouseDown(e, i));
            canvas.addEventListener('mousemove', e => this.onMouseMove(e, i));
            canvas.addEventListener('mouseup', () => this.onMouseUp());
            canvas.addEventListener('mouseleave', () => this.onMouseUp());
            canvas.addEventListener('dblclick', e => this.onDoubleClick(e, i));

            // Touch support
            canvas.addEventListener('touchstart', e => this.onTouchStart(e, i));
            canvas.addEventListener('touchmove', e => this.onTouchMove(e, i));
            canvas.addEventListener('touchend', () => this.onMouseUp());
        });

        this.selectPanel(0);
    }

    // Draw character previews in sidebar
    setupPreviews() {
        document.querySelectorAll('.char-btn').forEach(btn => {
            const canvas = btn.querySelector('.char-preview');
            const ctx = canvas.getContext('2d');
            const pose = btn.dataset.pose;
            const figure = new StickFigure(ctx, 25, 35, 0.35);
            figure.draw(pose);
        });
    }

    // Setup all event listeners
    setupEvents() {
        // Layout buttons
        document.querySelectorAll('.layout-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.layout-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.changeLayout(btn.dataset.layout);
            });
        });

        // Character buttons
        document.querySelectorAll('.char-btn').forEach(btn => {
            btn.addEventListener('click', () => this.addCharacter(btn.dataset.pose));
        });

        // Bubble buttons
        document.querySelectorAll('.bubble-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const type = btn.dataset.type;
                if (this.bubbleMode === type) {
                    this.setBubbleMode(null);
                } else {
                    this.setBubbleMode(type);
                }
            });
        });

        // Header actions
        document.getElementById('undoBtn').addEventListener('click', () => this.undo());
        document.getElementById('clearBtn').addEventListener('click', () => this.clear());
        document.getElementById('saveBtn').addEventListener('click', () => this.save());

        // Text editor
        document.getElementById('saveEdit').addEventListener('click', () => this.saveText());
        document.getElementById('cancelEdit').addEventListener('click', () => this.cancelEdit());
        document.getElementById('textInput').addEventListener('keydown', e => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.saveText();
            }
            if (e.key === 'Escape') {
                this.cancelEdit();
            }
        });

        // Click outside to close editor
        document.addEventListener('click', e => {
            const editor = document.getElementById('textEditor');
            if (!editor.classList.contains('hidden') &&
                !editor.contains(e.target) &&
                !e.target.closest('.panel')) {
                this.cancelEdit();
            }
        });

        // Window resize
        window.addEventListener('resize', () => this.resizePanels());
    }

    // Panel selection
    selectPanel(index) {
        this.panels.forEach(p => p.el.classList.remove('active'));
        if (this.panels[index]) {
            this.panels[index].el.classList.add('active');
            this.currentPanel = index;
        }
    }

    // Change layout
    changeLayout(layout) {
        const container = document.getElementById('comic');
        container.className = `comic layout-${layout}`;
        container.innerHTML = '';
        this.panels = [];
        this.elements.clear();

        const count = layout === '1' ? 1 : layout === '4' ? 4 : 2;

        for (let i = 0; i < count; i++) {
            const panel = document.createElement('div');
            panel.className = 'panel';
            panel.dataset.index = i;
            panel.innerHTML = '<canvas></canvas>';
            container.appendChild(panel);
        }

        this.setupPanels();
        this.setBubbleMode(this.bubbleMode);
    }

    // Resize panels on window resize
    resizePanels() {
        this.panels.forEach((p, i) => {
            const rect = p.el.getBoundingClientRect();
            p.canvas.width = rect.width;
            p.canvas.height = rect.height;
            this.redraw(i);
        });
    }

    // Add character to current panel
    addCharacter(pose) {
        const panel = this.panels[this.currentPanel];
        if (!panel) return;

        const elements = this.elements.get(this.currentPanel);
        const offsetX = elements.filter(e => e.type === 'character').length * 30;

        const char = {
            type: 'character',
            pose,
            x: panel.canvas.width / 2 + (Math.random() - 0.5) * 60 + offsetX,
            y: panel.canvas.height / 2 + (Math.random() - 0.5) * 30,
            id: Date.now()
        };

        this.saveHistory();
        elements.push(char);
        this.redraw(this.currentPanel);
    }

    // Set bubble placement mode
    setBubbleMode(type) {
        this.bubbleMode = type;

        document.querySelectorAll('.bubble-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.type === type);
        });

        this.panels.forEach(p => {
            p.el.classList.toggle('bubble-mode', !!type);
        });
    }

    // Add bubble at position
    addBubble(x, y, panelIndex, type) {
        const bubble = {
            type: 'bubble',
            bubbleType: type,
            text: '',
            x,
            y,
            width: 120,
            height: 50,
            id: Date.now()
        };

        this.saveHistory();
        this.elements.get(panelIndex).push(bubble);
        this.redraw(panelIndex);
        this.editBubble(bubble, panelIndex);
    }

    // Panel click handler
    onPanelClick(e, panelIndex) {
        this.selectPanel(panelIndex);

        if (this.bubbleMode) {
            const rect = e.target.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            this.addBubble(x, y, panelIndex, this.bubbleMode);
            this.setBubbleMode(null);
        }
    }

    // Mouse down - start drag
    onMouseDown(e, panelIndex) {
        const rect = e.target.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        const element = this.findElementAt(x, y, panelIndex);
        if (element) {
            this.isDragging = true;
            this.dragTarget = { element, panelIndex };
            this.dragOffset = { x: x - element.x, y: y - element.y };
            e.target.style.cursor = 'grabbing';
        }
    }

    // Mouse move - drag
    onMouseMove(e, panelIndex) {
        const rect = e.target.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        if (this.isDragging && this.dragTarget) {
            this.dragTarget.element.x = x - this.dragOffset.x;
            this.dragTarget.element.y = y - this.dragOffset.y;
            this.redraw(this.dragTarget.panelIndex);
        } else if (!this.bubbleMode) {
            const element = this.findElementAt(x, y, panelIndex);
            e.target.style.cursor = element ? 'grab' : 'default';
        }
    }

    // Mouse up - end drag
    onMouseUp() {
        if (this.isDragging) {
            this.saveHistory();
        }
        this.isDragging = false;
        this.dragTarget = null;
        this.panels.forEach(p => {
            p.canvas.style.cursor = this.bubbleMode ? 'crosshair' : 'default';
        });
    }

    // Touch support
    onTouchStart(e, panelIndex) {
        const touch = e.touches[0];
        const rect = e.target.getBoundingClientRect();
        const x = touch.clientX - rect.left;
        const y = touch.clientY - rect.top;

        const element = this.findElementAt(x, y, panelIndex);
        if (element) {
            e.preventDefault();
            this.isDragging = true;
            this.dragTarget = { element, panelIndex };
            this.dragOffset = { x: x - element.x, y: y - element.y };
        }
    }

    onTouchMove(e, panelIndex) {
        if (!this.isDragging || !this.dragTarget) return;
        e.preventDefault();

        const touch = e.touches[0];
        const rect = e.target.getBoundingClientRect();
        const x = touch.clientX - rect.left;
        const y = touch.clientY - rect.top;

        this.dragTarget.element.x = x - this.dragOffset.x;
        this.dragTarget.element.y = y - this.dragOffset.y;
        this.redraw(this.dragTarget.panelIndex);
    }

    // Double click to edit bubble
    onDoubleClick(e, panelIndex) {
        const rect = e.target.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        const element = this.findElementAt(x, y, panelIndex);
        if (element && element.type === 'bubble') {
            this.editBubble(element, panelIndex);
        }
    }

    // Find element at coordinates
    findElementAt(x, y, panelIndex) {
        const elements = this.elements.get(panelIndex);
        for (let i = elements.length - 1; i >= 0; i--) {
            const el = elements[i];
            if (el.type === 'character') {
                if (x >= el.x - 40 && x <= el.x + 40 &&
                    y >= el.y - 20 && y <= el.y + 100) {
                    return el;
                }
            } else if (el.type === 'bubble') {
                if (x >= el.x - el.width/2 && x <= el.x + el.width/2 &&
                    y >= el.y - el.height/2 && y <= el.y + el.height/2) {
                    return el;
                }
            }
        }
        return null;
    }

    // Edit bubble text
    editBubble(bubble, panelIndex) {
        const editor = document.getElementById('textEditor');
        const input = document.getElementById('textInput');
        const panel = this.panels[panelIndex];
        const rect = panel.canvas.getBoundingClientRect();

        editor.style.left = (rect.left + bubble.x - 110) + 'px';
        editor.style.top = (rect.top + bubble.y - 60) + 'px';
        editor.classList.remove('hidden');

        input.value = bubble.text;
        input.focus();
        input.select();

        this.editingBubble = { bubble, panelIndex };
    }

    // Save edited text
    saveText() {
        if (!this.editingBubble) return;

        const input = document.getElementById('textInput');
        const text = input.value.trim();

        if (text) {
            this.saveHistory();
            this.editingBubble.bubble.text = text;
            this.editingBubble.bubble.width = Math.max(100, text.length * 7 + 30);
            this.redraw(this.editingBubble.panelIndex);
        } else {
            // Remove empty bubble
            const elements = this.elements.get(this.editingBubble.panelIndex);
            const idx = elements.indexOf(this.editingBubble.bubble);
            if (idx > -1) {
                this.saveHistory();
                elements.splice(idx, 1);
                this.redraw(this.editingBubble.panelIndex);
            }
        }

        this.cancelEdit();
    }

    // Cancel editing
    cancelEdit() {
        document.getElementById('textEditor').classList.add('hidden');

        // Remove bubble if it was new and empty
        if (this.editingBubble && !this.editingBubble.bubble.text) {
            const elements = this.elements.get(this.editingBubble.panelIndex);
            const idx = elements.indexOf(this.editingBubble.bubble);
            if (idx > -1) {
                elements.splice(idx, 1);
                this.redraw(this.editingBubble.panelIndex);
            }
        }

        this.editingBubble = null;
    }

    // Redraw panel
    redraw(panelIndex) {
        const panel = this.panels[panelIndex];
        const ctx = panel.ctx;
        const elements = this.elements.get(panelIndex);

        ctx.clearRect(0, 0, panel.canvas.width, panel.canvas.height);

        elements.forEach(el => {
            if (el.type === 'character') {
                const figure = new StickFigure(ctx, el.x, el.y, 1);
                figure.draw(el.pose);
            } else if (el.type === 'bubble') {
                this.drawBubble(ctx, el);
            }
        });
    }

    // Draw speech bubble
    drawBubble(ctx, bubble) {
        ctx.save();
        ctx.fillStyle = '#fff';
        ctx.strokeStyle = '#000';
        ctx.lineWidth = 2;

        const x = bubble.x - bubble.width / 2;
        const y = bubble.y - bubble.height / 2;

        if (bubble.bubbleType === 'speech') {
            // Draw bubble and tail as one continuous shape
            const r = 12; // corner radius
            const tailW = 10;
            const tailH = 14;
            const tailX = bubble.x; // tail center

            ctx.beginPath();
            // Start at top-left after corner
            ctx.moveTo(x + r, y);
            // Top edge
            ctx.lineTo(x + bubble.width - r, y);
            // Top-right corner
            ctx.quadraticCurveTo(x + bubble.width, y, x + bubble.width, y + r);
            // Right edge
            ctx.lineTo(x + bubble.width, y + bubble.height - r);
            // Bottom-right corner
            ctx.quadraticCurveTo(x + bubble.width, y + bubble.height, x + bubble.width - r, y + bubble.height);
            // Bottom edge to tail
            ctx.lineTo(tailX + tailW, y + bubble.height);
            // Tail
            ctx.lineTo(tailX, y + bubble.height + tailH);
            ctx.lineTo(tailX - tailW, y + bubble.height);
            // Bottom edge from tail
            ctx.lineTo(x + r, y + bubble.height);
            // Bottom-left corner
            ctx.quadraticCurveTo(x, y + bubble.height, x, y + bubble.height - r);
            // Left edge
            ctx.lineTo(x, y + r);
            // Top-left corner
            ctx.quadraticCurveTo(x, y, x + r, y);
            ctx.closePath();

            ctx.fill();
            ctx.stroke();
        } else {
            // Thought bubble - smooth cloud using bezier curves
            const cx = bubble.x;
            const cy = bubble.y;
            const rx = bubble.width * 0.48;
            const ry = bubble.height * 0.42;
            const bumps = 8;
            const bumpDepth = 0.18;

            ctx.beginPath();
            for (let i = 0; i < bumps; i++) {
                const angle1 = (Math.PI * 2 / bumps) * i;
                const angle2 = (Math.PI * 2 / bumps) * (i + 1);
                const midAngle = (angle1 + angle2) / 2;

                const x1 = cx + Math.cos(angle1) * rx;
                const y1 = cy + Math.sin(angle1) * ry;
                const x2 = cx + Math.cos(angle2) * rx;
                const y2 = cy + Math.sin(angle2) * ry;

                // Control point bulges outward for cloud effect
                const cpx = cx + Math.cos(midAngle) * rx * (1 + bumpDepth);
                const cpy = cy + Math.sin(midAngle) * ry * (1 + bumpDepth);

                if (i === 0) ctx.moveTo(x1, y1);
                ctx.quadraticCurveTo(cpx, cpy, x2, y2);
            }
            ctx.closePath();
            ctx.fill();
            ctx.stroke();

            // Thought trail dots
            ctx.beginPath();
            ctx.arc(cx, y + bubble.height + 10, 5, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();

            ctx.beginPath();
            ctx.arc(cx, y + bubble.height + 22, 3, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
        }

        // Text
        if (bubble.text) {
            ctx.fillStyle = '#000';
            ctx.font = '14px sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';

            const words = bubble.text.split(' ');
            const lines = [];
            let line = '';

            words.forEach(word => {
                const test = line + word + ' ';
                if (ctx.measureText(test).width > bubble.width - 16) {
                    lines.push(line);
                    line = word + ' ';
                } else {
                    line = test;
                }
            });
            lines.push(line);

            const lineHeight = 16;
            const startY = bubble.y - (lines.length - 1) * lineHeight / 2;
            lines.forEach((l, i) => {
                ctx.fillText(l.trim(), bubble.x, startY + i * lineHeight);
            });
        }

        ctx.restore();
    }

    // History management
    saveHistory() {
        const state = {};
        this.elements.forEach((els, key) => {
            state[key] = JSON.parse(JSON.stringify(els));
        });
        this.history.push(state);
        if (this.history.length > 20) this.history.shift();
    }

    // Undo
    undo() {
        if (this.history.length === 0) return;

        const state = this.history.pop();
        this.elements.clear();
        Object.keys(state).forEach(key => {
            this.elements.set(parseInt(key), state[key]);
        });

        this.panels.forEach((_, i) => this.redraw(i));
    }

    // Clear all
    clear() {
        if (!confirm('Clear all panels?')) return;

        this.saveHistory();
        this.elements.forEach(els => els.length = 0);
        this.panels.forEach((_, i) => this.redraw(i));
    }

    // Save/download comic
    save() {
        const container = document.getElementById('comic');
        const exportCanvas = document.getElementById('exportCanvas');
        const ctx = exportCanvas.getContext('2d');

        exportCanvas.width = container.offsetWidth * 2;
        exportCanvas.height = container.offsetHeight * 2;

        // White background
        ctx.fillStyle = '#fff';
        ctx.fillRect(0, 0, exportCanvas.width, exportCanvas.height);

        // Border
        ctx.strokeStyle = '#000';
        ctx.lineWidth = 4;
        ctx.strokeRect(2, 2, exportCanvas.width - 4, exportCanvas.height - 4);

        // Get layout
        const layout = container.className.split('layout-')[1];
        let cols = 1, rows = 1;
        if (layout === '2h') { cols = 2; rows = 1; }
        else if (layout === '2v') { cols = 1; rows = 2; }
        else if (layout === '4') { cols = 2; rows = 2; }

        const pw = exportCanvas.width / cols;
        const ph = exportCanvas.height / rows;

        // Draw panels
        this.panels.forEach((panel, i) => {
            const col = i % cols;
            const row = Math.floor(i / cols);
            const px = col * pw;
            const py = row * ph;

            // Panel border
            ctx.strokeRect(px, py, pw, ph);

            // Elements
            const elements = this.elements.get(i);
            const scaleX = pw / panel.canvas.width;
            const scaleY = ph / panel.canvas.height;
            const scale = Math.min(scaleX, scaleY);

            ctx.save();
            ctx.translate(px, py);
            ctx.scale(scale, scale);

            elements.forEach(el => {
                if (el.type === 'character') {
                    const figure = new StickFigure(ctx, el.x, el.y, 1);
                    figure.draw(el.pose);
                } else if (el.type === 'bubble') {
                    this.drawBubble(ctx, el);
                }
            });

            ctx.restore();
        });

        // Download
        exportCanvas.toBlob(blob => {
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `comic-${Date.now()}.png`;
            a.click();
            URL.revokeObjectURL(url);
        });
    }
}

// Start app
document.addEventListener('DOMContentLoaded', () => {
    window.comic = new ComicMaker();
});
