// Simplified Comic Maker - Focus on Core Experience
class SimpleComicMaker {
    constructor() {
        this.panels = [];
        this.currentPanel = 0;
        this.selectedElement = null;
        this.isDragging = false;
        this.isDrawing = false;
        this.currentBubbleType = null;
        this.elements = new Map(); // Store elements by panel

        this.init();
    }

    init() {
        this.setupPanels();
        this.setupCharacterPreviews();
        this.setupEventListeners();
        this.updateHelp('Click a character to add to panel');
    }

    setupPanels() {
        const container = document.getElementById('comicCanvas');
        const panels = container.querySelectorAll('.panel');

        panels.forEach((panel, index) => {
            const canvas = panel.querySelector('canvas');
            const ctx = canvas.getContext('2d');

            // Set canvas size
            const rect = panel.getBoundingClientRect();
            canvas.width = rect.width;
            canvas.height = rect.height;

            this.panels.push({
                element: panel,
                canvas: canvas,
                ctx: ctx,
                index: index
            });

            this.elements.set(index, []);

            // Panel interactions
            canvas.addEventListener('click', (e) => {
                this.selectPanel(index);
                this.handleCanvasClick(e, index);
            });

            canvas.addEventListener('mousedown', (e) => this.handleMouseDown(e, index));
            canvas.addEventListener('mousemove', (e) => this.handleMouseMove(e, index));
            canvas.addEventListener('mouseup', () => this.handleMouseUp());
            canvas.addEventListener('dblclick', (e) => this.handleDoubleClick(e, index));
        });

        if (this.panels.length > 0) {
            this.selectPanel(0);
        }
    }

    setupCharacterPreviews() {
        // Draw mini previews in the character cards
        document.querySelectorAll('.char-card').forEach(card => {
            const canvas = card.querySelector('.char-preview');
            const ctx = canvas.getContext('2d');
            const pose = card.dataset.pose;

            // Draw mini stick figure
            const figure = new StickFigure(ctx, 30, 40, 0.4);
            figure.draw(pose);
        });
    }

    setupEventListeners() {
        // Panel layout buttons
        document.querySelectorAll('.panel-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                this.changeLayout(btn.dataset.layout);
                document.querySelectorAll('.panel-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
            });
        });

        // Character cards - simple click to add
        document.querySelectorAll('.char-card').forEach(card => {
            card.addEventListener('click', () => {
                const pose = card.dataset.pose;
                this.addCharacter(pose);

                // Visual feedback
                card.style.transform = 'scale(0.95)';
                setTimeout(() => {
                    card.style.transform = '';
                }, 150);
            });
        });

        // Speech buttons
        document.querySelectorAll('.speech-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                this.currentBubbleType = btn.dataset.type;
                document.querySelectorAll('.speech-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.updateHelp('Click in panel to add ' + btn.dataset.type + ' bubble');
            });
        });

        // Drawing mode toggle
        document.getElementById('penMode').addEventListener('change', (e) => {
            if (e.target.checked) {
                this.updateHelp('Draw in panels with mouse');
                this.panels.forEach(p => p.canvas.style.cursor = 'crosshair');
            } else {
                this.updateHelp('Click a character to add to panel');
                this.panels.forEach(p => p.canvas.style.cursor = 'default');
            }
        });

        // Templates
        document.querySelectorAll('.template-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                this.loadTemplate(btn.dataset.template);
            });
        });

        // Quick actions
        document.getElementById('undoBtn').addEventListener('click', () => this.undo());
        document.getElementById('clearBtn').addEventListener('click', () => this.clear());
        document.getElementById('downloadBtn').addEventListener('click', () => this.download());

        // Collapsible sections
        document.querySelectorAll('.tool-group.collapsed h3').forEach(header => {
            header.addEventListener('click', () => {
                header.parentElement.classList.toggle('collapsed');
            });
        });

        // Text editor
        const textInput = document.getElementById('textInput');
        textInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.saveInlineText();
            }
        });
        textInput.addEventListener('blur', () => {
            this.saveInlineText();
        });
    }

    selectPanel(index) {
        this.panels.forEach(p => p.element.classList.remove('active'));
        if (this.panels[index]) {
            this.panels[index].element.classList.add('active');
            this.currentPanel = index;

            // Update panel numbers
            this.panels.forEach((p, i) => {
                const num = p.element.querySelector('.panel-number');
                if (num) num.textContent = i + 1;
            });
        }
    }

    addCharacter(pose) {
        const panel = this.panels[this.currentPanel];
        if (!panel) return;

        const centerX = panel.canvas.width / 2;
        const centerY = panel.canvas.height / 2;

        // Add slight random offset if there are already characters
        const elements = this.elements.get(this.currentPanel);
        const offsetX = elements.length > 0 ? (Math.random() - 0.5) * 100 : 0;
        const offsetY = elements.length > 0 ? (Math.random() - 0.5) * 50 : 0;

        const character = {
            type: 'character',
            pose: pose,
            x: centerX + offsetX,
            y: centerY + offsetY,
            id: Date.now()
        };

        elements.push(character);
        this.redraw(this.currentPanel);
        this.updateHelp('Drag to move character');
    }

    handleCanvasClick(e, panelIndex) {
        const panel = this.panels[panelIndex];
        const rect = panel.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // Add bubble if bubble mode is active
        if (this.currentBubbleType) {
            this.addBubble(x, y, panelIndex);
            this.currentBubbleType = null;
            document.querySelectorAll('.speech-btn').forEach(b => b.classList.remove('active'));
        }
    }

    handleMouseDown(e, panelIndex) {
        const panel = this.panels[panelIndex];
        const rect = panel.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // Check if pen mode
        if (document.getElementById('penMode').checked) {
            this.isDrawing = true;
            panel.ctx.beginPath();
            panel.ctx.moveTo(x, y);
            return;
        }

        // Check if clicking on an element
        const elements = this.elements.get(panelIndex);
        const clicked = this.findElementAt(x, y, elements);

        if (clicked) {
            this.selectedElement = clicked;
            this.isDragging = true;
            this.dragOffset = { x: x - clicked.x, y: y - clicked.y };
            panel.canvas.style.cursor = 'grabbing';
        }
    }

    handleMouseMove(e, panelIndex) {
        const panel = this.panels[panelIndex];
        const rect = panel.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // Drawing mode
        if (this.isDrawing && document.getElementById('penMode').checked) {
            const ctx = panel.ctx;
            ctx.strokeStyle = document.getElementById('colorPicker').value;
            ctx.lineWidth = document.getElementById('brushSize').value;
            ctx.lineCap = 'round';
            ctx.lineTo(x, y);
            ctx.stroke();
            return;
        }

        // Update cursor based on hover
        if (!this.isDragging && !document.getElementById('penMode').checked) {
            const elements = this.elements.get(panelIndex);
            const hover = this.findElementAt(x, y, elements);
            panel.canvas.style.cursor = hover ? 'grab' : 'default';
        }

        // Dragging
        if (this.isDragging && this.selectedElement) {
            this.selectedElement.x = x - this.dragOffset.x;
            this.selectedElement.y = y - this.dragOffset.y;
            this.redraw(panelIndex);
        }
    }

    handleMouseUp() {
        this.isDrawing = false;
        this.isDragging = false;
        this.panels.forEach(p => {
            if (!document.getElementById('penMode').checked) {
                p.canvas.style.cursor = 'default';
            }
        });
    }

    handleDoubleClick(e, panelIndex) {
        const panel = this.panels[panelIndex];
        const rect = panel.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        const elements = this.elements.get(panelIndex);
        const clicked = this.findElementAt(x, y, elements);

        if (clicked && clicked.type === 'bubble') {
            this.editBubbleText(clicked, x, y);
        }
    }

    addBubble(x, y, panelIndex) {
        const bubble = {
            type: 'bubble',
            bubbleType: this.currentBubbleType,
            text: 'Click to edit',
            x: x,
            y: y,
            width: 120,
            height: 60,
            id: Date.now()
        };

        const elements = this.elements.get(panelIndex);
        elements.push(bubble);
        this.redraw(panelIndex);

        // Immediately edit the text
        this.editBubbleText(bubble, x, y);
    }

    editBubbleText(bubble, x, y) {
        const editor = document.getElementById('textEditor');
        const input = document.getElementById('textInput');

        // Position editor over bubble
        editor.style.display = 'block';
        editor.style.left = x + 'px';
        editor.style.top = y + 'px';

        input.value = bubble.text === 'Click to edit' ? '' : bubble.text;
        input.focus();
        input.select();

        // Store reference to bubble being edited
        this.editingBubble = bubble;
        this.editingPanelIndex = this.currentPanel;
    }

    saveInlineText() {
        const editor = document.getElementById('textEditor');
        const input = document.getElementById('textInput');

        if (this.editingBubble && input.value) {
            this.editingBubble.text = input.value;
            this.editingBubble.width = Math.max(120, input.value.length * 8);
            this.redraw(this.editingPanelIndex);
        }

        editor.style.display = 'none';
        this.editingBubble = null;
    }

    findElementAt(x, y, elements) {
        // Search in reverse order (top elements first)
        for (let i = elements.length - 1; i >= 0; i--) {
            const el = elements[i];

            if (el.type === 'character') {
                // Approximate character bounds
                const bounds = {
                    left: el.x - 40,
                    right: el.x + 40,
                    top: el.y - 20,
                    bottom: el.y + 100
                };

                if (x >= bounds.left && x <= bounds.right &&
                    y >= bounds.top && y <= bounds.bottom) {
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

    redraw(panelIndex) {
        const panel = this.panels[panelIndex];
        const ctx = panel.ctx;
        const elements = this.elements.get(panelIndex);

        // Clear and redraw
        ctx.clearRect(0, 0, panel.canvas.width, panel.canvas.height);

        elements.forEach(el => {
            if (el.type === 'character') {
                const figure = new StickFigure(ctx, el.x, el.y, 1);
                figure.draw(el.pose);

                // Draw selection indicator
                if (el === this.selectedElement) {
                    ctx.save();
                    ctx.strokeStyle = '#007aff';
                    ctx.lineWidth = 2;
                    ctx.setLineDash([5, 5]);
                    ctx.strokeRect(el.x - 45, el.y - 25, 90, 120);
                    ctx.restore();
                }
            } else if (el.type === 'bubble') {
                this.drawBubble(ctx, el);
            }
        });
    }

    drawBubble(ctx, bubble) {
        ctx.save();
        ctx.strokeStyle = '#000';
        ctx.fillStyle = '#fff';
        ctx.lineWidth = 2;

        const x = bubble.x - bubble.width/2;
        const y = bubble.y - bubble.height/2;

        if (bubble.bubbleType === 'speech') {
            // Rounded rect
            ctx.beginPath();
            ctx.roundRect(x, y, bubble.width, bubble.height, 10);
            ctx.fill();
            ctx.stroke();

            // Tail
            ctx.beginPath();
            ctx.moveTo(bubble.x - 10, y + bubble.height);
            ctx.lineTo(bubble.x, y + bubble.height + 15);
            ctx.lineTo(bubble.x + 10, y + bubble.height);
            ctx.fill();
        } else if (bubble.bubbleType === 'thought') {
            // Cloud bubble
            ctx.beginPath();
            for (let i = 0; i < 8; i++) {
                const angle = (Math.PI * 2 / 8) * i;
                const bx = bubble.x + Math.cos(angle) * bubble.width/3;
                const by = bubble.y + Math.sin(angle) * bubble.height/3;
                ctx.arc(bx, by, bubble.width/6, 0, Math.PI * 2);
            }
            ctx.fill();
            ctx.stroke();

            // Thought dots
            ctx.beginPath();
            ctx.arc(bubble.x, y + bubble.height + 10, 5, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
            ctx.beginPath();
            ctx.arc(bubble.x, y + bubble.height + 25, 3, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
        }

        // Text
        ctx.fillStyle = '#000';
        ctx.font = '14px Comic Sans MS, Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';

        // Simple word wrap
        const words = bubble.text.split(' ');
        const lines = [];
        let currentLine = '';

        words.forEach(word => {
            const testLine = currentLine + word + ' ';
            const metrics = ctx.measureText(testLine);
            if (metrics.width > bubble.width - 20 && currentLine !== '') {
                lines.push(currentLine);
                currentLine = word + ' ';
            } else {
                currentLine = testLine;
            }
        });
        lines.push(currentLine);

        const lineHeight = 18;
        const startY = bubble.y - (lines.length - 1) * lineHeight/2;

        lines.forEach((line, i) => {
            ctx.fillText(line.trim(), bubble.x, startY + i * lineHeight);
        });

        ctx.restore();
    }

    changeLayout(layout) {
        const container = document.getElementById('comicCanvas');
        container.className = `comic-canvas layout-${layout}`;

        // Clear and recreate panels
        container.innerHTML = '';
        this.panels = [];
        this.elements.clear();

        let panelCount = 1;
        if (layout === '2h' || layout === '2v') panelCount = 2;
        if (layout === '4') panelCount = 4;

        for (let i = 0; i < panelCount; i++) {
            const panel = document.createElement('div');
            panel.className = 'panel';
            panel.dataset.panel = i;

            const canvas = document.createElement('canvas');
            panel.appendChild(canvas);

            const number = document.createElement('div');
            number.className = 'panel-number';
            number.textContent = i + 1;
            panel.appendChild(number);

            container.appendChild(panel);
        }

        this.setupPanels();
    }

    loadTemplate(template) {
        if (template === 'conversation') {
            this.changeLayout('2h');
            setTimeout(() => {
                this.selectPanel(0);
                this.addCharacter('standing');
                this.selectPanel(1);
                this.addCharacter('standing');
                this.updateHelp('Add speech bubbles to create dialogue');
            }, 100);
        } else if (template === 'joke') {
            this.changeLayout('2v');
            setTimeout(() => {
                this.selectPanel(0);
                this.addCharacter('standing');
                this.selectPanel(1);
                this.addCharacter('happy');
                this.updateHelp('Setup in panel 1, punchline in panel 2');
            }, 100);
        } else {
            this.clear();
        }
    }

    undo() {
        const elements = this.elements.get(this.currentPanel);
        if (elements.length > 0) {
            elements.pop();
            this.redraw(this.currentPanel);
        }
    }

    clear() {
        if (confirm('Clear all panels?')) {
            this.elements.forEach((els, index) => {
                els.length = 0;
                this.redraw(index);
            });
        }
    }

    download() {
        const exportCanvas = document.getElementById('exportCanvas');
        const exportCtx = exportCanvas.getContext('2d');
        const container = document.getElementById('comicCanvas');

        // Set size
        exportCanvas.width = container.offsetWidth;
        exportCanvas.height = container.offsetHeight;

        // White background
        exportCtx.fillStyle = '#fff';
        exportCtx.fillRect(0, 0, exportCanvas.width, exportCanvas.height);

        // Draw panels
        const layout = container.className.split('layout-')[1];
        let cols = 1, rows = 1;

        if (layout === '2h') { cols = 2; rows = 1; }
        else if (layout === '2v') { cols = 1; rows = 2; }
        else if (layout === '4') { cols = 2; rows = 2; }

        const panelWidth = exportCanvas.width / cols;
        const panelHeight = exportCanvas.height / rows;

        // Draw border
        exportCtx.strokeStyle = '#000';
        exportCtx.lineWidth = 3;
        exportCtx.strokeRect(0, 0, exportCanvas.width, exportCanvas.height);

        this.panels.forEach((panel, index) => {
            const col = index % cols;
            const row = Math.floor(index / cols);
            const x = col * panelWidth;
            const y = row * panelHeight;

            // Panel border
            exportCtx.strokeRect(x, y, panelWidth, panelHeight);

            // Draw elements
            const elements = this.elements.get(index);
            exportCtx.save();
            exportCtx.translate(x, y);

            const scale = Math.min(panelWidth / panel.canvas.width, panelHeight / panel.canvas.height);
            exportCtx.scale(scale, scale);

            elements.forEach(el => {
                if (el.type === 'character') {
                    const figure = new StickFigure(exportCtx, el.x, el.y, 1);
                    figure.draw(el.pose);
                } else if (el.type === 'bubble') {
                    this.drawBubble(exportCtx, el);
                }
            });

            exportCtx.restore();
        });

        // Download
        exportCanvas.toBlob((blob) => {
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `comic-${Date.now()}.png`;
            a.click();
            URL.revokeObjectURL(url);
        });

        this.updateHelp('Comic saved!');
    }

    updateHelp(text) {
        document.getElementById('helpText').textContent = text;
    }
}

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    new SimpleComicMaker();
});