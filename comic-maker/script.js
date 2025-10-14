// Comic Panel Maker - Main Application with Drag & Drop
class ComicMaker {
    constructor() {
        this.panels = [];
        this.currentPanel = 0;
        this.currentTool = 'select';
        this.isDrawing = false;
        this.currentBubble = null;
        this.brushSize = 2;
        this.color = '#000000';

        // New properties for drag and drop
        this.draggedElement = null;
        this.selectedElement = null;
        this.isDragging = false;
        this.dragOffset = { x: 0, y: 0 };
        this.hoverElement = null;
        this.previewPose = null;

        this.init();
    }

    init() {
        this.setupPanels();
        this.setupEventListeners();
        this.setupCanvas();
        this.showInstructions();
    }

    showInstructions() {
        // Add initial instructions overlay
        const instructions = document.createElement('div');
        instructions.className = 'instructions-overlay';
        instructions.innerHTML = `
            <div class="instructions-content">
                <h3>How to Use Comic Maker</h3>
                <ol>
                    <li><strong>Click a stick figure pose</strong> to add it to the active panel (outlined in purple)</li>
                    <li><strong>Click and drag</strong> figures to move them around</li>
                    <li><strong>Select the bubble tool</strong> then click in the panel to add speech bubbles</li>
                    <li><strong>Use the pen tool</strong> to draw freehand</li>
                    <li><strong>Export your comic</strong> as PNG when done!</li>
                </ol>
                <button id="closeInstructions">Got it!</button>
            </div>
        `;
        document.body.appendChild(instructions);

        document.getElementById('closeInstructions').addEventListener('click', () => {
            instructions.remove();
        });

        // Auto-remove after 10 seconds
        setTimeout(() => {
            if (instructions.parentNode) {
                instructions.remove();
            }
        }, 10000);
    }

    setupPanels() {
        const panelContainer = document.getElementById('comicPanels');
        const panels = panelContainer.querySelectorAll('.panel');

        panels.forEach((panel, index) => {
            const canvas = panel.querySelector('canvas');
            const ctx = canvas.getContext('2d');

            // Set canvas size
            canvas.width = panel.offsetWidth;
            canvas.height = panel.offsetHeight;

            this.panels.push({
                element: panel,
                canvas: canvas,
                ctx: ctx,
                elements: [], // Store objects, not just drawing commands
                history: []
            });

            // Panel click handler
            panel.addEventListener('click', (e) => {
                // Only select panel if clicking on the panel itself, not canvas
                if (e.target === panel) {
                    this.selectPanel(index);
                }
            });
        });

        // Select first panel
        if (this.panels.length > 0) {
            this.selectPanel(0);
        }
    }

    setupCanvas() {
        this.panels.forEach((panel, panelIndex) => {
            const canvas = panel.canvas;
            const ctx = panel.ctx;

            // Mouse events for drag and drop
            canvas.addEventListener('mousedown', (e) => this.handleMouseDown(e, panel, panelIndex));
            canvas.addEventListener('mousemove', (e) => this.handleMouseMove(e, panel, panelIndex));
            canvas.addEventListener('mouseup', (e) => this.handleMouseUp(e, panel, panelIndex));
            canvas.addEventListener('mouseout', (e) => this.handleMouseOut(e, panel, panelIndex));

            // Click handler for placing elements
            canvas.addEventListener('click', (e) => {
                // Select the panel when canvas is clicked
                this.selectPanel(panelIndex);
            });

            // Touch events for mobile
            canvas.addEventListener('touchstart', (e) => {
                const touch = e.touches[0];
                const rect = canvas.getBoundingClientRect();
                const mouseEvent = new MouseEvent('mousedown', {
                    clientX: touch.clientX,
                    clientY: touch.clientY
                });
                this.handleMouseDown(mouseEvent, panel, panelIndex);
                e.preventDefault();
            });

            canvas.addEventListener('touchmove', (e) => {
                const touch = e.touches[0];
                const mouseEvent = new MouseEvent('mousemove', {
                    clientX: touch.clientX,
                    clientY: touch.clientY
                });
                this.handleMouseMove(mouseEvent, panel, panelIndex);
                e.preventDefault();
            });

            canvas.addEventListener('touchend', (e) => {
                const mouseEvent = new MouseEvent('mouseup', {});
                this.handleMouseUp(mouseEvent, panel, panelIndex);
                e.preventDefault();
            });
        });
    }

    handleMouseDown(e, panel, panelIndex) {
        const rect = panel.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        if (this.currentTool === 'select') {
            // Check if clicking on an element
            const clickedElement = this.findElementAt(x, y, panel);
            if (clickedElement) {
                this.selectedElement = clickedElement;
                this.isDragging = true;
                this.dragOffset.x = x - clickedElement.x;
                this.dragOffset.y = y - clickedElement.y;
                panel.canvas.style.cursor = 'grabbing';
            }
        } else if (this.currentTool === 'pen' || this.currentTool === 'eraser') {
            this.isDrawing = true;
            panel.ctx.beginPath();
            panel.ctx.moveTo(x, y);
        } else if (this.currentTool === 'bubble' && this.currentBubble) {
            this.addBubbleAt(x, y, panel);
        } else if (this.currentTool === 'text') {
            this.addTextAt(x, y, panel);
        }
    }

    handleMouseMove(e, panel, panelIndex) {
        const rect = panel.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // Update cursor based on what's under mouse
        if (this.currentTool === 'select') {
            const elementUnder = this.findElementAt(x, y, panel);
            if (elementUnder) {
                panel.canvas.style.cursor = this.isDragging ? 'grabbing' : 'grab';
            } else {
                panel.canvas.style.cursor = 'default';
            }
        } else if (this.currentTool === 'pen' || this.currentTool === 'eraser') {
            panel.canvas.style.cursor = 'crosshair';
        } else if (this.currentTool === 'bubble' || this.currentTool === 'text') {
            panel.canvas.style.cursor = 'text';
        }

        // Handle dragging
        if (this.isDragging && this.selectedElement) {
            this.selectedElement.x = x - this.dragOffset.x;
            this.selectedElement.y = y - this.dragOffset.y;
            this.redrawPanel(panel);
        }

        // Handle drawing
        if (this.isDrawing) {
            const ctx = panel.ctx;
            ctx.lineWidth = this.brushSize;
            ctx.lineCap = 'round';

            if (this.currentTool === 'pen') {
                ctx.globalCompositeOperation = 'source-over';
                ctx.strokeStyle = this.color;
            } else if (this.currentTool === 'eraser') {
                ctx.globalCompositeOperation = 'destination-out';
            }

            ctx.lineTo(x, y);
            ctx.stroke();
        }
    }

    handleMouseUp(e, panel, panelIndex) {
        if (this.isDragging) {
            this.isDragging = false;
            panel.canvas.style.cursor = 'grab';
            this.saveHistory(panel);
        }

        if (this.isDrawing) {
            this.isDrawing = false;
            this.saveHistory(panel);
        }
    }

    handleMouseOut(e, panel, panelIndex) {
        if (this.isDragging) {
            this.isDragging = false;
            panel.canvas.style.cursor = 'default';
        }

        if (this.isDrawing) {
            this.isDrawing = false;
        }
    }

    findElementAt(x, y, panel) {
        // Search elements in reverse order (top elements first)
        for (let i = panel.elements.length - 1; i >= 0; i--) {
            const element = panel.elements[i];

            if (element.type === 'figure') {
                // Check if click is within figure bounds (approximate)
                const bounds = {
                    left: element.x - 40,
                    right: element.x + 40,
                    top: element.y - 20,
                    bottom: element.y + 100
                };

                if (x >= bounds.left && x <= bounds.right &&
                    y >= bounds.top && y <= bounds.bottom) {
                    return element;
                }
            } else if (element.type === 'bubble') {
                if (x >= element.x && x <= element.x + element.width &&
                    y >= element.y && y <= element.y + element.height) {
                    return element;
                }
            } else if (element.type === 'text') {
                // Approximate text bounds
                const textWidth = element.text.length * 8;
                const textHeight = 20;
                if (x >= element.x - textWidth/2 && x <= element.x + textWidth/2 &&
                    y >= element.y - textHeight/2 && y <= element.y + textHeight/2) {
                    return element;
                }
            }
        }
        return null;
    }

    setupEventListeners() {
        // Layout buttons
        document.querySelectorAll('.layout-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                this.changeLayout(btn.dataset.layout);
                document.querySelectorAll('.layout-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
            });
        });

        // Figure buttons - now just add to current panel on click
        document.querySelectorAll('.figure-btn').forEach(btn => {
            // Add hover effect
            btn.addEventListener('mouseenter', () => {
                btn.style.transform = 'scale(1.05)';
            });

            btn.addEventListener('mouseleave', () => {
                btn.style.transform = 'scale(1)';
            });

            btn.addEventListener('click', () => {
                const panel = this.panels[this.currentPanel];
                if (!panel) return;

                // Add figure at center or slightly offset if multiple
                const offsetX = (Math.random() - 0.5) * 100;
                const offsetY = (Math.random() - 0.5) * 50;

                this.addStickFigureAt(
                    panel.canvas.width / 2 + offsetX,
                    panel.canvas.height / 2 + offsetY,
                    btn.dataset.pose,
                    panel
                );

                // Visual feedback
                btn.style.backgroundColor = '#667eea';
                btn.style.color = 'white';
                setTimeout(() => {
                    btn.style.backgroundColor = '';
                    btn.style.color = '';
                }, 300);
            });
        });

        // Bubble buttons
        document.querySelectorAll('.bubble-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                this.currentBubble = btn.dataset.type;
                this.currentTool = 'bubble';
                this.updateToolButtons();

                // Show hint
                this.showHint('Click in the panel to add a ' + btn.dataset.type + ' bubble');
            });
        });

        // Text button
        document.getElementById('addTextBtn').addEventListener('click', () => {
            const text = document.getElementById('textInput').value;
            if (text) {
                this.currentTool = 'text';
                this.textToAdd = text;
                document.getElementById('textInput').value = '';
                this.showHint('Click in the panel to place the text');
            }
        });

        // Drawing tools
        document.querySelectorAll('.tool-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                this.currentTool = btn.dataset.tool;
                this.updateToolButtons();
            });
        });

        // Brush size and color
        document.getElementById('brushSize').addEventListener('input', (e) => {
            this.brushSize = e.target.value;
        });

        document.getElementById('colorPicker').addEventListener('input', (e) => {
            this.color = e.target.value;
        });

        // Action buttons
        document.getElementById('clearPanelBtn').addEventListener('click', () => {
            this.clearCurrentPanel();
        });

        document.getElementById('clearAllBtn').addEventListener('click', () => {
            if (confirm('Clear all panels?')) {
                this.clearAll();
            }
        });

        document.getElementById('undoBtn').addEventListener('click', () => {
            this.undo();
        });

        document.getElementById('redoBtn').addEventListener('click', () => {
            this.redo();
        });

        // Export buttons
        document.getElementById('downloadBtn').addEventListener('click', () => {
            this.downloadComic();
        });

        document.getElementById('copyBtn').addEventListener('click', () => {
            this.copyToClipboard();
        });

        document.getElementById('printBtn').addEventListener('click', () => {
            window.print();
        });

        // Template buttons
        document.querySelectorAll('.template-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                this.loadTemplate(btn.textContent.toLowerCase());
            });
        });
    }

    updateToolButtons() {
        document.querySelectorAll('.tool-btn').forEach(btn => {
            if (btn.dataset.tool === this.currentTool) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });

        // Update cursor for all panels
        this.panels.forEach(panel => {
            if (this.currentTool === 'select') {
                panel.canvas.style.cursor = 'default';
            } else if (this.currentTool === 'pen' || this.currentTool === 'eraser') {
                panel.canvas.style.cursor = 'crosshair';
            } else if (this.currentTool === 'bubble' || this.currentTool === 'text') {
                panel.canvas.style.cursor = 'text';
            }
        });
    }

    showHint(message) {
        // Remove existing hint
        const existingHint = document.querySelector('.hint-message');
        if (existingHint) existingHint.remove();

        const hint = document.createElement('div');
        hint.className = 'hint-message';
        hint.textContent = message;
        document.body.appendChild(hint);

        setTimeout(() => {
            if (hint.parentNode) {
                hint.remove();
            }
        }, 3000);
    }

    changeLayout(layout) {
        const container = document.getElementById('comicPanels');
        container.className = `comic-panels layout-${layout}`;

        // Clear existing panels
        container.innerHTML = '';
        this.panels = [];

        // Create new panels based on layout
        let panelCount = 1;
        if (layout === '2h' || layout === '2v') panelCount = 2;
        if (layout === '4') panelCount = 4;

        for (let i = 0; i < panelCount; i++) {
            const panel = document.createElement('div');
            panel.className = 'panel';
            panel.dataset.panel = i;

            const canvas = document.createElement('canvas');
            panel.appendChild(canvas);
            container.appendChild(panel);
        }

        // Re-setup panels
        this.setupPanels();
        this.setupCanvas();
    }

    selectPanel(index) {
        // Remove active class from all panels
        this.panels.forEach(p => p.element.classList.remove('active'));

        // Add active class to selected panel
        if (this.panels[index]) {
            this.panels[index].element.classList.add('active');
            this.currentPanel = index;
        }
    }

    addStickFigureAt(x, y, pose, panel) {
        // Get options
        const hasHair = document.getElementById('addHair').checked;
        const hasDress = document.getElementById('addDress').checked;

        // Add to elements array as an object
        const figureElement = {
            type: 'figure',
            pose: pose,
            x: x,
            y: y,
            hasHair: hasHair,
            hasDress: hasDress,
            color: this.color,
            id: Date.now() // Unique ID for tracking
        };

        panel.elements.push(figureElement);
        this.redrawPanel(panel);
        this.saveHistory(panel);

        // Select the newly added figure
        this.selectedElement = figureElement;
        this.currentTool = 'select';
        this.updateToolButtons();
    }

    addBubbleAt(x, y, panel) {
        const text = prompt('Enter text for the ' + this.currentBubble + ' bubble:');
        if (text) {
            const width = Math.max(150, text.length * 8);
            const height = 80;

            const bubbleElement = {
                type: 'bubble',
                bubbleType: this.currentBubble,
                text: text,
                x: x - width/2,
                y: y - height/2,
                width: width,
                height: height,
                id: Date.now()
            };

            panel.elements.push(bubbleElement);
            this.redrawPanel(panel);
            this.saveHistory(panel);

            // Reset tool to select
            this.currentTool = 'select';
            this.updateToolButtons();
        }
    }

    addTextAt(x, y, panel) {
        if (this.textToAdd) {
            const textElement = {
                type: 'text',
                text: this.textToAdd,
                x: x,
                y: y,
                color: this.color,
                id: Date.now()
            };

            panel.elements.push(textElement);
            this.redrawPanel(panel);
            this.saveHistory(panel);

            this.textToAdd = null;
            this.currentTool = 'select';
            this.updateToolButtons();
        } else {
            const text = prompt('Enter text:');
            if (text) {
                const textElement = {
                    type: 'text',
                    text: text,
                    x: x,
                    y: y,
                    color: this.color,
                    id: Date.now()
                };

                panel.elements.push(textElement);
                this.redrawPanel(panel);
                this.saveHistory(panel);
            }
        }
    }

    drawSpeechBubble(ctx, x, y, width, height, text, type = 'speech') {
        ctx.save();
        ctx.strokeStyle = '#000000';
        ctx.fillStyle = '#FFFFFF';
        ctx.lineWidth = 2;

        // Draw bubble based on type
        if (type === 'speech') {
            // Regular speech bubble
            this.drawRoundedRect(ctx, x, y, width, height, 10);

            // Draw tail
            ctx.beginPath();
            ctx.moveTo(x + width/2 - 10, y + height);
            ctx.lineTo(x + width/2, y + height + 15);
            ctx.lineTo(x + width/2 + 10, y + height);
            ctx.fill();
            ctx.stroke();
        } else if (type === 'thought') {
            // Thought bubble (cloud shape)
            this.drawCloudBubble(ctx, x, y, width, height);

            // Draw small circles for tail
            ctx.beginPath();
            ctx.arc(x + width/2 - 5, y + height + 10, 5, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
            ctx.beginPath();
            ctx.arc(x + width/2, y + height + 20, 3, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
        } else if (type === 'shout') {
            // Jagged shout bubble
            this.drawShoutBubble(ctx, x, y, width, height);
        }

        // Add text
        ctx.fillStyle = '#000000';
        ctx.font = '14px Comic Sans MS, Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';

        // Word wrap
        const words = text.split(' ');
        const lines = [];
        let currentLine = '';

        words.forEach(word => {
            const testLine = currentLine + word + ' ';
            const metrics = ctx.measureText(testLine);
            if (metrics.width > width - 20 && currentLine !== '') {
                lines.push(currentLine);
                currentLine = word + ' ';
            } else {
                currentLine = testLine;
            }
        });
        lines.push(currentLine);

        // Draw text lines
        const lineHeight = 20;
        const startY = y + height/2 - (lines.length - 1) * lineHeight/2;

        lines.forEach((line, index) => {
            ctx.fillText(line.trim(), x + width/2, startY + index * lineHeight);
        });

        ctx.restore();
    }

    drawRoundedRect(ctx, x, y, width, height, radius) {
        ctx.beginPath();
        ctx.moveTo(x + radius, y);
        ctx.lineTo(x + width - radius, y);
        ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
        ctx.lineTo(x + width, y + height - radius);
        ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
        ctx.lineTo(x + radius, y + height);
        ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
        ctx.lineTo(x, y + radius);
        ctx.quadraticCurveTo(x, y, x + radius, y);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
    }

    drawCloudBubble(ctx, x, y, width, height) {
        ctx.beginPath();
        // Draw cloud shape with arcs
        const bumps = 8;
        for (let i = 0; i < bumps; i++) {
            const angle = (Math.PI * 2 / bumps) * i;
            const bumpX = x + width/2 + Math.cos(angle) * width/2.5;
            const bumpY = y + height/2 + Math.sin(angle) * height/2.5;
            const radius = width/8;
            ctx.arc(bumpX, bumpY, radius, 0, Math.PI * 2);
        }
        ctx.fill();
        ctx.stroke();
    }

    drawShoutBubble(ctx, x, y, width, height) {
        ctx.beginPath();
        ctx.moveTo(x + width/2, y);

        // Jagged edges
        const points = 12;
        for (let i = 0; i <= points; i++) {
            const angle = (Math.PI * 2 / points) * i - Math.PI/2;
            const radius = i % 2 === 0 ? width/2 : width/2.5;
            const px = x + width/2 + Math.cos(angle) * radius;
            const py = y + height/2 + Math.sin(angle) * radius;
            ctx.lineTo(px, py);
        }

        ctx.closePath();
        ctx.fill();
        ctx.stroke();
    }

    saveHistory(panel) {
        // Save the elements array state
        panel.history = JSON.parse(JSON.stringify(panel.elements));
    }

    undo() {
        const panel = this.panels[this.currentPanel];
        if (!panel) return;

        // Remove last element
        if (panel.elements.length > 0) {
            panel.elements.pop();
            this.redrawPanel(panel);
        }
    }

    redo() {
        // Would need more complex implementation
        console.log('Redo functionality to be implemented');
    }

    clearPanel(panel) {
        panel.ctx.clearRect(0, 0, panel.canvas.width, panel.canvas.height);
        panel.elements = [];
        panel.history = [];
    }

    clearCurrentPanel() {
        const panel = this.panels[this.currentPanel];
        if (panel) {
            this.clearPanel(panel);
        }
    }

    clearAll() {
        this.panels.forEach(panel => {
            this.clearPanel(panel);
        });
    }

    redrawPanel(panel) {
        // Clear canvas
        panel.ctx.clearRect(0, 0, panel.canvas.width, panel.canvas.height);

        // Redraw all elements
        panel.elements.forEach(element => {
            // Highlight selected element
            if (element === this.selectedElement && this.currentTool === 'select') {
                panel.ctx.save();
                panel.ctx.strokeStyle = '#667eea';
                panel.ctx.lineWidth = 2;
                panel.ctx.setLineDash([5, 5]);

                if (element.type === 'figure') {
                    panel.ctx.strokeRect(element.x - 45, element.y - 25, 90, 120);
                } else if (element.type === 'bubble') {
                    panel.ctx.strokeRect(element.x - 5, element.y - 5, element.width + 10, element.height + 10);
                }

                panel.ctx.restore();
            }

            if (element.type === 'figure') {
                const figure = new StickFigure(panel.ctx, element.x, element.y, 1);
                figure.setOptions({
                    hasHair: element.hasHair,
                    hasDress: element.hasDress,
                    color: element.color
                });
                figure.draw(element.pose);
            } else if (element.type === 'bubble') {
                this.drawSpeechBubble(
                    panel.ctx,
                    element.x,
                    element.y,
                    element.width,
                    element.height,
                    element.text,
                    element.bubbleType
                );
            } else if (element.type === 'text') {
                panel.ctx.save();
                panel.ctx.font = '16px Comic Sans MS, Arial';
                panel.ctx.fillStyle = element.color;
                panel.ctx.textAlign = 'center';
                panel.ctx.fillText(element.text, element.x, element.y);
                panel.ctx.restore();
            }
        });
    }

    loadTemplate(templateName) {
        // Clear all panels first
        this.clearAll();

        switch(templateName) {
            case 'conversation':
                this.changeLayout('2h');
                setTimeout(() => {
                    // Add figures to both panels
                    this.selectPanel(0);
                    this.addStickFigureAt(
                        this.panels[0].canvas.width / 3,
                        this.panels[0].canvas.height / 2,
                        'standing',
                        this.panels[0]
                    );

                    this.selectPanel(1);
                    this.addStickFigureAt(
                        this.panels[1].canvas.width * 2 / 3,
                        this.panels[1].canvas.height / 2,
                        'standing',
                        this.panels[1]
                    );

                    this.showHint('Conversation template loaded! Click figures to move them, add speech bubbles to create dialogue.');
                }, 100);
                break;

            case 'story':
                this.changeLayout('4');
                setTimeout(() => {
                    this.showHint('4-panel story template loaded! Click figure poses to add characters.');
                }, 100);
                break;

            case 'joke':
                this.changeLayout('2v');
                setTimeout(() => {
                    this.selectPanel(0);
                    this.addStickFigureAt(
                        this.panels[0].canvas.width / 2,
                        this.panels[0].canvas.height / 2,
                        'standing',
                        this.panels[0]
                    );

                    this.selectPanel(1);
                    this.addStickFigureAt(
                        this.panels[1].canvas.width / 2,
                        this.panels[1].canvas.height / 2,
                        'happy',
                        this.panels[1]
                    );

                    this.showHint('Joke template loaded! Setup in panel 1, punchline in panel 2!');
                }, 100);
                break;
        }
    }

    downloadComic() {
        const exportCanvas = document.getElementById('exportCanvas');
        const exportCtx = exportCanvas.getContext('2d');
        const container = document.getElementById('comicPanels');

        // Set export canvas size
        exportCanvas.width = container.offsetWidth;
        exportCanvas.height = container.offsetHeight;

        // Draw white background
        exportCtx.fillStyle = '#FFFFFF';
        exportCtx.fillRect(0, 0, exportCanvas.width, exportCanvas.height);

        // Draw border
        exportCtx.strokeStyle = '#000000';
        exportCtx.lineWidth = 3;
        exportCtx.strokeRect(0, 0, exportCanvas.width, exportCanvas.height);

        // Get layout info
        const layout = container.className.split('layout-')[1];
        let cols = 1, rows = 1;

        if (layout === '2h') { cols = 2; rows = 1; }
        else if (layout === '2v') { cols = 1; rows = 2; }
        else if (layout === '4') { cols = 2; rows = 2; }

        const panelWidth = exportCanvas.width / cols;
        const panelHeight = exportCanvas.height / rows;

        // Draw each panel
        this.panels.forEach((panel, index) => {
            const col = index % cols;
            const row = Math.floor(index / cols);
            const x = col * panelWidth;
            const y = row * panelHeight;

            // Draw panel border
            exportCtx.strokeRect(x, y, panelWidth, panelHeight);

            // Draw panel content by redrawing elements
            exportCtx.save();
            exportCtx.translate(x, y);

            // Scale if needed
            const scaleX = panelWidth / panel.canvas.width;
            const scaleY = panelHeight / panel.canvas.height;
            exportCtx.scale(scaleX, scaleY);

            // Redraw elements without selection highlight
            const tempSelected = this.selectedElement;
            this.selectedElement = null;

            panel.elements.forEach(element => {
                if (element.type === 'figure') {
                    const figure = new StickFigure(exportCtx, element.x, element.y, 1);
                    figure.setOptions({
                        hasHair: element.hasHair,
                        hasDress: element.hasDress,
                        color: element.color
                    });
                    figure.draw(element.pose);
                } else if (element.type === 'bubble') {
                    this.drawSpeechBubble(
                        exportCtx,
                        element.x,
                        element.y,
                        element.width,
                        element.height,
                        element.text,
                        element.bubbleType
                    );
                } else if (element.type === 'text') {
                    exportCtx.font = '16px Comic Sans MS, Arial';
                    exportCtx.fillStyle = element.color;
                    exportCtx.textAlign = 'center';
                    exportCtx.fillText(element.text, element.x, element.y);
                }
            });

            this.selectedElement = tempSelected;
            exportCtx.restore();
        });

        // Create download link
        exportCanvas.toBlob((blob) => {
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `comic-${Date.now()}.png`;
            a.click();
            URL.revokeObjectURL(url);

            // Save to recent comics
            this.saveToRecent(exportCanvas.toDataURL());
        });
    }

    async copyToClipboard() {
        const exportCanvas = document.getElementById('exportCanvas');
        const exportCtx = exportCanvas.getContext('2d');
        const container = document.getElementById('comicPanels');

        // Same export process as download
        exportCanvas.width = container.offsetWidth;
        exportCanvas.height = container.offsetHeight;

        exportCtx.fillStyle = '#FFFFFF';
        exportCtx.fillRect(0, 0, exportCanvas.width, exportCanvas.height);

        exportCtx.strokeStyle = '#000000';
        exportCtx.lineWidth = 3;
        exportCtx.strokeRect(0, 0, exportCanvas.width, exportCanvas.height);

        const layout = container.className.split('layout-')[1];
        let cols = 1, rows = 1;

        if (layout === '2h') { cols = 2; rows = 1; }
        else if (layout === '2v') { cols = 1; rows = 2; }
        else if (layout === '4') { cols = 2; rows = 2; }

        const panelWidth = exportCanvas.width / cols;
        const panelHeight = exportCanvas.height / rows;

        this.panels.forEach((panel, index) => {
            const col = index % cols;
            const row = Math.floor(index / cols);
            const x = col * panelWidth;
            const y = row * panelHeight;

            exportCtx.strokeRect(x, y, panelWidth, panelHeight);

            exportCtx.save();
            exportCtx.translate(x, y);

            const scaleX = panelWidth / panel.canvas.width;
            const scaleY = panelHeight / panel.canvas.height;
            exportCtx.scale(scaleX, scaleY);

            const tempSelected = this.selectedElement;
            this.selectedElement = null;

            panel.elements.forEach(element => {
                if (element.type === 'figure') {
                    const figure = new StickFigure(exportCtx, element.x, element.y, 1);
                    figure.setOptions({
                        hasHair: element.hasHair,
                        hasDress: element.hasDress,
                        color: element.color
                    });
                    figure.draw(element.pose);
                } else if (element.type === 'bubble') {
                    this.drawSpeechBubble(
                        exportCtx,
                        element.x,
                        element.y,
                        element.width,
                        element.height,
                        element.text,
                        element.bubbleType
                    );
                } else if (element.type === 'text') {
                    exportCtx.font = '16px Comic Sans MS, Arial';
                    exportCtx.fillStyle = element.color;
                    exportCtx.textAlign = 'center';
                    exportCtx.fillText(element.text, element.x, element.y);
                }
            });

            this.selectedElement = tempSelected;
            exportCtx.restore();
        });

        try {
            const blob = await new Promise(resolve => exportCanvas.toBlob(resolve));
            await navigator.clipboard.write([
                new ClipboardItem({
                    'image/png': blob
                })
            ]);
            this.showHint('Comic copied to clipboard!');
        } catch (err) {
            console.error('Failed to copy to clipboard:', err);
            this.showHint('Copy to clipboard failed. Try downloading instead.');
        }
    }

    saveToRecent(dataUrl) {
        let recentComics = JSON.parse(localStorage.getItem('recentComics') || '[]');
        recentComics.unshift({
            data: dataUrl,
            timestamp: Date.now()
        });
        recentComics = recentComics.slice(0, 5);
        localStorage.setItem('recentComics', JSON.stringify(recentComics));
        this.displayRecentComics();
    }

    displayRecentComics() {
        const container = document.getElementById('recentComics');
        const recentComics = JSON.parse(localStorage.getItem('recentComics') || '[]');

        container.innerHTML = '';
        recentComics.forEach((comic, index) => {
            const div = document.createElement('div');
            div.className = 'recent-comic';
            div.style.backgroundImage = `url(${comic.data})`;
            div.title = `Click to load comic ${index + 1}`;

            div.addEventListener('click', () => {
                if (confirm('Load this comic? Current work will be lost.')) {
                    this.loadComic(comic.data);
                }
            });

            container.appendChild(div);
        });
    }

    loadComic(dataUrl) {
        const img = new Image();
        img.onload = () => {
            const panel = this.panels[0];
            if (panel) {
                panel.ctx.clearRect(0, 0, panel.canvas.width, panel.canvas.height);
                panel.ctx.drawImage(img, 0, 0, panel.canvas.width, panel.canvas.height);
            }
        };
        img.src = dataUrl;
    }
}

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const comicMaker = new ComicMaker();
    comicMaker.displayRecentComics();
    window.comicMaker = comicMaker;
});