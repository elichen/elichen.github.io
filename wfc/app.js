class WFCVisualizer {
    constructor() {
        this.canvas = document.getElementById('wfcCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.cellSize = 20;
        this.gridSize = 25;
        this.wfc = null;
        this.isPlaying = false;
        this.animationSpeed = 50;
        this.lastStepTime = 0;
        
        this.initializeElements();
        this.initialize();
        this.bindEvents();
        this.animate();
    }
    
    initializeElements() {
        this.generateBtn = document.getElementById('generateBtn');
        this.stepBtn = document.getElementById('stepBtn');
        this.playBtn = document.getElementById('playBtn');
        this.resetBtn = document.getElementById('resetBtn');
        this.speedSlider = document.getElementById('speedSlider');
        this.speedValue = document.getElementById('speedValue');
        this.sizeSlider = document.getElementById('sizeSlider');
        this.sizeValue = document.getElementById('sizeValue');
        this.entropyValue = document.getElementById('entropyValue');
        this.collapsedValue = document.getElementById('collapsedValue');
        this.totalValue = document.getElementById('totalValue');
    }
    
    initialize() {
        this.wfc = new WaveFunctionCollapse(this.gridSize, this.gridSize);
        this.canvas.width = this.gridSize * this.cellSize;
        this.canvas.height = this.gridSize * this.cellSize;
        this.totalValue.textContent = this.wfc.totalCells;
        this.updateInfo();
    }
    
    bindEvents() {
        this.generateBtn.addEventListener('click', () => this.generate());
        this.stepBtn.addEventListener('click', () => this.step());
        this.playBtn.addEventListener('click', () => this.togglePlay());
        this.resetBtn.addEventListener('click', () => this.reset());
        
        this.speedSlider.addEventListener('input', (e) => {
            this.animationSpeed = parseInt(e.target.value);
            this.speedValue.textContent = this.animationSpeed;
        });
        
        this.sizeSlider.addEventListener('input', (e) => {
            this.gridSize = parseInt(e.target.value);
            this.sizeValue.textContent = this.gridSize;
            this.initialize();
        });
        
        this.canvas.addEventListener('click', (e) => this.handleCanvasClick(e));
    }
    
    handleCanvasClick(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = Math.floor((e.clientX - rect.left) / this.cellSize);
        const y = Math.floor((e.clientY - rect.top) / this.cellSize);
        
        if (this.wfc.manualCollapse(x, y)) {
            this.updateInfo();
        }
    }
    
    generate() {
        this.isPlaying = false;
        this.playBtn.textContent = 'Play';
        this.wfc.generate();
        this.updateInfo();
    }
    
    step() {
        if (this.wfc.step()) {
            this.updateInfo();
        }
    }
    
    togglePlay() {
        this.isPlaying = !this.isPlaying;
        this.playBtn.textContent = this.isPlaying ? 'Pause' : 'Play';
    }
    
    reset() {
        this.isPlaying = false;
        this.playBtn.textContent = 'Play';
        this.wfc.reset();
        this.updateInfo();
    }
    
    updateInfo() {
        this.entropyValue.textContent = this.wfc.getTotalEntropy();
        this.collapsedValue.textContent = this.wfc.collapsedCount;
    }
    
    drawTile(x, y, tileIndex) {
        const tile = this.wfc.tiles[tileIndex];
        const px = x * this.cellSize;
        const py = y * this.cellSize;
        
        // Clear cell
        this.ctx.fillStyle = '#000000';
        this.ctx.fillRect(px, py, this.cellSize, this.cellSize);
        
        // Draw tile based on type
        this.ctx.fillStyle = '#7fbf7f';
        const padding = 2;
        const innerSize = this.cellSize - padding * 2;
        
        switch (tile.type) {
            case 'empty':
                // Draw nothing
                break;
                
            case 'wall-h':
                this.ctx.fillRect(px, py + this.cellSize/3, this.cellSize, this.cellSize/3);
                break;
                
            case 'wall-v':
                this.ctx.fillRect(px + this.cellSize/3, py, this.cellSize/3, this.cellSize);
                break;
                
            case 'corner-tl':
                this.ctx.fillRect(px + this.cellSize/3, py + this.cellSize/3, this.cellSize*2/3, this.cellSize/3);
                this.ctx.fillRect(px + this.cellSize/3, py + this.cellSize/3, this.cellSize/3, this.cellSize*2/3);
                break;
                
            case 'corner-tr':
                this.ctx.fillRect(px, py + this.cellSize/3, this.cellSize*2/3, this.cellSize/3);
                this.ctx.fillRect(px + this.cellSize/3, py + this.cellSize/3, this.cellSize/3, this.cellSize*2/3);
                break;
                
            case 'corner-bl':
                this.ctx.fillRect(px + this.cellSize/3, py, this.cellSize/3, this.cellSize*2/3);
                this.ctx.fillRect(px + this.cellSize/3, py + this.cellSize/3, this.cellSize*2/3, this.cellSize/3);
                break;
                
            case 'corner-br':
                this.ctx.fillRect(px + this.cellSize/3, py, this.cellSize/3, this.cellSize*2/3);
                this.ctx.fillRect(px, py + this.cellSize/3, this.cellSize*2/3, this.cellSize/3);
                break;
                
            case 'junction-t':
                this.ctx.fillRect(px, py + this.cellSize/3, this.cellSize, this.cellSize/3);
                this.ctx.fillRect(px + this.cellSize/3, py + this.cellSize/3, this.cellSize/3, this.cellSize*2/3);
                break;
                
            case 'junction-r':
                this.ctx.fillRect(px + this.cellSize/3, py, this.cellSize/3, this.cellSize);
                this.ctx.fillRect(px, py + this.cellSize/3, this.cellSize*2/3, this.cellSize/3);
                break;
                
            case 'junction-b':
                this.ctx.fillRect(px, py + this.cellSize/3, this.cellSize, this.cellSize/3);
                this.ctx.fillRect(px + this.cellSize/3, py, this.cellSize/3, this.cellSize*2/3);
                break;
                
            case 'junction-l':
                this.ctx.fillRect(px + this.cellSize/3, py, this.cellSize/3, this.cellSize);
                this.ctx.fillRect(px + this.cellSize/3, py + this.cellSize/3, this.cellSize*2/3, this.cellSize/3);
                break;
                
            case 'cross':
                this.ctx.fillRect(px + this.cellSize/3, py, this.cellSize/3, this.cellSize);
                this.ctx.fillRect(px, py + this.cellSize/3, this.cellSize, this.cellSize/3);
                break;
        }
        
        // Draw cell border
        this.ctx.strokeStyle = '#1a4d1a';
        this.ctx.strokeRect(px, py, this.cellSize, this.cellSize);
    }
    
    render() {
        // Clear canvas
        this.ctx.fillStyle = '#000000';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw grid
        for (let y = 0; y < this.gridSize; y++) {
            for (let x = 0; x < this.gridSize; x++) {
                const cell = this.wfc.getCell(x, y);
                
                if (cell.collapsed) {
                    this.drawTile(x, y, cell.options[0]);
                } else {
                    // Draw entropy visualization
                    const entropy = cell.entropy;
                    const maxEntropy = this.wfc.tiles.length;
                    const alpha = 1 - (entropy / maxEntropy);
                    
                    this.ctx.fillStyle = `rgba(127, 191, 127, ${alpha * 0.3})`;
                    this.ctx.fillRect(
                        x * this.cellSize,
                        y * this.cellSize,
                        this.cellSize,
                        this.cellSize
                    );
                    
                    // Draw entropy number for low entropy cells
                    if (entropy < 5 && entropy > 0) {
                        this.ctx.fillStyle = '#7fbf7f';
                        this.ctx.font = '10px Source Code Pro';
                        this.ctx.textAlign = 'center';
                        this.ctx.textBaseline = 'middle';
                        this.ctx.fillText(
                            entropy.toString(),
                            x * this.cellSize + this.cellSize / 2,
                            y * this.cellSize + this.cellSize / 2
                        );
                    }
                    
                    // Draw cell border
                    this.ctx.strokeStyle = '#1a4d1a';
                    this.ctx.strokeRect(
                        x * this.cellSize,
                        y * this.cellSize,
                        this.cellSize,
                        this.cellSize
                    );
                }
            }
        }
    }
    
    animate(timestamp) {
        // Handle animation
        if (this.isPlaying && timestamp - this.lastStepTime > (100 - this.animationSpeed) * 10) {
            this.step();
            this.lastStepTime = timestamp;
            
            // Stop playing if generation is complete
            if (this.wfc.collapsedCount >= this.wfc.totalCells) {
                this.isPlaying = false;
                this.playBtn.textContent = 'Play';
            }
        }
        
        // Render
        this.render();
        
        // Continue animation loop
        requestAnimationFrame((t) => this.animate(t));
    }
}

// Initialize visualizer when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new WFCVisualizer();
});