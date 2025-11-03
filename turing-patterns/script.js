// Turing Pattern Generator using Gray-Scott Reaction-Diffusion Model

class TuringPatternSimulation {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');

        // Grid dimensions - larger for full screen
        this.gridSize = 400;

        // Set canvas to window size
        this.resizeCanvas();

        // Simulation parameters (Gray-Scott model)
        this.feedRate = 0.055;
        this.killRate = 0.062;
        this.diffusionRateA = 1.0;
        this.diffusionRateB = 0.5;
        this.deltaT = 1.0;

        // Grids for chemicals A and B (current and next states)
        this.gridA = [];
        this.gridB = [];
        this.nextA = [];
        this.nextB = [];

        // Visualization
        this.colorScheme = 'grayscale';
        this.showChemicalB = true;

        // Animation
        this.isRunning = true;
        this.animationId = null;

        this.initialize();
        this.setupEventListeners();
    }

    resizeCanvas() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
    }

    initialize() {
        // Initialize grids
        for (let i = 0; i < this.gridSize; i++) {
            this.gridA[i] = [];
            this.gridB[i] = [];
            this.nextA[i] = [];
            this.nextB[i] = [];

            for (let j = 0; j < this.gridSize; j++) {
                // Start with chemical A everywhere
                this.gridA[i][j] = 1.0;
                this.gridB[i][j] = 0.0;
                this.nextA[i][j] = 1.0;
                this.nextB[i][j] = 0.0;
            }
        }

        // Add some random initial seeds
        this.addRandomSeeds(8);
    }

    addRandomSeeds(count) {
        for (let n = 0; n < count; n++) {
            const x = Math.floor(Math.random() * this.gridSize);
            const y = Math.floor(Math.random() * this.gridSize);
            this.addSeed(x, y, 15);
        }
    }

    addSeed(centerX, centerY, radius = 10) {
        // Add a circular region of chemical B
        for (let i = -radius; i <= radius; i++) {
            for (let j = -radius; j <= radius; j++) {
                if (i * i + j * j <= radius * radius) {
                    const x = (centerX + i + this.gridSize) % this.gridSize;
                    const y = (centerY + j + this.gridSize) % this.gridSize;
                    this.gridB[x][y] = 1.0;
                    this.gridA[x][y] = 0.0;
                }
            }
        }
    }

    laplacian(grid, x, y) {
        // Calculate discrete Laplacian with periodic boundary conditions
        let sum = 0;

        // Direct neighbors (weight = 0.2)
        sum += grid[(x + 1) % this.gridSize][y] * 0.2;
        sum += grid[(x - 1 + this.gridSize) % this.gridSize][y] * 0.2;
        sum += grid[x][(y + 1) % this.gridSize] * 0.2;
        sum += grid[x][(y - 1 + this.gridSize) % this.gridSize] * 0.2;

        // Diagonal neighbors (weight = 0.05)
        sum += grid[(x + 1) % this.gridSize][(y + 1) % this.gridSize] * 0.05;
        sum += grid[(x - 1 + this.gridSize) % this.gridSize][(y + 1) % this.gridSize] * 0.05;
        sum += grid[(x + 1) % this.gridSize][(y - 1 + this.gridSize) % this.gridSize] * 0.05;
        sum += grid[(x - 1 + this.gridSize) % this.gridSize][(y - 1 + this.gridSize) % this.gridSize] * 0.05;

        // Center point
        sum += grid[x][y] * -1;

        return sum;
    }

    step() {
        // Update the reaction-diffusion system
        for (let x = 0; x < this.gridSize; x++) {
            for (let y = 0; y < this.gridSize; y++) {
                const a = this.gridA[x][y];
                const b = this.gridB[x][y];

                // Reaction term
                const reaction = a * b * b;

                // Laplacian for diffusion
                const laplaceA = this.laplacian(this.gridA, x, y);
                const laplaceB = this.laplacian(this.gridB, x, y);

                // Gray-Scott equations
                this.nextA[x][y] = a + this.deltaT * (
                    this.diffusionRateA * laplaceA -
                    reaction +
                    this.feedRate * (1 - a)
                );

                this.nextB[x][y] = b + this.deltaT * (
                    this.diffusionRateB * laplaceB +
                    reaction -
                    (this.killRate + this.feedRate) * b
                );

                // Clamp values
                this.nextA[x][y] = Math.max(0, Math.min(1, this.nextA[x][y]));
                this.nextB[x][y] = Math.max(0, Math.min(1, this.nextB[x][y]));
            }
        }

        // Swap grids
        [this.gridA, this.nextA] = [this.nextA, this.gridA];
        [this.gridB, this.nextB] = [this.nextB, this.gridB];
    }

    render() {
        // Create image data at grid resolution
        const imageData = this.ctx.createImageData(this.gridSize, this.gridSize);
        const data = imageData.data;

        for (let x = 0; x < this.gridSize; x++) {
            for (let y = 0; y < this.gridSize; y++) {
                const idx = (y * this.gridSize + x) * 4;

                // Get chemical concentrations
                const a = this.gridA[x][y];
                const b = this.showChemicalB ? this.gridB[x][y] : this.gridA[x][y];

                // Apply color scheme
                const color = this.applyColorScheme(b);

                data[idx] = color[0];
                data[idx + 1] = color[1];
                data[idx + 2] = color[2];
                data[idx + 3] = 255;
            }
        }

        // Create temporary canvas at grid resolution
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = this.gridSize;
        tempCanvas.height = this.gridSize;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.putImageData(imageData, 0, 0);

        // Clear and scale to full window
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Calculate scaling to fill screen while maintaining aspect ratio
        const scale = Math.max(this.canvas.width / this.gridSize, this.canvas.height / this.gridSize);
        const offsetX = (this.canvas.width - this.gridSize * scale) / 2;
        const offsetY = (this.canvas.height - this.gridSize * scale) / 2;

        this.ctx.imageSmoothingEnabled = false;
        this.ctx.drawImage(tempCanvas, offsetX, offsetY, this.gridSize * scale, this.gridSize * scale);
    }

    applyColorScheme(value) {
        // Clamp value between 0 and 1
        value = Math.max(0, Math.min(1, value));

        switch (this.colorScheme) {
            case 'grayscale':
                const gray = Math.floor((1 - value) * 255);
                return [gray, gray, gray];

            case 'heatmap':
                // Red to yellow to white
                if (value < 0.5) {
                    const t = value * 2;
                    return [255, Math.floor(t * 255), Math.floor(t * t * 255)];
                } else {
                    const t = (value - 0.5) * 2;
                    return [255, 255, Math.floor(128 + t * 127)];
                }

            case 'ocean':
                // Deep blue to cyan to white
                if (value < 0.5) {
                    const t = value * 2;
                    return [Math.floor(t * 100), Math.floor(t * 150), 200 + Math.floor(t * 55)];
                } else {
                    const t = (value - 0.5) * 2;
                    return [100 + Math.floor(t * 155), 150 + Math.floor(t * 105), 255];
                }

            case 'plasma':
                // Purple to pink to yellow
                const r = Math.floor(Math.min(255, 50 + value * 400));
                const g = Math.floor(Math.min(255, value * value * 500));
                const b = Math.floor(255 - value * 200);
                return [r, g, b];

            case 'viridis':
                // Dark blue to green to yellow
                const vr = Math.floor(68 + value * 187);
                const vg = Math.floor(1 + value * 200 + value * value * 54);
                const vb = Math.floor(84 - value * 84);
                return [vr, vg, vb];

            default:
                return [0, 0, 0];
        }
    }

    animate() {
        if (this.isRunning) {
            // Run multiple steps per frame for faster evolution
            for (let i = 0; i < 10; i++) {
                this.step();
            }
            this.render();
        }

        this.animationId = requestAnimationFrame(() => this.animate());
    }

    setupEventListeners() {
        // Canvas click to add seeds
        this.canvas.addEventListener('click', (e) => {
            // Calculate position in grid space
            const rect = this.canvas.getBoundingClientRect();
            const scale = Math.max(this.canvas.width / this.gridSize, this.canvas.height / this.gridSize);
            const offsetX = (this.canvas.width - this.gridSize * scale) / 2;
            const offsetY = (this.canvas.height - this.gridSize * scale) / 2;

            const x = Math.floor((e.clientX - rect.left - offsetX) / scale);
            const y = Math.floor((e.clientY - rect.top - offsetY) / scale);

            if (x >= 0 && x < this.gridSize && y >= 0 && y < this.gridSize) {
                this.addSeed(x, y, 12);
            }
        });

        // Window resize
        window.addEventListener('resize', () => {
            this.resizeCanvas();
        });
    }

    reset() {
        this.initialize();
    }

    clear() {
        // Clear to all chemical A
        for (let i = 0; i < this.gridSize; i++) {
            for (let j = 0; j < this.gridSize; j++) {
                this.gridA[i][j] = 1.0;
                this.gridB[i][j] = 0.0;
            }
        }
    }

    setParameters(feedRate, killRate) {
        this.feedRate = feedRate;
        this.killRate = killRate;
    }

    setDiffusionRates(diffA, diffB) {
        this.diffusionRateA = diffA;
        this.diffusionRateB = diffB;
    }

    setColorScheme(scheme) {
        this.colorScheme = scheme;
    }

    toggleChemicalDisplay() {
        this.showChemicalB = !this.showChemicalB;
    }

    start() {
        this.isRunning = true;
        if (!this.animationId) {
            this.animate();
        }
    }

    stop() {
        this.isRunning = false;
    }

    destroy() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
    }
}

// Initialize simulation when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('patternCanvas');
    const sim = new TuringPatternSimulation(canvas);

    // Start animation
    sim.animate();

    // Handle scroll-based title visibility
    const titleOverlay = document.querySelector('.title-overlay');
    let lastScrollY = 0;

    window.addEventListener('scroll', () => {
        const scrollY = window.scrollY;

        // Fade title when scrolling down
        if (scrollY > 100) {
            titleOverlay.classList.add('title-hidden');
        } else {
            titleOverlay.classList.remove('title-hidden');
        }

        lastScrollY = scrollY;
    });

    // Control buttons
    document.getElementById('playPause').addEventListener('click', (e) => {
        if (sim.isRunning) {
            sim.stop();
            e.target.textContent = 'Play';
        } else {
            sim.start();
            e.target.textContent = 'Pause';
        }
    });

    document.getElementById('reset').addEventListener('click', () => {
        sim.reset();
    });

    document.getElementById('clear').addEventListener('click', () => {
        sim.clear();
    });

    // Parameter sliders
    document.getElementById('feedRate').addEventListener('input', (e) => {
        const value = parseFloat(e.target.value);
        document.getElementById('feedRateValue').textContent = value.toFixed(3);
        sim.setParameters(value, sim.killRate);
    });

    document.getElementById('killRate').addEventListener('input', (e) => {
        const value = parseFloat(e.target.value);
        document.getElementById('killRateValue').textContent = value.toFixed(3);
        sim.setParameters(sim.feedRate, value);
    });

    document.getElementById('diffusionA').addEventListener('input', (e) => {
        const value = parseFloat(e.target.value);
        document.getElementById('diffusionAValue').textContent = value.toFixed(2);
        sim.setDiffusionRates(value, sim.diffusionRateB);
    });

    document.getElementById('diffusionB').addEventListener('input', (e) => {
        const value = parseFloat(e.target.value);
        document.getElementById('diffusionBValue').textContent = value.toFixed(2);
        sim.setDiffusionRates(sim.diffusionRateA, value);
    });

    // Preset buttons - updated selector
    document.querySelectorAll('.preset-btn').forEach(button => {
        button.addEventListener('click', () => {
            const f = parseFloat(button.dataset.f);
            const k = parseFloat(button.dataset.k);

            // Update sliders and values
            document.getElementById('feedRate').value = f;
            document.getElementById('feedRateValue').textContent = f.toFixed(3);
            document.getElementById('killRate').value = k;
            document.getElementById('killRateValue').textContent = k.toFixed(3);

            // Apply to simulation
            sim.setParameters(f, k);
            sim.reset();
        });
    });

    // Color scheme selector
    document.getElementById('colorScheme').addEventListener('change', (e) => {
        sim.setColorScheme(e.target.value);
    });

    // Chemical display toggle
    document.getElementById('showChemicalB').addEventListener('change', (e) => {
        sim.showChemicalB = e.target.checked;
    });

    // Cleanup on page unload
    window.addEventListener('beforeunload', () => {
        sim.destroy();
    });
});