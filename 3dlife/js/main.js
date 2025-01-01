class Conway3D {
    constructor() {
        this.lastTime = 0;
        this.frameCount = 0;
        this.lastFpsUpdate = 0;
        this.density = 0.20;
        
        this.init();
        this.setupEventListeners();
    }

    init() {
        this.renderer = new Renderer();
        this.grid = new Grid(this.renderer.WORLD_SIZE, this.renderer.WORLD_SIZE, this.renderer.WORLD_SIZE);
        this.gameLogic = new GameLogic(this.grid);
        this.cameraController = new CameraController(this.renderer.camera, this.renderer.renderer.domElement);
        
        this.grid.randomize(this.density);
        this.animate();
    }

    setupEventListeners() {
        document.getElementById('reset').addEventListener('click', () => {
            this.grid.reset();
            this.grid.randomize(this.density);
        });

        const densitySlider = document.getElementById('density');
        const densityValue = document.getElementById('densityValue');
        
        densitySlider.addEventListener('input', (e) => {
            this.density = parseFloat(e.target.value);
            densityValue.textContent = this.density.toFixed(2);
        });
    }

    updateFPS(now) {
        this.frameCount++;
        if (now - this.lastFpsUpdate > 1000) {
            const fps = Math.round(this.frameCount * 1000 / (now - this.lastFpsUpdate));
            document.getElementById('fps').textContent = fps;
            this.frameCount = 0;
            this.lastFpsUpdate = now;
        }
    }

    animate(now = 0) {
        requestAnimationFrame(this.animate.bind(this));
        
        this.updateFPS(now);

        // Always update if enough time has passed
        if (now - this.lastTime > 100) {
            this.gameLogic.update();
            this.lastTime = now;
        }

        document.getElementById('cellCount').textContent = this.grid.getAliveCellCount();
        
        this.renderer.render(this.grid);
    }
}

// Start the application when the page loads
window.addEventListener('load', () => {
    new Conway3D();
}); 