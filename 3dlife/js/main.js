class Conway3D {
    constructor() {
        this.isRunning = false;
        this.lastTime = 0;
        this.frameCount = 0;
        this.lastFpsUpdate = 0;
        
        this.init();
        this.setupEventListeners();
    }

    init() {
        this.renderer = new Renderer();
        this.grid = new Grid(this.renderer.WORLD_SIZE, this.renderer.WORLD_SIZE, this.renderer.WORLD_SIZE);
        this.gameLogic = new GameLogic(this.grid);
        this.cameraController = new CameraController(this.renderer.camera, this.renderer.renderer.domElement);
        
        this.grid.randomize(0.05);  // Changed from 0.4 to 0.05 (5% initial fill)
        this.animate();
    }

    setupEventListeners() {
        document.getElementById('startStop').addEventListener('click', () => {
            this.isRunning = !this.isRunning;
        });

        document.getElementById('reset').addEventListener('click', () => {
            this.grid.reset();
            this.grid.randomize(0.05);  // Now matches initial load
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

        if (this.isRunning && now - this.lastTime > 100) {
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