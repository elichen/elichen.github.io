class GameVisualizer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.trainer = new PongTrainer();
        this.isRunning = false;
        this.mode = 'training'; // 'training' or 'testing'
        
        // Initialize keyboard state first
        this.keys = {
            ArrowUp: false,
            ArrowDown: false
        };
        
        // Setup all event listeners
        this.setupKeyboardControls();
        this.setupEventListeners();
        
        console.log("GameVisualizer initialized");
        this.animationFrameId = null;
        
        // Start training after everything is initialized
        this.startMode('training');
    }

    setupEventListeners() {
        document.getElementById('toggle-mode').addEventListener('click', () => {
            // Stop current mode
            this.stop();
            // Switch and start new mode
            this.mode = this.mode === 'training' ? 'testing' : 'training';
            this.startMode(this.mode);
        });
    }

    setupKeyboardControls() {
        window.addEventListener('keydown', (e) => {
            if (this.keys.hasOwnProperty(e.key)) {
                this.keys[e.key] = true;
            }
        });

        window.addEventListener('keyup', (e) => {
            if (this.keys.hasOwnProperty(e.key)) {
                this.keys[e.key] = false;
            }
        });
    }

    // Add method to get human action
    getHumanAction() {
        if (this.keys.ArrowUp) return 0;     // Move up (maps to -1 in paddle)
        if (this.keys.ArrowDown) return 2;    // Move down (maps to 1 in paddle)
        return 1;                             // Stay still (maps to 0 in paddle)
    }

    startMode(mode) {
        this.isRunning = true;
        this.mode = mode;  // Make sure mode is set before anything else
        
        document.getElementById('toggle-mode').textContent = 
            mode === 'training' ? 'Switch to Testing' : 'Switch to Training';
            
        if (mode === 'training') {
            console.log("Starting training mode");
            this.trainer.setTestingMode(false, null);
            this.startTraining();
        } else {
            console.log("Starting testing mode (Human vs AI)");
            // Set testing mode first, then start animation, then start testing
            this.trainer.setTestingMode(true, this);
            this.startAnimation();
            this.startTesting();
        }
        
        if (mode === 'training') {
            this.startAnimation();
        }
    }

    async startTraining() {
        this.trainer.resume();
        try {
            await this.trainer.train();
        } catch (error) {
            console.error("Error during training:", error);
            this.stop();
        }
    }

    async startTesting() {
        try {
            // Small delay to ensure everything is initialized
            await new Promise(resolve => setTimeout(resolve, 100));
            await this.trainer.test();
        } catch (error) {
            console.error("Error during testing:", error);
            this.stop();
        }
    }

    stop() {
        console.log("Stopping current mode");
        this.isRunning = false;
        if (this.mode === 'training') {
            this.trainer.pause();
        }
        this.trainer.setTestingMode(false, null);
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
    }

    startAnimation() {
        const animate = () => {
            this.draw();
            if (this.isRunning) {
                this.animationFrameId = requestAnimationFrame(animate);
            }
        };
        animate();
    }

    draw() {
        const game = this.trainer.env.game;
        const ctx = this.ctx;

        // Clear canvas
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw middle line
        ctx.strokeStyle = 'white';
        ctx.setLineDash([5, 15]);
        ctx.beginPath();
        ctx.moveTo(this.canvas.width/2, 0);
        ctx.lineTo(this.canvas.width/2, this.canvas.height);
        ctx.stroke();
        ctx.setLineDash([]);

        // Draw paddles
        ctx.fillStyle = 'white';
        ctx.fillRect(game.leftPaddle.x, game.leftPaddle.y, 
                    game.leftPaddle.width, game.leftPaddle.height);
        ctx.fillRect(game.rightPaddle.x, game.rightPaddle.y, 
                    game.rightPaddle.width, game.rightPaddle.height);

        // Draw ball
        ctx.beginPath();
        ctx.arc(game.ball.x, game.ball.y, game.ball.radius, 0, Math.PI * 2);
        ctx.fill();

        // Draw scores
        ctx.font = '32px Arial';
        ctx.fillText(game.leftPaddle.score, this.canvas.width/4, 50);
        ctx.fillText(game.rightPaddle.score, 3*this.canvas.width/4, 50);

        // Update episode counter
        document.getElementById('episode-counter').textContent = 
            `Episode: ${this.trainer.episodeCount}`;

        // Draw mode indicator
        ctx.font = '20px Arial';
        ctx.fillText(this.mode === 'training' ? 'Training Mode' : 'Testing Mode (Human vs AI)', 10, 30);
    }
}

// Initialize when page loads
window.addEventListener('load', () => {
    const canvas = document.getElementById('game-canvas');
    const visualizer = new GameVisualizer(canvas);
    visualizer.draw();
}); 