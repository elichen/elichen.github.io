class GameVisualizer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.trainer = new PongTrainer();
        this.isRunning = false;
        this.mode = 'training'; // 'training' or 'testing'
        this.setupEventListeners();
        console.log("GameVisualizer initialized");
        this.animationFrameId = null;
        
        // Start training immediately
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

    startMode(mode) {
        this.isRunning = true;
        document.getElementById('toggle-mode').textContent = 
            mode === 'training' ? 'Switch to Testing' : 'Switch to Training';
            
        if (mode === 'training') {
            console.log("Starting training mode");
            this.startTraining();
        } else {
            console.log("Starting testing mode");
            this.startTesting();
        }
        this.startAnimation();
    }

    async startTraining() {
        this.trainer.resume();
        try {
            await this.trainer.train(1000);
        } catch (error) {
            console.error("Error during training:", error);
            this.stop();
        }
    }

    async startTesting() {
        this.trainer.setTestingMode(true);
        try {
            await this.trainer.test();
        } catch (error) {
            console.error("Error during testing:", error);
            this.stop();
        }
    }

    stop() {
        console.log("Stopping current mode");
        this.isRunning = false;
        this.trainer.pause();
        this.trainer.setTestingMode(false);
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
    }
}

// Initialize when page loads
window.addEventListener('load', () => {
    const canvas = document.getElementById('game-canvas');
    const visualizer = new GameVisualizer(canvas);
    visualizer.draw();
}); 