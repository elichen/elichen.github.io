class GameVisualizer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.trainer = new PongTrainer();
        this.isTraining = false;
        this.setupEventListeners();
        console.log("GameVisualizer initialized");
        this.animationFrameId = null;
    }

    setupEventListeners() {
        document.getElementById('start-training').addEventListener('click', () => {
            console.log("Start training button clicked");
            this.startTraining();
        });
        document.getElementById('pause-training').addEventListener('click', () => {
            console.log("Pause training button clicked");
            this.pauseTraining();
        });
    }

    async startTraining() {
        console.log("startTraining called, isTraining:", this.isTraining);
        if (!this.isTraining) {
            this.isTraining = true;
            this.trainer.resume();
            this.startAnimation();
            try {
                await this.trainer.train(1000);
            } catch (error) {
                console.error("Error during training:", error);
                this.pauseTraining();
            }
        }
    }

    pauseTraining() {
        console.log("Pausing training");
        this.isTraining = false;
        this.trainer.pause();
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
    }

    startAnimation() {
        const animate = () => {
            this.draw();
            if (this.isTraining) {
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

        // Continue animation if training
        if (this.isTraining) {
            requestAnimationFrame(() => this.draw());
        }
    }
}

// Initialize when page loads
window.addEventListener('load', () => {
    const canvas = document.getElementById('game-canvas');
    const visualizer = new GameVisualizer(canvas);
    visualizer.draw();
}); 