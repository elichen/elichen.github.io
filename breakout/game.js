class Game {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.width = canvas.width;
        this.height = canvas.height;
        this.paddle = { width: 75, height: 10, x: this.width / 2 - 37.5, y: this.height - 20 };
        this.ball = { radius: 5, x: this.width / 2, y: this.height - 30, dx: 2, dy: -2 };
        this.bricks = [];
        this.score = 0;
        this.gameOver = false;
        this.colors = ['#FF0000', '#FF7F00', '#FFFF00', '#00FF00', '#0000FF', '#4B0082', '#9400D3'];
        this.initBricks();
    }

    initBricks() {
        const brickRowCount = 6;
        const brickColumnCount = 13;
        const brickWidth = 61; // Slightly increased to fill the entire width
        const brickHeight = 20;
        const brickPadding = 2;
        const brickOffsetTop = 30;
        const brickOffsetLeft = 0; // Changed from 15 to 0

        for (let r = 0; r < brickRowCount; r++) {
            for (let c = 0; c < brickColumnCount; c++) {
                const brickX = c * (brickWidth + brickPadding) + brickOffsetLeft;
                const brickY = r * (brickHeight + brickPadding) + brickOffsetTop;
                this.bricks.push({
                    x: brickX,
                    y: brickY,
                    width: brickWidth,
                    height: brickHeight,
                    status: 1,
                    color: this.colors[r % this.colors.length]
                });
            }
        }
    }

    movePaddle(direction) {
        const speed = 15; // Increased from 7 to 15
        if (direction === 'left' && this.paddle.x > 0) {
            this.paddle.x -= speed;
        } else if (direction === 'right' && this.paddle.x + this.paddle.width < this.width) {
            this.paddle.x += speed;
        }

        // Ensure the paddle doesn't go out of bounds
        this.paddle.x = Math.max(0, Math.min(this.width - this.paddle.width, this.paddle.x));
    }

    update() {
        if (this.gameOver) return;

        // Move the ball
        this.ball.x += this.ball.dx;
        this.ball.y += this.ball.dy;

        // Ball collision with walls
        if (this.ball.x + this.ball.radius > this.width || this.ball.x - this.ball.radius < 0) {
            this.ball.dx = -this.ball.dx;
        }
        if (this.ball.y - this.ball.radius < 0) {
            this.ball.dy = -this.ball.dy;
        }

        // Ball collision with paddle
        if (
            this.ball.y + this.ball.radius > this.paddle.y &&
            this.ball.x > this.paddle.x &&
            this.ball.x < this.paddle.x + this.paddle.width
        ) {
            this.ball.dy = -this.ball.dy;
        }

        // Ball collision with bricks
        for (let i = 0; i < this.bricks.length; i++) {
            const brick = this.bricks[i];
            if (brick.status === 1) {
                if (
                    this.ball.x > brick.x &&
                    this.ball.x < brick.x + brick.width &&
                    this.ball.y > brick.y &&
                    this.ball.y < brick.y + brick.height
                ) {
                    this.ball.dy = -this.ball.dy;
                    brick.status = 0;
                    this.score++;
                    if (this.score === this.bricks.length) {
                        this.gameOver = true;
                    }
                }
            }
        }

        // Game over if ball touches bottom
        if (this.ball.y + this.ball.radius > this.height) {
            this.gameOver = true;
        }
    }

    draw() {
        this.ctx.clearRect(0, 0, this.width, this.height);

        // Draw bricks
        for (let i = 0; i < this.bricks.length; i++) {
            const brick = this.bricks[i];
            if (brick.status === 1) {
                this.ctx.fillStyle = brick.color;
                this.ctx.fillRect(brick.x, brick.y, brick.width, brick.height);
            }
        }

        // Draw paddle
        this.ctx.fillStyle = '#FFFFFF';
        this.ctx.fillRect(this.paddle.x, this.paddle.y, this.paddle.width, this.paddle.height);

        // Draw ball
        this.ctx.beginPath();
        this.ctx.arc(this.ball.x, this.ball.y, this.ball.radius, 0, Math.PI * 2);
        this.ctx.fillStyle = '#FFFFFF';
        this.ctx.fill();
        this.ctx.closePath();

        // Update score
        document.getElementById('scoreValue').textContent = this.score;
    }

    reset() {
        this.paddle = { width: 75, height: 10, x: this.width / 2 - 37.5, y: this.height - 20 };
        this.ball = { radius: 5, x: this.width / 2, y: this.height - 30, dx: 2, dy: -2 };
        this.bricks = [];
        this.score = 0;
        this.gameOver = false;
        this.initBricks();
    }
}