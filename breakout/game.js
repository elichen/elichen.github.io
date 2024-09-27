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
        
        // Initialize ball with random trajectory going downwards, starting below the blocks
        const angle = this.getRandomAngle();
        const speed = 4;
        this.ball = {
            radius: 5,
            x: this.width / 2,
            y: this.height / 2,  // Start in the middle of the screen, below the blocks
            dx: Math.cos(angle) * speed,
            dy: Math.abs(Math.sin(angle) * speed)  // Use absolute value to ensure downward motion
        };

        this.bricks = [];
        this.score = 0;
        this.gameOver = false;
        this.initBricks();
    }

    getRandomAngle() {
        // Generate a random angle between 210 and 330 degrees (in radians)
        return (Math.random() * 120 + 210) * Math.PI / 180;
    }

    // Add a helper method to resize the state
    resizeState(state, newWidth = 42, newHeight = 42) {
        const resizedState = [];
        const scaleX = state[0].length / newWidth;
        const scaleY = state.length / newHeight;

        for (let y = 0; y < newHeight; y++) {
            const row = [];
            for (let x = 0; x < newWidth; x++) {
                const origX = Math.floor(x * scaleX);
                const origY = Math.floor(y * scaleY);
                row.push(state[origY][origX]);
            }
            resizedState.push(row);
        }
        return resizedState;
    }

    getState() {
        const state = new Array(this.height).fill(0).map(() => new Array(this.width).fill(0));
        
        // Add paddle
        for (let x = Math.floor(this.paddle.x); x < Math.floor(this.paddle.x + this.paddle.width); x++) {
            if (x >= 0 && x < this.width) {
                state[Math.floor(this.paddle.y)][x] = 1;
            }
        }
        
        // Add ball
        const ballX = Math.floor(this.ball.x);
        const ballY = Math.floor(this.ball.y);
        if (ballX >= 0 && ballX < this.width && ballY >= 0 && ballY < this.height) {
            state[ballY][ballX] = 2;
        }
        
        // Add bricks
        for (const brick of this.bricks) {
            if (brick.status === 1) {
                for (let y = Math.floor(brick.y); y < Math.floor(brick.y + brick.height); y++) {
                    for (let x = Math.floor(brick.x); x < Math.floor(brick.x + brick.width); x++) {
                        if (x >= 0 && x < this.width && y >= 0 && y < this.height) {
                            state[y][x] = 3;
                        }
                    }
                }
            }
        }
        
        // Resize the state
        let resizedState = this.resizeState(state, 42, 42).map(row => row.map(value => value / 3)); // Normalize between 0 and 1

        return resizedState;
    }
}