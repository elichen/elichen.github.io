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
        this.lastBallY = this.ball.y; // Add this line to track the ball's last Y position
        this.penaltyForLosingBall = 0; // Negative reward for losing the ball
        this.rewardForHittingPaddle = .1; // New reward for hitting the paddle
        this.ballHitPaddle = false; // New flag to track if the ball hit the paddle
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

        this.lastBallY = this.ball.y; // Store the current Y position before updating

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
            this.ball.y - this.ball.radius < this.paddle.y + this.paddle.height &&
            this.ball.x > this.paddle.x &&
            this.ball.x < this.paddle.x + this.paddle.width
        ) {
            // Only reverse direction if the ball is moving downward
            if (this.ball.dy > 0) {
                this.ball.dy = -this.ball.dy;
                
                // Add some variation to the ball's direction based on where it hits the paddle
                const hitPosition = (this.ball.x - this.paddle.x) / this.paddle.width;
                const maxAngleOffset = 1;
                this.ball.dx = this.ball.dx + (hitPosition - 0.5) * maxAngleOffset;
            }
            this.ballHitPaddle = true;
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

    getReward() {
        let reward = 0;

        // Reward for breaking bricks
        if (this.score > 0) {
            console.log("rewarding for breaking bricks", this.score);
            reward += this.score;
            this.score = 0; // Reset the score after adding it to the reward
        }

        // Reward for hitting the paddle
        if (this.ballHitPaddle) {
            console.log("rewarding for hitting paddle", this.rewardForHittingPaddle);
            reward += this.rewardForHittingPaddle;
            this.ballHitPaddle = false; // Reset the flag after giving the reward
        }

        // Penalty for losing the ball
        if (this.gameOver) {
            console.log("penalizing for losing ball", this.penaltyForLosingBall);
            reward += this.penaltyForLosingBall;
        }

        return reward;
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
        this.lastBallY = this.ball.y;
        this.ballHitPaddle = false;
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
        // Match Python state representation:
        // [paddle_x, ball_x, ball_y, ball_dx, ball_dy]
        const state = [
            this.paddle.x / this.width,    // Normalized paddle x position
            this.ball.x / this.width,      // Normalized ball x position
            this.ball.y / this.height,     // Normalized ball y position
            this.ball.dx / 4,              // Normalized ball x velocity
            this.ball.dy / 4               // Normalized ball y velocity
        ];
        
        return state;
    }
}
