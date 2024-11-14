class PongGame {
    constructor(width = 600, height = 400) {
        this.width = width;
        this.height = height;
        this.ball = new Ball(width/2, height/2, width, height);
        this.leftPaddle = new Paddle(20, true, height);
        this.rightPaddle = new Paddle(width - 30, false, height);
        this.reset();
    }

    reset() {
        this.ball.reset();
        this.leftPaddle.reset();
        this.rightPaddle.reset();
        return this.getState();
    }

    getState() {
        return [
            this.ball.x / this.width,  // Normalized positions
            this.ball.y / this.height,
            this.ball.dx / 10,         // Normalized velocities
            this.ball.dy / 10,
            this.leftPaddle.y / this.height,
            this.rightPaddle.y / this.height
        ];
    }

    step(action1, action2) {
        let hitPaddle = false;

        // Update paddles
        this.leftPaddle.move(action1);
        this.rightPaddle.move(action2);

        // Update ball
        this.ball.update();

        // Check paddle collisions
        if (this.checkPaddleCollision(this.leftPaddle) || 
            this.checkPaddleCollision(this.rightPaddle)) {
            hitPaddle = true;
        }

        // Check scoring
        let done = false;
        if (this.ball.x <= 0) {
            this.rightPaddle.score++;  // Right paddle scores
            done = true;
        } else if (this.ball.x >= this.width) {
            this.leftPaddle.score++;   // Left paddle scores
            done = true;
        }

        return {
            state: this.getState(),
            done: done,
            hitPaddle: hitPaddle
        };
    }

    checkPaddleCollision(paddle) {
        // Calculate paddle center and ball's next position
        const paddleCenter = paddle.y + paddle.height/2;
        const nextBallX = this.ball.x + this.ball.dx;
        const nextBallY = this.ball.y + this.ball.dy;

        if (paddle.isLeft) {
            // Check if ball will collide with left paddle
            if (nextBallX - this.ball.radius <= paddle.x + paddle.width &&
                this.ball.x - this.ball.radius > paddle.x + paddle.width &&  // Wasn't colliding previously
                nextBallY + this.ball.radius >= paddle.y &&
                nextBallY - this.ball.radius <= paddle.y + paddle.height) {
                
                // Move ball to paddle surface to prevent sticking
                this.ball.x = paddle.x + paddle.width + this.ball.radius;
                this.ball.bouncePaddle(paddleCenter, paddle.height);
                return true;
            }
        } else {
            // Check if ball will collide with right paddle
            if (nextBallX + this.ball.radius >= paddle.x &&
                this.ball.x + this.ball.radius < paddle.x &&  // Wasn't colliding previously
                nextBallY + this.ball.radius >= paddle.y &&
                nextBallY - this.ball.radius <= paddle.y + paddle.height) {
                
                // Move ball to paddle surface to prevent sticking
                this.ball.x = paddle.x - this.ball.radius;
                this.ball.bouncePaddle(paddleCenter, paddle.height);
                return true;
            }
        }
        return false;
    }
} 