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
        let reward1 = 0;
        let reward2 = 0;
        let done = false;

        // Update paddles
        this.leftPaddle.move(action1);
        this.rightPaddle.move(action2);

        // Small negative reward for movement
        if (action1 !== 0) reward1 -= 0.01;
        if (action2 !== 0) reward2 -= 0.01;

        // Update ball
        this.ball.update();

        // Check wall collisions
        if (this.ball.y <= 0 || this.ball.y >= this.height) {
            this.ball.bounceWall();
        }

        // Check paddle collisions
        if (this.checkPaddleCollision(this.leftPaddle)) {
            reward1 += 0.1; // Reward for hitting
        }
        if (this.checkPaddleCollision(this.rightPaddle)) {
            reward2 += 0.1; // Reward for hitting
        }

        // Check scoring
        if (this.ball.x <= 0) {
            this.rightPaddle.score++;
            reward1 -= 1;
            reward2 += 1;
            done = true;
        } else if (this.ball.x >= this.width) {
            this.leftPaddle.score++;
            reward1 += 1;
            reward2 -= 1;
            done = true;
        }

        return {
            state: this.getState(),
            reward1,
            reward2,
            done
        };
    }

    checkPaddleCollision(paddle) {
        if (paddle.isLeft) {
            if (this.ball.x - this.ball.radius <= paddle.x + paddle.width &&
                this.ball.x - this.ball.radius >= paddle.x &&
                this.ball.y >= paddle.y &&
                this.ball.y <= paddle.y + paddle.height) {
                this.ball.bouncePaddle(paddle.y, paddle.height);
                return true;
            }
        } else {
            if (this.ball.x + this.ball.radius >= paddle.x &&
                this.ball.x + this.ball.radius <= paddle.x + paddle.width &&
                this.ball.y >= paddle.y &&
                this.ball.y <= paddle.y + paddle.height) {
                this.ball.bouncePaddle(paddle.y, paddle.height);
                return true;
            }
        }
        return false;
    }
} 