const GOAL_WIDTH = 200, GOAL_POSTS = 20, friction = 0.997, maxSpeed = 30, wallBounce = 0.9, paddleBounce = 0.8, substeps = 4;

class AirHockeyEnvironment {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.canvas.width = 600;
        this.canvas.height = 800;
        this.reset();
    }

    drawTableMarkings() {
        this.ctx.beginPath();
        this.ctx.arc(this.canvas.width/2, this.canvas.height/2, 100, 0, Math.PI*2);
        this.ctx.strokeStyle = '#ffffff22';
        this.ctx.lineWidth = 4;
        this.ctx.stroke();
        this.ctx.beginPath();
        this.ctx.moveTo(0, this.canvas.height/2);
        this.ctx.lineTo(this.canvas.width, this.canvas.height/2);
        this.ctx.stroke();
    }

    drawGoals() {
        this.ctx.fillStyle = '#ffffff22';
        this.ctx.fillRect((this.canvas.width-GOAL_WIDTH)/2, -GOAL_POSTS/2, GOAL_WIDTH, GOAL_POSTS);
        this.ctx.fillRect((this.canvas.width-GOAL_WIDTH)/2, this.canvas.height-GOAL_POSTS/2, GOAL_WIDTH, GOAL_POSTS);
    }

    drawCircle(x, y, radius, color) {
        this.ctx.beginPath();
        this.ctx.arc(x, y, radius, 0, Math.PI*2);
        this.ctx.fillStyle = color;
        this.ctx.fill();
    }

    drawScore() {
        this.ctx.font = 'bold 48px Arial';
        this.ctx.fillStyle = '#ffffff44';
        this.ctx.textAlign = 'center';
        this.ctx.fillText(`${this.state.aiScore} - ${this.state.playerScore}`, this.canvas.width/2, this.canvas.height/2);
    }

    resetPaddles() {
        Object.assign(this.playerPaddle, { x: this.canvas.width/2, y: this.canvas.height - 50, dx: 0, dy: 0 });
        Object.assign(this.aiPaddle, { x: this.canvas.width/2, y: 50, dx: 0, dy: 0 });
    }

    resetPuck(scoredOnTop = null, resetPlayers = false) {
        if (resetPlayers) this.resetPaddles();
        this.puck.x = this.canvas.width/2;
        this.puck.dx = 0;
        this.puck.dy = 0;
        this.puck.y = scoredOnTop === true ? this.canvas.height/4 : scoredOnTop === false ? this.canvas.height*3/4 : this.canvas.height/2;
        this.state.roundFrames = 0;
    }

    isInGoal() {
        const inX = this.puck.x > (this.canvas.width-GOAL_WIDTH)/2 && this.puck.x < (this.canvas.width+GOAL_WIDTH)/2;
        if (this.puck.y - this.puck.radius < GOAL_POSTS && inX) return 'top';
        if (this.puck.y + this.puck.radius > this.canvas.height - GOAL_POSTS && inX) return 'bottom';
        return false;
    }

    handleWallCollision() {
        const p = this.puck, r = p.radius;
        if (p.x - r < 0) { p.x = r; p.dx = Math.abs(p.dx) * wallBounce; }
        if (p.x + r > this.canvas.width) { p.x = this.canvas.width - r; p.dx = -Math.abs(p.dx) * wallBounce; }
        if (!this.isInGoal()) {
            if (p.y - r < 0) { p.y = r; p.dy = Math.abs(p.dy) * wallBounce; }
            if (p.y + r > this.canvas.height) { p.y = this.canvas.height - r; p.dy = -Math.abs(p.dy) * wallBounce; }
        }
    }

    // Paddle is hand-driven (infinite mass): reflect the puck's velocity relative to the paddle along the contact normal.
    handlePaddleCollision(paddle, frac) {
        const px = paddle.x - (paddle.dx || 0) * (1 - frac), py = paddle.y - (paddle.dy || 0) * (1 - frac);
        const dx = this.puck.x - px, dy = this.puck.y - py, dist = Math.sqrt(dx*dx + dy*dy), minDist = paddle.radius + this.puck.radius;
        if (dist >= minDist || dist === 0) return;
        const nx = dx / dist, ny = dy / dist;
        this.puck.x = px + nx * minDist;
        this.puck.y = py + ny * minDist;
        const vn = (this.puck.dx - (paddle.dx || 0)) * nx + (this.puck.dy - (paddle.dy || 0)) * ny;
        if (vn < 0) {
            this.puck.dx -= (1 + paddleBounce) * vn * nx;
            this.puck.dy -= (1 + paddleBounce) * vn * ny;
        }
        const speed = Math.sqrt(this.puck.dx*this.puck.dx + this.puck.dy*this.puck.dy);
        if (speed > maxSpeed) { this.puck.dx *= maxSpeed / speed; this.puck.dy *= maxSpeed / speed; }
    }

    reset() {
        this.state = { playerScore: 0, aiScore: 0, roundFrames: 0 };
        this.playerPaddle = { x: this.canvas.width/2, y: this.canvas.height-50, radius: 20, color: '#3498db', speed: 10, dx: 0, dy: 0 };
        this.aiPaddle = { x: this.canvas.width/2, y: 50, radius: 20, color: '#2ecc71', speed: 10, dx: 0, dy: 0 };
        this.puck = { x: this.canvas.width/2, y: this.canvas.height/2, radius: 15, dx: 0, dy: 0, color: '#e74c3c' };
    }

    // Paddles have already moved this frame; sweep them and the puck together in substeps.
    update() {
        this.state.roundFrames++;
        let goalHit = false;
        for (let s = 1; s <= substeps && !goalHit; s++) {
            this.puck.x += this.puck.dx / substeps;
            this.puck.y += this.puck.dy / substeps;
            this.handlePaddleCollision(this.playerPaddle, s / substeps);
            this.handlePaddleCollision(this.aiPaddle, s / substeps);
            this.handleWallCollision();
            goalHit = this.isInGoal();
        }
        this.puck.dx *= friction;
        this.puck.dy *= friction;

        if (goalHit === 'top') {
            this.state.playerScore++;
            this.resetPuck(true, true);
        } else if (goalHit === 'bottom') {
            this.state.aiScore++;
            this.resetPuck(false, true);
        } else if (this.state.roundFrames >= 1200) {
            this.resetPuck(null, true);
            return 'timeout';
        }
        return goalHit;
    }

    draw() {
        this.ctx.fillStyle = '#2c3e50';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.strokeStyle = '#34495e';
        this.ctx.lineWidth = 10;
        this.ctx.strokeRect(0, 0, this.canvas.width, this.canvas.height);

        this.drawTableMarkings();
        this.drawGoals();
        this.drawScore();
        this.drawCircle(this.playerPaddle.x, this.playerPaddle.y, this.playerPaddle.radius, this.playerPaddle.color);
        this.drawCircle(this.aiPaddle.x, this.aiPaddle.y, this.aiPaddle.radius, this.aiPaddle.color);
        this.drawCircle(this.puck.x, this.puck.y, this.puck.radius, this.puck.color);
    }
}
