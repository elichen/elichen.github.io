const GOAL_WIDTH = 200, GOAL_POSTS = 20, friction = 0.98, maxSpeed = 25;

class AirHockeyEnvironment {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.canvas.width = 600;
        this.canvas.height = 800;

        this.state = { playerScore: 0, aiScore: 0, stuckTime: 0, lastPuckPos: {x:0,y:0}, samePositionTime: 0 };
        this.playerPaddle = { x: canvas.width/2, y: canvas.height-50, radius: 20, color: '#3498db', speed: 10 };
        this.aiPaddle = { x: canvas.width/2, y: 50, radius: 20, color: '#2ecc71', speed: 10 };
        this.puck = { x: canvas.width/2, y: canvas.height/2, radius: 15, dx: 0, dy: 0, color: '#e74c3c', isStuck: false, stuckEffectSize: 0 };
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

    resetPuck(scoredOnTop = null) {
        this.puck.x = this.canvas.width/2;
        this.puck.dx = 0;
        this.puck.dy = 0;
        this.puck.y = scoredOnTop === true ? this.canvas.height/4 : scoredOnTop === false ? this.canvas.height*3/4 : this.canvas.height/2;
        this.state.samePositionTime = 0;
        this.state.lastPuckPos = {x: this.puck.x, y: this.puck.y};
    }

    isInGoal() {
        const inX = this.puck.x > (this.canvas.width-GOAL_WIDTH)/2 && this.puck.x < (this.canvas.width+GOAL_WIDTH)/2;
        if (this.puck.y - this.puck.radius < GOAL_POSTS && inX) return 'top';
        if (this.puck.y + this.puck.radius > this.canvas.height - GOAL_POSTS && inX) return 'bottom';
        return false;
    }

    handleWallCollision() {
        if (this.puck.x - this.puck.radius < 0) { this.puck.x = this.puck.radius; this.puck.dx *= -0.8; }
        if (this.puck.x + this.puck.radius > this.canvas.width) { this.puck.x = this.canvas.width - this.puck.radius; this.puck.dx *= -0.8; }
        if (!this.isInGoal()) {
            if (this.puck.y - this.puck.radius < 0) { this.puck.y = this.puck.radius; this.puck.dy *= -0.8; }
            if (this.puck.y + this.puck.radius > this.canvas.height) { this.puck.y = this.canvas.height - this.puck.radius; this.puck.dy *= -0.8; }
        }
    }

    handlePaddleCollision(paddle) {
        const dx = this.puck.x - paddle.x, dy = this.puck.y - paddle.y;
        const dist = Math.sqrt(dx*dx + dy*dy);
        if (dist < paddle.radius + this.puck.radius) {
            const angle = Math.atan2(dy, dx);
            const minDist = paddle.radius + this.puck.radius;
            this.puck.x = paddle.x + Math.cos(angle) * minDist;
            this.puck.y = paddle.y + Math.sin(angle) * minDist;

            this.puck.dx = (paddle.dx || 0) * 1.8;
            this.puck.dy = (paddle.dy || 0) * 1.8;

            const speed = Math.sqrt(this.puck.dx*this.puck.dx + this.puck.dy*this.puck.dy);
            if (speed < 5) {
                const scale = 5 / (speed || 1);
                this.puck.dx *= scale;
                this.puck.dy *= scale;
            }
            if (speed > maxSpeed) {
                const scale = maxSpeed / speed;
                this.puck.dx *= scale;
                this.puck.dy *= scale;
            }
        }
    }

    isPuckStuck() {
        const isSlowMoving = Math.abs(this.puck.dx) < 0.1 && Math.abs(this.puck.dy) < 0.1;
        const nearWall = this.puck.x - this.puck.radius < 10 || this.puck.x + this.puck.radius > this.canvas.width - 10 ||
                        (this.puck.y - this.puck.radius < 10 && !this.isInGoal()) ||
                        (this.puck.y + this.puck.radius > this.canvas.height - 10 && !this.isInGoal());

        if (!nearWall) return false;

        const distFromLast = Math.sqrt(Math.pow(this.puck.x - this.state.lastPuckPos.x, 2) + Math.pow(this.puck.y - this.state.lastPuckPos.y, 2));
        if (distFromLast < 1) {
            this.state.samePositionTime++;
        } else {
            this.state.samePositionTime = 0;
            this.state.lastPuckPos = {x: this.puck.x, y: this.puck.y};
        }
        return isSlowMoving || this.state.samePositionTime > 30;
    }

    unstickPuck() {
        this.puck.x = Math.random() * (this.canvas.width - 2*this.puck.radius) + this.puck.radius;
        this.puck.y = Math.random() * (this.canvas.height - 2*this.puck.radius) + this.puck.radius;
        this.puck.dx = (Math.random() - 0.5) * 5;
        this.puck.dy = (Math.random() - 0.5) * 5;
        this.state.samePositionTime = 0;
        this.puck.isStuck = true;
        this.puck.stuckEffectSize = 20;
    }

    reset() {
        this.state = { playerScore: 0, aiScore: 0, stuckTime: 0, lastPuckPos: {x:0,y:0}, samePositionTime: 0 };
        this.playerPaddle = { x: this.canvas.width/2, y: this.canvas.height-50, radius: 20, color: '#3498db', speed: 10, dx: 0, dy: 0 };
        this.aiPaddle = { x: this.canvas.width/2, y: 50, radius: 20, color: '#2ecc71', speed: 10, dx: 0, dy: 0 };
        this.puck = { x: this.canvas.width/2, y: this.canvas.height/2, radius: 15, dx: 0, dy: 0, color: '#e74c3c', isStuck: false, stuckEffectSize: 0 };
    }

    update() {
        this.puck.x += this.puck.dx;
        this.puck.y += this.puck.dy;
        this.puck.dx *= friction;
        this.puck.dy *= friction;

        this.handleWallCollision();
        this.handlePaddleCollision(this.playerPaddle);
        this.handlePaddleCollision(this.aiPaddle);

        const goalHit = this.isInGoal();
        if (goalHit === 'top') {
            this.state.playerScore++;
            this.resetPuck(true);
        } else if (goalHit === 'bottom') {
            this.state.aiScore++;
            this.resetPuck(false);
        } else if (this.isPuckStuck()) {
            this.unstickPuck();
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

        if (this.puck.isStuck) {
            this.ctx.beginPath();
            this.ctx.arc(this.puck.x, this.puck.y, this.puck.radius + this.puck.stuckEffectSize, 0, Math.PI*2);
            this.ctx.strokeStyle = `rgba(255, 255, 255, ${this.puck.stuckEffectSize/20})`;
            this.ctx.lineWidth = 2;
            this.ctx.stroke();
            this.puck.stuckEffectSize *= 0.9;
            if (this.puck.stuckEffectSize < 0.5) this.puck.isStuck = false;
        }
    }
}