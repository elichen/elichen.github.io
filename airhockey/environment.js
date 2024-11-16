// Constants for the environment
const GOAL_WIDTH = 200;
const GOAL_POSTS = 20;
const TABLE_COLOR = '#2c3e50';
const TABLE_BORDER = '#34495e';
const CENTER_CIRCLE_RADIUS = 100;
const friction = 0.99;
const maxSpeed = 20;

class AirHockeyEnvironment {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        
        // Set canvas size
        this.canvas.width = 600;
        this.canvas.height = 800;
        
        // Game state
        this.state = {
            playerScore: 0,
            aiScore: 0,
            stuckTime: 0,
            lastPuckPos: { x: 0, y: 0 },
            samePositionTime: 0
        };
        
        // Game objects
        this.playerPaddle = {
            x: canvas.width / 2,
            y: canvas.height - 50,
            radius: 20,
            color: '#3498db',
            speed: 5
        };

        this.aiPaddle = {
            x: canvas.width / 2,
            y: 50,
            radius: 20,
            color: '#2ecc71',
            speed: 5
        };

        this.puck = {
            x: canvas.width / 2,
            y: canvas.height / 2,
            radius: 15,
            dx: 0,
            dy: 0,
            color: '#e74c3c',
            isStuck: false,
            stuckEffectSize: 0
        };
    }

    createGradient(x, y, radius, colorStart, colorEnd) {
        const gradient = this.ctx.createRadialGradient(x, y, 0, x, y, radius);
        gradient.addColorStop(0, colorStart);
        gradient.addColorStop(1, colorEnd);
        return gradient;
    }

    drawTableMarkings() {
        // Draw center circle
        this.ctx.beginPath();
        this.ctx.arc(this.canvas.width/2, this.canvas.height/2, CENTER_CIRCLE_RADIUS, 0, Math.PI * 2);
        this.ctx.strokeStyle = '#ffffff22';
        this.ctx.lineWidth = 4;
        this.ctx.stroke();

        // Draw center line
        this.ctx.beginPath();
        this.ctx.moveTo(0, this.canvas.height/2);
        this.ctx.lineTo(this.canvas.width, this.canvas.height/2);
        this.ctx.strokeStyle = '#ffffff22';
        this.ctx.stroke();
    }

    drawGoals() {
        this.ctx.fillStyle = '#ffffff22';
        
        // Top goal
        this.ctx.fillRect((this.canvas.width - GOAL_WIDTH) / 2, -GOAL_POSTS/2, GOAL_WIDTH, GOAL_POSTS);
        // Bottom goal
        this.ctx.fillRect((this.canvas.width - GOAL_WIDTH) / 2, this.canvas.height - GOAL_POSTS/2, GOAL_WIDTH, GOAL_POSTS);
    }

    drawCircle(x, y, radius, color) {
        this.ctx.beginPath();
        this.ctx.arc(x, y, radius, 0, Math.PI * 2);
        this.ctx.fillStyle = color;
        this.ctx.fill();
    }

    drawScore() {
        this.ctx.font = 'bold 48px Arial';
        this.ctx.fillStyle = '#ffffff44';
        this.ctx.textAlign = 'center';
        this.ctx.fillText(`${this.state.aiScore} - ${this.state.playerScore}`, this.canvas.width/2, this.canvas.height/2);
    }

    checkCollision(circle1, circle2) {
        const dx = circle1.x - circle2.x;
        const dy = circle1.y - circle2.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        return distance < circle1.radius + circle2.radius;
    }

    resetPuck(scoredOnTop = null) {
        this.puck.x = this.canvas.width / 2;
        this.puck.dx = 0;
        this.puck.dy = 0;

        if (scoredOnTop === true) {
            this.puck.y = this.canvas.height / 4; // Position in AI's side
        } else if (scoredOnTop === false) {
            this.puck.y = this.canvas.height * 3/4; // Position in player's side
        } else {
            this.puck.y = this.canvas.height / 2; // Center for other resets
        }
        
        // Reset stuck detection state
        this.state.samePositionTime = 0;
        this.state.lastPuckPos.x = this.puck.x;
        this.state.lastPuckPos.y = this.puck.y;
    }

    isInGoal() {
        const inXRange = this.puck.x > (this.canvas.width - GOAL_WIDTH) / 2 && 
                        this.puck.x < (this.canvas.width + GOAL_WIDTH) / 2;
        
        if (this.puck.y - this.puck.radius < GOAL_POSTS && inXRange) return 'top';
        if (this.puck.y + this.puck.radius > this.canvas.height - GOAL_POSTS && inXRange) return 'bottom';
        return false;
    }

    handleWallCollision() {
        // Left and right walls
        if (this.puck.x - this.puck.radius < 0) {
            this.puck.x = this.puck.radius;
            this.puck.dx *= -0.8;
        }
        if (this.puck.x + this.puck.radius > this.canvas.width) {
            this.puck.x = this.canvas.width - this.puck.radius;
            this.puck.dx *= -0.8;
        }

        // Top and bottom walls (except for goals)
        const goalHit = this.isInGoal();
        if (!goalHit) {
            if (this.puck.y - this.puck.radius < 0) {
                this.puck.y = this.puck.radius;
                this.puck.dy *= -0.8;
            }
            if (this.puck.y + this.puck.radius > this.canvas.height) {
                this.puck.y = this.canvas.height - this.puck.radius;
                this.puck.dy *= -0.8;
            }
        }
    }

    handlePaddleCollision(paddle, prevX, prevY) {
        // Continuous collision detection
        const dx = this.puck.x - paddle.x;
        const dy = this.puck.y - paddle.y;
        const distance = Math.sqrt(dx * dx + dy * dy);

        if (distance < paddle.radius + this.puck.radius) {
            // Calculate collision point
            const angle = Math.atan2(dy, dx);
            const minDistance = paddle.radius + this.puck.radius;
            
            // Move puck outside paddle
            this.puck.x = paddle.x + Math.cos(angle) * minDistance;
            this.puck.y = paddle.y + Math.sin(angle) * minDistance;
            
            // Calculate new velocity based on paddle movement
            const paddleSpeed = {
                x: paddle.x - prevX,
                y: paddle.y - prevY
            };
            
            const dotProduct = (this.puck.dx * dx + this.puck.dy * dy) / distance;
            
            // Combine paddle momentum with puck direction
            this.puck.dx = (Math.cos(angle) * Math.abs(dotProduct) + paddleSpeed.x * 0.9);
            this.puck.dy = (Math.sin(angle) * Math.abs(dotProduct) + paddleSpeed.y * 0.9);
            
            // Add minimum speed after collision
            const speed = Math.sqrt(this.puck.dx * this.puck.dx + this.puck.dy * this.puck.dy);
            if (speed < 3) {
                this.puck.dx *= 3 / speed;
                this.puck.dy *= 3 / speed;
            }
            
            // Enforce speed limit
            if (speed > maxSpeed) {
                this.puck.dx = (this.puck.dx / speed) * maxSpeed;
                this.puck.dy = (this.puck.dy / speed) * maxSpeed;
            }
        }
    }

    getRandomPosition() {
        return {
            x: Math.random() * (this.canvas.width - 2 * this.puck.radius) + this.puck.radius,
            y: Math.random() * (this.canvas.height - 2 * this.puck.radius) + this.puck.radius,
            dx: (Math.random() - 0.5) * 5,
            dy: (Math.random() - 0.5) * 5
        };
    }

    isPuckStuck() {
        const speed = Math.sqrt(this.puck.dx * this.puck.dx + this.puck.dy * this.puck.dy);
        const isSlowMoving = Math.abs(this.puck.dx) < 0.1 && Math.abs(this.puck.dy) < 0.1;
        
        // Check if puck is near any wall
        const nearWall = (
            this.puck.x - this.puck.radius < 10 || // Left wall
            this.puck.x + this.puck.radius > this.canvas.width - 10 || // Right wall
            (this.puck.y - this.puck.radius < 10 && !this.isInGoal()) || // Top wall (not goal)
            (this.puck.y + this.puck.radius > this.canvas.height - 10 && !this.isInGoal()) // Bottom wall (not goal)
        );
        
        if (!nearWall) {
            return false;
        }

        const distFromLast = Math.sqrt(
            Math.pow(this.puck.x - this.state.lastPuckPos.x, 2) + 
            Math.pow(this.puck.y - this.state.lastPuckPos.y, 2)
        );
        
        if (distFromLast < 1) {
            this.state.samePositionTime++;
        } else {
            this.state.samePositionTime = 0;
            this.state.lastPuckPos.x = this.puck.x;
            this.state.lastPuckPos.y = this.puck.y;
        }

        return isSlowMoving || this.state.samePositionTime > 30;
    }

    unstickPuck() {
        const newPos = this.getRandomPosition();
        this.puck.x = newPos.x;
        this.puck.y = newPos.y;
        this.puck.dx = newPos.dx;
        this.puck.dy = newPos.dy;
        this.state.samePositionTime = 0;
        this.puck.isStuck = true;
        this.puck.stuckEffectSize = 20;
    }

    update(mouseX, mouseY, isTrainingMode) {
        // Store previous positions for collision detection
        const prevPlayerX = this.playerPaddle.x;
        const prevPlayerY = this.playerPaddle.y;
        const prevAIX = this.aiPaddle.x;
        const prevAIY = this.aiPaddle.y;

        // Update player paddle position only if not in training mode
        if (!isTrainingMode) {
            this.playerPaddle.x = mouseX;
            this.playerPaddle.y = mouseY;

            // Restrict player to bottom half
            this.playerPaddle.x = Math.max(this.playerPaddle.radius, 
                Math.min(this.canvas.width - this.playerPaddle.radius, this.playerPaddle.x));
            this.playerPaddle.y = Math.max(this.canvas.height/2 + this.playerPaddle.radius, 
                Math.min(this.canvas.height - this.playerPaddle.radius, this.playerPaddle.y));
        }

        // Update puck position
        this.puck.x += this.puck.dx;
        this.puck.y += this.puck.dy;

        // Apply friction
        this.puck.dx *= friction;
        this.puck.dy *= friction;

        // Handle collisions
        this.handleWallCollision();
        this.handlePaddleCollision(this.playerPaddle, prevPlayerX, prevPlayerY);
        this.handlePaddleCollision(this.aiPaddle, prevAIX, prevAIY);

        // Check for goals and stuck puck
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
        // Clear canvas with flat color
        this.ctx.fillStyle = TABLE_COLOR;
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Add table border
        this.ctx.strokeStyle = TABLE_BORDER;
        this.ctx.lineWidth = 10;
        this.ctx.strokeRect(0, 0, this.canvas.width, this.canvas.height);
        
        this.drawTableMarkings();
        this.drawGoals();
        this.drawScore();
        
        // Draw game objects
        this.drawCircle(this.playerPaddle.x, this.playerPaddle.y, this.playerPaddle.radius, this.playerPaddle.color);
        this.drawCircle(this.aiPaddle.x, this.aiPaddle.y, this.aiPaddle.radius, this.aiPaddle.color);
        
        // Draw puck with stuck effect if needed
        this.drawCircle(this.puck.x, this.puck.y, this.puck.radius, this.puck.color);
        if (this.puck.isStuck) {
            this.ctx.beginPath();
            this.ctx.arc(this.puck.x, this.puck.y, this.puck.radius + this.puck.stuckEffectSize, 0, Math.PI * 2);
            this.ctx.strokeStyle = `rgba(255, 255, 255, ${this.puck.stuckEffectSize / 20})`;
            this.ctx.lineWidth = 2;
            this.ctx.stroke();
            
            this.puck.stuckEffectSize *= 0.9;
            if (this.puck.stuckEffectSize < 0.5) {
                this.puck.isStuck = false;
            }
        }
    }
} 