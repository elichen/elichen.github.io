class Ball {
    constructor(x, y, gameWidth, gameHeight, radius = 5) {
        this.x = x;
        this.y = y;
        this.gameWidth = gameWidth;
        this.gameHeight = gameHeight;
        this.radius = radius;
        this.reset();
    }

    reset() {
        this.x = this.gameWidth / 2;
        this.y = this.gameHeight / 2;
        // Random initial direction (between -45 and 45 degrees)
        const angle = (Math.random() * Math.PI/4) - Math.PI/8;
        const direction = Math.random() < 0.5 ? 1 : -1;
        this.dx = Math.cos(angle) * 5 * direction;
        this.dy = Math.sin(angle) * 5;
        this.speed = 5;
        console.log("Ball reset - dx:", this.dx, "dy:", this.dy);
    }

    update() {
        this.x += this.dx;
        this.y += this.dy;

        // Check wall collisions
        if (this.y - this.radius <= 0) {
            console.log("Hit top wall - Before bounce dy:", this.dy);
            this.y = this.radius; // Prevent sticking to wall
            this.bounceWall();
            console.log("After bounce dy:", this.dy);
        } 
        else if (this.y + this.radius >= this.gameHeight) {
            console.log("Hit bottom wall - Before bounce dy:", this.dy);
            this.y = this.gameHeight - this.radius; // Prevent sticking to wall
            this.bounceWall();
            console.log("After bounce dy:", this.dy);
        }
    }

    bounceWall() {
        // Simply reverse dy direction
        this.dy = -this.dy;
        
        // Add tiny random variation to dx to prevent loops
        this.dx *= 0.99 + Math.random() * 0.02;
        
        // Ensure speed doesn't grow or diminish too much
        const currentSpeed = Math.sqrt(this.dx * this.dx + this.dy * this.dy);
        if (currentSpeed > this.speed * 1.1 || currentSpeed < this.speed * 0.9) {
            const factor = this.speed / currentSpeed;
            this.dx *= factor;
            this.dy *= factor;
        }
        
        console.log("Wall bounce - new dx:", this.dx, "new dy:", this.dy);
    }

    bouncePaddle(paddleCenter, paddleHeight) {
        // Calculate relative intersection point (-1 to 1)
        const relativeIntersectY = (this.y - paddleCenter) / (paddleHeight/2);
        
        // Constrain the angle (maximum 45 degrees)
        const maxBounceAngle = Math.PI / 4;  // 45 degrees
        const bounceAngle = relativeIntersectY * maxBounceAngle;
        
        // Increase speed slightly (cap at maximum speed)
        this.speed = Math.min(this.speed * 1.05, 15);
        
        // Calculate new velocities
        const direction = this.dx > 0 ? -1 : 1;  // Reverse x direction
        
        // Use bounceAngle to determine new velocity components
        this.dx = direction * this.speed * Math.cos(bounceAngle);
        this.dy = this.speed * Math.sin(bounceAngle);
        
        console.log("Paddle bounce - new dx:", this.dx, "new dy:", this.dy);
    }
} 