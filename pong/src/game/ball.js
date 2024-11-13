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
        // Random initial direction
        const angle = (Math.random() * Math.PI/4) + Math.PI/8;
        const direction = Math.random() < 0.5 ? 1 : -1;
        this.dx = Math.cos(angle) * 5 * direction;
        this.dy = Math.sin(angle) * 5;
        this.speed = 5;
    }

    update() {
        this.x += this.dx;
        this.y += this.dy;
    }

    bounceWall() {
        this.dy = -this.dy;
        // Add small random variation to prevent loops
        this.dx += (Math.random() - 0.5) * 0.5;
    }

    bouncePaddle(paddleCenter, paddleHeight) {
        // Calculate relative intersection point (-1 to 1)
        const relativeIntersectY = (this.y - paddleCenter) / (paddleHeight/2);
        
        // Constrain the angle (maximum 75 degrees)
        const maxBounceAngle = 0.75 * Math.PI / 2;
        const bounceAngle = relativeIntersectY * maxBounceAngle;
        
        // Increase speed slightly (cap at maximum speed)
        this.speed = Math.min(this.speed * 1.05, 15);
        
        // Calculate new velocities
        const direction = this.dx > 0 ? -1 : 1;  // Reverse x direction
        
        // Use bounceAngle to determine new velocity components
        this.dx = direction * this.speed * Math.cos(bounceAngle);
        this.dy = this.speed * Math.sin(bounceAngle);
        
        // Add small random variation to prevent loops
        this.dy += (Math.random() - 0.5) * 0.5;
    }
} 