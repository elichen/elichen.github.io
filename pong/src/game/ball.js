class Ball {
    constructor(x, y, gameWidth, gameHeight, radius = 5) {
        this.x = x;
        this.y = y;
        this.gameWidth = gameWidth;
        this.gameHeight = gameHeight;
        this.radius = radius;
        this.serveDirection = 1;  // Track who to serve towards (1 or -1)
        this.reset();
    }

    reset() {
        this.x = this.gameWidth / 2;
        this.y = this.gameHeight / 2;
        
        // Alternate serve direction
        this.serveDirection *= -1;
        
        // Random angle between -45 and 45 degrees
        const angle = (Math.random() * Math.PI/4) - Math.PI/8;
        
        // Use serveDirection to determine initial direction
        this.dx = Math.cos(angle) * 5 * this.serveDirection;
        this.dy = Math.sin(angle) * 5;
        this.speed = 5;
    }

    update() {
        this.x += this.dx;
        this.y += this.dy;

        // Check wall collisions
        if (this.y - this.radius <= 0) {
            this.y = this.radius; // Prevent sticking to wall
            this.bounceWall();
        } 
        else if (this.y + this.radius >= this.gameHeight) {
            this.y = this.gameHeight - this.radius; // Prevent sticking to wall
            this.bounceWall();
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
    }
} 