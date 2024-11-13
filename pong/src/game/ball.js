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
        // Slightly randomize the angle
        this.dy *= 0.9 + Math.random() * 0.2;
    }

    bouncePaddle(paddleY, paddleHeight) {
        // Calculate relative intersection point
        const relativeIntersectY = (paddleY + (paddleHeight/2)) - this.y;
        const normalizedIntersectY = relativeIntersectY / (paddleHeight/2);
        
        // Calculate bounce angle (maximum 75 degrees)
        const maxAngle = Math.PI * 0.75;
        const bounceAngle = normalizedIntersectY * maxAngle;
        
        // Reverse x direction and apply new angle
        this.dx = -this.dx;
        this.speed *= 1.05; // Increase speed slightly
        const direction = this.dx < 0 ? -1 : 1;
        
        this.dx = direction * this.speed * Math.cos(bounceAngle);
        this.dy = -this.speed * Math.sin(bounceAngle);
    }
} 