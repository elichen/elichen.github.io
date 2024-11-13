class Paddle {
    constructor(x, isLeft, gameHeight) {
        this.width = 10;
        this.height = 60;
        this.x = x;
        this.gameHeight = gameHeight;
        this.y = gameHeight/2 - this.height/2;
        this.isLeft = isLeft;
        this.speed = 5;
        this.score = 0;
    }

    move(action) {
        // action: -1 (up), 0 (stay), 1 (down)
        this.y += action * this.speed;
        
        // Keep paddle within bounds
        this.y = Math.max(0, Math.min(this.gameHeight - this.height, this.y));
    }

    reset() {
        this.y = this.gameHeight/2 - this.height/2;
    }
} 