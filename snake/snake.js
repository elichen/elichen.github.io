class SnakeGame {
    constructor(canvasId, gridSize = 20) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.gridSize = gridSize;
        this.tileSize = this.canvas.width / this.gridSize;
        this.reset();
        this.movesSinceLastFood = 0;
        this.maxMovesWithoutFood = gridSize * 2; // Adjust this value as needed
    }

    reset() {
        // Random initial direction (matching Python training env)
        const directions = [
            { x: 0, y: -1 },  // up
            { x: 1, y: 0 },   // right
            { x: 0, y: 1 },   // down
            { x: -1, y: 0 }   // left
        ];
        this.direction = directions[Math.floor(Math.random() * 4)];

        // Start snake in center with 3 segments (matching Python training env)
        const centerX = Math.floor(this.gridSize / 2);
        const centerY = Math.floor(this.gridSize / 2);
        this.snake = [];
        for (let i = 0; i < 3; i++) {
            // Head first, then body segments behind (opposite of direction)
            const x = centerX - i * this.direction.x;
            const y = centerY - i * this.direction.y;
            // Clamp to grid
            this.snake.push({
                x: Math.max(0, Math.min(this.gridSize - 1, x)),
                y: Math.max(0, Math.min(this.gridSize - 1, y))
            });
        }

        this.food = this.generateFood();
        this.score = 0;
        this.gameOver = false;
        this.movesSinceLastFood = 0;
    }

    generateFood() {
        let food;
        do {
            food = {
                x: Math.floor(Math.random() * this.gridSize),
                y: Math.floor(Math.random() * this.gridSize)
            };
        } while (this.snake.some(segment => segment.x === food.x && segment.y === food.y));
        return food;
    }

    update() {
        if (this.gameOver) return false;

        // Move snake
        const head = { x: this.snake[0].x + this.direction.x, y: this.snake[0].y + this.direction.y };

        // Check collision with walls
        if (head.x < 0 || head.x >= this.gridSize || head.y < 0 || head.y >= this.gridSize) {
            this.gameOver = true;
            this.collisionType = 'wall';
            return false;
        }

        // Check collision with self (excluding tail, which will move away)
        // Match Python: new_head in self.snake[:-1]
        const bodyWithoutTail = this.snake.slice(0, -1);
        if (bodyWithoutTail.some(segment => segment.x === head.x && segment.y === head.y)) {
            this.gameOver = true;
            this.collisionType = 'self';
            return false;
        }
        // Also check if head hits tail AND we're not eating food at that position
        // (if eating food, snake grows so tail doesn't move)
        const tail = this.snake[this.snake.length - 1];
        if (head.x === tail.x && head.y === tail.y) {
            // This is only a collision if we're eating food (tail won't move)
            if (head.x === this.food.x && head.y === this.food.y) {
                this.gameOver = true;
                this.collisionType = 'self';
                return false;
            }
            // Otherwise tail will move, so it's safe
        }

        this.snake.unshift(head);

        // Check if food is eaten
        if (head.x === this.food.x && head.y === this.food.y) {
            this.score++;
            this.food = this.generateFood();
            return true; // Food was eaten
        } else {
            this.snake.pop();
            return false; // Food was not eaten
        }
    }

    step(action) {
        // Translate action to direction
        const directions = [
            { x: 0, y: -1 }, // Up
            { x: 1, y: 0 },  // Right
            { x: 0, y: 1 },  // Down
            { x: -1, y: 0 }  // Left
        ];
        let newDirection = directions[action];

        // Prevent moving backward into itself (matching Python training env)
        if (this.snake.length > 1) {
            if (newDirection.x === -this.direction.x && newDirection.y === -this.direction.y) {
                newDirection = this.direction; // Keep current direction
            }
        }

        this.direction = newDirection;

        const foodEaten = this.update();
        this.movesSinceLastFood++;

        let reward = 0;

        if (this.gameOver) {
            reward = -1;
        } else if (foodEaten) {
            reward = 10;
            this.movesSinceLastFood = 0;
        } else {
            reward -= 0.01;
        }

        // Check if the snake has gone too long without eating
        if (this.movesSinceLastFood >= this.maxMovesWithoutFood) {
            this.gameOver = true;
            reward = -1;
            this.collisionType = 'starvation';
        }

        this.draw();

        return {
            reward: reward,
            done: this.gameOver
        };
    }

    draw() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw game area background
        this.ctx.fillStyle = 'white';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw snake
        this.ctx.fillStyle = 'green';
        this.snake.forEach(segment => {
            this.ctx.fillRect(segment.x * this.tileSize, segment.y * this.tileSize, this.tileSize, this.tileSize);
        });

        // Draw food
        this.ctx.fillStyle = 'red';
        this.ctx.fillRect(this.food.x * this.tileSize, this.food.y * this.tileSize, this.tileSize, this.tileSize);

        // Draw grid lines
        this.ctx.strokeStyle = 'rgba(0, 0, 0, 0.1)';
        this.ctx.lineWidth = 1;
        for (let i = 0; i <= this.gridSize; i++) {
            this.ctx.beginPath();
            this.ctx.moveTo(i * this.tileSize, 0);
            this.ctx.lineTo(i * this.tileSize, this.canvas.height);
            this.ctx.stroke();

            this.ctx.beginPath();
            this.ctx.moveTo(0, i * this.tileSize);
            this.ctx.lineTo(this.canvas.width, i * this.tileSize);
            this.ctx.stroke();
        }
    }

    setDirection(direction) {
        this.direction = direction;
    }
}