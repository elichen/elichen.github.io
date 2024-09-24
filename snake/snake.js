class SnakeGame {
    constructor(canvasId, gridSize = 20) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.gridSize = gridSize;
        this.tileSize = this.canvas.width / (this.gridSize + 2); // Add 2 for borders
        this.reset();
    }

    reset() {
        this.snake = [{ x: Math.floor(this.gridSize / 2), y: Math.floor(this.gridSize / 2) }];
        this.direction = { x: 0, y: -1 };
        this.food = this.generateFood();
        this.score = 0;
        this.gameOver = false;
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
            this.collisionType = 'wall'; // Track collision type
            return false;
        }

        // Check collision with self
        if (this.snake.some(segment => segment.x === head.x && segment.y === head.y)) {
            this.gameOver = true;
            this.collisionType = 'self'; // Track collision type
            return false;
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
        this.direction = directions[action];

        const oldHead = this.snake[0];
        const oldDistance = this.calculateDistanceToFood(oldHead);

        const foodEaten = this.update(); // Capture if food was eaten

        const newHead = this.snake[0];
        const newDistance = this.calculateDistanceToFood(newHead);

        let reward = 0;

        if (this.gameOver) {
            if (this.collisionType === 'wall') {
                console.log('Hit a wall! Penalty assigned: -1');
                reward = -1; // Penalty for hitting wall
            } else if (this.collisionType === 'self') {
                console.log('Hit itself! Penalty assigned: -1');
                reward = -1; // Higher penalty for self-collision
            }
        } else if (foodEaten) {
            console.log('Food Eaten! Reward assigned: 1');
            reward = 1; // Reward for eating food
        } else {
            // Small reward/penalty based on distance to food
            const distanceDifference = oldDistance - newDistance;
            reward = distanceDifference * 0.1; // Scale the reward/penalty
            
            // Small penalty for each move to encourage efficiency
            reward -= 0.01;
            
            // Additional penalty for moving away from food
            if (distanceDifference < 0) {
                reward -= 0.1;
            }
        }

        this.draw();

        return {
            state: this.getState(),
            reward: reward,
            done: this.gameOver
        };
    }

    draw() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw border
        this.ctx.fillStyle = 'black';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw game area
        this.ctx.fillStyle = 'white';
        this.ctx.fillRect(this.tileSize, this.tileSize, this.canvas.width - 2 * this.tileSize, this.canvas.height - 2 * this.tileSize);

        // Draw snake
        this.ctx.fillStyle = 'green';
        this.snake.forEach(segment => {
            this.ctx.fillRect((segment.x + 1) * this.tileSize, (segment.y + 1) * this.tileSize, this.tileSize, this.tileSize);
        });

        // Draw food
        this.ctx.fillStyle = 'red';
        this.ctx.fillRect((this.food.x + 1) * this.tileSize, (this.food.y + 1) * this.tileSize, this.tileSize, this.tileSize);
    }

    setDirection(direction) {
        this.direction = direction;
    }

    getState() {
        // Implement state representation for AI
        // This is a placeholder and needs to be expanded
        return {
            snake: this.snake,
            food: this.food,
            direction: this.direction
        };
    }

    calculateDistanceToFood(head) {
        return Math.abs(head.x - this.food.x) + Math.abs(head.y - this.food.y);
    }
}