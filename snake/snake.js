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
        this.snake = [{ x: Math.floor(this.gridSize / 2), y: Math.floor(this.gridSize / 2) }];
        this.direction = { x: 0, y: -1 }; // Start moving upwards
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
            console.log('Snake hit a wall!'); // New log
            return false;
        }

        // Check collision with self
        if (this.snake.some(segment => segment.x === head.x && segment.y === head.y)) {
            this.gameOver = true;
            this.collisionType = 'self';
            console.log('Snake hit itself!'); // New log
            return false;
        }

        this.snake.unshift(head);

        // Check if food is eaten
        if (head.x === this.food.x && head.y === this.food.y) {
            this.score++;
            this.food = this.generateFood();
            console.log('Snake ate food! Score:', this.score); // New log
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
            console.log('Snake starved!');
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