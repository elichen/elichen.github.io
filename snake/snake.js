class SnakeGame {
    constructor(canvasId, gridSize = 20) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.gridSize = gridSize;
        this.tileSize = this.canvas.width / this.gridSize;
        this.bg = null; // cached background+grid layer, built lazily on first draw
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
        this.won = false;
        this.collisionType = null;
        this.movesSinceLastFood = 0;
    }

    generateFood() {
        // Collect empty cells; if none, the board is full (a win) -> no food.
        const occupied = new Set(this.snake.map(s => s.y * this.gridSize + s.x));
        if (occupied.size >= this.gridSize * this.gridSize) return null;
        let food;
        do {
            food = {
                x: Math.floor(Math.random() * this.gridSize),
                y: Math.floor(Math.random() * this.gridSize)
            };
        } while (occupied.has(food.y * this.gridSize + food.x));
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
        if (this.food && head.x === this.food.x && head.y === this.food.y) {
            this.score++;
            // Board full after growing = win (snake fills all gridSize^2 cells)
            if (this.snake.length >= this.gridSize * this.gridSize) {
                this.won = true;
                this.gameOver = true;
                this.food = null;
                return true;
            }
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
        const ctx = this.ctx;
        if (!this.bg) this._buildBackground();

        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        ctx.drawImage(this.bg, 0, 0);

        // A slow sine drives the food's pulse and glow.
        const now = (typeof performance !== 'undefined') ? performance.now() : 0;
        const pulse = 0.5 + 0.5 * Math.sin(now / 360);

        if (this.food) this._drawFood(this.food, pulse);
        this._drawSnake();
    }

    // Bake the dark board, glowing grid and vignette once into an offscreen
    // canvas so every frame is a single drawImage instead of ~40 line strokes.
    _buildBackground() {
        const W = this.canvas.width, H = this.canvas.height, ts = this.tileSize;
        const off = document.createElement('canvas');
        off.width = W;
        off.height = H;
        const x = off.getContext('2d');

        const bg = x.createRadialGradient(W / 2, H * 0.42, 30, W / 2, H / 2, W * 0.78);
        bg.addColorStop(0, '#102a20');
        bg.addColorStop(1, '#04080a');
        x.fillStyle = bg;
        x.fillRect(0, 0, W, H);

        // A crisp coordinate lattice: visible minor cells, brighter every 5th
        // line, so the board reads as an instrumented simulation grid.
        x.lineWidth = 1;
        for (let i = 0; i <= this.gridSize; i++) {
            const major = (i % 5 === 0);
            x.strokeStyle = major ? 'rgba(110, 240, 205, 0.34)' : 'rgba(100, 230, 195, 0.13)';
            x.lineWidth = major ? 1.25 : 1;
            x.beginPath(); x.moveTo(i * ts, 0); x.lineTo(i * ts, H); x.stroke();
            x.beginPath(); x.moveTo(0, i * ts); x.lineTo(W, i * ts); x.stroke();
        }

        // Keep a gentle vignette for depth, but light enough that the grid
        // stays readable all the way to the edges.
        const vig = x.createRadialGradient(W / 2, H / 2, W * 0.46, W / 2, H / 2, W * 0.86);
        vig.addColorStop(0, 'rgba(0, 0, 0, 0)');
        vig.addColorStop(1, 'rgba(0, 0, 0, 0.34)');
        x.fillStyle = vig;
        x.fillRect(0, 0, W, H);

        this.bg = off;
    }

    // The snake is one rounded neon tube. Stacked additive strokes fake a glow
    // localised to the body, which scales to a full board far better than a
    // full-canvas shadowBlur would.
    _drawSnake() {
        const ctx = this.ctx, ts = this.tileSize, c = ts / 2;
        const path = new Path2D();
        this.snake.forEach((seg, i) => {
            const x = seg.x * ts + c, y = seg.y * ts + c;
            if (i === 0) path.moveTo(x, y); else path.lineTo(x, y);
        });

        ctx.save();
        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';
        ctx.globalCompositeOperation = 'lighter';
        ctx.strokeStyle = 'rgba(20, 150, 100, 0.35)'; ctx.lineWidth = ts * 0.86; ctx.stroke(path);
        ctx.strokeStyle = 'rgba(29, 200, 140, 0.40)'; ctx.lineWidth = ts * 0.60; ctx.stroke(path);
        ctx.strokeStyle = 'rgba(90, 255, 190, 0.55)'; ctx.lineWidth = ts * 0.42; ctx.stroke(path);
        ctx.globalCompositeOperation = 'source-over';
        ctx.strokeStyle = 'rgba(216, 255, 238, 0.95)'; ctx.lineWidth = ts * 0.26; ctx.stroke(path);
        ctx.restore();

        this._drawHead();
    }

    _drawHead() {
        const ctx = this.ctx, ts = this.tileSize, c = ts / 2;
        const h = this.snake[0];
        const x = h.x * ts + c, y = h.y * ts + c;

        ctx.save();
        ctx.shadowColor = '#7dffd0';
        ctx.shadowBlur = ts * 0.8;
        const g = ctx.createRadialGradient(x, y, 0, x, y, ts * 0.5);
        g.addColorStop(0, 'rgba(240, 255, 250, 1)');
        g.addColorStop(0.55, 'rgba(120, 255, 200, 0.95)');
        g.addColorStop(1, 'rgba(29, 255, 160, 0.18)');
        ctx.fillStyle = g;
        ctx.beginPath(); ctx.arc(x, y, ts * 0.42, 0, Math.PI * 2); ctx.fill();
        ctx.restore();

        // Eyes face the direction of travel.
        const d = this.direction;
        const perp = { x: -d.y, y: d.x };
        const fwd = ts * 0.10, sep = ts * 0.17;
        for (const side of [-1, 1]) {
            const ex = x + d.x * fwd + perp.x * sep * side;
            const ey = y + d.y * fwd + perp.y * sep * side;
            ctx.fillStyle = '#06231a';
            ctx.beginPath(); ctx.arc(ex, ey, ts * 0.075, 0, Math.PI * 2); ctx.fill();
        }
    }

    _drawFood(food, pulse) {
        const ctx = this.ctx, ts = this.tileSize, c = ts / 2;
        const x = food.x * ts + c, y = food.y * ts + c;
        const r = ts * (0.30 + 0.06 * pulse);

        ctx.save();
        ctx.shadowColor = '#ff2d55';
        ctx.shadowBlur = ts * (0.55 + 0.55 * pulse);
        const g = ctx.createRadialGradient(x - ts * 0.09, y - ts * 0.09, 0, x, y, r);
        g.addColorStop(0, 'rgba(255, 238, 238, 1)');
        g.addColorStop(0.35, 'rgba(255, 93, 115, 1)');
        g.addColorStop(1, 'rgba(255, 45, 85, 0.85)');
        ctx.fillStyle = g;
        ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
        ctx.restore();
    }

    // A glowing cyan comet wake behind the head: brightest and widest at the
    // newest position, tapering and fading toward the oldest. Drawn additively
    // so it reads as light over the board and the snake.
    drawActionTrail(headPositions) {
        if (!headPositions || headPositions.length < 2) return;

        const ts = this.tileSize, c = ts / 2, ctx = this.ctx;
        const n = headPositions.length;
        const px = (p) => p.x * ts + c;
        const py = (p) => p.y * ts + c;

        ctx.save();
        ctx.globalCompositeOperation = 'lighter';
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        // Streak: connect the recent positions with a tapering bright line.
        for (let i = 1; i < n; i++) {
            const t = i / (n - 1); // 0 = oldest, 1 = newest
            const a = headPositions[i - 1], b = headPositions[i];
            ctx.beginPath();
            ctx.moveTo(px(a), py(a));
            ctx.lineTo(px(b), py(b));
            ctx.lineWidth = ts * (0.08 + 0.34 * t);
            ctx.strokeStyle = `rgba(120, 240, 255, ${0.04 + 0.16 * t})`;
            ctx.stroke();
        }

        // Comet glow blobs, growing toward the head.
        for (let i = 0; i < n; i++) {
            const t = i / (n - 1);
            const x = px(headPositions[i]), y = py(headPositions[i]);
            const r = ts * (0.10 + 0.5 * t * t);
            const core = 0.08 + 0.55 * t;
            const g = ctx.createRadialGradient(x, y, 0, x, y, r);
            g.addColorStop(0, `rgba(235, 253, 255, ${core})`);
            g.addColorStop(0.4, `rgba(56, 232, 255, ${core * 0.55})`);
            g.addColorStop(1, 'rgba(40, 180, 255, 0)');
            ctx.fillStyle = g;
            ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
        }

        ctx.restore();
    }

    setDirection(direction) {
        this.direction = direction;
    }
}
