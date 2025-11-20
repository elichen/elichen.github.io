/**
 * GEMINI DUNGEON
 * A Procedural Raycasting Engine written in Vanilla JS.
 * Demonstrates: Maze Generation, Raycasting (DDA), Dynamic Lighting, Collision Detection.
 */

const CONFIG = {
    fov: 60 * (Math.PI / 180),
    mapWidth: 32,
    mapHeight: 32,
    wallSize: 64,
    res: 2, // Resolution divisor (1 = full, 2 = half res for performance)
    renderDist: 20,
    lightDecay: 0.15,
    minimapScale: 6
};

// --- 1. Maze Generation (Recursive Backtracker) ---
class MazeGenerator {
    constructor(width, height) {
        this.width = width;
        this.height = height;
        this.grid = new Uint8Array(width * height);
        this.rng = Math.random; // Could seed this
    }

    generate() {
        // Fill with walls (1)
        this.grid.fill(1);

        const stack = [];
        const startX = 1;
        const startY = 1;

        this.grid[startY * this.width + startX] = 0;
        stack.push({ x: startX, y: startY });

        while (stack.length > 0) {
            const current = stack[stack.length - 1];
            const neighbors = this.getNeighbors(current.x, current.y);

            if (neighbors.length > 0) {
                const next = neighbors[Math.floor(this.rng() * neighbors.length)];
                // Remove wall between
                const wallX = (current.x + next.x) / 2;
                const wallY = (current.y + next.y) / 2;
                this.grid[wallY * this.width + wallX] = 0;
                this.grid[next.y * this.width + next.x] = 0;
                
                stack.push(next);
            } else {
                stack.pop();
            }
        }

        // Add some randomness (loops)
        for (let i = 0; i < this.width * this.height * 0.05; i++) {
            const x = Math.floor(this.rng() * (this.width - 2)) + 1;
            const y = Math.floor(this.rng() * (this.height - 2)) + 1;
            if (this.grid[y * this.width + x] === 1) {
                // Check if it connects two open spaces
                if (this.grid[y * this.width + (x + 1)] === 0 && this.grid[y * this.width + (x - 1)] === 0) {
                    this.grid[y * this.width + x] = 0;
                }
            }
        }

        return this.grid;
    }

    getNeighbors(x, y) {
        const neighbors = [];
        const dirs = [
            { x: 0, y: -2 }, { x: 2, y: 0 }, { x: 0, y: 2 }, { x: -2, y: 0 }
        ];

        for (const d of dirs) {
            const nx = x + d.x;
            const ny = y + d.y;
            if (nx > 0 && nx < this.width - 1 && ny > 0 && ny < this.height - 1) {
                if (this.grid[ny * this.width + nx] === 1) {
                    neighbors.push({ x: nx, y: ny });
                }
            }
        }
        return neighbors;
    }
}

// --- 2. Engine Core ---
class Game {
    constructor() {
        this.canvas = document.getElementById('gameCanvas');
        this.ctx = this.canvas.getContext('2d', { alpha: false });
        this.miniCanvas = document.getElementById('minimap');
        this.miniCtx = this.miniCanvas.getContext('2d');
        
        this.resize();
        window.addEventListener('resize', () => this.resize());

        this.keys = {};
        this.bindInput();

        this.lights = []; // Dynamic light sources
        this.init();
    }

    init() {
        this.gen = new MazeGenerator(CONFIG.mapWidth, CONFIG.mapHeight);
        this.map = this.gen.generate();
        
        // Find empty spot for player
        let px = 1.5, py = 1.5;
        while(this.map[Math.floor(py) * CONFIG.mapWidth + Math.floor(px)] !== 0) {
            px += 1;
        }

        this.player = {
            x: px,
            y: py,
            dir: 0, // Radians
            plane: { x: 0, y: 0.66 }, // FOV plane
            speed: 3.5,
            rotSpeed: 2.5
        };

        this.lastTime = 0;
        this.frameCount = 0;
        this.lastFpsTime = 0;

        // Add initial light at start
        this.lights = [{ x: px, y: py, intensity: 2.5, r: 1.0, g: 0.8, b: 0.6 }];

        document.getElementById('seed-display').textContent = Math.floor(Math.random() * 9999);
        
        requestAnimationFrame((t) => this.loop(t));
    }

    resize() {
        const w = this.canvas.clientWidth;
        const h = this.canvas.clientHeight;
        this.canvas.width = w / CONFIG.res;
        this.canvas.height = h / CONFIG.res;
        this.width = this.canvas.width;
        this.height = this.canvas.height;

        // Minimap
        this.miniCanvas.width = CONFIG.mapWidth * CONFIG.minimapScale;
        this.miniCanvas.height = CONFIG.mapHeight * CONFIG.minimapScale;
    }

    bindInput() {
        window.addEventListener('keydown', e => {
            this.keys[e.code] = true;
            if(e.code === 'Space') this.placeLight();
            if(e.code === 'KeyR') this.init();
            if(e.code === 'KeyM') this.toggleMap();
        });
        window.addEventListener('keyup', e => this.keys[e.code] = false);
    }

    placeLight() {
        if (this.lights.length < 20) {
            this.lights.push({
                x: this.player.x,
                y: this.player.y,
                intensity: 2.0,
                r: Math.random() * 0.5 + 0.5,
                g: Math.random() * 0.5 + 0.5,
                b: Math.random() * 0.5 + 0.5
            });
            document.getElementById('light-count').textContent = this.lights.length;
        }
    }

    toggleMap() {
        this.miniCanvas.style.display = this.miniCanvas.style.display === 'none' ? 'block' : 'none';
    }

    update(dt) {
        const moveStep = this.player.speed * dt;
        const rotStep = this.player.rotSpeed * dt;

        // Rotation
        if (this.keys['ArrowLeft']) this.player.dir -= rotStep;
        if (this.keys['ArrowRight']) this.player.dir += rotStep;

        // Movement with collision
        const newX = this.player.x + Math.cos(this.player.dir) * moveStep * (this.keys['KeyW'] ? 1 : this.keys['KeyS'] ? -1 : 0);
        const newY = this.player.y + Math.sin(this.player.dir) * moveStep * (this.keys['KeyW'] ? 1 : this.keys['KeyS'] ? -1 : 0);
        
        // Strafe
        const strafeX = Math.cos(this.player.dir + Math.PI/2) * moveStep * (this.keys['KeyA'] ? -1 : this.keys['KeyD'] ? 1 : 0);
        const strafeY = Math.sin(this.player.dir + Math.PI/2) * moveStep * (this.keys['KeyA'] ? -1 : this.keys['KeyD'] ? 1 : 0);

        const finalX = (this.keys['KeyW'] || this.keys['KeyS'] ? newX : this.player.x) + (this.keys['KeyA'] || this.keys['KeyD'] ? strafeX : 0);
        const finalY = (this.keys['KeyW'] || this.keys['KeyS'] ? newY : this.player.y) + (this.keys['KeyA'] || this.keys['KeyD'] ? strafeY : 0);

        // Simple collision check
        if (this.map[Math.floor(finalY) * CONFIG.mapWidth + Math.floor(finalX)] === 0) {
            this.player.x = finalX;
            this.player.y = finalY;
        } else {
            // Sliding collision (try X only)
            if (this.map[Math.floor(this.player.y) * CONFIG.mapWidth + Math.floor(finalX)] === 0) this.player.x = finalX;
            // Try Y only
            else if (this.map[Math.floor(finalY) * CONFIG.mapWidth + Math.floor(this.player.x)] === 0) this.player.y = finalY;
        }
    }

    // --- 3. Raycasting Core (DDA Algorithm) ---
    render() {
        // Clear floor and ceiling
        // Gradient ceiling
        const grad = this.ctx.createLinearGradient(0, 0, 0, this.height / 2);
        grad.addColorStop(0, '#000');
        grad.addColorStop(1, '#1a1a1a');
        this.ctx.fillStyle = grad;
        this.ctx.fillRect(0, 0, this.width, this.height / 2);

        // Gradient floor
        const floorGrad = this.ctx.createLinearGradient(0, this.height / 2, 0, this.height);
        floorGrad.addColorStop(0, '#1a1a1a');
        floorGrad.addColorStop(1, '#2d2d2d');
        this.ctx.fillStyle = floorGrad;
        this.ctx.fillRect(0, this.height / 2, this.width, this.height / 2);

        // Raycasting
        const dirX = Math.cos(this.player.dir);
        const dirY = Math.sin(this.player.dir);
        const planeX = -dirY * 0.66; // FOV plane perpendicular to direction
        const planeY = dirX * 0.66;

        for (let x = 0; x < this.width; x++) {
            const cameraX = 2 * x / this.width - 1; // -1 to 1
            const rayDirX = dirX + planeX * cameraX;
            const rayDirY = dirY + planeY * cameraX;

            let mapX = Math.floor(this.player.x);
            let mapY = Math.floor(this.player.y);

            let sideDistX, sideDistY;
            
            const deltaDistX = Math.abs(1 / rayDirX);
            const deltaDistY = Math.abs(1 / rayDirY);
            let perpWallDist;

            let stepX, stepY;
            let hit = 0;
            let side; // 0 for NS, 1 for EW

            if (rayDirX < 0) {
                stepX = -1;
                sideDistX = (this.player.x - mapX) * deltaDistX;
            } else {
                stepX = 1;
                sideDistX = (mapX + 1.0 - this.player.x) * deltaDistX;
            }
            if (rayDirY < 0) {
                stepY = -1;
                sideDistY = (this.player.y - mapY) * deltaDistY;
            } else {
                stepY = 1;
                sideDistY = (mapY + 1.0 - this.player.y) * deltaDistY;
            }

            // DDA Hit Loop
            let dist = 0;
            while (hit === 0 && dist < CONFIG.renderDist) {
                if (sideDistX < sideDistY) {
                    sideDistX += deltaDistX;
                    mapX += stepX;
                    side = 0;
                } else {
                    sideDistY += deltaDistY;
                    mapY += stepY;
                    side = 1;
                }
                
                if (mapX >= 0 && mapX < CONFIG.mapWidth && mapY >= 0 && mapY < CONFIG.mapHeight) {
                    if (this.map[mapY * CONFIG.mapWidth + mapX] > 0) hit = 1;
                } else {
                    hit = 1; // Hit edge of world
                }
            }

            if (side === 0) perpWallDist = (mapX - this.player.x + (1 - stepX) / 2) / rayDirX;
            else           perpWallDist = (mapY - this.player.y + (1 - stepY) / 2) / rayDirY;

            // Calculate height of line to draw
            const lineHeight = Math.floor(this.height / perpWallDist);
            let drawStart = -lineHeight / 2 + this.height / 2;
            if (drawStart < 0) drawStart = 0;
            let drawEnd = lineHeight / 2 + this.height / 2;
            if (drawEnd >= this.height) drawEnd = this.height - 1;

            // --- Lighting Calculation ---
            let wallXExact = side === 0 ? this.player.y + perpWallDist * rayDirY : this.player.x + perpWallDist * rayDirX;
            let wallYExact = side === 0 ? mapY : mapX; // Not exact but sufficient for grid check
            // Actually we need the precise world hit coordinates for dynamic light distance
            let hitX = this.player.x + rayDirX * perpWallDist;
            let hitY = this.player.y + rayDirY * perpWallDist;

            // Ambient light (distance fade)
            let light = 1.0 / (1.0 + perpWallDist * perpWallDist * 0.1);
            
            // Dynamic lights contribution
            let r = 0, g = 0, b = 0;
            
            for(let l of this.lights) {
                const dx = hitX - l.x;
                const dy = hitY - l.y;
                const distSq = dx*dx + dy*dy;
                const att = l.intensity / (1.0 + distSq * 2.0); // Inverse square law approx
                r += l.r * att;
                g += l.g * att;
                b += l.b * att;
            }

            // Texture shading (procedural stripes/noise)
            const texX = side === 0 ? (this.player.y + perpWallDist * rayDirY) : (this.player.x + perpWallDist * rayDirX);
            const texCoord = Math.floor((texX - Math.floor(texX)) * 64);
            const pattern = (texCoord % 8 === 0 || texCoord % 13 === 0) ? 0.8 : 1.0;

            // Final Color Mixing
            // Base wall color
            let baseR = side === 1 ? 100 : 140;
            let baseG = side === 1 ? 100 : 140;
            let baseB = side === 1 ? 100 : 140;

            // Apply lighting
            // light var is ambient white
            r += light; g += light; b += light;

            // Clamp and mix
            let finR = Math.min(255, baseR * r * pattern);
            let finG = Math.min(255, baseG * g * pattern);
            let finB = Math.min(255, baseB * b * pattern);

            this.ctx.fillStyle = `rgb(${Math.floor(finR)}, ${Math.floor(finG)}, ${Math.floor(finB)})`;
            this.ctx.fillRect(x, drawStart, 1, drawEnd - drawStart);
        }
        
        this.renderMinimap();
    }

    renderMinimap() {
        const mw = CONFIG.mapWidth;
        const mh = CONFIG.mapHeight;
        const s = CONFIG.minimapScale;

        this.miniCtx.fillStyle = '#000';
        this.miniCtx.fillRect(0, 0, this.miniCanvas.width, this.miniCanvas.height);

        // Draw walls
        this.miniCtx.fillStyle = '#444';
        for(let y=0; y<mh; y++) {
            for(let x=0; x<mw; x++) {
                if(this.map[y*mw+x] > 0) {
                    this.miniCtx.fillRect(x*s, y*s, s, s);
                }
            }
        }

        // Draw Lights
        for(let l of this.lights) {
            this.miniCtx.fillStyle = `rgb(${l.r*255}, ${l.g*255}, ${l.b*255})`;
            this.miniCtx.beginPath();
            this.miniCtx.arc(l.x*s, l.y*s, 2, 0, Math.PI*2);
            this.miniCtx.fill();
        }

        // Draw Player
        this.miniCtx.fillStyle = '#0f0';
        this.miniCtx.beginPath();
        this.miniCtx.arc(this.player.x*s, this.player.y*s, 3, 0, Math.PI*2);
        this.miniCtx.fill();

        // Draw View Cone
        this.miniCtx.strokeStyle = '#0f0';
        this.miniCtx.beginPath();
        this.miniCtx.moveTo(this.player.x*s, this.player.y*s);
        this.miniCtx.lineTo(
            (this.player.x + Math.cos(this.player.dir) * 5) * s,
            (this.player.y + Math.sin(this.player.dir) * 5) * s
        );
        this.miniCtx.stroke();
    }

    loop(timestamp) {
        const dt = Math.min((timestamp - this.lastTime) / 1000, 0.1);
        this.lastTime = timestamp;

        this.update(dt);
        this.render();

        // FPS calc
        this.frameCount++;
        if (timestamp - this.lastFpsTime >= 1000) {
            document.getElementById('fps-counter').textContent = this.frameCount;
            this.frameCount = 0;
            this.lastFpsTime = timestamp;
        }

        requestAnimationFrame((t) => this.loop(t));
    }
}

// Start
window.onload = () => new Game();
