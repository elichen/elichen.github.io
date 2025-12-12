/**
 * Swarm Simulation - Mouse-guided flocking
 */

class SwarmSimulation {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.agent = new MurmurationAgent();

        // Config (must match training parameters!)
        this.numBirds = 150;
        this.numNeighbors = 7;
        this.birdSpeed = 0.02;
        this.birdAcceleration = 0.008;

        // Action directions (8 + stay)
        this.actionDirs = [];
        for (let i = 0; i < 8; i++) {
            const angle = (i / 8) * Math.PI * 2;
            this.actionDirs.push([Math.cos(angle), Math.sin(angle)]);
        }
        this.actionDirs.push([0, 0]);

        // State
        this.birds = [];
        this.running = false;
        this.lastTime = 0;

        // Mouse attractor (in arena coords)
        this.attractor = { x: 0.5, y: 0.5 };
        this.mouseInCanvas = false;

        // Trail effect
        this.trailCanvas = document.createElement('canvas');
        this.trailCtx = null;

        this.resize();
        window.addEventListener('resize', () => this.resize());

        // Mouse tracking
        this.canvas.addEventListener('mousemove', (e) => this.onMouseMove(e));
        this.canvas.addEventListener('mouseenter', () => this.mouseInCanvas = true);
        this.canvas.addEventListener('mouseleave', () => this.mouseInCanvas = false);
    }

    onMouseMove(e) {
        const rect = this.canvas.getBoundingClientRect();
        const canvasX = e.clientX - rect.left;
        const canvasY = e.clientY - rect.top;

        // Convert to arena coords
        this.attractor.x = (canvasX / this.width) * this.arenaWidth;
        this.attractor.y = (canvasY / this.height) * this.arenaHeight;
    }

    async init() {
        await this.agent.load();
        this.reset();
        document.getElementById('loading').classList.add('hidden');
        this.start();
    }

    resize() {
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width * window.devicePixelRatio;
        this.canvas.height = rect.height * window.devicePixelRatio;
        this.ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        this.width = rect.width;
        this.height = rect.height;

        // Arena dimensions based on aspect ratio
        this.arenaHeight = 1.0;
        this.arenaWidth = this.width / this.height;

        // Update attractor to center
        this.attractor.x = this.arenaWidth / 2;
        this.attractor.y = this.arenaHeight / 2;

        // Trail canvas
        this.trailCanvas.width = this.width;
        this.trailCanvas.height = this.height;
        this.trailCtx = this.trailCanvas.getContext('2d');
    }

    reset() {
        const centerX = this.arenaWidth / 2;
        const centerY = this.arenaHeight / 2;
        this.birds = [];

        for (let i = 0; i < this.numBirds; i++) {
            const angle = Math.random() * Math.PI * 2;
            const speed = (0.5 + Math.random() * 0.5) * this.birdSpeed;
            this.birds.push({
                id: i,
                x: centerX + (Math.random() - 0.5) * 0.15,
                y: centerY + (Math.random() - 0.5) * 0.15,
                vx: Math.cos(angle) * speed,
                vy: Math.sin(angle) * speed,
            });
        }

        if (this.trailCtx) {
            this.trailCtx.fillStyle = '#1a1a2e';
            this.trailCtx.fillRect(0, 0, this.width, this.height);
        }

        this.render();
    }

    getNeighbors(birdIdx) {
        const bird = this.birds[birdIdx];
        const distances = this.birds.map((other, i) => ({
            idx: i,
            dist: i === birdIdx ? Infinity :
                Math.sqrt((other.x - bird.x) ** 2 + (other.y - bird.y) ** 2)
        }));
        distances.sort((a, b) => a.dist - b.dist);
        return distances.slice(0, this.numNeighbors).map(d => d.idx);
    }

    buildObservations() {
        const observations = [];

        for (let i = 0; i < this.birds.length; i++) {
            const bird = this.birds[i];

            // My velocity (normalized)
            const myVel = [bird.vx / this.birdSpeed, bird.vy / this.birdSpeed];

            // Direction to attractor (normalized)
            const toAttractorX = this.attractor.x - bird.x;
            const toAttractorY = this.attractor.y - bird.y;
            const distToAttractor = Math.sqrt(toAttractorX ** 2 + toAttractorY ** 2);
            let attractorDir;
            if (distToAttractor > 0.01) {
                attractorDir = [toAttractorX / distToAttractor, toAttractorY / distToAttractor];
            } else {
                attractorDir = [0, 0];
            }

            // Neighbors
            const neighbors = this.getNeighbors(i);
            const neighborObs = [];

            for (const n of neighbors) {
                const nRelPos = [
                    (this.birds[n].x - bird.x) / this.arenaHeight,
                    (this.birds[n].y - bird.y) / this.arenaHeight
                ];
                const nRelVel = [
                    (this.birds[n].vx - bird.vx) / this.birdSpeed,
                    (this.birds[n].vy - bird.vy) / this.birdSpeed
                ];
                neighborObs.push(...nRelPos, ...nRelVel);
            }

            observations.push([...myVel, ...attractorDir, ...neighborObs]);
        }

        return observations;
    }

    step() {
        const observations = this.buildObservations();
        const actions = this.agent.sample(observations);

        for (let i = 0; i < this.birds.length; i++) {
            const bird = this.birds[i];
            const actionDir = this.actionDirs[actions[i]];

            const turbulence = 0.002;
            bird.vx += actionDir[0] * this.birdAcceleration + (Math.random() - 0.5) * turbulence;
            bird.vy += actionDir[1] * this.birdAcceleration + (Math.random() - 0.5) * turbulence;

            const speed = Math.sqrt(bird.vx ** 2 + bird.vy ** 2);
            if (speed > this.birdSpeed) {
                bird.vx = bird.vx / speed * this.birdSpeed;
                bird.vy = bird.vy / speed * this.birdSpeed;
            } else if (speed < this.birdSpeed * 0.3 && speed > 0) {
                bird.vx = bird.vx / speed * this.birdSpeed * 0.3;
                bird.vy = bird.vy / speed * this.birdSpeed * 0.3;
            }

            bird.x += bird.vx;
            bird.y += bird.vy;

            // Soft boundary
            const margin = 0.15;
            const turnForce = 0.004;
            if (bird.x < margin) bird.vx += turnForce;
            if (bird.x > this.arenaWidth - margin) bird.vx -= turnForce;
            if (bird.y < margin) bird.vy += turnForce;
            if (bird.y > this.arenaHeight - margin) bird.vy -= turnForce;

            bird.x = Math.max(0.02, Math.min(this.arenaWidth - 0.02, bird.x));
            bird.y = Math.max(0.02, Math.min(this.arenaHeight - 0.02, bird.y));
        }
    }

    render() {
        const ctx = this.ctx;

        const toCanvas = (x, y) => [
            (x / this.arenaWidth) * this.width,
            (y / this.arenaHeight) * this.height
        ];

        // Trail fade
        this.trailCtx.fillStyle = 'rgba(26, 26, 46, 0.1)';
        this.trailCtx.fillRect(0, 0, this.width, this.height);

        // Birds on trail
        for (const bird of this.birds) {
            const [x, y] = toCanvas(bird.x, bird.y);
            this.trailCtx.fillStyle = 'rgba(255, 255, 255, 0.3)';
            this.trailCtx.beginPath();
            this.trailCtx.arc(x, y, 1.5, 0, Math.PI * 2);
            this.trailCtx.fill();
        }

        ctx.drawImage(this.trailCanvas, 0, 0);

        // Draw birds
        for (const bird of this.birds) {
            this.drawBird(bird, toCanvas);
        }

        // Draw attractor hint when mouse is in canvas
        if (this.mouseInCanvas) {
            const [ax, ay] = toCanvas(this.attractor.x, this.attractor.y);
            ctx.strokeStyle = 'rgba(124, 58, 237, 0.5)';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(ax, ay, 15, 0, Math.PI * 2);
            ctx.stroke();
        }
    }

    drawBird(bird, toCanvas) {
        const ctx = this.ctx;
        const [x, y] = toCanvas(bird.x, bird.y);

        ctx.fillStyle = '#fff';
        ctx.beginPath();
        ctx.arc(x, y, 2, 0, Math.PI * 2);
        ctx.fill();
    }

    start() {
        this.running = true;
        this.lastTime = performance.now();
        this.animate();
    }

    animate() {
        if (!this.running) return;

        const now = performance.now();
        const dt = (now - this.lastTime) / 1000;

        if (dt > 0.033) {
            this.step();
            this.lastTime = now;
        }

        this.render();
        requestAnimationFrame(() => this.animate());
    }
}

document.addEventListener('DOMContentLoaded', async () => {
    const sim = new SwarmSimulation('canvas');
    await sim.init();
});
