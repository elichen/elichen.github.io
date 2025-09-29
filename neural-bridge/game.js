class AntBridgeSimulation {
    constructor() {
        this.canvas = document.getElementById('simulation');
        this.ctx = this.canvas.getContext('2d');
        this.pixelRatio = window.devicePixelRatio || 1;
        this.worldWidth = 960;
        this.worldHeight = 540;
        this.resizeCanvas();

        this.ants = [];
        this.pheromones = [];
        this.bridgeLinks = [];
        this.linkLength = 42;
        this.maxSag = 110;
        this.bridgeProgress = 0;
        this.bridgeComplete = false;

        this.debug = false;
        this.debugEvents = [];
        this.debugLastFrame = null;
        this.debugFrame = null;
        this.lastLoggedProgress = 0;
        this.startTime = performance.now();
        this.nextAntId = 0;

        this.groundY = 420;
        this.chasm = { start: 360, end: 600 };
        this.anchors = [
            { x: this.chasm.start, y: this.groundY },
            { x: this.chasm.end, y: this.groundY }
        ];
        this.gapDistance = this.distance(this.anchors[0], this.anchors[1]);

        this.stats = {
            coverage: document.getElementById('coverage'),
            attached: document.getElementById('attached'),
            foragers: document.getElementById('foragers'),
            crossing: document.getElementById('crossing'),
            status: document.getElementById('status')
        };

        document.getElementById('resetBtn').addEventListener('click', () => this.reset());
        document.getElementById('shuffleBtn').addEventListener('click', () => this.shuffleTerrain());
        window.addEventListener('resize', () => this.resizeCanvas());
        window.addEventListener('keydown', (event) => {
            if (event.key && event.key.toLowerCase() === 'd') {
                this.toggleDebug();
            }
        });

        this.reset();
        this.lastTime = performance.now();
        requestAnimationFrame((t) => this.loop(t));
    }

    resizeCanvas() {
        const ratio = this.pixelRatio;
        this.canvas.width = this.worldWidth * ratio;
        this.canvas.height = this.worldHeight * ratio;
        this.canvas.style.width = this.worldWidth + 'px';
        this.canvas.style.height = this.worldHeight + 'px';
        this.ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    }

    reset() {
        this.ants = [];
        this.pheromones = [];
        this.bridgeLinks = [];
        this.bridgeProgress = 0;
        this.bridgeComplete = false;
        this.statusMessage = 'Scouting the edge...';
        this.nextAntId = 0;
        this.startTime = performance.now();
        this.debugEvents = [];
        this.debugLastFrame = null;
        this.lastLoggedProgress = 0;

        for (let i = 0; i < 70; i++) {
            this.ants.push(this.createAnt('left'));
        }
        for (let i = 0; i < 40; i++) {
            this.ants.push(this.createAnt('right'));
        }
        this.updateHud();
        this.logDebug('Swarm reset', {
            gapDistance: parseFloat(this.gapDistance.toFixed(1)),
            totalAnts: this.ants.length
        });
    }

    shuffleTerrain() {
        const gapWidth = this.randomRange(210, 320);
        const gapCenter = this.randomRange(0.42, 0.58) * this.worldWidth;
        this.chasm.start = Math.max(220, gapCenter - gapWidth / 2);
        this.chasm.end = Math.min(this.worldWidth - 220, gapCenter + gapWidth / 2);
        this.anchors[0].x = this.chasm.start;
        this.anchors[1].x = this.chasm.end;
        this.gapDistance = this.distance(this.anchors[0], this.anchors[1]);
        this.reset();
        this.logDebug('Terrain shuffled', {
            newGapDistance: parseFloat(this.gapDistance.toFixed(1))
        });
    }

    createAnt(side) {
        const margin = 80;
        const x = side === 'left'
            ? this.randomRange(margin, this.chasm.start - 60)
            : this.randomRange(this.chasm.end + 60, this.worldWidth - margin);
        return {
            id: this.nextAntId++,
            x,
            y: this.groundY - 4 + Math.random() * 4,
            vx: 0,
            vy: 0,
            state: 'foraging',
            side,
            heading: Math.random() * Math.PI * 2,
            bridgeIndex: -1,
            crossingT: 0
        };
    }

    loop(timestamp) {
        const dt = Math.min((timestamp - this.lastTime) / 1000, 0.033);
        this.lastTime = timestamp;
        this.update(dt);
        this.draw();
        requestAnimationFrame((t) => this.loop(t));
    }

    update(dt) {
        this.updateBridgeTargets();
        let crossingCount = 0;

        if (this.debug) {
            const tailPoint = this.getAttachmentPoint();
            this.debugFrame = {
                nearTail: 0,
                leftEdgeClamp: 0,
                tailX: parseFloat(tailPoint.x.toFixed(1)),
                tailY: parseFloat(tailPoint.y.toFixed(1)),
                tailToFarCliff: parseFloat(this.distance(tailPoint, this.anchors[1]).toFixed(1)),
                links: this.bridgeLinks.length
            };
        }

        for (const ant of this.ants) {
            if (ant.state === 'foraging') {
                this.updateForagingAnt(ant, dt);
            } else if (ant.state === 'bridge') {
                this.updateBridgeAnt(ant, dt);
            } else if (ant.state === 'crossing') {
                crossingCount++;
                this.updateCrossingAnt(ant, dt);
            }
        }

        this.pheromones = this.pheromones.filter((p) => {
            p.life -= dt;
            return p.life > 0;
        });

        this.updateHud(crossingCount);

        if (this.debug) {
            this.debugFrame.runtime = parseFloat(((performance.now() - this.startTime) / 1000).toFixed(2));
            this.debugLastFrame = this.debugFrame;
        } else {
            this.debugLastFrame = null;
            this.debugFrame = null;
        }
    }

    updateForagingAnt(ant, dt) {
        const isLeft = ant.side === 'left';
        const edgeTarget = isLeft ? this.chasm.start - 16 : this.chasm.end + 16;
        const dir = Math.sign(edgeTarget - ant.x) || (isLeft ? 1 : -1);
        ant.vx += dir * 18 * dt;
        ant.vx += (Math.random() - 0.5) * 70 * dt;
        ant.vx = this.clamp(ant.vx, -60, 60);
        ant.x += ant.vx * dt;
        ant.heading = Math.atan2(0, ant.vx || dir);

        if (isLeft && ant.x > this.chasm.start - 8) {
            ant.x = this.chasm.start - 8;
            ant.vx *= -0.2;
            if (this.debugFrame) {
                this.debugFrame.leftEdgeClamp++;
            }
        }
        if (!isLeft && ant.x < this.chasm.end + 8) {
            ant.x = this.chasm.end + 8;
            ant.vx *= -0.2;
        }

        const pheromoneChance = this.bridgeProgress > 0.2 ? 0.15 : 0.3;
        if (Math.random() < pheromoneChance * dt) {
            this.pheromones.push({
                x: ant.x + this.randomRange(-6, 6),
                y: ant.y + this.randomRange(-10, 6),
                life: this.randomRange(0.8, 1.6),
                strength: this.randomRange(0.4, 1)
            });
        }

        if (!this.bridgeComplete) {
            const attachmentPoint = this.getAttachmentPoint();
            const distToTail = this.distance({ x: ant.x, y: ant.y }, attachmentPoint);
            if (this.debugFrame && isLeft) {
                if (distToTail < 45) {
                    this.debugFrame.nearTail++;
                }
                this.debugFrame.tailToFarCliff = parseFloat(this.distance(attachmentPoint, this.anchors[1]).toFixed(1));
            }
            if (isLeft && distToTail < 26 && this.bridgeLinks.length < 60) {
                this.attachAntToBridge(ant);
                return;
            }
        } else if (isLeft && Math.abs(ant.x - this.anchors[0].x) < 12 && Math.random() < 0.4) {
            ant.state = 'crossing';
            ant.crossingT = 0;
            this.logDebug('Ant entering crossing flow', { antId: ant.id });
        }

        const floorMin = isLeft ? 70 : this.chasm.end + 40;
        const floorMax = isLeft ? this.chasm.start - 8 : this.worldWidth - 70;
        if (ant.x < floorMin) {
            ant.x = floorMin;
            ant.vx *= -0.3;
        }
        if (ant.x > floorMax) {
            ant.x = floorMax;
            ant.vx *= -0.3;
        }

        ant.y = this.groundY - 4 + Math.sin((ant.x + ant.heading) * 0.08) * 1.5;
    }

    attachAntToBridge(ant) {
        ant.state = 'bridge';
        ant.bridgeIndex = this.bridgeLinks.length;
        ant.vx = 0;
        ant.vy = 0;
        const count = this.bridgeLinks.push({ ant });
        this.statusMessage = 'Chain extending...';
        this.logDebug('Ant attached to bridge', {
            antId: ant.id,
            bridgeIndex: ant.bridgeIndex,
            totalLinks: count
        });
    }

    updateBridgeAnt(ant, dt) {
        if (!this.bridgeLinks.length) {
            ant.state = 'foraging';
            return;
        }
        const progress = this.bridgeProgress;
        const links = this.bridgeLinks.length;
        const segments = links + 1;
        const dirX = (this.anchors[1].x - this.anchors[0].x) / this.gapDistance;
        const dirY = (this.anchors[1].y - this.anchors[0].y) / this.gapDistance;
        const totalReach = this.linkLength * links;
        const normalizedIndex = links ? (ant.bridgeIndex + 1) / links : 1;
        let targetDist = this.bridgeComplete
            ? this.gapDistance * ((ant.bridgeIndex + 1) / segments)
            : totalReach * normalizedIndex;
        targetDist = Math.min(targetDist, this.gapDistance * 0.98);
        const sag = (1 - progress) * this.maxSag * Math.sin(Math.PI * (ant.bridgeIndex + 1) / segments);
        const targetX = this.anchors[0].x + dirX * targetDist;
        const targetY = this.anchors[0].y + dirY * targetDist + sag;
        const dx = targetX - ant.x;
        const dy = targetY - ant.y;
        ant.x += dx * this.clamp(8 * dt, 0, 0.3);
        ant.y += dy * this.clamp(10 * dt, 0, 0.38);
        ant.heading = Math.atan2(dy, dx);
    }

    updateCrossingAnt(ant, dt) {
        const path = this.getBridgeNodes();
        if (path.length < 2) {
            ant.state = 'foraging';
            return;
        }
        ant.crossingT += dt * 0.25;
        if (ant.crossingT >= 1) {
            ant.side = 'right';
            ant.state = 'foraging';
            ant.x = this.chasm.end + 24 + Math.random() * 10;
            ant.y = this.groundY - 4;
            ant.vx = 0;
            ant.heading = 0;
            this.logDebug('Ant finished crossing', { antId: ant.id });
            return;
        }
        const pos = this.interpolatePath(path, ant.crossingT);
        ant.x = pos.x;
        ant.y = pos.y - 6;
        ant.heading = pos.heading;
    }

    updateBridgeTargets() {
        const links = this.bridgeLinks.length;
        if (!links) {
            this.lastLoggedProgress = 0;
        }
        const prevProgress = this.bridgeProgress;
        const spanCapacity = links * this.linkLength;
        const progress = links ? this.clamp(spanCapacity / this.gapDistance, 0, 1) : 0;
        if (progress > this.bridgeProgress) {
            this.bridgeProgress = progress;
            if (progress >= 1 && !this.bridgeComplete) {
                this.bridgeComplete = true;
                this.statusMessage = 'Bridge locked - colony crossing!';
                this.logDebug('Bridge span completed', {
                    links,
                    totalLength: parseFloat((links * this.linkLength).toFixed(1))
                });
            }
        } else {
            this.bridgeProgress = links ? Math.max(this.bridgeProgress * 0.995, progress) : 0;
        }

        if (this.bridgeProgress > prevProgress && this.bridgeProgress - this.lastLoggedProgress >= 0.05) {
            this.lastLoggedProgress = this.bridgeProgress;
            this.logDebug('Bridge progress increased', {
                progressPercent: parseFloat((this.bridgeProgress * 100).toFixed(1)),
                links
            });
        }
    }

    getAttachmentPoint() {
        if (!this.bridgeLinks.length) {
            return this.anchors[0];
        }
        const lastAnt = this.bridgeLinks[this.bridgeLinks.length - 1].ant;
        return { x: lastAnt.x, y: lastAnt.y };
    }

    getBridgeNodes() {
        const nodes = [this.anchors[0]];
        for (const link of this.bridgeLinks) {
            nodes.push({ x: link.ant.x, y: link.ant.y });
        }
        if (this.bridgeProgress >= 1) {
            nodes.push(this.anchors[1]);
        }
        return nodes;
    }

    interpolatePath(nodes, t) {
        let total = 0;
        const segments = [];
        for (let i = 0; i < nodes.length - 1; i++) {
            const a = nodes[i];
            const b = nodes[i + 1];
            const d = this.distance(a, b);
            segments.push({ a, b, d });
            total += d;
        }
        const targetDist = t * total;
        let accum = 0;
        for (const seg of segments) {
            if (accum + seg.d >= targetDist) {
                const localT = (targetDist - accum) / seg.d;
                const x = seg.a.x + (seg.b.x - seg.a.x) * localT;
                const y = seg.a.y + (seg.b.y - seg.a.y) * localT;
                const heading = Math.atan2(seg.b.y - seg.a.y, seg.b.x - seg.a.x);
                return { x, y, heading };
            }
            accum += seg.d;
        }
        const lastSeg = segments[segments.length - 1];
        return {
            x: lastSeg.b.x,
            y: lastSeg.b.y,
            heading: Math.atan2(lastSeg.b.y - lastSeg.a.y, lastSeg.b.x - lastSeg.a.x)
        };
    }

    updateHud(crossingCount = 0) {
        this.stats.coverage.textContent = Math.round(this.bridgeProgress * 100) + '%';
        this.stats.attached.textContent = this.bridgeLinks.length.toString();
        const foragers = this.ants.filter((a) => a.state === 'foraging').length;
        this.stats.foragers.textContent = foragers.toString();
        this.stats.crossing.textContent = crossingCount.toString();
        this.stats.status.textContent = this.statusMessage;
    }

    draw() {
        this.ctx.clearRect(0, 0, this.worldWidth, this.worldHeight);
        this.drawBackground();
        this.drawPheromones();
        this.drawTerrain();
        this.drawBridge();
        this.drawAnts();
        this.drawDebugOverlay();
    }

    drawBackground() {
        const grd = this.ctx.createLinearGradient(0, 0, 0, this.worldHeight);
        grd.addColorStop(0, '#0e1621');
        grd.addColorStop(0.5, '#0a1015');
        grd.addColorStop(1, '#050608');
        this.ctx.fillStyle = grd;
        this.ctx.fillRect(0, 0, this.worldWidth, this.worldHeight);
    }

    drawTerrain() {
        this.ctx.fillStyle = '#090c11';
        this.ctx.fillRect(0, this.groundY, this.worldWidth, this.worldHeight - this.groundY);

        const cliffGradient = this.ctx.createLinearGradient(0, this.groundY - 160, 0, this.worldHeight);
        cliffGradient.addColorStop(0, '#1d272f');
        cliffGradient.addColorStop(1, '#080b0e');
        this.ctx.fillStyle = cliffGradient;
        this.ctx.fillRect(0, this.groundY - 160, this.chasm.start, this.worldHeight - (this.groundY - 160));
        this.ctx.fillRect(this.chasm.end, this.groundY - 160, this.worldWidth - this.chasm.end, this.worldHeight - (this.groundY - 160));

        const depth = 120;
        const waterGradient = this.ctx.createLinearGradient(0, this.groundY, 0, this.groundY + depth);
        waterGradient.addColorStop(0, 'rgba(30, 60, 90, 0.45)');
        waterGradient.addColorStop(1, 'rgba(10, 15, 20, 0.9)');
        this.ctx.fillStyle = waterGradient;
        this.ctx.beginPath();
        this.ctx.moveTo(this.chasm.start, this.groundY);
        this.ctx.lineTo(this.chasm.end, this.groundY);
        this.ctx.lineTo(this.chasm.end - 30, this.groundY + depth);
        this.ctx.lineTo(this.chasm.start + 30, this.groundY + depth);
        this.ctx.closePath();
        this.ctx.fill();

        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
        this.ctx.lineWidth = 2;
        for (let i = 0; i < 8; i++) {
            const y = this.groundY + i * 14;
            const alpha = 0.25 - i * 0.02;
            this.ctx.strokeStyle = `rgba(110, 160, 210, ${alpha})`;
            this.ctx.beginPath();
            this.ctx.moveTo(this.chasm.start + 20, y);
            this.ctx.lineTo(this.chasm.end - 20, y + 6);
            this.ctx.stroke();
        }

        this.ctx.fillStyle = '#364450';
        this.ctx.beginPath();
        this.ctx.arc(this.chasm.start, this.groundY, 7, 0, Math.PI * 2);
        this.ctx.arc(this.chasm.end, this.groundY, 7, 0, Math.PI * 2);
        this.ctx.fill();
    }

    drawPheromones() {
        for (const p of this.pheromones) {
            this.ctx.fillStyle = `rgba(77, 209, 161, ${p.life * 0.15})`;
            this.ctx.beginPath();
            this.ctx.arc(p.x, p.y, 12 * p.strength, 0, Math.PI * 2);
            this.ctx.fill();
        }
    }

    drawBridge() {
        const nodes = this.getBridgeNodes();
        if (nodes.length < 2) {
            return;
        }
        this.ctx.lineWidth = 8;
        const color = this.bridgeComplete ? '#c8883e' : '#8c5527';
        this.ctx.strokeStyle = color;
        this.ctx.lineCap = 'round';
        this.ctx.beginPath();
        this.ctx.moveTo(nodes[0].x, nodes[0].y);
        for (let i = 1; i < nodes.length; i++) {
            this.ctx.lineTo(nodes[i].x, nodes[i].y);
        }
        this.ctx.stroke();

        this.ctx.lineWidth = 3;
        this.ctx.strokeStyle = 'rgba(255, 204, 128, 0.25)';
        this.ctx.beginPath();
        this.ctx.moveTo(nodes[0].x, nodes[0].y - 4);
        for (let i = 1; i < nodes.length; i++) {
            const wobble = Math.sin((performance.now() * 0.002) + i) * 4 * (1 - this.bridgeProgress);
            this.ctx.lineTo(nodes[i].x, nodes[i].y - 4 + wobble);
        }
        this.ctx.stroke();
    }

    drawAnts() {
        for (const ant of this.ants) {
            this.drawAnt(ant);
        }
    }

    drawAnt(ant) {
        this.ctx.save();
        this.ctx.translate(ant.x, ant.y);
        this.ctx.rotate(ant.heading);
        const bodyColor = ant.state === 'bridge' ? '#2f1d14' : '#3a2617';
        const headColor = ant.state === 'bridge' ? '#5f3a22' : '#7a4b2a';

        this.ctx.fillStyle = bodyColor;
        this.ctx.beginPath();
        this.ctx.ellipse(-4, 0, 6, 4, 0, 0, Math.PI * 2);
        this.ctx.fill();

        this.ctx.fillStyle = headColor;
        this.ctx.beginPath();
        this.ctx.ellipse(4, 0, 4.5, 3.4, 0, 0, Math.PI * 2);
        this.ctx.fill();

        this.ctx.strokeStyle = 'rgba(0, 0, 0, 0.45)';
        this.ctx.lineWidth = 1;
        this.ctx.beginPath();
        this.ctx.moveTo(-2, -3);
        this.ctx.lineTo(-7, -6);
        this.ctx.moveTo(-1, 3);
        this.ctx.lineTo(-7, 6);
        this.ctx.stroke();

        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.35)';
        this.ctx.beginPath();
        this.ctx.moveTo(7, -1);
        this.ctx.lineTo(10, -5);
        this.ctx.moveTo(7, 1);
        this.ctx.lineTo(10, 5);
        this.ctx.stroke();

        this.ctx.restore();
    }

    drawDebugOverlay() {
        if (!this.debug) {
            return;
        }

        const frame = this.debugLastFrame || {};
        const runtime = frame.runtime !== undefined
            ? frame.runtime
            : parseFloat(((performance.now() - this.startTime) / 1000).toFixed(2));
        const lines = [
            'DEBUG MODE (press D to toggle)',
            `runtime: ${runtime}s`,
            `links: ${frame.links !== undefined ? frame.links : this.bridgeLinks.length}`,
            `progress: ${(this.bridgeProgress * 100).toFixed(1)}%`,
            `tail->far cliff: ${frame.tailToFarCliff !== undefined ? frame.tailToFarCliff : parseFloat(this.distance(this.getAttachmentPoint(), this.anchors[1]).toFixed(1))}`,
            `near-tail this frame: ${frame.nearTail || 0}`,
            `edge clamp hits: ${frame.leftEdgeClamp || 0}`,
            `bridgeComplete: ${this.bridgeComplete}`
        ];

        if (this.debugEvents.length) {
            lines.push('recent events:');
            const recent = this.debugEvents.slice(-4).reverse();
            for (const entry of recent) {
                lines.push(`[${entry.timestamp}s] ${entry.message}`);
            }
        }

        this.ctx.save();
        this.ctx.font = '12px "Source Code Pro", monospace';
        this.ctx.textBaseline = 'top';
        this.ctx.textAlign = 'left';
        const padding = 10;
        const lineHeight = 16;
        let boxWidth = 0;
        for (const line of lines) {
            const width = this.ctx.measureText(line).width;
            if (width > boxWidth) {
                boxWidth = width;
            }
        }
        const boxHeight = lineHeight * lines.length + padding * 2;
        const x = 16;
        const y = 16;

        this.ctx.fillStyle = 'rgba(6, 10, 14, 0.74)';
        this.ctx.fillRect(x, y, boxWidth + padding * 2, boxHeight);
        this.ctx.strokeStyle = 'rgba(245, 175, 25, 0.45)';
        this.ctx.lineWidth = 1;
        this.ctx.strokeRect(x, y, boxWidth + padding * 2, boxHeight);

        lines.forEach((line, index) => {
            let color;
            if (index === 0) {
                color = '#4dd1a1';
            } else if (line === 'recent events:') {
                color = '#9aa5b1';
            } else if (line.startsWith('[')) {
                color = '#f5af19';
            } else {
                color = '#f3f3f5';
            }
            this.ctx.fillStyle = color;
            this.ctx.fillText(line, x + padding, y + padding + index * lineHeight);
        });

        this.ctx.restore();
    }

    logDebug(message, data = {}, forceConsole = false) {
        const timestamp = parseFloat(((performance.now() - this.startTime) / 1000).toFixed(2));
        const entry = { timestamp, message, data };
        this.debugEvents.push(entry);
        if (this.debugEvents.length > 12) {
            this.debugEvents.shift();
        }
        if (this.debug || forceConsole) {
            if (data && Object.keys(data).length) {
                console.log(`[AntBridge ${timestamp}s] ${message}`, data);
            } else {
                console.log(`[AntBridge ${timestamp}s] ${message}`);
            }
        }
    }

    toggleDebug() {
        this.debug = !this.debug;
        this.logDebug(this.debug ? 'Debug overlay enabled (press D to hide)' : 'Debug overlay disabled', {}, true);
    }

    randomRange(min, max) {
        return min + Math.random() * (max - min);
    }

    distance(a, b) {
        const dx = a.x - b.x;
        const dy = a.y - b.y;
        return Math.sqrt(dx * dx + dy * dy);
    }

    clamp(value, min, max) {
        return Math.max(min, Math.min(max, value));
    }
}

window.addEventListener('DOMContentLoaded', () => {
    new AntBridgeSimulation();
});
