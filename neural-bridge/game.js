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
        this.bridgeLeft = [];
        this.bridgeRight = [];
        this.linkLength = 42;
        this.maxSag = 110;
        this.bridgeProgress = 0;
        this.bridgeComplete = false;
        this.bridgeTipGap = 0;

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

        this.debug = false;
        this.debugEvents = [];
        this.debugFrame = null;
        this.debugLastFrame = null;
        this.lastLoggedProgress = 0;
        this.startTime = performance.now();
        this.nextAntId = 0;

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
        this.bridgeLeft = [];
        this.bridgeRight = [];
        this.bridgeProgress = 0;
        this.bridgeComplete = false;
        this.bridgeTipGap = this.gapDistance;
        this.statusMessage = 'Scouting the edge...';
        this.nextAntId = 0;
        this.startTime = performance.now();
        this.debugEvents = [];
        this.debugFrame = null;
        this.debugLastFrame = null;
        this.lastLoggedProgress = 0;

        for (let i = 0; i < 70; i++) {
            this.ants.push(this.createAnt('left'));
        }
        for (let i = 0; i < 70; i++) {
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
            chain: null,
            crossingT: 0,
            crossingDirection: 'leftToRight'
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
        let crossingCount = 0;

        if (this.debug) {
            const leftTail = this.getTailPosition('left');
            const rightTail = this.getTailPosition('right');
            this.debugFrame = {
                nearTailLeft: 0,
                nearTailRight: 0,
                leftEdgeClamp: 0,
                rightEdgeClamp: 0,
                leftTailX: parseFloat(leftTail.x.toFixed(1)),
                leftTailY: parseFloat(leftTail.y.toFixed(1)),
                rightTailX: parseFloat(rightTail.x.toFixed(1)),
                rightTailY: parseFloat(rightTail.y.toFixed(1)),
                tipGap: parseFloat(this.distance(leftTail, rightTail).toFixed(1)),
                leftLinks: this.bridgeLeft.length,
                rightLinks: this.bridgeRight.length,
                links: this.getTotalLinks()
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

        this.updateBridgeMetrics();
        this.updateHud(crossingCount);

        if (this.debug) {
            this.debugFrame.runtime = parseFloat(((performance.now() - this.startTime) / 1000).toFixed(2));
            this.debugFrame.progress = parseFloat((this.bridgeProgress * 100).toFixed(1));
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
            if (this.debugFrame) {
                this.debugFrame.rightEdgeClamp++;
            }
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
            const attachmentPoint = this.getTailPosition(ant.side);
            const distToTail = this.distance({ x: ant.x, y: ant.y }, attachmentPoint);
            if (this.debugFrame) {
                if (isLeft && distToTail < 45) {
                    this.debugFrame.nearTailLeft++;
                }
                if (!isLeft && distToTail < 45) {
                    this.debugFrame.nearTailRight++;
                }
            }

            const ownChain = isLeft ? this.bridgeLeft : this.bridgeRight;
            const otherChain = isLeft ? this.bridgeRight : this.bridgeLeft;
            const chainBalanceOK = ownChain.length <= otherChain.length + 2;

            if (chainBalanceOK && distToTail < 26 && ownChain.length < 80) {
                this.attachAntToBridge(ant, ant.side);
                return;
            }
        } else {
            const anchor = isLeft ? this.anchors[0] : this.anchors[1];
            if (Math.abs(ant.x - anchor.x) < 12 && Math.random() < 0.4) {
                ant.state = 'crossing';
                ant.crossingT = 0;
                ant.crossingDirection = isLeft ? 'leftToRight' : 'rightToLeft';
                this.logDebug('Ant entering crossing flow', { antId: ant.id, side: ant.side });
                return;
            }
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

    attachAntToBridge(ant, side) {
        const chain = side === 'left' ? this.bridgeLeft : this.bridgeRight;
        ant.state = 'bridge';
        ant.chain = side;
        ant.bridgeIndex = chain.length;
        ant.vx = 0;
        ant.vy = 0;
        chain.push({ ant });
        this.statusMessage = 'Chains extending...';
        this.logDebug('Ant attached to bridge', {
            antId: ant.id,
            side,
            bridgeIndex: ant.bridgeIndex,
            leftLinks: this.bridgeLeft.length,
            rightLinks: this.bridgeRight.length
        });
    }

    updateBridgeAnt(ant, dt) {
        const totalLinks = this.getTotalLinks();
        if (!totalLinks) {
            ant.state = 'foraging';
            ant.chain = null;
            ant.bridgeIndex = -1;
            return;
        }
        const segments = totalLinks + 1;
        const dirX = (this.anchors[1].x - this.anchors[0].x) / this.gapDistance;
        const dirY = (this.anchors[1].y - this.anchors[0].y) / this.gapDistance;
        const globalIndex = ant.chain === 'left'
            ? ant.bridgeIndex + 1
            : segments - ant.bridgeIndex - 1;
        const t = globalIndex / segments;
        const spanX = this.anchors[0].x + dirX * this.gapDistance * t;
        const spanY = this.anchors[0].y + dirY * this.gapDistance * t;
        const sag = (1 - this.bridgeProgress) * this.maxSag * Math.sin(Math.PI * t);
        const targetX = spanX;
        const targetY = spanY + sag;
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
            ant.crossingDirection = 'leftToRight';
            return;
        }
        ant.crossingT += dt * 0.25;
        if (ant.crossingT >= 1) {
            const direction = ant.crossingDirection;
            if (direction === 'leftToRight') {
                ant.side = 'right';
                ant.x = this.anchors[1].x + 24 + Math.random() * 10;
            } else {
                ant.side = 'left';
                ant.x = this.anchors[0].x - 24 - Math.random() * 10;
            }
            ant.state = 'foraging';
            ant.crossingDirection = 'leftToRight';
            ant.y = this.groundY - 4;
            ant.vx = 0;
            ant.heading = direction === 'leftToRight' ? 0 : Math.PI;
            this.logDebug('Ant finished crossing', { antId: ant.id, direction });
            return;
        }
        const direction = ant.crossingDirection;
        const travelT = direction === 'leftToRight' ? ant.crossingT : 1 - ant.crossingT;
        const pos = this.interpolatePath(path, travelT);
        ant.x = pos.x;
        ant.y = pos.y - 6;
        ant.heading = direction === 'leftToRight' ? pos.heading : pos.heading + Math.PI;
    }

    updateBridgeMetrics() {
        const leftTail = this.getTailPosition('left');
        const rightTail = this.getTailPosition('right');
        const tipGap = this.distance(leftTail, rightTail);
        const coverage = this.gapDistance - tipGap;
        const progress = this.clamp(coverage / this.gapDistance, 0, 1);
        this.bridgeProgress = progress;
        this.bridgeTipGap = tipGap;

        if (!this.bridgeComplete) {
            if (this.bridgeLeft.length > 0 && this.bridgeRight.length > 0 && tipGap <= this.linkLength * 1.25) {
                this.bridgeComplete = true;
                this.statusMessage = 'Bridge locked - colony crossing!';
                this.logDebug('Bridge span completed', {
                    tipGap: parseFloat(tipGap.toFixed(1)),
                    leftLinks: this.bridgeLeft.length,
                    rightLinks: this.bridgeRight.length
                });
            }
        }

        if (!this.bridgeComplete && progress - this.lastLoggedProgress >= 0.05) {
            this.lastLoggedProgress = progress;
            this.logDebug('Bridge progress increased', {
                progressPercent: parseFloat((progress * 100).toFixed(1)),
                tipGap: parseFloat(tipGap.toFixed(1)),
                leftLinks: this.bridgeLeft.length,
                rightLinks: this.bridgeRight.length
            });
        }
    }

    getTailPosition(side) {
        if (side === 'left') {
            if (!this.bridgeLeft.length) {
                return { x: this.anchors[0].x, y: this.anchors[0].y };
            }
            const ant = this.bridgeLeft[this.bridgeLeft.length - 1].ant;
            return { x: ant.x, y: ant.y };
        }
        if (!this.bridgeRight.length) {
            return { x: this.anchors[1].x, y: this.anchors[1].y };
        }
        const ant = this.bridgeRight[this.bridgeRight.length - 1].ant;
        return { x: ant.x, y: ant.y };
    }

    getBridgeNodes() {
        if (!this.bridgeComplete) {
            return [];
        }
        const nodes = [this.anchors[0]];
        for (const link of this.bridgeLeft) {
            nodes.push({ x: link.ant.x, y: link.ant.y });
        }
        const rightNodes = [];
        for (const link of this.bridgeRight) {
            rightNodes.push({ x: link.ant.x, y: link.ant.y });
        }
        rightNodes.reverse();
        nodes.push(...rightNodes);
        nodes.push(this.anchors[1]);
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
        this.stats.attached.textContent = this.getTotalLinks().toString();
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
        const leftNodes = [this.anchors[0]];
        for (const link of this.bridgeLeft) {
            leftNodes.push({ x: link.ant.x, y: link.ant.y });
        }
        const rightNodes = [this.anchors[1]];
        for (const link of this.bridgeRight) {
            rightNodes.push({ x: link.ant.x, y: link.ant.y });
        }

        const drawPath = (nodes) => {
            if (nodes.length < 2) {
                return;
            }
            this.ctx.beginPath();
            this.ctx.moveTo(nodes[0].x, nodes[0].y);
            for (let i = 1; i < nodes.length; i++) {
                this.ctx.lineTo(nodes[i].x, nodes[i].y);
            }
            this.ctx.stroke();
        };

        this.ctx.lineCap = 'round';
        this.ctx.lineWidth = 8;
        this.ctx.strokeStyle = '#8c5527';
        drawPath(leftNodes);
        this.ctx.strokeStyle = '#8c5527';
        drawPath(rightNodes);

        if (!this.bridgeComplete && leftNodes.length > 1 && rightNodes.length > 1) {
            const leftTip = leftNodes[leftNodes.length - 1];
            const rightTip = rightNodes[rightNodes.length - 1];
            this.ctx.setLineDash([6, 6]);
            this.ctx.lineWidth = 2.5;
            this.ctx.strokeStyle = 'rgba(255, 204, 128, 0.35)';
            this.ctx.beginPath();
            this.ctx.moveTo(leftTip.x, leftTip.y);
            this.ctx.lineTo(rightTip.x, rightTip.y);
            this.ctx.stroke();
            this.ctx.setLineDash([]);
        }

        if (this.bridgeComplete) {
            const nodes = this.getBridgeNodes();
            if (nodes.length >= 2) {
                this.ctx.lineWidth = 10;
                this.ctx.strokeStyle = '#c8883e';
                drawPath(nodes);

                this.ctx.lineWidth = 3;
                this.ctx.strokeStyle = 'rgba(255, 204, 128, 0.25)';
                this.ctx.beginPath();
                this.ctx.moveTo(nodes[0].x, nodes[0].y - 4);
                for (let i = 1; i < nodes.length; i++) {
                    const wobble = Math.sin((performance.now() * 0.002) + i) * 2;
                    this.ctx.lineTo(nodes[i].x, nodes[i].y - 4 + wobble);
                }
                this.ctx.stroke();
            }
        }
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
            `left links: ${frame.leftLinks !== undefined ? frame.leftLinks : this.bridgeLeft.length}`,
            `right links: ${frame.rightLinks !== undefined ? frame.rightLinks : this.bridgeRight.length}`,
            `total links: ${frame.links !== undefined ? frame.links : this.getTotalLinks()}`,
            `progress: ${(this.bridgeProgress * 100).toFixed(1)}%`,
            `tip gap: ${frame.tipGap !== undefined ? frame.tipGap : parseFloat(this.bridgeTipGap.toFixed(1))}`,
            `near-tail left: ${frame.nearTailLeft || 0}`,
            `near-tail right: ${frame.nearTailRight || 0}`,
            `edge clamp left: ${frame.leftEdgeClamp || 0}`,
            `edge clamp right: ${frame.rightEdgeClamp || 0}`,
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

    getTotalLinks() {
        return this.bridgeLeft.length + this.bridgeRight.length;
    }
}

window.addEventListener('DOMContentLoaded', () => {
    new AntBridgeSimulation();
});
