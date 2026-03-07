// Double Pendulum on Cart — policy inference
// Physics: Lagrangian dynamics, RK4 integration
// Policy: 2×256 MLP with ReLU, tanh output — SAC base refined by closed-loop distillation

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

function smoothstep(edge0, edge1, x) {
    const t = clamp((x - edge0) / (edge1 - edge0), 0, 1);
    return t * t * (3 - 2 * t);
}

// ── Environment ──────────────────────────────────────────────────────────────

class DoublePendulumEnv {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');

        this.gravity = 9.8;
        this.cartMass = 1.0;
        this.m1 = 0.5;
        this.m2 = 0.5;
        this.L1 = 1.0;
        this.L2 = 1.0;
        this.forceMag = 100.0;
        this.dt = 0.02;
        this.jointDamping = 1.0;
        this.xLimit = 5.0;
        this.maxSteps = 500;

        this.scale = (canvas.width - 120) / (2 * this.xLimit);
        // Keep the trained dynamics intact while giving the rods a larger on-canvas span.
        this.visualLengthScale = 1.45;

        this.state = null;
        this.steps = 0;
        this.episodeReturn = 0;
        this.episodeAbsCart = 0;

        // Trail for tip position (visual only)
        this.trail = [];
        this.maxTrail = 80;
    }

    reset() {
        this.state = [
            (Math.random() - 0.5) * 0.1, 0.0,
            Math.PI + (Math.random() - 0.5) * 0.2, 0.0,
            Math.PI + (Math.random() - 0.5) * 0.2, 0.0
        ];
        this.steps = 0;
        this.episodeReturn = 0;
        this.episodeAbsCart = 0;
        this.trail = [];
        return this.getObs();
    }

    getObs() {
        const [x, xd, t1, t1d, t2, t2d] = this.state;
        return [x, xd, Math.sin(t1), Math.cos(t1), t1d, Math.sin(t2), Math.cos(t2), t2d];
    }

    bothUpright() {
        return Math.cos(this.state[2]) > 0.9 && Math.cos(this.state[4]) > 0.9;
    }

    uprightness() {
        return 0.5 * (Math.cos(this.state[2]) + Math.cos(this.state[4]));
    }

    derivatives(s, F) {
        const [x, xd, t1, t1d, t2, t2d] = s;
        const { gravity: g, cartMass: M, m1, m2, L1: l1, L2: l2, jointDamping: damp } = this;

        const c1 = Math.cos(t1), s1 = Math.sin(t1);
        const c2 = Math.cos(t2), s2 = Math.sin(t2);
        const c12 = Math.cos(t1 - t2), s12 = Math.sin(t1 - t2);

        const d1 = M + m1 + m2;
        const d2 = (m1 / 2 + m2) * l1;
        const d3 = m2 * l2 / 2;
        const d4 = (m1 / 3 + m2) * l1 * l1;
        const d5 = m2 * l1 * l2 / 2;
        const d6 = m2 * l2 * l2 / 3;

        const M11 = d1, M12 = d2 * c1, M13 = d3 * c2;
        const M21 = d2 * c1, M22 = d4, M23 = d5 * c12;
        const M31 = d3 * c2, M32 = d5 * c12, M33 = d6;

        const f1 = F + d2 * t1d * t1d * s1 + d3 * t2d * t2d * s2;
        const f2 = d5 * t2d * t2d * s12 + (m1 / 2 + m2) * g * l1 * s1 - damp * t1d;
        const f3 = -d5 * t1d * t1d * s12 + m2 * g * l2 * s2 / 2 - damp * t2d;

        const det = M11 * (M22 * M33 - M23 * M32)
                  - M12 * (M21 * M33 - M23 * M31)
                  + M13 * (M21 * M32 - M22 * M31);

        if (Math.abs(det) < 1e-10) return [xd, F / d1, t1d, 0, t2d, 0];

        const xdd = (f1 * (M22 * M33 - M23 * M32) - M12 * (f2 * M33 - M23 * f3) + M13 * (f2 * M32 - M22 * f3)) / det;
        const t1dd = (M11 * (f2 * M33 - M23 * f3) - f1 * (M21 * M33 - M23 * M31) + M13 * (M21 * f3 - f2 * M31)) / det;
        const t2dd = (M11 * (M22 * f3 - f2 * M32) - M12 * (M21 * f3 - f2 * M31) + f1 * (M21 * M32 - M22 * M31)) / det;

        return [xd, xdd, t1d, t1dd, t2d, t2dd];
    }

    step(action) {
        this.steps++;
        const a = clamp(action, -1, 1);
        const F = a * this.forceMag;
        const s = [...this.state];
        const dt = this.dt;

        const k1 = this.derivatives(s, F);
        const k2 = this.derivatives(s.map((v, i) => v + 0.5 * dt * k1[i]), F);
        const k3 = this.derivatives(s.map((v, i) => v + 0.5 * dt * k2[i]), F);
        const k4 = this.derivatives(s.map((v, i) => v + dt * k3[i]), F);
        const ns = s.map((v, i) => v + (dt / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));

        ns[2] = ((ns[2] + Math.PI) % (2 * Math.PI)) - Math.PI;
        if (ns[2] < -Math.PI) ns[2] += 2 * Math.PI;
        ns[4] = ((ns[4] + Math.PI) % (2 * Math.PI)) - Math.PI;
        if (ns[4] < -Math.PI) ns[4] += 2 * Math.PI;

        this.state = ns;

        if (Math.abs(this.state[0]) > this.xLimit) {
            this.state[0] = this.xLimit * Math.sign(this.state[0]);
            this.state[1] = 0;
        }

        const reward = this.computeReward(a);
        this.episodeReturn += reward;
        this.episodeAbsCart += Math.abs(this.state[0]);
        const done = this.steps >= this.maxSteps;

        return {
            obs: this.getObs(),
            reward,
            done,
            episodeReturn: this.episodeReturn,
            meanAbsCart: this.episodeAbsCart / this.steps,
        };
    }

    computeReward(action) {
        const [x, xd, t1, t1d, t2, t2d] = this.state;
        const uprightReward = 1.0 + 0.5 * Math.cos(t1) + 0.5 * Math.cos(t2);
        const balanceBlend = smoothstep(0.55, 0.92, this.uprightness());
        const centerPenalty = (0.06 + 0.42 * balanceBlend) * (x / this.xLimit) ** 2;
        const cartVelocityPenalty = (0.004 + 0.018 * balanceBlend) * Math.min(xd * xd, 9);
        const angularVelocityPenalty = 0.0025 * Math.min(t1d * t1d + t2d * t2d, 49);
        const controlPenalty = (0.004 + 0.010 * balanceBlend) * action * action;
        const railPenalty = (0.04 + 0.14 * balanceBlend) * smoothstep(0.72, 1.0, Math.abs(x) / this.xLimit);
        const recenterBonus = 0.15 * balanceBlend * Math.exp(-0.5 * (x / 1.25) ** 2);
        return uprightReward - centerPenalty - cartVelocityPenalty - angularVelocityPenalty - controlPenalty - railPenalty + recenterBonus;
    }

    render() {
        const [x, , t1, , t2] = this.state;
        const ctx = this.ctx;
        const W = this.canvas.width, H = this.canvas.height;
        const cx = x * this.scale + W / 2;
        const cy = H * 0.52;
        const poleScale = this.scale * this.visualLengthScale;

        // ── Background
        ctx.fillStyle = '#111114';
        ctx.fillRect(0, 0, W, H);

        // ── Subtle grid
        ctx.strokeStyle = 'rgba(255,255,255,0.025)';
        ctx.lineWidth = 1;
        const gridSpacing = 40;
        for (let gx = gridSpacing; gx < W; gx += gridSpacing) {
            ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, H); ctx.stroke();
        }
        for (let gy = gridSpacing; gy < H; gy += gridSpacing) {
            ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(W, gy); ctx.stroke();
        }

        // ── Track
        const trackL = W / 2 - this.xLimit * this.scale;
        const trackR = W / 2 + this.xLimit * this.scale;
        const trackY = cy + 18;
        ctx.beginPath();
        ctx.moveTo(trackL, trackY);
        ctx.lineTo(trackR, trackY);
        ctx.strokeStyle = '#2a2a30';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Track ticks
        ctx.fillStyle = '#2a2a30';
        ctx.font = '9px JetBrains Mono, monospace';
        ctx.textAlign = 'center';
        for (let m = -4; m <= 4; m += 2) {
            const tx = W / 2 + m * this.scale;
            ctx.beginPath();
            ctx.moveTo(tx, trackY - 3);
            ctx.lineTo(tx, trackY + 3);
            ctx.stroke();
        }

        // ── Boundary lines
        ctx.strokeStyle = 'rgba(199, 93, 72, 0.2)';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 6]);
        ctx.beginPath(); ctx.moveTo(trackL, 0); ctx.lineTo(trackL, H); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(trackR, 0); ctx.lineTo(trackR, H); ctx.stroke();
        ctx.setLineDash([]);

        // ── Compute positions
        const j1x = cx + Math.sin(t1) * this.L1 * poleScale;
        const j1y = cy - Math.cos(t1) * this.L1 * poleScale;
        const tipx = j1x + Math.sin(t2) * this.L2 * poleScale;
        const tipy = j1y - Math.cos(t2) * this.L2 * poleScale;

        // ── Tip trail
        this.trail.push({ x: tipx, y: tipy });
        if (this.trail.length > this.maxTrail) this.trail.shift();

        if (this.trail.length > 2) {
            for (let i = 1; i < this.trail.length; i++) {
                const alpha = (i / this.trail.length) * 0.35;
                ctx.beginPath();
                ctx.moveTo(this.trail[i - 1].x, this.trail[i - 1].y);
                ctx.lineTo(this.trail[i].x, this.trail[i].y);
                ctx.strokeStyle = `rgba(212, 160, 57, ${alpha})`;
                ctx.lineWidth = 1.5;
                ctx.stroke();
            }
        }

        // ── Cart
        const cartW = 40, cartH = 22;
        ctx.fillStyle = '#3a3a42';
        ctx.strokeStyle = '#4a4a54';
        ctx.lineWidth = 1;
        const cartR = 3;
        ctx.beginPath();
        ctx.roundRect(cx - cartW / 2, cy - cartH / 2, cartW, cartH, cartR);
        ctx.fill();
        ctx.stroke();

        // Cart wheels
        ctx.fillStyle = '#2a2a30';
        ctx.beginPath();
        ctx.arc(cx - 12, cy + cartH / 2 + 2, 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.beginPath();
        ctx.arc(cx + 12, cy + cartH / 2 + 2, 4, 0, Math.PI * 2);
        ctx.fill();

        // ── Pole 1
        const up1 = (1 + Math.cos(t1)) / 2;
        const r1 = 180 - 120 * up1, g1 = 80 + 100 * up1, b1 = 60 + 40 * up1;

        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(j1x, j1y);
        ctx.strokeStyle = `rgb(${r1|0}, ${g1|0}, ${b1|0})`;
        ctx.lineWidth = 5;
        ctx.lineCap = 'round';
        ctx.stroke();

        // ── Joint
        ctx.beginPath();
        ctx.arc(j1x, j1y, 5, 0, Math.PI * 2);
        ctx.fillStyle = '#555';
        ctx.fill();
        ctx.strokeStyle = '#666';
        ctx.lineWidth = 1;
        ctx.stroke();

        // ── Pole 2
        const up2 = (1 + Math.cos(t2)) / 2;
        const r2 = 180 - 120 * up2, g2 = 80 + 100 * up2, b2 = 60 + 40 * up2;

        ctx.beginPath();
        ctx.moveTo(j1x, j1y);
        ctx.lineTo(tipx, tipy);
        ctx.strokeStyle = `rgb(${r2|0}, ${g2|0}, ${b2|0})`;
        ctx.lineWidth = 5;
        ctx.lineCap = 'round';
        ctx.stroke();

        // ── Tip mass
        ctx.beginPath();
        ctx.arc(tipx, tipy, 7, 0, Math.PI * 2);
        const tipGlow = this.bothUpright() ? 'rgba(212, 160, 57, 0.3)' : 'rgba(0,0,0,0)';
        ctx.fillStyle = `rgb(${r2|0}, ${g2|0}, ${b2|0})`;
        ctx.fill();
        ctx.strokeStyle = tipGlow;
        ctx.lineWidth = 3;
        ctx.stroke();

        // ── Cart pivot dot
        ctx.beginPath();
        ctx.arc(cx, cy, 3, 0, Math.PI * 2);
        ctx.fillStyle = '#666';
        ctx.fill();

        ctx.lineCap = 'butt';
    }
}

// ── SAC Policy (pure JS, no TF.js) ──────────────────────────────────────────

class SACPolicy {
    constructor(weights) {
        this.w0 = new Float32Array(weights.layer_0.kernel);
        this.b0 = new Float32Array(weights.layer_0.bias);
        this.w1 = new Float32Array(weights.layer_1.kernel);
        this.b1 = new Float32Array(weights.layer_1.bias);
        this.w2 = new Float32Array(weights.mu.kernel);
        this.b2 = new Float32Array(weights.mu.bias);
        this.h0 = new Float32Array(256);
        this.h1 = new Float32Array(256);
    }

    predict(obs) {
        for (let j = 0; j < 256; j++) {
            let sum = this.b0[j];
            for (let i = 0; i < 8; i++) sum += obs[i] * this.w0[i * 256 + j];
            this.h0[j] = sum > 0 ? sum : 0;
        }
        for (let j = 0; j < 256; j++) {
            let sum = this.b1[j];
            for (let i = 0; i < 256; i++) sum += this.h0[i] * this.w1[i * 256 + j];
            this.h1[j] = sum > 0 ? sum : 0;
        }
        let sum = this.b2[0];
        for (let i = 0; i < 256; i++) sum += this.h1[i] * this.w2[i];
        return Math.tanh(sum);
    }
}

// ── Main ─────────────────────────────────────────────────────────────────────

async function main() {
    const canvas = document.getElementById('canvas');
    const loading = document.getElementById('loading');
    const phaseBadge = document.getElementById('phaseBadge');
    const telEp = document.getElementById('telEp');
    const telReturn = document.getElementById('telReturn');
    const telAvg = document.getElementById('telAvg');
    const telBias = document.getElementById('telBias');

    const env = new DoublePendulumEnv(canvas);

    let policy;
    try {
        const resp = await fetch('trained-weights-double-sac.json');
        const weights = await resp.json();
        policy = new SACPolicy(weights);
    } catch (e) {
        loading.textContent = `Failed to load: ${e.message}`;
        return;
    }

    loading.classList.add('hidden');

    const returns = [];
    let epCount = 0;

    let obs = env.reset();
    let lastFrameTime = null;
    let accumulator = 0;

    function updatePhaseBadge() {
        const up = env.bothUpright();
        phaseBadge.textContent = up ? 'Balanced' : 'Swing-up';
        phaseBadge.className = 'phase-badge ' + (up ? 'balanced' : 'swingup');
    }

    function updateTelemetry(result) {
        epCount++;
        returns.push(result.episodeReturn);
        if (returns.length > 20) returns.shift();
        const avg = returns.reduce((a, b) => a + b, 0) / returns.length;

        telEp.textContent = epCount;
        telReturn.textContent = result.episodeReturn.toFixed(1);
        telAvg.textContent = avg.toFixed(1);
        telBias.innerHTML = result.meanAbsCart.toFixed(2) + '<span class="unit">m</span>';
    }

    function startEpisode() {
        obs = env.reset();
        accumulator = 0;
        updatePhaseBadge();
        env.render();
    }

    function animate(frameTime) {
        if (lastFrameTime === null) lastFrameTime = frameTime;
        const elapsed = clamp((frameTime - lastFrameTime) / 1000, 0, 0.1);
        lastFrameTime = frameTime;
        accumulator += elapsed;

        let result = null;
        while (accumulator >= env.dt) {
            result = env.step(policy.predict(obs));
            obs = result.obs;
            accumulator -= env.dt;

            if (result.done) {
                updateTelemetry(result);
                startEpisode();
                result = null;
                break;
            }
        }

        updatePhaseBadge();
        env.render();
        requestAnimationFrame(animate);
    }

    startEpisode();
    requestAnimationFrame(animate);
}

window.onload = main;
