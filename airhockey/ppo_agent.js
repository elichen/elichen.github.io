class PPOAgent {
    async load(path) {
        const w = new Float32Array(await (await fetch(path)).arrayBuffer());
        const sizes = Array.from(w.subarray(1, 1 + w[0]));
        let o = 1 + w[0];
        this.layers = sizes.slice(1).map((n, i) => {
            const m = sizes[i], W = w.subarray(o, o + n * m), b = w.subarray(o + n * m, o + n * m + n);
            o += n * m + n;
            return { W, b, m, n };
        });
    }

    act(state) {
        let x = Float32Array.from(state, v => v * 2 - 1);
        this.layers.forEach(({ W, b, m, n }, l) => {
            const y = new Float32Array(n);
            for (let j = 0; j < n; j++) {
                let s = b[j];
                for (let i = 0; i < m; i++) s += W[j * m + i] * x[i];
                y[j] = l < this.layers.length - 1 ? Math.tanh(s) : Math.max(-1, Math.min(1, s));
            }
            x = y;
        });
        return [x[0], x[1]];
    }

    // 12 features from the player's own side: own paddle, puck, opponent paddle (positions, then velocities); y=0 is the own goal line.
    getState(puck, playerPaddle, aiPaddle, isTopPlayer, W, H) {
        const own = isTopPlayer ? aiPaddle : playerPaddle, opp = isTopPlayer ? playerPaddle : aiPaddle, s = isTopPlayer ? 1 : -1;
        const p = o => [o.x / W, isTopPlayer ? o.y / H : (H - o.y) / H];
        const v = (dx, dy) => [dx, s * dy].map(c => Math.max(-1, Math.min(1, c / 25)) * 0.5 + 0.5);
        return [...p(own), ...p(puck), ...v(own.dx, own.dy), ...v(puck.dx, puck.dy), ...p(opp), ...v(opp.dx, opp.dy)];
    }
}
