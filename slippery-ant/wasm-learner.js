// Runs a StreamAC's per-step math in WebAssembly (stream-ac.wasm, built from
// tools/wasm-learner). stream-ac.js stays the reference: it initializes,
// saves and loads agents, and this class copies their state in and out.
//
// Usage per step, mirroring the JavaScript loop:
//   learner.resetObs(env.reset())               after every reset
//   const a = learner.act()                     Float32Array view, valid until the next call
//   learner.learn(r.obs, r.reward, r.terminated, r.truncated)

const BUF = { actorW: 0, criticW: 1, actorScale: 2, criticScale: 3, obsMean: 10, obsVar: 11, obsP: 12,
              retMean: 13, retVar: 14, retP: 15, raw: 20, action: 21 };

export class WasmLearner {
    static async create(wasmSource, agent, seed = 1) {
        const bytes = wasmSource instanceof ArrayBuffer || ArrayBuffer.isView(wasmSource)
            ? wasmSource
            : await (await fetch(wasmSource)).arrayBuffer();
        const { instance } = await WebAssembly.instantiate(bytes, {});
        return new WasmLearner(instance.exports, agent, seed);
    }

    constructor(x, agent, seed) {
        this.x = x;
        this.obsDim = agent.obsDim;
        this.actDim = agent.actDim;
        const hp = agent.hp;
        x.init(agent.obsDim, agent.actDim, hp.hidden, hp.gamma, hp.lambda, hp.beta, hp.lrPolicy, hp.lrValue, hp.entropy, seed >>> 0);
        this.inp = x.padded_input();
        this.learning = true;
        this.copyIn(agent);
    }

    f32(id, n) { return new Float32Array(this.x.memory.buffer, this.x.buf(id), n); }
    f64(id, n) { return new Float64Array(this.x.memory.buffer, this.x.buf(id), n); }

    // JS trunks are unpadded; the wasm layout pads layer-1 rows and every segment to 4 floats
    trunkCopy(trunk, net, toWasm, id) {
        const nSeg = trunk.offsets.length;
        const wasm = this.f32(id, this.x.offset(net, nSeg));
        for (let k = 0; k < nSeg; k++) {
            const jo = trunk.offsets[k], jn = (trunk.offsets[k + 1] ?? trunk.n) - jo, wo = this.x.offset(net, k);
            if (k === 0) {
                for (let j = 0; j < trunk.h; j++) {
                    for (let i = 0; i < trunk.inDim; i++) {
                        if (toWasm) wasm[wo + j * this.inp + i] = trunk.w[jo + j * trunk.inDim + i];
                        else trunk.w[jo + j * trunk.inDim + i] = wasm[wo + j * this.inp + i];
                    }
                }
            } else if (toWasm) {
                wasm.set(trunk.w.subarray(jo, jo + jn), wo);
            } else {
                trunk.w.set(wasm.subarray(wo, wo + jn), jo);
            }
        }
    }

    // The optimizers' step scales share the weights' layout; borrow trunkCopy via a stand-in
    scaleCopy(trunk, scale, net, toWasm, id) {
        this.trunkCopy({ ...trunk, w: scale, offsets: trunk.offsets, n: trunk.n, h: trunk.h, inDim: trunk.inDim }, net, toWasm, id);
    }

    copyIn(agent) {
        this.trunkCopy(agent.actor, 0, true, BUF.actorW);
        this.trunkCopy(agent.critic, 1, true, BUF.criticW);
        this.scaleCopy(agent.actor, agent.optPi.v, 0, true, BUF.actorScale);
        this.scaleCopy(agent.critic, agent.optV.v, 1, true, BUF.criticScale);
        const d = this.obsDim;
        this.f64(BUF.obsMean, d).set(agent.obsStats.mean);
        this.f64(BUF.obsVar, d).set(agent.obsStats.var);
        this.f64(BUF.obsP, d).set(agent.obsStats.p);
        this.f64(BUF.retMean, 1).set(agent.retStats.mean);
        this.f64(BUF.retVar, 1).set(agent.retStats.var);
        this.f64(BUF.retP, 1).set(agent.retStats.p);
        this.x.set_counts(agent.obsStats.count, agent.retStats.count);
        this.x.reset_traces();
    }

    // Write the learned state back into a StreamAC (e.g. to save it)
    copyOut(agent) {
        this.trunkCopy(agent.actor, 0, false, BUF.actorW);
        this.trunkCopy(agent.critic, 1, false, BUF.criticW);
        this.scaleCopy(agent.actor, agent.optPi.v, 0, false, BUF.actorScale);
        this.scaleCopy(agent.critic, agent.optV.v, 1, false, BUF.criticScale);
        const d = this.obsDim;
        agent.obsStats.mean.set(this.f64(BUF.obsMean, d));
        agent.obsStats.var.set(this.f64(BUF.obsVar, d));
        agent.obsStats.p.set(this.f64(BUF.obsP, d));
        agent.obsStats.count = this.x.counts(0);
        agent.retStats.mean.set(this.f64(BUF.retMean, 1));
        agent.retStats.var.set(this.f64(BUF.retVar, 1));
        agent.retStats.p.set(this.f64(BUF.retP, 1));
        agent.retStats.count = this.x.counts(1);
        return agent;
    }

    resetObs(rawObs) {
        this.f64(BUF.raw, this.obsDim).set(rawObs);
        this.x.reset_obs(this.learning ? 1 : 0);
    }

    act() {
        this.x.act();
        return this.f32(BUF.action, this.actDim);
    }

    learn(rawNextObs, reward, terminated, truncated) {
        this.f64(BUF.raw, this.obsDim).set(rawNextObs);
        return this.x.learn(reward, terminated ? 1 : 0, truncated ? 1 : 0, this.learning ? 1 : 0);
    }

    resetTraces() { this.x.reset_traces(); }

    get lastDelta() { return this.x.stat(0); }
    get lastValue() { return this.x.stat(1); }
    get actorPush() { return this.x.stat(2); }
    get criticPush() { return this.x.stat(3); }
}
