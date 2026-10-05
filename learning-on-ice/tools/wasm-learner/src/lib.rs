//! Stream-AC's per-step math, mirroring ../../stream-ac.js line for line but in
//! f32 with WebAssembly SIMD (4 lanes). One global agent; JavaScript owns
//! initialization and checkpoints and reaches the buffers through `buf(id)`.
//!
//! Build: cargo build --release   (see build.sh, which copies the .wasm next to the page)
#![allow(static_mut_refs, clippy::missing_safety_doc)]
use core::arch::wasm32::*;

const SLOPE: f32 = 0.01;
const LN_EPS: f32 = 1e-5;

fn pad4(n: usize) -> usize {
    (n + 3) & !3
}

#[inline(always)]
unsafe fn ld(p: *const f32) -> v128 {
    v128_load(p as *const v128)
}

#[inline(always)]
unsafe fn st(p: *mut f32, v: v128) {
    v128_store(p as *mut v128, v)
}

#[inline(always)]
fn hsum(v: v128) -> f32 {
    f32x4_extract_lane::<0>(v) + f32x4_extract_lane::<1>(v) + f32x4_extract_lane::<2>(v) + f32x4_extract_lane::<3>(v)
}

// y[j] = b[j] + W[j,:]·x, rows of width n (a multiple of 4)
unsafe fn dense(w: *const f32, b: *const f32, x: *const f32, n: usize, out: usize, y: *mut f32) {
    for j in 0..out {
        let row = w.add(j * n);
        let mut a0 = f32x4_splat(0.0);
        let mut a1 = f32x4_splat(0.0);
        let mut i = 0;
        while i + 8 <= n {
            a0 = f32x4_add(a0, f32x4_mul(ld(row.add(i)), ld(x.add(i))));
            a1 = f32x4_add(a1, f32x4_mul(ld(row.add(i + 4)), ld(x.add(i + 4))));
            i += 8;
        }
        if i < n {
            a0 = f32x4_add(a0, f32x4_mul(ld(row.add(i)), ld(x.add(i))));
        }
        *y.add(j) = *b.add(j) + hsum(f32x4_add(a0, a1));
    }
}

// g[row + i] = dj·a[i] and d[i] += W[row + i]·dj over a row of width n
#[inline(always)]
unsafe fn row_back(w: *const f32, g: *mut f32, a: *const f32, d: *mut f32, n: usize, dj: f32, accumulate: bool) {
    let djv = f32x4_splat(dj);
    let mut i = 0;
    while i < n {
        st(g.add(i), f32x4_mul(djv, ld(a.add(i))));
        if accumulate {
            st(d.add(i), f32x4_add(ld(d.add(i)), f32x4_mul(ld(w.add(i)), djv)));
        }
        i += 4;
    }
}

// LayerNorm (no affine) then LeakyReLU; returns 1/std
fn ln_act(z: &[f32], n: &mut [f32], a: &mut [f32]) -> f32 {
    let h = z.len() as f32;
    let m = z.iter().sum::<f32>() / h;
    let v = z.iter().map(|&q| (q - m) * (q - m)).sum::<f32>() / h;
    let s = 1.0 / (v + LN_EPS).sqrt();
    for i in 0..z.len() {
        n[i] = (z[i] - m) * s;
        a[i] = if n[i] > 0.0 { n[i] } else { SLOPE * n[i] };
    }
    s
}

// Gradient w.r.t. the activation -> gradient w.r.t. the pre-LayerNorm input, in place
fn ln_act_back(n: &[f32], s: f32, d: &mut [f32]) {
    let h = n.len() as f32;
    let (mut m1, mut m2) = (0.0f32, 0.0f32);
    for i in 0..n.len() {
        d[i] *= if n[i] > 0.0 { 1.0 } else { SLOPE };
        m1 += d[i];
        m2 += d[i] * n[i];
    }
    m1 /= h;
    m2 /= h;
    for i in 0..n.len() {
        d[i] = s * (d[i] - m1 - n[i] * m2);
    }
}

struct Trunk {
    inp: usize,
    h: usize,
    heads: Vec<usize>,
    // W1, b1, W2, b2, then (W, b) per head; every segment padded to a multiple of 4
    off: Vec<usize>,
    w: Vec<f32>,
    g: Vec<f32>,
    x: Vec<f32>,
    z1: Vec<f32>,
    n1: Vec<f32>,
    a1: Vec<f32>,
    z2: Vec<f32>,
    n2: Vec<f32>,
    a2: Vec<f32>,
    s1: f32,
    s2: f32,
    out: Vec<Vec<f32>>,
    d1: Vec<f32>,
    d2: Vec<f32>,
    tz: Vec<f32>,
    tn: Vec<f32>,
    ta: Vec<f32>,
    tout: Vec<Vec<f32>>,
}

impl Trunk {
    fn new(inp: usize, h: usize, heads: Vec<usize>) -> Self {
        let mut sizes = vec![h * inp, h, h * h, h];
        for &k in &heads {
            sizes.push(k * h);
            sizes.push(k);
        }
        let mut off = Vec::new();
        let mut o = 0;
        for s in sizes {
            off.push(o);
            o += pad4(s);
        }
        let z = |n: usize| vec![0.0f32; n];
        Trunk {
            inp, h,
            off,
            w: z(o), g: z(o),
            x: z(inp), z1: z(h), n1: z(h), a1: z(h), z2: z(h), n2: z(h), a2: z(h),
            s1: 0.0, s2: 0.0,
            out: heads.iter().map(|&k| z(pad4(k))).collect(),
            d1: z(h), d2: z(h),
            tz: z(h), tn: z(h), ta: z(h),
            tout: heads.iter().map(|&k| z(pad4(k))).collect(),
            heads,
        }
    }

    unsafe fn forward_cached(&mut self, x: &[f32]) {
        let (h, inp) = (self.h, self.inp);
        let w = self.w.as_ptr();
        self.x.copy_from_slice(x);
        dense(w.add(self.off[0]), w.add(self.off[1]), self.x.as_ptr(), inp, h, self.z1.as_mut_ptr());
        self.s1 = ln_act(&self.z1, &mut self.n1, &mut self.a1);
        dense(w.add(self.off[2]), w.add(self.off[3]), self.a1.as_ptr(), h, h, self.z2.as_mut_ptr());
        self.s2 = ln_act(&self.z2, &mut self.n2, &mut self.a2);
        for (i, &k) in self.heads.iter().enumerate() {
            dense(w.add(self.off[4 + 2 * i]), w.add(self.off[5 + 2 * i]), self.a2.as_ptr(), h, k, self.out[i].as_mut_ptr());
        }
    }

    // Evaluate without touching the cached activations; writes self.tout
    unsafe fn forward_scratch(&mut self, x: &[f32]) {
        let (h, inp) = (self.h, self.inp);
        let w = self.w.as_ptr();
        dense(w.add(self.off[0]), w.add(self.off[1]), x.as_ptr(), inp, h, self.tz.as_mut_ptr());
        ln_act(&self.tz, &mut self.tn, &mut self.ta);
        dense(w.add(self.off[2]), w.add(self.off[3]), self.ta.as_ptr(), h, h, self.tz.as_mut_ptr());
        ln_act(&self.tz, &mut self.tn, &mut self.ta);
        for (i, &k) in self.heads.iter().enumerate() {
            dense(w.add(self.off[4 + 2 * i]), w.add(self.off[5 + 2 * i]), self.ta.as_ptr(), h, k, self.tout[i].as_mut_ptr());
        }
    }

    // Given d(loss)/d(heads), write d(loss)/d(parameters) into self.g
    unsafe fn backward(&mut self, dheads: &[&[f32]]) {
        let (h, inp) = (self.h, self.inp);
        let w = self.w.as_ptr();
        let g = self.g.as_mut_ptr();
        self.d1.fill(0.0);
        self.d2.fill(0.0);
        for (hi, &k) in self.heads.iter().enumerate() {
            let (wo, bo) = (self.off[4 + 2 * hi], self.off[5 + 2 * hi]);
            for j in 0..k {
                let dj = dheads[hi][j];
                *g.add(bo + j) = dj;
                row_back(w.add(wo + j * h), g.add(wo + j * h), self.a2.as_ptr(), self.d2.as_mut_ptr(), h, dj, true);
            }
        }
        ln_act_back(&self.n2, self.s2, &mut self.d2);
        for j in 0..h {
            let dj = self.d2[j];
            *g.add(self.off[3] + j) = dj;
            let row = self.off[2] + j * h;
            row_back(w.add(row), g.add(row), self.a1.as_ptr(), self.d1.as_mut_ptr(), h, dj, true);
        }
        ln_act_back(&self.n1, self.s1, &mut self.d1);
        for j in 0..h {
            let dj = self.d1[j];
            *g.add(self.off[1] + j) = dj;
            let row = self.off[0] + j * inp;
            row_back(w.add(row), g.add(row), self.x.as_ptr(), core::ptr::null_mut(), inp, dj, false);
        }
    }
}

// StreamingOptimizer: e = γλe + g; v = max(βv, |δe|); w += lr·δ·e/(v + eps)
struct Opt {
    lr: f32,
    gl: f32,
    beta: f32,
    e: Vec<f32>,
    v: Vec<f32>,
    pushed: f32,
}

impl Opt {
    unsafe fn step(&mut self, t: &mut Trunk, delta: f32, reset: bool) {
        let n = t.w.len();
        let (e, v, w, g) = (self.e.as_mut_ptr(), self.v.as_mut_ptr(), t.w.as_mut_ptr(), t.g.as_ptr());
        let gl = f32x4_splat(self.gl);
        let beta = f32x4_splat(self.beta);
        let ad = f32x4_splat(delta.abs());
        let k = f32x4_splat(self.lr * delta);
        let eps = f32x4_splat(1e-8);
        let zero = f32x4_splat(0.0);
        let mut push = zero;
        let mut i = 0;
        while i < n {
            let ei = f32x4_add(f32x4_mul(gl, ld(e.add(i))), ld(g.add(i)));
            let de = f32x4_mul(f32x4_abs(ei), ad);
            let vi = f32x4_max(f32x4_mul(beta, ld(v.add(i))), de);
            st(v.add(i), vi);
            let den = f32x4_add(vi, eps);
            st(w.add(i), f32x4_add(ld(w.add(i)), f32x4_div(f32x4_mul(k, ei), den)));
            push = f32x4_add(push, f32x4_div(de, den));
            st(e.add(i), if reset { zero } else { ei });
            i += 4;
        }
        self.pushed = hsum(push) / n as f32;
    }
}

// SampleMeanStd (sample variance; var = 1 until two samples)
struct Stats {
    mean: Vec<f64>,
    var: Vec<f64>,
    p: Vec<f64>,
    count: f64,
}

impl Stats {
    fn new(n: usize) -> Self {
        Stats { mean: vec![0.0; n], var: vec![1.0; n], p: vec![0.0; n], count: 0.0 }
    }
    fn update(&mut self, x: &[f64]) {
        self.count += 1.0;
        let n = self.count;
        for i in 0..self.mean.len() {
            if n == 1.0 {
                self.mean[i] = x[i];
                self.p[i] = 0.0;
                self.var[i] = 1.0;
                continue;
            }
            let m = self.mean[i] + (x[i] - self.mean[i]) / n;
            self.p[i] += (x[i] - self.mean[i]) * (x[i] - m);
            self.mean[i] = m;
            self.var[i] = self.p[i] / (n - 1.0);
        }
    }
}

struct Agent {
    obs_dim: usize,
    act_dim: usize,
    gamma: f32,
    entropy: f32,
    actor: Trunk,
    critic: Trunk,
    opt_pi: Opt,
    opt_v: Opt,
    obs: Stats,
    ret: Stats,
    ret_trace: f64,
    raw: Vec<f64>,
    s: Vec<f32>,
    s2: Vec<f32>,
    a: Vec<f32>,
    mu: Vec<f32>,
    std: Vec<f32>,
    dmu: Vec<f32>,
    dpre: Vec<f32>,
    rng: u32,
    last_delta: f32,
    last_value: f32,
}

static mut AGENT: Option<Agent> = None;

fn agent() -> &'static mut Agent {
    unsafe { AGENT.as_mut().unwrap() }
}

impl Agent {
    // mulberry32
    fn uniform(&mut self) -> f64 {
        self.rng = self.rng.wrapping_add(0x6D2B79F5);
        let mut t = self.rng;
        t = (t ^ (t >> 15)).wrapping_mul(t | 1);
        t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61));
        ((t ^ (t >> 14)) as f64) / 4294967296.0
    }

    fn gauss(&mut self) -> f32 {
        let mut u = 0.0;
        while u == 0.0 {
            u = self.uniform();
        }
        let v = self.uniform();
        ((-2.0 * u.ln()).sqrt() * (2.0 * core::f64::consts::PI * v).cos()) as f32
    }

    // NormalizeObservation on self.raw into `into` (s or s2)
    fn normalize(&mut self, update: bool, into_next: bool) {
        if update {
            self.obs.update(&self.raw);
        }
        let out = if into_next { &mut self.s2 } else { &mut self.s };
        for i in 0..self.obs_dim {
            out[i] = ((self.raw[i] - self.obs.mean[i]) / (self.obs.var[i] + 1e-8).sqrt()) as f32;
        }
    }

    // ScaleReward: divide by the running std of the discounted return
    fn scale_reward(&mut self, r: f64, done: bool) -> f32 {
        self.ret_trace = self.ret_trace * self.gamma as f64 + r;
        let t = [self.ret_trace];
        self.ret.update(&t);
        let scaled = r / (self.ret.var[0] + 1e-8).sqrt();
        if done {
            self.ret_trace = 0.0;
        }
        scaled as f32
    }
}

#[no_mangle]
pub extern "C" fn init(obs_dim: u32, act_dim: u32, hidden: u32, gamma: f32, lambda: f32, beta: f32,
                       lr_pi: f32, lr_v: f32, entropy: f32, seed: u32) {
    let (od, ad, h) = (obs_dim as usize, act_dim as usize, hidden as usize);
    let inp = pad4(od);
    let actor = Trunk::new(inp, h, vec![ad, ad]);
    let critic = Trunk::new(inp, h, vec![1]);
    let opt = |lr: f32, n: usize| Opt { lr, gl: gamma * lambda, beta, e: vec![0.0; n], v: vec![0.0; n], pushed: 0.0 };
    let opt_pi = opt(lr_pi, actor.w.len());
    let opt_v = opt(lr_v, critic.w.len());
    let z = |n: usize| vec![0.0f32; n];
    unsafe {
        AGENT = Some(Agent {
            obs_dim: od, act_dim: ad, gamma, entropy,
            actor, critic, opt_pi, opt_v,
            obs: Stats::new(od), ret: Stats::new(1), ret_trace: 0.0,
            raw: vec![0.0; od], s: z(inp), s2: z(inp),
            a: z(pad4(ad)), mu: z(pad4(ad)), std: z(pad4(ad)), dmu: z(pad4(ad)), dpre: z(pad4(ad)),
            rng: seed, last_delta: 0.0, last_value: 0.0,
        });
    }
}

// Pointers into the agent's buffers, so JavaScript can read and write them in place
#[no_mangle]
pub extern "C" fn buf(id: u32) -> *mut u8 {
    let a = agent();
    match id {
        0 => a.actor.w.as_mut_ptr() as *mut u8,
        1 => a.critic.w.as_mut_ptr() as *mut u8,
        2 => a.opt_pi.v.as_mut_ptr() as *mut u8,
        3 => a.opt_v.v.as_mut_ptr() as *mut u8,
        10 => a.obs.mean.as_mut_ptr() as *mut u8,
        11 => a.obs.var.as_mut_ptr() as *mut u8,
        12 => a.obs.p.as_mut_ptr() as *mut u8,
        13 => a.ret.mean.as_mut_ptr() as *mut u8,
        14 => a.ret.var.as_mut_ptr() as *mut u8,
        15 => a.ret.p.as_mut_ptr() as *mut u8,
        20 => a.raw.as_mut_ptr() as *mut u8,
        21 => a.a.as_mut_ptr() as *mut u8,
        22 => a.mu.as_mut_ptr() as *mut u8,
        23 => a.std.as_mut_ptr() as *mut u8,
        _ => core::ptr::null_mut(),
    }
}

// Segment offsets (in floats) of the padded layouts: net 0 = actor, 1 = critic
#[no_mangle]
pub extern "C" fn offset(net: u32, i: u32) -> u32 {
    let a = agent();
    let t = if net == 0 { &a.actor } else { &a.critic };
    *t.off.get(i as usize).unwrap_or(&t.w.len()) as u32
}

#[no_mangle]
pub extern "C" fn padded_input() -> u32 {
    agent().actor.inp as u32
}

#[no_mangle]
pub extern "C" fn set_counts(obs_count: f64, ret_count: f64) {
    let a = agent();
    a.obs.count = obs_count;
    a.ret.count = ret_count;
}

#[no_mangle]
pub extern "C" fn counts(which: u32) -> f64 {
    let a = agent();
    if which == 0 { a.obs.count } else { a.ret.count }
}

// After an env reset: s = normalize(raw)
#[no_mangle]
pub extern "C" fn reset_obs(learning: u32) {
    agent().normalize(learning != 0, false);
}

// Sample a ~ N(μ(s), σ(s)) into buf(21)
#[no_mangle]
pub extern "C" fn act() {
    let a = agent();
    // Raw slices: the trunk and the state live in different fields; no per-step allocation
    let s = unsafe { core::slice::from_raw_parts(a.s.as_ptr(), a.s.len()) };
    unsafe { a.actor.forward_cached(s) };
    for i in 0..a.act_dim {
        let pre = a.actor.out[1][i];
        a.mu[i] = a.actor.out[0][i];
        a.std[i] = if pre > 20.0 { pre } else { pre.exp().ln_1p() };
        let n = a.gauss();
        a.a[i] = a.mu[i] + a.std[i] * n;
    }
}

// One streaming update after env.step(): raw next observation in buf(20).
// Returns the TD error (0 when learning is off).
#[no_mangle]
pub extern "C" fn learn(reward: f64, terminated: u32, truncated: u32, learning: u32) -> f32 {
    let a = agent();
    let (term, done) = (terminated != 0, terminated != 0 || truncated != 0);
    a.normalize(learning != 0, true);
    if learning == 0 {
        if done {
            a.ret_trace = 0.0;
        }
        core::mem::swap(&mut a.s, &mut a.s2);
        return 0.0;
    }
    let r = a.scale_reward(reward, done);
    unsafe {
        // A time-limit cutoff still bootstraps; only a fall is terminal
        let v_next = if term {
            0.0
        } else {
            let s2 = core::slice::from_raw_parts(a.s2.as_ptr(), a.s2.len());
            a.critic.forward_scratch(s2);
            a.critic.tout[0][0]
        };
        let s = core::slice::from_raw_parts(a.s.as_ptr(), a.s.len());
        a.critic.forward_cached(s);
        let v = a.critic.out[0][0];
        let delta = r + a.gamma * v_next - v;
        a.last_delta = delta;
        a.last_value = v;

        // Critic follows ∇V(s); actor follows ∇[log π(a|s) + c·sign(δ)·H(π(·|s))]
        let one = [1.0f32];
        a.critic.backward(&[&one]);
        let sgn = if delta > 0.0 { 1.0 } else if delta < 0.0 { -1.0 } else { 0.0 };
        for i in 0..a.act_dim {
            let (sd, u) = (a.std[i], a.a[i] - a.mu[i]);
            a.dmu[i] = u / (sd * sd);
            let dstd = (u * u) / (sd * sd * sd) - 1.0 / sd + a.entropy * sgn / sd;
            let pre = a.actor.out[1][i];
            a.dpre[i] = dstd / (1.0 + (-pre).exp());
        }
        let dmu = core::slice::from_raw_parts(a.dmu.as_ptr(), a.dmu.len());
        let dpre = core::slice::from_raw_parts(a.dpre.as_ptr(), a.dpre.len());
        a.actor.backward(&[dmu, dpre]);
        a.opt_pi.step(&mut a.actor, delta, done);
        a.opt_v.step(&mut a.critic, delta, done);
        core::mem::swap(&mut a.s, &mut a.s2);
        delta
    }
}

#[no_mangle]
pub extern "C" fn reset_traces() {
    let a = agent();
    a.opt_pi.e.fill(0.0);
    a.opt_v.e.fill(0.0);
}

// Diagnostics: 0 = last TD error, 1 = last V(s), 2/3 = actor/critic push (mean |δe|/v)
#[no_mangle]
pub extern "C" fn stat(which: u32) -> f32 {
    let a = agent();
    match which {
        0 => a.last_delta,
        1 => a.last_value,
        2 => a.opt_pi.pushed,
        3 => a.opt_v.pushed,
        _ => 0.0,
    }
}
