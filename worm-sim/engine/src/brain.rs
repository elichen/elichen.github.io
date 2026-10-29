//! BAAIWorm's nervous system: 136 multicompartment neurons (23,848 segments),
//! graded synapses and gap junctions, stepped the way NEURON's fixed-step
//! backward Euler does it (secondorder = 0):
//!
//!   1. currents and their conductances at v(t), synapses and gap junctions
//!      reading the presynaptic v(t);
//!   2. one linear solve over each cell's tree (Hines) for v(t + dt);
//!   3. gating states, synapse states and calcium advanced with v(t + dt)
//!      (cnexp: exact exponential step; derivimplicit: backward Euler).
//!
//! Channel kinetics are the Nicoletti et al. (2019) models in
//! `eworm/components/mechanism/modfile`, transcribed with NEURON's reading of
//! their units: gbar in S/cm², currents in mA/cm² (the .mod files say nS/µm²
//! and pA/µm², which NEURON does not convert). Reversal potentials are the
//! ones hard-coded in the .mod files; kqt3 uses NEURON's default ek = -77 mV.
//!
//! Units here: area µm², capacitance nF, conductance µS, current nA, mV, ms.

pub const N_CHANNELS: usize = 16;
// order of the gbars in brain.bin (see tools/export_brain.py)
const SHL1: usize = 0;
const SHK1: usize = 1;
const KVS1: usize = 2;
const EGL2: usize = 3;
const EGL36: usize = 4;
const KQT3: usize = 5;
const EGL19: usize = 6;
const UNC2: usize = 7;
const CCA1: usize = 8;
const SLO1_EGL19: usize = 9;
const SLO1_UNC2: usize = 10;
const SLO2_EGL19: usize = 11;
const SLO2_UNC2: usize = 12;
const KCNL: usize = 13;
const NCA: usize = 14;
const IRK: usize = 15;

const EK: f64 = -80.0; // hard-coded in the .mod files
const EK_ION: f64 = -77.0; // NEURON's default ek (kqt3 reads the ion)
const ECA: f64 = 60.0;
const ENA: f64 = 30.0;
const FARADAY: f64 = 96485.0;

#[inline]
fn sig(x: f64) -> f64 {
    1.0 / (1.0 + x.exp())
}

#[inline]
fn cnexp(x: f64, inf: f64, tau: f64, dt: f64) -> f64 {
    x + (1.0 - (-dt / tau).exp()) * (inf - x)
}

// egl19 (and the egl19 half of slo*_egl19): minf, hinf, tm, th
#[inline]
fn egl19_rates(v: f64) -> (f64, f64, f64, f64) {
    let minf = sig(-(v - 5.6) / 7.5);
    let hinf = (1.43 * sig(-(v - 24.9) / 12.0) + 0.14) * (5.96 * sig((v + 20.5) / 8.1) + 0.60);
    let tm = 2.9 * (-((v - 5.2) / 6.0).powi(2)).exp() + 1.9 * (-((v - 1.4) / 30.0).powi(2)).exp() + 2.3;
    let th = 0.4 * (44.6 * sig((v + 23.0) / 5.0) + 36.4 * sig((v - 28.7) / 3.7) + 43.1);
    (minf, hinf, tm, th)
}

#[inline]
fn unc2_rates(v: f64) -> (f64, f64, f64, f64) {
    let minf = sig(-(v + 12.2) / 4.0);
    let hinf = sig((v + 52.5) / 5.6);
    let tm = 1.5 / ((-(v + 8.2) / 9.1).exp() + ((v + 8.2) / 15.4).exp()) + 0.1;
    let th = 83.8 * sig(-(v - 52.9) / -3.5) + 72.1 * sig((v - 23.9) / -3.6);
    (minf, hinf, tm, th)
}

// The BK/SLO channels see calcium in a nanodomain around the calcium channel.
// Returns (minf, tm) of the open state given the calcium channel's mcav.
struct Slo {
    wyx: f64,
    wxy: f64,
    wom: f64,
    wop: f64,
    kxy: f64,
    nxy: f64,
    kyx: f64,
    nyx: f64,
}

const SLO1: Slo = Slo { wyx: 0.013, wxy: -0.028, wom: 3.15, wop: 0.16, kxy: 55.73, nxy: 1.30, kyx: 34.34, nyx: 0.0001 };
const SLO2: Slo = Slo { wyx: 0.019, wxy: -0.024, wom: 0.90, wop: 0.027, kxy: 93.45, nxy: 1.84, kyx: 3294.55, nyx: 0.00001 };

#[inline]
fn slo_rates(p: &Slo, v: f64, mcav: f64, mcavinf: f64, tmcav: f64) -> (f64, f64) {
    const GSC: f64 = 0.04;
    const R: f64 = 0.013;
    const DCA: f64 = 250.0;
    const KB: f64 = 500.0;
    const BTOT: f64 = 30.0;
    const CANCI: f64 = 0.05;
    let cain = if v < 60.0 {
        -GSC * (v - 60.0) * 1e9 / (8.0 * 3.1415926 * R * DCA * FARADAY) * (-R / (DCA / (KB * BTOT)).sqrt()).exp()
    } else {
        0.0001
    };
    let alpha = mcavinf / tmcav;
    let beta = 1.0 / tmcav - alpha;
    let wm = p.wom * (-p.wyx * v).exp();
    let wp = p.wop * (-p.wxy * v).exp();
    let fm = 1.0 / (1.0 + (cain / p.kyx).powf(p.nyx));
    let fp = 1.0 / (1.0 + (p.kxy / cain).powf(p.nxy));
    let kom = wm * fm;
    let kop = wp * fp;
    let kcm = wm / (1.0 + (CANCI / p.kyx).powf(p.nyx));
    let den = (kop + kom) * (kcm + alpha) + beta * kcm;
    let minf = mcav * kop * (alpha + beta + kcm) / den;
    let tm = (alpha + beta + kcm) / den;
    (minf, tm)
}

/// Gating states of one soma (all channels live on the soma; neurites are passive).
#[derive(Clone, Default)]
struct SomaState {
    shl1: [f64; 3], // m, hf, hs
    shk1: [f64; 2],
    kvs1: [f64; 2],
    egl2: f64,
    egl36: [f64; 3],
    kqt3: [f64; 4], // mf, ms, w, s
    egl19: [f64; 2],
    unc2: [f64; 2],
    cca1: [f64; 2],
    slo: [[f64; 3]; 4], // per slo variant: m, hcav, mcav
    kcnl: f64,
    irk: f64,
    cai: f64,
}

pub struct Brain {
    pub n: usize,
    parent: Vec<i32>,
    cap: Vec<f64>,  // nF
    gpas: Vec<f64>, // µS
    epas: Vec<f64>,
    gax: Vec<f64>, // µS between node and parent
    d0: Vec<f64>,  // cap/dt + gpas + axial conductances to parent and children
    /// node voltages in solver order; `order[k]` = brain.bin node of solver node k
    pub v: Vec<f64>,
    pub order: Vec<usize>,
    d: Vec<f64>,
    rhs: Vec<f64>,
    pub soma: Vec<usize>,
    soma_area: Vec<f64>,
    gbar: Vec<[f64; N_CHANNELS]>,
    st: Vec<SomaState>,
    ica: Vec<f64>,
    // gap junctions
    gj_pre: Vec<u32>,
    gj_post: Vec<u32>,
    gj_w: Vec<f64>,
    // graded synapses
    sy_pre: Vec<u32>,
    sy_post: Vec<u32>,
    sy_g: Vec<f64>, // weight × conductance, µS
    sy_exc: Vec<bool>,
    sy_s: Vec<f64>,
    /// injected current per cell (nA, into the soma), like NEURON's IClamp
    pub iclamp: Vec<f64>,
    /// per-synapse gain on the conductance (1 = as built); lets the page cut connections
    pub sy_gain: Vec<f64>,
    pub gj_gain: Vec<f64>,
    pub dt: f64,
    pub t: f64,
}

// synapse constants: neuron_to_neuron_{exc,inh}_syn.mod
const EXC_G: f64 = 4.9e-4;
const EXC_K: f64 = 0.5;
const EXC_E: f64 = 30.0;
const INH_G: f64 = 2.0e-4;
const INH_K: f64 = 0.015000001;
const INH_E: f64 = -70.0;
const SYN_VTH: f64 = -20.0;
const SYN_DELTA: f64 = 5.0;

#[inline]
fn syn_inf(vpre: f64) -> f64 {
    1.0 / (1.0 + ((SYN_VTH - vpre) / SYN_DELTA).exp())
}

impl Brain {
    /// `data` is brain.bin as f64s (layout in tools/export_brain.py).
    pub fn new(data: &[f64], dt: f64) -> Brain {
        let n = data[0] as usize;
        let nc = data[1] as usize;
        let nk = data[2] as usize;
        let mut o = 3;
        let mut b = Brain {
            n,
            parent: vec![0; n],
            cap: vec![0.0; n],
            gpas: vec![0.0; n],
            epas: vec![0.0; n],
            gax: vec![0.0; n],
            d0: vec![0.0; n],
            v: vec![0.0; n],
            order: (0..n).collect(),
            d: vec![0.0; n],
            rhs: vec![0.0; n],
            soma: Vec::with_capacity(nc),
            soma_area: Vec::with_capacity(nc),
            gbar: Vec::with_capacity(nc),
            st: vec![SomaState::default(); nc],
            ica: vec![0.0; nc],
            gj_pre: vec![],
            gj_post: vec![],
            gj_w: vec![],
            sy_pre: vec![],
            sy_post: vec![],
            sy_g: vec![],
            sy_exc: vec![],
            sy_s: vec![],
            iclamp: vec![0.0; nc],
            sy_gain: vec![],
            gj_gain: vec![],
            dt,
            t: 0.0,
        };
        for i in 0..n {
            let r = &data[o..o + 6];
            o += 6;
            let area = r[1];
            b.parent[i] = r[0] as i32;
            b.gax[i] = if r[0] >= 0.0 { 1.0 / r[2] } else { 0.0 };
            b.cap[i] = r[3] * area * 1e-5;
            b.gpas[i] = r[4] * area * 1e-2;
            b.epas[i] = r[5];
            debug_assert!(b.parent[i] < i as i32);
        }
        // Order nodes by depth from their cell's root (then by original index): parents
        // still come before children, and consecutive nodes now belong to different
        // cells, so the solver's dependent divisions overlap instead of queueing.
        let mut depth = vec![0u32; n];
        for i in 0..n {
            if b.parent[i] >= 0 {
                depth[i] = depth[b.parent[i] as usize] + 1;
            }
        }
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&i| (depth[i], i));
        let mut new_of = vec![0usize; n];
        for (k, &i) in order.iter().enumerate() {
            new_of[i] = k;
        }
        let perm = |x: &Vec<f64>| order.iter().map(|&i| x[i]).collect::<Vec<f64>>();
        b.cap = perm(&b.cap);
        b.gpas = perm(&b.gpas);
        b.epas = perm(&b.epas);
        b.gax = perm(&b.gax);
        b.parent = order.iter().map(|&i| if b.parent[i] >= 0 { new_of[b.parent[i] as usize] as i32 } else { -1 }).collect();
        b.order = order;
        b.set_dt(dt);
        let mut area = vec![0.0; n];
        for i in 0..n {
            area[new_of[i]] = data[3 + i * 6 + 1];
        }
        for _ in 0..nc {
            let s = new_of[data[o] as usize];
            b.soma.push(s);
            b.soma_area.push(area[s]);
            let mut g = [0.0; N_CHANNELS];
            g.copy_from_slice(&data[o + 1..o + 1 + N_CHANNELS]);
            b.gbar.push(g);
            o += 1 + N_CHANNELS;
        }
        for _ in 0..nk {
            let kind = data[o] as u32;
            let pre = new_of[data[o + 1] as usize] as u32;
            let post = new_of[data[o + 2] as usize] as u32;
            let w = data[o + 3];
            o += 4;
            if kind == 0 {
                b.gj_pre.push(pre);
                b.gj_post.push(post);
                b.gj_w.push(w);
            } else {
                b.sy_pre.push(pre);
                b.sy_post.push(post);
                b.sy_exc.push(kind == 1);
                b.sy_g.push(w * if kind == 1 { EXC_G } else { INH_G });
            }
        }
        b.sy_s = vec![0.0; b.sy_pre.len()];
        b.sy_gain = vec![1.0; b.sy_pre.len()];
        b.gj_gain = vec![1.0; b.gj_pre.len()];
        b.init(-65.0);
        b
    }

    pub fn set_dt(&mut self, dt: f64) {
        self.dt = dt;
        for i in 0..self.n {
            self.d0[i] = self.cap[i] / dt + self.gpas[i];
        }
        for i in 0..self.n {
            let p = self.parent[i];
            if p >= 0 {
                self.d0[i] += self.gax[i];
                self.d0[p as usize] += self.gax[i];
            }
        }
    }

    pub fn n_syn(&self) -> usize {
        self.sy_pre.len()
    }

    pub fn n_gap(&self) -> usize {
        self.gj_pre.len()
    }

    pub fn syn_pre_post(&self, k: usize) -> (u32, u32, bool) {
        (self.sy_pre[k], self.sy_post[k], self.sy_exc[k])
    }

    pub fn gap_pre_post(&self, k: usize) -> (u32, u32) {
        (self.gj_pre[k], self.gj_post[k])
    }

    /// NEURON's finitialize(v0)
    pub fn init(&mut self, v0: f64) {
        self.t = 0.0;
        for x in self.v.iter_mut() {
            *x = v0;
        }
        for c in 0..self.soma.len() {
            let v = v0;
            let s = &mut self.st[c];
            s.shl1 = [sig(-(v - 11.2) / 14.1), sig((v + 33.1) / 8.3), sig((v + 33.1) / 8.3)];
            s.shk1 = [sig(-(v - 20.4) / 7.7), sig((v + 7.0) / 5.8)];
            s.kvs1 = [sig(-(v - 57.1) / 25.0), sig((v - 47.3) / 11.1)];
            s.egl2 = sig(-(v + 6.9) / 14.9);
            let e36 = sig(-(v - 63.0) / 28.5);
            s.egl36 = [e36, e36, e36];
            s.kqt3 = [0.0, 0.0, 0.49 + 0.51 * sig((v + 1.084) / 28.78), 0.34 + 0.66 * sig((v + 45.3) / 12.3)];
            let (m, h, _, _) = egl19_rates(v);
            s.egl19 = [m, h];
            let (mu, hu, _, _) = unc2_rates(v);
            s.unc2 = [mu, hu];
            s.cca1 = [sig(-(v + 43.32) / 7.6), sig((v + 58.0) / 7.0)];
            // slo m starts at 0 (slo2's m = minf is evaluated before mcav is set, so also 0)
            s.slo = [[0.0, h, m], [0.0, hu, mu], [0.0, h, m], [0.0, hu, mu]];
            s.cai = 5e-5; // NEURON's cai0_ca_ion; cainternm's INITIAL leaves it
            s.kcnl = s.cai / (0.33 + s.cai);
            s.irk = sig((v + 82.0) / 13.0);
        }
        for k in 0..self.sy_s.len() {
            self.sy_s[k] = syn_inf(v0);
        }
    }

    /// One fadvance.
    pub fn step(&mut self) {
        let dt = self.dt;
        let n = self.n;
        let v = &self.v;
        let d = &mut self.d;
        let rhs = &mut self.rhs;
        // passive membrane; capacitance and axial conductances are in d0
        for i in 0..n {
            d[i] = self.d0[i];
            rhs[i] = -self.gpas[i] * (v[i] - self.epas[i]);
        }
        // soma channels
        let ica = &mut self.ica;
        for c in 0..self.soma.len() {
            let i = self.soma[c];
            let vi = v[i];
            let gb = &self.gbar[c];
            let s = &self.st[c];
            let gk = gb[SHL1] * s.shl1[0].powi(3) * (0.7 * s.shl1[1] + 0.3 * s.shl1[2])
                + gb[SHK1] * s.shk1[0] * s.shk1[1]
                + gb[KVS1] * s.kvs1[0] * s.kvs1[1]
                + gb[EGL2] * s.egl2
                + gb[EGL36] * (0.33 * s.egl36[0] + 0.36 * s.egl36[1] + 0.39 * s.egl36[2])
                + gb[SLO1_EGL19] * s.slo[0][0] * s.slo[0][1]
                + gb[SLO1_UNC2] * s.slo[1][0] * s.slo[1][1]
                + gb[SLO2_EGL19] * s.slo[2][0] * s.slo[2][1]
                + gb[SLO2_UNC2] * s.slo[3][0] * s.slo[3][1]
                + gb[KCNL] * s.kcnl
                + gb[IRK] * s.irk;
            let gkq = gb[KQT3] * (0.3 * s.kqt3[0] + 0.7 * s.kqt3[1]) * s.kqt3[2] * s.kqt3[3];
            let gca = gb[EGL19] * s.egl19[0] * s.egl19[1]
                + gb[UNC2] * s.unc2[0] * s.unc2[1]
                + gb[CCA1] * s.cca1[0] * s.cca1[0] * s.cca1[1];
            let gna = gb[NCA];
            ica[c] = gca * (vi - ECA);
            let idens = gk * (vi - EK) + gkq * (vi - EK_ION) + ica[c] + gna * (vi - ENA);
            let a = self.soma_area[c] * 1e-2;
            rhs[i] -= idens * a - self.iclamp[c];
            d[i] += (gk + gkq + gca + gna) * a;
        }
        // gap junctions: i = w (v - vpre), implicit in the own v only
        for k in 0..self.gj_w.len() {
            let post = self.gj_post[k] as usize;
            let w = self.gj_w[k] * self.gj_gain[k];
            rhs[post] -= w * (v[post] - v[self.gj_pre[k] as usize]);
            d[post] += w;
        }
        for k in 0..self.sy_g.len() {
            let post = self.sy_post[k] as usize;
            let g = self.sy_g[k] * self.sy_gain[k] * self.sy_s[k];
            let e = if self.sy_exc[k] { EXC_E } else { INH_E };
            rhs[post] -= g * (v[post] - e);
            d[post] += g;
        }
        // Hines: axial currents and elimination in one sweep from the leaves (a
        // node's children all have larger indices), then substitution and update
        for i in (0..n).rev() {
            let p = self.parent[i];
            if p >= 0 {
                let p = p as usize;
                let g = self.gax[i];
                let flow = g * (v[p] - v[i]);
                let ri = rhs[i] + flow;
                rhs[p] -= flow;
                let inv = 1.0 / d[i];
                d[i] = inv;
                let f = g * inv;
                d[p] -= f * g;
                rhs[p] += f * ri;
                rhs[i] = ri;
            } else {
                d[i] = 1.0 / d[i];
            }
        }
        for i in 0..n {
            let p = self.parent[i];
            let dv = if p >= 0 { (rhs[i] + self.gax[i] * rhs[p as usize]) * d[i] } else { rhs[i] * d[i] };
            rhs[i] = dv;
            self.v[i] += dv;
        }
        self.t += dt;
        // states with v(t + dt)
        let v = &self.v;
        // (a channel with zero conductance in this cell is never read, so its states are skipped)
        for c in 0..self.soma.len() {
            let vi = v[self.soma[c]];
            let gb = &self.gbar[c];
            let s = &mut self.st[c];
            // cainternm first (mechanism order), derivimplicit = backward Euler
            {
                const F: f64 = 0.001;
                const TCA: f64 = 50.0;
                const CAEQ: f64 = 0.05;
                const VCELL: f64 = 31.16;
                let src = if vi <= 60.0 { -F * ica[c] * 1e6 / (2.0 * FARADAY * VCELL) } else { 0.0 };
                s.cai = (s.cai + dt * (src + CAEQ / TCA)) / (1.0 + dt / TCA);
            }
            if gb[CCA1] != 0.0 {
                let (minf, hinf) = (sig(-(vi + 43.32) / 7.6), sig((vi + 58.0) / 7.0));
                let tm = 40.0 * sig(-(vi + 62.5) / -12.6) + 0.7;
                let th = 280.0 * sig((vi + 60.7) / 8.5) + 19.8;
                s.cca1 = [cnexp(s.cca1[0], minf, tm, dt), cnexp(s.cca1[1], hinf, th, dt)];
            }
            let need19 = gb[EGL19] != 0.0 || gb[SLO1_EGL19] != 0.0 || gb[SLO2_EGL19] != 0.0;
            let needu = gb[UNC2] != 0.0 || gb[SLO1_UNC2] != 0.0 || gb[SLO2_UNC2] != 0.0;
            let (m19, h19, tm19, th19) = if need19 { egl19_rates(vi) } else { (0.0, 0.0, 1.0, 1.0) };
            let (mu, hu, tmu, thu) = if needu { unc2_rates(vi) } else { (0.0, 0.0, 1.0, 1.0) };
            if gb[EGL19] != 0.0 {
                s.egl19 = [cnexp(s.egl19[0], m19, tm19, dt), cnexp(s.egl19[1], h19, th19, dt)];
            }
            if gb[EGL2] != 0.0 {
                s.egl2 = cnexp(s.egl2, sig(-(vi + 6.9) / 14.9), 1845.8 * sig((vi + 122.6) / 13.8) + 1517.74, dt);
            }
            if gb[EGL36] != 0.0 {
                let inf = sig(-(vi - 63.0) / 28.5);
                s.egl36 = [cnexp(s.egl36[0], inf, 13.0, dt), cnexp(s.egl36[1], inf, 63.0, dt), cnexp(s.egl36[2], inf, 355.0, dt)];
            }
            if gb[IRK] != 0.0 {
                let minf = sig((vi + 82.0) / 13.0);
                let tm = 17.1 / ((-(vi + 17.8) / 20.3).exp() + ((vi + 43.4) / 11.2).exp()) + 3.8;
                s.irk = cnexp(s.irk, minf, tm, dt);
            }
            if gb[KCNL] != 0.0 {
                s.kcnl = cnexp(s.kcnl, s.cai / (0.33 + s.cai), 6.3, dt);
            }
            if gb[KQT3] != 0.0 {
                let minf = sig(-(vi + 12.6726) / 15.8008);
                let tmf = 395.3 / (1.0 + ((vi + 38.1) / 33.59).powi(2));
                let tms = 5503.0 - 5345.4 / (1.0 + 10f64.powf(-0.02827 * (-23.9 - vi)))
                    - 4590.6 / (1.0 + 10f64.powf(-0.0357 * (14.15 + vi)));
                let winf = 0.49 + 0.51 * sig((vi + 1.084) / 28.78);
                let sinf = 0.34 + 0.66 * sig((vi + 45.3) / 12.3);
                let tw = 0.544 + 29.2 / (1.0 + ((vi + 48.09) / 48.83).powi(2));
                s.kqt3 = [
                    cnexp(s.kqt3[0], minf, tmf, dt),
                    cnexp(s.kqt3[1], minf, tms, dt),
                    cnexp(s.kqt3[2], winf, tw, dt),
                    cnexp(s.kqt3[3], sinf, 500e3, dt),
                ];
            }
            if gb[KVS1] != 0.0 {
                let minf = sig(-(vi - 57.1) / 25.0);
                let hinf = sig((vi - 47.3) / 11.1);
                let tm = 30.0 * sig((vi - 18.1) / 20.0) + 1.0;
                let th = 88.5 * sig((vi - 50.0) / 15.0) + 53.4;
                s.kvs1 = [cnexp(s.kvs1[0], minf, tm, dt), cnexp(s.kvs1[1], hinf, th, dt)];
            }
            if gb[SHK1] != 0.0 {
                let minf = sig(-(vi - 20.4) / 7.7);
                let hinf = sig((vi + 7.0) / 5.8);
                let tm = 26.6 / ((-(vi + 33.7) / 15.8).exp() + ((vi + 33.7) / 15.4).exp()) + 2.0;
                s.shk1 = [cnexp(s.shk1[0], minf, tm, dt), cnexp(s.shk1[1], hinf, 1400.0, dt)];
            }
            if gb[SHL1] != 0.0 {
                let minf = sig(-(vi - 11.2) / 14.1);
                let hinf = sig((vi + 33.1) / 8.3);
                let tm = 13.8 / ((-(vi + 17.5) / 12.9).exp() + ((vi + 3.7) / 6.5).exp()) + 1.9;
                let thf = 539.2 * sig((vi + 28.2) / 4.9) + 27.3;
                let ths = 8422.0 * sig((vi + 37.7) / 6.4) + 118.9;
                s.shl1 = [
                    cnexp(s.shl1[0], minf, tm * 0.4, dt),
                    cnexp(s.shl1[1], hinf, thf * 0.08, dt),
                    cnexp(s.shl1[2], hinf, ths * 0.3, dt),
                ];
            }
            // slo1_egl19, slo1_unc2, slo2_egl19, slo2_unc2: rates use mcav before its update
            for (j, (p, egl, g)) in [(&SLO1, true, SLO1_EGL19), (&SLO1, false, SLO1_UNC2), (&SLO2, true, SLO2_EGL19), (&SLO2, false, SLO2_UNC2)].iter().enumerate() {
                if gb[*g] == 0.0 {
                    continue;
                }
                let (mi, hi, tmc, thc) = if *egl { (m19, h19, tm19, th19) } else { (mu, hu, tmu, thu) };
                let x = &mut s.slo[j];
                let (minf, tm) = slo_rates(p, vi, x[2], mi, tmc);
                x[2] = cnexp(x[2], mi, tmc, dt);
                x[1] = cnexp(x[1], hi, thc, dt);
                x[0] = cnexp(x[0], minf, tm, dt);
            }
            if gb[UNC2] != 0.0 {
                s.unc2 = [cnexp(s.unc2[0], mu, tmu, dt), cnexp(s.unc2[1], hu, thu, dt)];
            }
        }
        for k in 0..self.sy_s.len() {
            let inf = syn_inf(v[self.sy_pre[k] as usize]);
            let kk = if self.sy_exc[k] { EXC_K } else { INH_K };
            self.sy_s[k] = cnexp(self.sy_s[k], inf, (1.0 - inf) / kk, dt);
        }
    }

    pub fn soma_v(&self, c: usize) -> f64 {
        self.v[self.soma[c]]
    }
}
