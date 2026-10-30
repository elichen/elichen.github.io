//! Shiu et al. 2024's whole-brain leaky integrate-and-fire model on the FlyWire connectome,
//! stepped the way Brian 2 steps it (dt 0.1 ms):
//!
//! 1. groups: not_refractory = (n - lastspike) >= rfc; if so, exact update of
//!    dv/dt = (v0 - v + g)/t_mbr, dg/dt = -g/tau ('linear' method), else v, g frozen
//! 2. thresholds: v > v_th and not refractory -> spike, lastspike = n
//! 3. synapses: spikes from step n - 18 (1.8 ms delay) add count * w_syn to g, except
//!    while the target is refractory (Brian ignores synaptic updates to a variable marked
//!    'unless refractory'); Poisson inputs add w_syn * f_poi to v
//! 4. resets: spiking neurons get v = v_rst, g = 0 (inputs that arrived this step are lost)
//!
//! Only neurons that could reach threshold are stepped. Without new input, v - v0 can rise to
//! at most max(v - v0, 0) + max(g, 0) * h_max (h_max = peak of the response to a unit g), so a
//! neuron below that bound is dormant: it keeps its state and the step it was last updated,
//! and is brought up to date in closed form when input next arrives. `full = true` steps every
//! neuron every step, as Brian does.

const MV: f64 = 1e-3;
const MS: f64 = 1e-3;
pub const DT: f64 = 0.1 * MS;
pub const DELAY: usize = 18; // round(1.8 ms / dt)
const SLOTS: usize = DELAY + 1;
pub const RFC: i64 = 22; // timestep(2.2 ms)
/// Closed-form catch-up table length; after this many steps a dormant neuron is at rest.
const CATCH: usize = 1 << 16;

pub struct Params {
    pub v0: f64,
    pub v_rst: f64,
    pub v_th: f64,
    pub t_mbr: f64,
    pub tau: f64,
    pub w_syn: f64,
    pub w_kick: f64,
}

impl Default for Params {
    fn default() -> Self {
        let w_syn = 0.275 * MV;
        Params {
            v0: -52.0 * MV,
            v_rst: -52.0 * MV,
            v_th: -45.0 * MV,
            t_mbr: 20.0 * MS,
            tau: 5.0 * MS,
            w_syn,
            w_kick: w_syn * 250.0,
        }
    }
}

/// One neuron's state, packed so a synaptic event touches one cache line.
#[derive(Clone, Copy)]
#[repr(C, align(32))]
pub struct Cell {
    pub v: f64,
    pub g: f64,
    /// step of the last spike
    pub last: i32,
    /// first step whose update a dormant neuron has not had yet
    since: i32,
    /// refractory steps (0 for Poisson-driven neurons, as in Shiu's model)
    pub rfc: u8,
    active: bool,
    excited: bool,
    pub silenced: bool,
}

pub struct Brain {
    pub n: usize,
    pub offsets: Vec<u32>,
    pub targets: Vec<u32>,
    pub counts: Vec<i16>,
    pub p: Params,
    // one step: g' = b g, v' = rest + (gv g + a v)
    a: f64,
    b: f64,
    gv: f64,
    rest: f64,
    // k steps without input: v - v0 -> a_k (v - v0) + s_k g, g -> b_k g
    catch: Vec<[f64; 3]>,
    h_max: f64,
    pub cells: Vec<Cell>,
    /// Poisson rate per neuron (Hz); nonzero = excited
    pub rate: Vec<f64>,
    excited: Vec<u32>,
    queue: Vec<Vec<u32>>,
    active: Vec<u32>,
    pub full: bool,
    pub step_n: i32,
    pub spikes: Vec<u32>,
    pub spike_count: Vec<u32>,
    rng: u64,
}

/// Decode model/connectome.gz (already gunzipped): see tools/export_brain.py.
pub fn parse(bytes: &[u8]) -> (Vec<u32>, Vec<u32>, Vec<i16>) {
    assert_eq!(&bytes[0..4], b"FLY2", "not a connectome");
    let u32_at = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
    let n = u32_at(4) as usize;
    let m = u32_at(8) as usize;
    let mut o = 12;
    let offsets: Vec<u32> = (0..=n).map(|i| u32_at(o + 4 * i)).collect();
    o += 4 * (n + 1);
    let mut varint = || {
        let (mut x, mut shift) = (0u64, 0);
        loop {
            let b = bytes[o];
            o += 1;
            x |= ((b & 0x7f) as u64) << shift;
            if b < 0x80 {
                return x;
            }
            shift += 7;
        }
    };
    let mut targets = Vec::with_capacity(m);
    for i in 0..n {
        let mut t = 0u64;
        for e in offsets[i]..offsets[i + 1] {
            let d = varint();
            t = if e == offsets[i] { d } else { t + d };
            targets.push(t as u32);
        }
    }
    let counts = (0..m).map(|_| {
        let z = varint();
        ((z >> 1) as i64 ^ -((z & 1) as i64)) as i16
    }).collect();
    (offsets, targets, counts)
}

/// Read and gunzip a connectome file (native tools).
#[cfg(not(target_arch = "wasm32"))]
pub fn load(path: &str) -> (Vec<u32>, Vec<u32>, Vec<i16>) {
    use std::io::Read;
    let mut bytes = Vec::new();
    flate2::read::GzDecoder::new(std::fs::File::open(path).expect(path)).read_to_end(&mut bytes).unwrap();
    parse(&bytes)
}

impl Brain {
    pub fn new(offsets: Vec<u32>, targets: Vec<u32>, counts: Vec<i16>, seed: u64) -> Brain {
        let n = offsets.len() - 1;
        let p = Params::default();
        // Brian's 'linear' solution, with the constants hoisted the way its Cython code does:
        // _g = _lio_2*g, _v = _lio_8 + ((_lio_9*g) + (_lio_10*v))
        let lio1 = -DT / p.tau;
        let b = lio1.exp();
        let lio3 = p.tau / (p.t_mbr - p.tau);
        let lio6 = -DT / p.t_mbr;
        let a = lio6.exp();
        let gv = ((lio3 * (-(DT / p.t_mbr).exp() + (DT / p.tau).exp())) * lio6.exp()) * lio1.exp();
        let rest = p.v0 - p.v0 * lio6.exp();
        let mut catch = vec![[1.0, 1.0, 0.0]; CATCH];
        for k in 1..CATCH {
            let [ak, bk, sk] = catch[k - 1];
            catch[k] = [ak * a, bk * b, a * sk + gv * bk];
        }
        // response of v - v0 to g = 1 is tau/(t_mbr - tau) (e^(-t/t_mbr) - e^(-t/tau)); its peak:
        let t_peak = (p.t_mbr / p.tau).ln() / (1.0 / p.tau - 1.0 / p.t_mbr);
        let h_max = p.tau / (p.t_mbr - p.tau) * ((-t_peak / p.t_mbr).exp() - (-t_peak / p.tau).exp());
        let rest_cell = Cell { v: p.v0, g: 0.0, last: i32::MIN / 2, since: 0, rfc: RFC as u8, active: false, excited: false, silenced: false };
        let mut br = Brain {
            n,
            offsets,
            targets,
            counts,
            a,
            b,
            gv,
            rest,
            catch,
            h_max: h_max * (1.0 + 1e-9),
            cells: vec![rest_cell; n],
            rate: vec![0.0; n],
            excited: Vec::new(),
            queue: (0..SLOTS).map(|_| Vec::new()).collect(),
            active: Vec::new(),
            full: false,
            step_n: 0,
            spikes: Vec::new(),
            spike_count: vec![0; n],
            rng: seed.wrapping_mul(0x9E3779B97F4A7C15) | 1,
            p,
        };
        br.reset();
        br
    }

    /// Back to rest; excitation and silencing stay.
    pub fn reset(&mut self) {
        for c in self.cells.iter_mut() {
            c.v = self.p.v0;
            c.g = 0.0;
            c.last = i32::MIN / 2;
            c.since = 0;
            c.active = false;
        }
        for q in self.queue.iter_mut() {
            q.clear();
        }
        self.active.clear();
        self.step_n = 0;
        self.spikes.clear();
        self.spike_count.fill(0);
    }

    /// Shiu's PoissonInput: an excited neuron is kicked at `hz` and has no refractory period.
    pub fn set_rate(&mut self, i: usize, hz: f64) {
        self.catch_up(i, self.step_n - 1);
        self.rate[i] = hz;
        let c = &mut self.cells[i];
        c.excited = hz > 0.0;
        c.rfc = if c.excited { 0 } else { RFC as u8 };
        self.excited = (0..self.n as u32).filter(|&j| self.rate[j as usize] > 0.0).collect();
    }

    /// Shiu's silencing: every synapse out of the neuron has weight 0.
    pub fn set_silenced(&mut self, i: usize, s: bool) {
        self.cells[i].silenced = s;
    }

    /// Refractory period in steps (parity runs: Poisson targets get 0 without a rate).
    pub fn set_rfc(&mut self, i: usize, steps: u8) {
        self.cells[i].rfc = steps;
    }

    fn rand(&mut self) -> f64 {
        // xorshift64*
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        (self.rng.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Bring a dormant neuron's state up to date through step `n`'s update.
    #[inline]
    fn catch_up(&mut self, i: usize, n: i32) {
        if self.full {
            return;
        }
        let v0 = self.p.v0;
        let c = &mut self.cells[i];
        if c.active {
            return;
        }
        // refractory steps leave the state frozen
        let from = c.since.max(c.last + c.rfc as i32);
        let k = n - from + 1;
        if k > 0 {
            if k as usize >= CATCH {
                c.v = v0;
                c.g = 0.0;
            } else {
                let [ak, bk, sk] = self.catch[k as usize];
                c.v = v0 + (ak * (c.v - v0) + sk * c.g);
                c.g *= bk;
            }
        }
        c.since = c.since.max(n + 1);
    }

    #[inline]
    fn could_fire(&self, c: &Cell) -> bool {
        c.excited || (c.v - self.p.v0).max(0.0) + c.g.max(0.0) * self.h_max >= self.p.v_th - self.p.v0 - 1e-12
    }

    #[inline]
    fn wake(&mut self, i: usize) {
        if !self.cells[i].active {
            self.cells[i].active = true;
            self.active.push(i as u32);
        }
    }

    /// Brian's state update; returns not_refractory.
    #[inline]
    fn update(&mut self, i: usize, n: i32) -> bool {
        let (a, b, gv, rest) = (self.a, self.b, self.gv, self.rest);
        let c = &mut self.cells[i];
        let not_refractory = n - c.last >= c.rfc as i32;
        if not_refractory {
            let (v, g) = (c.v, c.g);
            c.g = b * g;
            c.v = rest + (gv * g + a * v);
        }
        not_refractory
    }

    /// One 0.1 ms step with Poisson input. Returns the neurons that spiked.
    pub fn step(&mut self) -> &[u32] {
        let mut hit = Vec::new();
        for k in 0..self.excited.len() {
            let i = self.excited[k];
            if self.rand() < self.rate[i as usize] * DT {
                hit.push(i);
            }
        }
        self.step_with(&hit);
        &self.spikes
    }

    /// One step with the given Poisson kicks (for parity runs fed Brian's own input).
    pub fn step_with(&mut self, kicks: &[u32]) {
        let n = self.step_n;
        let v_th = self.p.v_th;
        self.spikes.clear();
        // 1-2. groups + thresholds
        if self.full {
            for i in 0..self.n {
                if self.update(i, n) && self.cells[i].v > v_th {
                    self.spikes.push(i as u32);
                }
            }
        } else {
            for k in 0..self.excited.len() {
                let i = self.excited[k] as usize;
                if !self.cells[i].active {
                    self.catch_up(i, n - 1);
                    self.wake(i);
                }
            }
            let mut k = 0;
            while k < self.active.len() {
                let i = self.active[k] as usize;
                let nr = self.update(i, n);
                let c = self.cells[i];
                if nr && c.v > v_th {
                    self.spikes.push(i as u32);
                }
                if nr && !self.could_fire(&c) {
                    let c = &mut self.cells[i];
                    c.active = false;
                    c.since = n + 1;
                    self.active.swap_remove(k);
                } else {
                    k += 1;
                }
            }
            self.spikes.sort_unstable();
        }
        for &s in &self.spikes {
            self.cells[s as usize].last = n;
            self.spike_count[s as usize] += 1;
        }
        // 3. synapses: deliver spikes from DELAY steps ago, queue this step's
        let slot = (n as usize) % SLOTS;
        let mut arriving = std::mem::take(&mut self.queue[slot]);
        let w_syn = self.p.w_syn;
        for &s in &arriving {
            let s = s as usize;
            if self.cells[s].silenced {
                continue;
            }
            for e in self.offsets[s] as usize..self.offsets[s + 1] as usize {
                let j = self.targets[e] as usize;
                let c = &self.cells[j];
                if n - c.last >= c.rfc as i32 && c.last != n {
                    self.catch_up(j, n);
                    let c = &mut self.cells[j];
                    c.g += self.counts[e] as f64 * w_syn;
                    if !self.full && !c.active {
                        let c = *c;
                        if self.could_fire(&c) {
                            self.wake(j);
                        }
                    }
                }
            }
        }
        arriving.clear();
        self.queue[slot] = arriving;
        let out = (n as usize + DELAY) % SLOTS;
        self.queue[out].extend_from_slice(&self.spikes);
        for &i in kicks {
            let i = i as usize;
            self.catch_up(i, n);
            self.cells[i].v += self.p.w_kick;
            if !self.full {
                self.wake(i);
            }
        }
        // 4. resets
        for &s in &self.spikes {
            let c = &mut self.cells[s as usize];
            c.v = self.p.v_rst;
            c.g = 0.0;
        }
        self.step_n += 1;
    }

    /// Bring every dormant neuron up to date (after this, cells[i].v is current for all i).
    pub fn sync(&mut self) {
        let n = self.step_n - 1;
        for i in 0..self.n {
            self.catch_up(i, n);
        }
    }

    pub fn active_len(&self) -> usize {
        self.active.len()
    }
}
