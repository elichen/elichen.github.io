//! Food seeking on an agar plate, as a layer between the brain's muscle output and
//! the body (ours, not BAAIWorm's). The brain still makes the forward gait; this
//! layer smells an attractant at the nose and steers with the two strategies real
//! worms use:
//!
//!   weathervane (Iino & Yoshida 2009): while the head sweeps to one side, a rising
//!     concentration adds bend toward that side, so the path curves up the gradient;
//!   pirouettes (Pierce-Shimomura et al. 1999): sharp turns come at a rate that
//!     rises when the concentration falls; each is a reversal (the brain's gait
//!     run tail first) and then an omega turn (a deep ventral bend).
//!
//! The plate is the body's rest plane (x = dorsal, z = head to tail at rest), with
//! normal y. Concentration is a Gaussian hill around the food; the worm senses its
//! log (adaptation makes C. elegans respond to relative change).
use crate::body::Body;

pub const SEGMENTS: usize = 24;

#[derive(Clone, Copy, Debug)]
pub struct Params {
    /// width of the attractant hill (body units; 0.63 = 1 mm)
    pub sigma: f64,
    /// shape of the hill: false = Gaussian (ln C = -r²/2σ², as a spot of salt spreads),
    /// true = cone (ln C = -r/σ: the same relative gradient everywhere)
    pub cone: bool,
    /// weathervane gain: muscle bias per (d log C / dt × head sweep rate)
    pub k_vane: f64,
    /// time constant of the weathervane bias (s)
    pub tau_vane: f64,
    /// head segments the bias acts on
    pub vane_segments: usize,
    /// sharp turns per second with a steady concentration
    pub turn_rate: f64,
    /// how strongly a falling concentration raises it: rate × exp(-k_turn · d log C/dt)
    pub k_turn: f64,
    /// d log C/dt is smoothed over this long before it sets the turn rate (s)
    pub tau_turn: f64,
    /// reversal and omega durations (s)
    pub reverse_time: f64,
    pub omega_time: f64,
    /// ventral bias of the omega bend, and how far back it reaches (segments)
    pub omega_bias: f64,
    pub omega_segments: usize,
    /// shortest time between two sharp turns (s)
    pub refractory: f64,
    /// a steady head bias (+ = dorsal) that cancels the body's own turning: SoftCon's
    /// irregular mesh is not a mirror image of itself, so even a perfectly symmetric
    /// wave crawls in circles; calibrated so the worm keeps its heading on average
    pub straighten: f64,
}

impl Default for Params {
    fn default() -> Params {
        Params {
            sigma: 6.3,
            cone: false,
            k_vane: 0.0,
            tau_vane: 1.0,
            vane_segments: 6,
            turn_rate: 0.0,
            k_turn: 0.0,
            tau_turn: 2.0,
            reverse_time: 3.0,
            omega_time: 3.0,
            omega_bias: 0.8,
            omega_segments: 18, // 0.8, 18, 3 s: reversal + omega turn the heading by ~160°
            refractory: 2.0,
            straighten: 0.015, // tools/turning.py over 3 min: -0.6 deg/s at 0.005, -0.1 at 0.013, +0.1 at 0.016, +0.3 at 0.02
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mode {
    Forward,
    Reverse,
    Omega,
}

pub struct Seek {
    pub p: Params,
    pub food: [f64; 3],
    pub mode: Mode,
    mode_time: f64,
    since_turn: f64,
    pub log_c: f64,
    pub dlogc: f64,      // per second, last step
    pub dlogc_slow: f64, // smoothed for the turn rate
    pub head_angle: f64, // head bend (rad, + = dorsal)
    pub head_rate: f64,
    pub bias: f64, // weathervane bias (+ = dorsal)
    pub turns: usize,
    /// subtracted from each segment's dorsal-minus-ventral activation: the brain's gait
    /// has a small steady ventral bend in the head that makes it crawl in circles
    pub trim: [f64; SEGMENTS],
    started: bool,
    rng: u64,
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// the mean dorsal-minus-ventral activation of each segment over a run of the brain's
/// muscle outputs (96 per step), skipping the first `skip` steps
pub fn gait_trim(muscles: &[f64], skip: usize) -> [f64; SEGMENTS] {
    let n = muscles.len() / 96;
    let mut t = [0.0; SEGMENTS];
    for k in skip..n {
        let m = &muscles[k * 96..k * 96 + 96];
        for s in 0..SEGMENTS {
            t[s] += (m[s] + m[2 * SEGMENTS + s] - m[SEGMENTS + s] - m[3 * SEGMENTS + s]) / 2.0 / (n - skip) as f64;
        }
    }
    t
}

impl Seek {
    pub fn new(food: [f64; 3], p: Params, seed: u64) -> Seek {
        Seek {
            p,
            food,
            mode: Mode::Forward,
            mode_time: 0.0,
            since_turn: 0.0,
            log_c: 0.0,
            dlogc: 0.0,
            dlogc_slow: 0.0,
            head_angle: 0.0,
            head_rate: 0.0,
            bias: 0.0,
            turns: 0,
            trim: [0.0; SEGMENTS],
            started: false,
            rng: seed.wrapping_mul(0x9E3779B97F4A7C15) | 1,
        }
    }

    fn uniform(&mut self) -> f64 {
        // xorshift64*
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        (self.rng.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64
    }

    /// log concentration at p (in-plane distance to the food)
    pub fn log_conc(&self, p: [f64; 3]) -> f64 {
        let d = sub(p, self.food);
        let r2 = d[0] * d[0] + d[2] * d[2];
        if self.p.cone {
            -r2.sqrt() / self.p.sigma
        } else {
            -r2 / (2.0 * self.p.sigma * self.p.sigma)
        }
    }

    /// signed angle of the head (slices 0-3) against the neck (slices 4-9), + = toward dorsal
    fn head_bend(body: &Body) -> f64 {
        let c = body.centerline();
        let mean = |a: usize, b: usize| {
            let mut m = [0.0; 3];
            for p in &c[a..b] {
                for k in 0..3 {
                    m[k] += p[k] / (b - a) as f64;
                }
            }
            m
        };
        let (h, n, t) = (mean(0, 3), mean(4, 7), mean(7, 10));
        let th = sub(h, n);
        let tn = sub(n, t);
        // in the plate (x, z); dorsal = turning from -z toward +x (straight body: tn = -z, th = +x - z gives +)
        let cross = tn[0] * th[2] - tn[2] * th[0];
        let dot = tn[0] * th[0] + tn[2] * th[2];
        cross.atan2(dot)
    }

    /// Takes the brain's muscles for this control step (dt seconds of body) and returns
    /// the ones to apply.
    pub fn apply(&mut self, brain: &[f64], body: &Body, nose: [f64; 3], dt: f64) -> [f64; 96] {
        let mut m = [0.0; 96];
        m.copy_from_slice(brain);
        for s in 0..SEGMENTS {
            let h = self.trim[s] / 2.0;
            for (q, sign) in [(0, -1.0), (1, 1.0), (2, -1.0), (3, 1.0)] {
                let i = q * SEGMENTS + s;
                m[i] = (m[i] + sign * h).clamp(0.0, 0.8);
            }
        }

        // sense
        let lc = self.log_conc(nose);
        let ang = Seek::head_bend(body);
        if !self.started {
            self.started = true;
            self.log_c = lc;
            self.head_angle = ang;
        }
        self.dlogc = (lc - self.log_c) / dt;
        self.log_c = lc;
        self.head_rate = (ang - self.head_angle) / dt;
        self.head_angle = ang;
        let a = dt / self.p.tau_turn;
        self.dlogc_slow += a * (self.dlogc - self.dlogc_slow);

        self.mode_time += dt;
        self.since_turn += dt;
        match self.mode {
            Mode::Forward => {
                // weathervane: rising concentration while the head sweeps one way bends it further that way
                let target = self.p.k_vane * self.dlogc * self.head_rate.signum();
                self.bias += dt / self.p.tau_vane * (target - self.bias);
                let rate = self.p.turn_rate * (-self.p.k_turn * self.dlogc_slow).exp();
                if self.since_turn > self.p.refractory && self.uniform() < rate * dt {
                    self.mode = Mode::Reverse;
                    self.mode_time = 0.0;
                    self.bias = 0.0;
                }
            }
            Mode::Reverse => {
                if self.mode_time >= self.p.reverse_time {
                    self.mode = Mode::Omega;
                    self.mode_time = 0.0;
                }
            }
            Mode::Omega => {
                if self.mode_time >= self.p.omega_time {
                    self.mode = Mode::Forward;
                    self.mode_time = 0.0;
                    self.since_turn = 0.0;
                    self.turns += 1;
                }
            }
        }

        match self.mode {
            Mode::Forward => bend(&mut m, self.bias + self.p.straighten, self.p.vane_segments),
            Mode::Reverse => {
                // the brain's gait run tail first: segment s takes segment 23 - s's activation,
                // so the wave starts large at the tail and travels to the head
                let f = m;
                for q in 0..4 {
                    for s in 0..SEGMENTS {
                        m[q * SEGMENTS + s] = f[q * SEGMENTS + SEGMENTS - 1 - s];
                    }
                }
            }
            Mode::Omega => {
                // the gait goes on with a deep ventral bend that fades in and out
                let s = (std::f64::consts::PI * self.mode_time / self.p.omega_time).sin();
                bend(&mut m, -self.p.omega_bias * s, self.p.omega_segments);
            }
        }
        m
    }
}

/// adds `b` to the dorsal and takes it from the ventral muscles of the first `n`
/// segments, tapering to zero at segment n (muscles: DR, VR, DL, VL × 24, head first)
fn bend(m: &mut [f64; 96], b: f64, n: usize) {
    for s in 0..n.min(SEGMENTS) {
        let w = b * (1.0 - s as f64 / n as f64);
        for (q, sign) in [(0, 1.0), (1, -1.0), (2, 1.0), (3, -1.0)] {
            let i = q * SEGMENTS + s;
            m[i] = (m[i] + sign * w).clamp(0.0, 0.8);
        }
    }
}
