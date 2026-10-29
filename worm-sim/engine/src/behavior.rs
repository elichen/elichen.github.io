//! Food seeking on an agar plate, as a layer between the brain's muscle output and
//! the body (ours, not BAAIWorm's). The brain still makes the forward gait; this
//! layer smells an attractant at the nose and steers with the two strategies real
//! worms use, at the rates measured in them:
//!
//!   weathervane (Iino & Yoshida 2009): while the head sweeps to one side, a rising
//!     concentration adds bend toward that side, so the path curves up the gradient
//!     (fit to their 12.7 °/mm of curving per mM/mm of gradient across the heading),
//!     on top of a slow random wander (their SD 32 °/mm, correlated over ~6 s);
//!   pirouettes (Pierce-Shimomura et al. 1999): sharp turns at Iino & Yoshida's fitted
//!     rate 0.023/(0.40 + e^(140 dC/dt)) + 0.0033 per s (dC/dt in mM/s), so they come
//!     more often while the concentration falls. Each is a reversal (the brain's gait
//!     run tail first), followed in a share of them by an omega turn (a deep ventral
//!     bend); durations from Zhao et al. 2003 (2.3 s, or 3.4 s before an omega).
//!
//! The plate is the body's rest plane (x = dorsal, z = head to tail at rest), with
//! normal y. The attractant is a Gaussian hill around the food on a flat floor, like
//! Pierce-Shimomura's radial 2-6 mM gradient.
use crate::body::Body;

pub const SEGMENTS: usize = 24;

/// gait_trim of the brain's muscles from 3 s to 63 s after v = -65 mV on BAAIWorm's
/// recorded input (seek checks it against its own run)
pub const GAIT_TRIM: [f64; SEGMENTS] = [
    -0.015372612738218064, -0.015336205304888176, -0.0067896154846119865, 0.0007098410469227907, 0.0030434810275001374, 0.002293714271064852, 0.0003570827520865053, 0.0001975132462501387, 0.00028884826898236977, 0.0003122861516259236, 0.0002719399824269109, 0.00017462434173892632, 4.9495191549946416e-05, -6.995885807817588e-05, -0.00015185361985098122, -0.00018663490974994628, -0.00017109239247303584, -0.00011945015898621338, -4.876114848616923e-05, 2.126099379729495e-05, 7.544725548324804e-05, 0.00010382466848570286, 0.0001031182578368294, 7.993191250782923e-05,
];

#[derive(Clone, Copy, Debug)]
pub struct Params {
    /// the hill: c_floor + c_peak exp(-r²/2σ²) mM; σ in body units (0.63 = 1 mm)
    pub sigma: f64,
    pub c_floor: f64,
    pub c_peak: f64,
    /// weathervane gain: muscle bias per (dC/dt in mM/s × sign of the head sweep)
    pub k_vane: f64,
    /// time constant of the weathervane bias (s)
    pub tau_vane: f64,
    /// head segments the bias acts on
    pub vane_segments: usize,
    /// random wander of the head bias: SD and correlation time (s)
    pub wander: f64,
    pub tau_wander: f64,
    /// sharp-turn rate a / (b + e^(k dC/dt)) + c per s; 0 × gain = no turns
    pub turn_a: f64,
    pub turn_b: f64,
    pub turn_k: f64,
    pub turn_c: f64,
    /// dC/dt is smoothed over this long before it sets the turn rate (s)
    pub tau_turn: f64,
    /// share of reversals that end in an omega turn
    pub p_omega: f64,
    /// reversal durations without and before an omega, omega duration (s)
    pub reverse_time: f64,
    pub reverse_omega_time: f64,
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
            c_floor: 2.0,
            c_peak: 4.0,
            k_vane: 0.7, // 0.5 gives 7.1 °/mm per mM/mm, 1 gives 24.5 (Iino & Yoshida: 12.7)
            tau_vane: 1.0,
            vane_segments: 6,
            wander: 0.09, // curving SD 23 °/mm at 0.06, 35 at 0.1 (Iino & Yoshida: 32.3)
            tau_wander: 6.0,
            turn_a: 0.023,
            turn_b: 0.40,
            turn_k: 140.0,
            turn_c: 0.0033,
            tau_turn: 2.0,
            p_omega: 0.5,
            reverse_time: 2.3,
            reverse_omega_time: 3.4,
            omega_time: 3.0,
            omega_bias: 0.8,
            omega_segments: 18, // 0.8, 18, 3 s: reversal + omega turn the heading by ~160°
            refractory: 1.0,
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
    pub c: f64,       // mM at the nose
    pub dcdt: f64,    // mM/s, last step
    pub dcdt_slow: f64, // smoothed for the turn rate
    pub wander: f64,
    omega_next: bool,
    pub head_angle: f64, // head bend (rad, + = dorsal)
    pub head_rate: f64,
    pub bias: f64, // weathervane bias (+ = dorsal)
    pub turns: usize,
    /// subtracted from each segment's dorsal-minus-ventral activation: the brain's gait
    /// has a small steady ventral bend in the head that makes it crawl in circles
    pub trim: [f64; SEGMENTS],
    started: bool,
    pub seed: u64,
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
            c: 0.0,
            dcdt: 0.0,
            dcdt_slow: 0.0,
            wander: 0.0,
            omega_next: false,
            head_angle: 0.0,
            head_rate: 0.0,
            bias: 0.0,
            turns: 0,
            trim: [0.0; SEGMENTS],
            started: false,
            seed,
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

    fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform().max(1e-300), self.uniform());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }

    /// concentration (mM) at p, from the in-plane distance to the food
    pub fn conc(&self, p: [f64; 3]) -> f64 {
        let d = sub(p, self.food);
        let r2 = d[0] * d[0] + d[2] * d[2];
        self.p.c_floor + self.p.c_peak * (-r2 / (2.0 * self.p.sigma * self.p.sigma)).exp()
    }

    /// sharp turns per second at a smoothed dC/dt (mM/s)
    pub fn turn_rate(&self, dcdt: f64) -> f64 {
        let p = &self.p;
        p.turn_a / (p.turn_b + (p.turn_k * dcdt).min(50.0).exp()) + p.turn_c
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
        let c = self.conc(nose);
        let ang = Seek::head_bend(body);
        if !self.started {
            self.started = true;
            self.c = c;
            self.head_angle = ang;
        }
        self.dcdt = (c - self.c) / dt;
        self.c = c;
        self.head_rate = (ang - self.head_angle) / dt;
        self.head_angle = ang;
        self.dcdt_slow += dt / self.p.tau_turn * (self.dcdt - self.dcdt_slow);
        // Ornstein-Uhlenbeck wander
        let a = dt / self.p.tau_wander;
        let n = self.normal();
        self.wander += -a * self.wander + self.p.wander * (2.0 * a).sqrt() * n;

        self.mode_time += dt;
        self.since_turn += dt;
        match self.mode {
            Mode::Forward => {
                // weathervane: rising concentration while the head sweeps one way bends it further that way
                let target = self.p.k_vane * self.dcdt * self.head_rate.signum();
                self.bias += dt / self.p.tau_vane * (target - self.bias);
                let rate = self.turn_rate(self.dcdt_slow);
                if self.since_turn > self.p.refractory && self.uniform() < rate * dt {
                    self.mode = Mode::Reverse;
                    self.mode_time = 0.0;
                    self.bias = 0.0;
                    self.omega_next = self.uniform() < self.p.p_omega;
                }
            }
            Mode::Reverse => {
                let t = if self.omega_next { self.p.reverse_omega_time } else { self.p.reverse_time };
                if self.mode_time >= t {
                    self.mode_time = 0.0;
                    if self.omega_next {
                        self.mode = Mode::Omega;
                    } else {
                        self.mode = Mode::Forward;
                        self.since_turn = 0.0;
                        self.turns += 1;
                    }
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
            Mode::Forward => bend(&mut m, self.bias + self.wander + self.p.straighten, self.p.vane_segments),
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
