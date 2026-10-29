//! The closed loop of BAAIWorm's worm_in_env.py / worm_neural_network.py ("fc"
//! control), one control step at a time:
//!
//!   sense:  the head's change in attractant concentration (-distance to the
//!           food), normalized against the last 11 changes, becomes a current
//!           clamp on 15 sensory neurons (in nA, clipped to [-0.08, 0.02]);
//!           "offline" mode instead plays the recorded 2 s cycle the circuit
//!           was tuned on.
//!   think:  60 neural steps of 5/3 ms (100 ms);
//!   move:   the 80 motor neurons' mean soma voltage through the 80×96 readout,
//!           muscle = clip((v·W + 80)/100, 0, 0.8), then 8 body steps of 1/240 s.
use crate::body::Body;
use crate::brain::Brain;

pub const INPUT: [&str; 15] = ["AWAL", "AWAR", "AWCL", "AWCR", "ASKL", "ASKR", "ALNL", "ALNR", "PLML", "PHAL", "PHAR", "URYDL", "URYDR", "URYVL", "URYVR"];
pub const OUTPUT: [&str; 80] = ["RIML", "RIMR", "RMEL", "RMER", "RMED", "RMEV", "RMDDL", "RMDDR", "RMDL", "RMDR", "RMDVL", "RMDVR", "RIVL", "RIVR", "SMDDL", "SMDDR", "SMDVL", "SMDVR", "SMBDL", "SMBDR", "SMBVL", "SMBVR", "DA01", "DA02", "DA03", "DA04", "DA05", "DA06", "DA07", "DA08", "DA09", "DB01", "DB02", "DB03", "DB04", "DB05", "DB06", "DB07", "DD01", "DD02", "DD03", "DD04", "DD05", "DD06", "VA01", "VA02", "VA03", "VA04", "VA05", "VA06", "VA07", "VA08", "VA09", "VA10", "VA11", "VA12", "VB01", "VB02", "VB03", "VB04", "VB05", "VB06", "VB07", "VB08", "VB09", "VB10", "VB11", "VD01", "VD02", "VD03", "VD04", "VD05", "VD06", "VD07", "VD08", "VD09", "VD10", "VD11", "VD12", "VD13"];

/// The recorded input the circuit's readout was fit on (forfigure_input.txt), one 2 s cycle.
pub const CYCLE: [f64; 20] = [
    -0.080620, -0.074593, 0.022032, 0.022032, -0.017174, -0.055962, -0.055962, -0.045709, -0.064428, -0.072575, -0.046820, -0.034368, -0.034368, -0.029772,
    -0.027130, -0.023020, -0.038887, -0.010831, -0.065437, -0.080620,
];

pub const NEURAL_STEPS: usize = 60;
pub const BODY_STEPS: usize = 8;
const HISTORY: usize = 11;

pub struct Worm {
    pub brain: Brain,
    pub body: Body,
    pub wout: Vec<f64>, // 80 × 96
    pub input_cells: Vec<usize>,
    pub output_cells: Vec<usize>,
    pub muscle: Vec<f64>,
    pub motor_v: Vec<f64>,
    pub step_count: usize,
    /// food in body coordinates; None = play the recorded input cycle
    pub food: Option<[f64; 3]>,
    pub head_vertex: usize,
    concentration: f64,
    history: [f64; HISTORY],
    pub stimulus: f64,
    pub delta_c: f64,
}

impl Worm {
    /// `cells` = the circuit's cell names in brain.bin order
    pub fn new(brain: Brain, body: Body, wout: Vec<f64>, cells: &[String]) -> Worm {
        let idx = |n: &str| cells.iter().position(|x| x == n).unwrap_or_else(|| panic!("{n}"));
        let input = INPUT.iter().map(|n| idx(n)).collect();
        let output = OUTPUT.iter().map(|n| idx(n)).collect();
        Worm::with_cells(brain, body, wout, input, output)
    }

    pub fn with_cells(brain: Brain, body: Body, wout: Vec<f64>, input_cells: Vec<usize>, output_cells: Vec<usize>) -> Worm {
        let head_vertex = body.contour[0] as usize;
        let mut w = Worm {
            input_cells,
            output_cells,
            brain,
            body,
            wout,
            muscle: vec![0.0; 96],
            motor_v: vec![0.0; 80],
            step_count: 0,
            food: None,
            head_vertex,
            concentration: 0.0,
            history: [0.0; HISTORY],
            stimulus: CYCLE[0],
            delta_c: 0.0,
        };
        w.reset();
        w
    }

    pub fn head(&self) -> [f64; 3] {
        let i = self.head_vertex * 3;
        [self.body.x[i], self.body.x[i + 1], self.body.x[i + 2]]
    }

    fn concentration_at(&self, p: [f64; 3]) -> f64 {
        match self.food {
            Some(f) => -((f[0] - p[0]).powi(2) + (f[1] - p[1]).powi(2) + (f[2] - p[2]).powi(2)).sqrt(),
            None => 0.0,
        }
    }

    pub fn reset(&mut self) {
        self.brain.init(-65.0);
        self.body.reset();
        self.step_count = 0;
        self.history = [0.0; HISTORY];
        self.concentration = self.concentration_at(self.head());
        for m in self.muscle.iter_mut() {
            *m = 0.0;
        }
        self.body.act.copy_from_slice(&self.muscle);
    }

    /// Moving the food restarts the concentration bookkeeping (the worm "smells" it fresh).
    /// Before the first step BAAIWorm takes the last smell at the start point (the body
    /// origin), not at the head vertex, so its first change in concentration is not zero.
    pub fn set_food(&mut self, food: Option<[f64; 3]>) {
        self.food = food;
        let at = if self.step_count == 0 { [0.0; 3] } else { self.head() };
        self.concentration = self.concentration_at(at);
    }

    /// WormNeuralNetwork.normalize_delta_concentration ("fc" branch)
    fn sense(&mut self) -> f64 {
        let c = self.concentration_at(self.head());
        let lo = self.history.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = self.history.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let delta = c - self.concentration;
        let s = ((delta - lo) * 100.0 / (hi - lo) - 80.0) / 1000.0;
        let s = if s.is_nan() { -0.08 } else { s.clamp(-0.08, 0.02) };
        self.history.copy_within(1.., 0);
        self.history[HISTORY - 1] = delta;
        self.concentration = c;
        self.delta_c = delta;
        s
    }

    /// The brain's half of a control step; returns the new muscle activations.
    pub fn think(&mut self) {
        self.step_count += 1;
        self.stimulus = if self.food.is_some() { self.sense() } else { CYCLE[(self.step_count - 1) % 20] };
        for &c in &self.input_cells {
            self.brain.iclamp[c] = self.stimulus;
        }
        for v in self.motor_v.iter_mut() {
            *v = 0.0;
        }
        for _ in 0..NEURAL_STEPS {
            self.brain.step();
            for (j, &c) in self.output_cells.iter().enumerate() {
                self.motor_v[j] += self.brain.soma_v(c);
            }
        }
        for v in self.motor_v.iter_mut() {
            *v /= NEURAL_STEPS as f64;
        }
        for m in 0..96 {
            let mut x = 0.0;
            for j in 0..80 {
                x += self.motor_v[j] * self.wout[j * 96 + m];
            }
            self.muscle[m] = ((x + 80.0) / 100.0).clamp(0.0, 0.8);
        }
    }

    /// Runs the brain with the body held straight (BAAIWorm drives the body from a
    /// recorded file meanwhile): from v = -65 mV everywhere the first second's
    /// transient would otherwise curl the worm round.
    pub fn warm_up(&mut self, control_steps: usize) {
        let food = self.food.take();
        for _ in 0..control_steps {
            self.think();
        }
        self.set_food(food);
    }

    pub fn moving(&mut self) {
        self.body.act.copy_from_slice(&self.muscle);
        for _ in 0..BODY_STEPS {
            self.body.step();
        }
    }

    pub fn control_step(&mut self) {
        self.think();
        self.moving();
    }
}
