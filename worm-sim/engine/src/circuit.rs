//! Muscles from the circuit through the worm's anatomy (ours, "plan B"), instead of
//! BAAIWorm's fitted 80×96 readout: each motor neuron drives only the muscles it has
//! neuromuscular junctions on (BAAIWorm's neuron_muscle.xlsx; D-class neurons are
//! GABAergic and relax them), and the motor neurons that sense body bending (Wen et al.
//! 2012; muscle_neuron.xlsx) get a current from the bend of the muscles they sense. The
//! loop runs in short sub-steps so this feedback is not 100 ms late.

/// the motor classes, each with its own NMJ and proprioceptive gain
pub const CLASSES: [&str; 12] = ["DA", "DB", "DD", "VA", "VB", "VD", "SMD", "RMD", "RME", "SMB", "RIM", "RIV"];

/// class of a cell name: digits, then L/R, then D/V are dropped (AVBL -> AVB, SMDDL -> SMD, DB05 -> DB)
pub fn class_of(name: &str) -> String {
    let mut s = name.trim_end_matches(|c: char| c.is_ascii_digit()).to_string();
    if s.len() > 3 && s.ends_with(['L', 'R']) {
        s.pop();
    }
    if s.len() > 3 && s.ends_with(['D', 'V']) {
        s.pop();
    }
    s
}

/// each class's mean soma voltage with the circuit at rest (tonic input, straight body)
pub const REST: [f64; 12] = [-57.1, -54.2, -53.2, -49.6, -47.0, -51.4, -24.3, -10.1, -20.4, -23.8, -40.5, 4.3];

#[derive(Clone, Debug)]
pub struct Params {
    /// muscle activation per fully active neuron, by class
    pub nmj_gain: [f64; 12],
    /// release = 1 / (1 + exp(-(v - v_half) / v_slope)), graded like the worm's NMJs; v_half by class
    pub v_half: [f64; 12],
    pub v_slope: f64,
    /// tonic activation of every muscle
    pub bias: f64,
    /// nA per unit of bending strain on the muscles a neuron senses, by class
    pub prop_gain: [f64; 12],
    /// muscle activation time constant (s)
    pub tau_muscle: f64,
    /// current on the sensory neurons when no food is smelled (nA)
    pub tonic: f64,
    /// play BAAIWorm's recorded 2 s input cycle on the sensory neurons instead of `tonic`
    pub cycle: bool,
}

impl Default for Params {
    fn default() -> Self {
        Params { nmj_gain: [0.3; 12], v_half: REST, v_slope: 3.0, bias: 0.0, prop_gain: [0.0; 12], tau_muscle: 0.05, tonic: -0.04, cycle: false }
    }
}

pub struct Circuit {
    nmj: Vec<(usize, usize)>, // (output neuron, muscle)
    prop: Vec<(usize, usize)>, // (muscle, output neuron)
    n_sensed: [f64; 80],
    pub class: [usize; 80], // index into CLASSES, of each output neuron
    pub p: Params,
    pub act: [f64; 96],
    pub substeps: usize,
}

impl Circuit {
    /// `nmj` = nmj.bin (tools/export_nmj.py); `names` = the 80 output neurons
    pub fn new(nmj: &[f64], names: &[&str], p: Params) -> Circuit {
        assert_eq!(nmj.len(), 2 * 80 * 96);
        let mut c = Circuit { nmj: vec![], prop: vec![], n_sensed: [0.0; 80], class: [0; 80], p, act: [0.0; 96], substeps: 6 };
        for j in 0..80 {
            let k = class_of(names[j]);
            c.class[j] = CLASSES.iter().position(|x| *x == k).unwrap_or_else(|| panic!("class {k}"));
            for m in 0..96 {
                if nmj[j * 96 + m] != 0.0 {
                    c.nmj.push((j, m));
                }
                if nmj[80 * 96 + m * 80 + j] != 0.0 {
                    c.prop.push((m, j));
                    c.n_sensed[j] += 1.0;
                }
            }
        }
        c
    }

    /// output neuron `j` drives no muscle and senses none (ablation)
    pub fn ablate(&mut self, j: usize) {
        self.nmj.retain(|&(n, _)| n != j);
        self.prop.retain(|&(_, n)| n != j);
    }

    /// proprioceptive current on each output neuron (added to `iclamp` of its cell):
    /// the bend toward the sensed muscle's side, half the strain difference between it
    /// and the opposite muscle (DR <-> VR, DL <-> VL), so even shortening does not count
    pub fn proprio(&self, stretch: &[f64; 96], out: &mut [f64; 80]) {
        *out = [0.0; 80];
        for &(m, j) in &self.prop {
            let opp = (m + 24) % 48 + m / 48 * 48;
            out[j] += self.p.prop_gain[self.class[j]] * (stretch[opp] - stretch[m]) / 2.0 / self.n_sensed[j];
        }
    }

    /// new activations from the output neurons' mean soma voltages over `dt` seconds
    pub fn muscles(&mut self, v: &[f64; 80], dt: f64) {
        let mut u = [self.p.bias; 96];
        for &(j, m) in &self.nmj {
            let k = self.class[j];
            let sign = if k == 2 || k == 5 { -1.0 } else { 1.0 }; // DD, VD
            u[m] += sign * self.p.nmj_gain[k] / (1.0 + (-(v[j] - self.p.v_half[k]) / self.p.v_slope).exp());
        }
        let a = (dt / self.p.tau_muscle).min(1.0);
        for m in 0..96 {
            self.act[m] += (u[m].clamp(0.0, 0.8) - self.act[m]) * a;
        }
    }
}
