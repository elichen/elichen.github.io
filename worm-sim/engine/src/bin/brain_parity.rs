//! Runs the circuit on the input NEURON was given in BAAIWorm's 02_train_cnn
//! (forfigure_input.txt, held for 100 ms per value, into 15 sensory neurons) and
//! compares the 80 motor-neuron somas with NEURON's recorded traces.
//! The input file was appended to after NEURON's run: from value 238 on (23.8 s)
//! it holds a later closed-loop run's input, so only the first 23.8 s compare.
//! usage: brain_parity <model dir> <ref dir from tools/export_refs.py> [steps]
use std::time::Instant;
use worm_engine::brain::Brain;

fn f64s(path: &str) -> Vec<f64> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    b.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect()
}

const INPUT: [&str; 15] = ["AWAL", "AWAR", "AWCL", "AWCR", "ASKL", "ASKR", "ALNL", "ALNR", "PLML", "PHAL", "PHAR", "URYDL", "URYDR", "URYVL", "URYVR"];
pub const OUTPUT: [&str; 80] = ["RIML", "RIMR", "RMEL", "RMER", "RMED", "RMEV", "RMDDL", "RMDDR", "RMDL", "RMDR", "RMDVL", "RMDVR", "RIVL", "RIVR", "SMDDL", "SMDDR", "SMDVL", "SMDVR", "SMBDL", "SMBDR", "SMBVL", "SMBVR", "DA01", "DA02", "DA03", "DA04", "DA05", "DA06", "DA07", "DA08", "DA09", "DB01", "DB02", "DB03", "DB04", "DB05", "DB06", "DB07", "DD01", "DD02", "DD03", "DD04", "DD05", "DD06", "VA01", "VA02", "VA03", "VA04", "VA05", "VA06", "VA07", "VA08", "VA09", "VA10", "VA11", "VA12", "VB01", "VB02", "VB03", "VB04", "VB05", "VB06", "VB07", "VB08", "VB09", "VB10", "VB11", "VD01", "VD02", "VD03", "VD04", "VD05", "VD06", "VD07", "VD08", "VD09", "VD10", "VD11", "VD12", "VD13"];

fn main() {
    let dir = std::env::args().nth(1).expect("model dir");
    let refdir = std::env::args().nth(2).expect("ref dir");
    let names: Vec<String> = {
        let s = std::fs::read_to_string(format!("{dir}/brain.json")).unwrap();
        let a = s.find("[").unwrap();
        let b = s.find("]").unwrap();
        s[a + 1..b].split(',').map(|x| x.trim().trim_matches('"').to_string()).collect()
    };
    let idx = |n: &str| names.iter().position(|x| x == n).unwrap_or_else(|| panic!("{n}"));
    let mut brain = Brain::new(&f64s(&format!("{dir}/brain.bin")), 5.0 / 3.0);
    println!("nodes {} synapses {} gap junctions {}", brain.n, brain.n_syn(), brain.n_gap());
    let input = f64s(&format!("{refdir}/input300.bin"));
    let refv = f64s(&format!("{refdir}/ref_soma.bin")); // (80, 18000)
    let wout = f64s(&format!("{dir}/wout_offline.bin")); // (80, 96)
    let steps: usize = std::env::args().nth(3).map(|s| s.parse().unwrap()).unwrap_or(238 * 60 - 1);
    let inp: Vec<usize> = INPUT.iter().map(|n| idx(n)).collect();
    let out: Vec<usize> = OUTPUT.iter().map(|n| idx(n)).collect();
    // NEURON's recorded vectors start with v(0): mine[k] holds v after k steps too
    let steps = steps.min(17999) + 1;
    let mut mine = vec![-65.0; 80 * steps];
    let t0 = Instant::now();
    for k in 0..steps - 1 {
        for &c in &inp {
            brain.iclamp[c] = input[k / 60];
        }
        brain.step();
        for (j, &c) in out.iter().enumerate() {
            mine[j * steps + k + 1] = brain.soma_v(c);
        }
    }
    let el = t0.elapsed().as_secs_f64();
    if std::env::var("TRACE").is_ok() {
        let j: usize = std::env::var("TRACE").unwrap().parse().unwrap();
        for k in (0..steps.min(40)).chain((40..steps).step_by(steps / 20 + 1)) {
            println!("  {} k {:5} mine {:9.4} NEURON {:9.4}", OUTPUT[j], k, mine[j * steps + k], refv[j * 18000 + k]);
        }
    }
    println!("{} steps in {:.2} s ({:.1}x real time)", steps, el, steps as f64 * 5.0 / 3.0 / 1000.0 / el);
    for sec in 0..steps / 600 {
        let mut e = 0.0f64;
        for j in 0..80 {
            for k in sec * 600..(sec + 1) * 600 {
                e = e.max((mine[j * steps + k] - refv[j * 18000 + k]).abs());
            }
        }
        print!("{}s:{:.2e} ", sec + 1, e);
    }
    println!();
    let mut worst = 0.0f64;
    let mut sq = 0.0;
    for j in 0..80 {
        let mut e = 0.0f64;
        for k in 0..steps {
            e = e.max((mine[j * steps + k] - refv[j * 18000 + k]).abs());
            sq += (mine[j * steps + k] - refv[j * 18000 + k]).powi(2);
        }
        worst = worst.max(e);
        if j < 6 || e > 1.0 {
            println!("{:6} max|dv| {:.4} mV  v(end) mine {:.3} NEURON {:.3}", OUTPUT[j], e, mine[j * steps + steps - 1], refv[j * 18000 + steps - 1]);
        }
    }
    println!("all 80: max |dv| {:.5} mV, rms {:.5} mV", worst, (sq / (80 * steps) as f64).sqrt());
    // muscle drive through the readout, as WormNeuralNetwork does
    let blocks = steps / 60;
    let mut dm = 0.0f64;
    for b in 0..blocks {
        for m in 0..96 {
            let (mut a, mut r) = (0.0, 0.0);
            for j in 0..80 {
                let (mut s1, mut s2) = (0.0, 0.0);
                for k in b * 60..b * 60 + 60 {
                    s1 += mine[j * steps + k];
                    s2 += refv[j * 18000 + k];
                }
                a += s1 / 60.0 * wout[j * 96 + m];
                r += s2 / 60.0 * wout[j * 96 + m];
            }
            let f = |x: f64| ((x + 80.0) / 100.0).clamp(0.0, 0.8);
            dm = dm.max((f(a) - f(r)).abs());
        }
    }
    println!("muscle activation max |diff| {:.5} (range 0-0.8)", dm);
    println!("{}", if worst < 1e-3 && dm < 1e-4 { "PASS" } else { "FAIL" });
}
