//! Closed-loop run: brain → readout → body, printing where the worm goes.
//! usage: loop_run <model dir> <offline|online|replay> <seconds of body time> [food x y z]
//! (replay plays SoftCon's recorded muscles from REF=<ref dir> instead of the brain's)
//! env: GJ / SYN = gain on gap junctions / chemical synapses (default 1)
//!      WARM=<control steps of brain warm-up, body held> (default 20)
//!      DUMP=<file> writes each step's 96 muscle activations
use std::io::Write;
use std::time::Instant;
use worm_engine::{body::Body, brain::Brain, frame::fit, worm::Worm};

fn f64s(path: &str) -> Vec<f64> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    b.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect()
}

fn com(w: &Worm) -> [f64; 3] {
    let mut c = [0.0; 3];
    for i in 0..w.body.nv {
        for k in 0..3 {
            c[k] += w.body.x[i * 3 + k] / w.body.nv as f64;
        }
    }
    c
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let dir = &a[1];
    let mode = a[2].as_str();
    let names: Vec<String> = {
        let s = std::fs::read_to_string(format!("{dir}/brain.json")).unwrap();
        let (i, j) = (s.find('[').unwrap(), s.find(']').unwrap());
        s[i + 1..j].split(',').map(|x| x.trim().trim_matches('"').to_string()).collect()
    };
    let brain = Brain::new(&f64s(&format!("{dir}/brain.bin")), 5.0 / 3.0);
    let body = Body::new(&f64s(&format!("{dir}/body.bin")));
    let wout = f64s(&format!("{dir}/wout_{}.bin", if mode == "replay" { "offline" } else { mode }));
    let replay = if mode == "replay" { Some(f64s(&format!("{}/ref_muscles.bin", std::env::var("REF").expect("REF=<ref dir>")))) } else { None };
    let mut w = Worm::new(brain, body, wout, &names);
    let secs: f64 = a[3].parse().unwrap();
    if a.len() > 6 {
        w.set_food(Some([a[4].parse().unwrap(), a[5].parse().unwrap(), a[6].parse().unwrap()]));
    }
    for (var, kinds) in [("GJ", &[0u32][..]), ("SYN", &[1, 2][..])] {
        if let Ok(g) = std::env::var(var) {
            let g: f64 = g.parse().unwrap();
            for &kind in kinds {
                let b = &mut w.brain;
                if kind == 0 {
                    b.gj_gain.iter_mut().for_each(|x| *x = g);
                } else {
                    for k in 0..b.n_syn() {
                        if b.syn_pre_post(k).2 == (kind == 1) {
                            b.sy_gain[k] = g;
                        }
                    }
                }
            }
        }
    }
    let warm: usize = std::env::var("WARM").map(|s| s.parse().unwrap()).unwrap_or(20);
    w.warm_up(warm);
    let mut dump = std::env::var("DUMP").ok().map(|p| std::fs::File::create(p).unwrap());
    let steps = (secs * 30.0) as usize;
    let t0 = Instant::now();
    let (mut tb, mut tn) = (0.0, 0.0);
    let mut prev = com(&w);
    let (mut lo, mut hi) = (f64::MAX, f64::MIN);
    for k in 0..steps {
        let t = Instant::now();
        w.think();
        tn += t.elapsed().as_secs_f64();
        if let Some(r) = &replay {
            w.muscle.copy_from_slice(&r[(k % 300) * 96..(k % 300) * 96 + 96]);
        }
        lo = lo.min(w.stimulus);
        hi = hi.max(w.stimulus);
        if let Some(f) = dump.as_mut() {
            writeln!(f, "{}", w.muscle.iter().map(|x| format!("{:.5}", x)).collect::<Vec<_>>().join(" ")).unwrap();
        }
        let t = Instant::now();
        w.moving();
        tb += t.elapsed().as_secs_f64();
        if k % 30 == 29 {
            let h = w.head();
            let c = com(&w);
            let (r, _) = fit(&w.body.rest, &w.body.x);
            let fwd = [-r[0][2], -r[1][2], -r[2][2]]; // the head end of the rest mesh is -z
            let d = [c[0] - prev[0], c[1] - prev[1], c[2] - prev[2]];
            let v_fwd = d[0] * fwd[0] + d[1] * fwd[1] + d[2] * fwd[2];
            prev = c;
            let food = w
                .food
                .map(|f| {
                    let df = ((f[0] - h[0]).powi(2) + (f[1] - h[1]).powi(2) + (f[2] - h[2]).powi(2)).sqrt();
                    let to = [(f[0] - c[0]), (f[1] - c[1]), (f[2] - c[2])];
                    let n = (to[0] * to[0] + to[1] * to[1] + to[2] * to[2]).sqrt();
                    format!("food dist {:.3} cos(fwd,food) {:+.2}", df, (to[0] * fwd[0] + to[1] * fwd[1] + to[2] * fwd[2]) / n)
                })
                .unwrap_or_default();
            println!("t {:5.1}s com [{:+.2} {:+.2} {:+.2}] v_fwd {:+.3}/s stim [{:+.3},{:+.3}] {}", (k + 1) as f64 / 30.0, c[0], c[1], c[2], v_fwd, lo, hi, food);
            lo = f64::MAX;
            hi = f64::MIN;
        }
    }
    let el = t0.elapsed().as_secs_f64();
    println!("{:.1} s body time in {:.2} s wall (brain {:.2} s, body {:.2} s): {:.2}x real time", secs, el, tn, tb, secs / el);
}
