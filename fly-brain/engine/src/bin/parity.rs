//! Spike-for-spike check against Brian 2 fed the same Poisson kicks (tools/parity_ref.py):
//!   cargo run --release --bin parity ../model/connectome.gz <ref dir>
//! Runs the engine with every neuron updated each step (as Brian does) and with the active
//! list the page uses, and compares every spike.
use fly_engine::{brain::{load, Brain}, read_events};
use std::collections::HashSet;

fn run(b: &mut Brain, kicks: &[(u32, u32)], steps: u32) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    let mut k = 0;
    let mut now = Vec::new();
    for n in 0..steps {
        now.clear();
        while k < kicks.len() && kicks[k].0 == n {
            now.push(kicks[k].1);
            k += 1;
        }
        b.step_with(&now);
        out.extend(b.spikes.iter().map(|&s| (n, s)));
    }
    out
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (offsets, targets, counts) = load(&a[1]);
    let dir = &a[2];
    let mut cases: Vec<String> = std::fs::read_dir(dir).unwrap()
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter(|f| f.ends_with(".kicks")).map(|f| f.trim_end_matches(".kicks").to_string()).collect();
    cases.sort();
    let mut ok = true;
    for case in &cases {
        let kicks = read_events(&format!("{dir}/{case}.kicks"));
        let reference = read_events(&format!("{dir}/{case}.spikes"));
        let silence = std::fs::read_to_string(format!("{dir}/{case}.silence")).unwrap_or_default();
        let steps = reference.iter().map(|e| e.0).max().unwrap_or(0).max(kicks.last().map_or(0, |e| e.0)) + 1;
        let steps = std::fs::read_to_string(format!("{dir}/{case}.steps")).map(|s| s.trim().parse().unwrap()).unwrap_or(steps);
        let reference_set: HashSet<_> = reference.iter().copied().collect();
        for full in [true, false] {
            let mut b = Brain::new(offsets.clone(), targets.clone(), counts.clone(), 1);
            b.full = full;
            for &(_, i) in &kicks {
                b.set_rfc(i as usize, 0); // Poisson targets have no refractory period
            }
            for s in silence.split_whitespace() {
                b.set_silenced(s.parse().unwrap(), true);
            }
            let t = std::time::Instant::now();
            let ours = run(&mut b, &kicks, steps);
            let secs = t.elapsed().as_secs_f64();
            let ours_set: HashSet<_> = ours.iter().copied().collect();
            let missing = reference_set.difference(&ours_set).count();
            let extra = ours_set.difference(&reference_set).count();
            let first = reference.iter().filter(|e| !ours_set.contains(e)).chain(ours.iter().filter(|e| !reference_set.contains(e)))
                .map(|e| e.0).min();
            let neurons: HashSet<_> = reference.iter().map(|e| e.1).collect();
            println!("{case:>20} {:>6}: Brian {} spikes ({} neurons), ours {}, missing {missing}, extra {extra}, first difference {}, {:.2} s for {:.1} s",
                if full { "full" } else { "active" }, reference.len(), neurons.len(), ours.len(),
                first.map_or("none".into(), |s| format!("step {s}")), secs, steps as f64 * 1e-4);
            if missing + extra > 0 { ok = false; }
        }
    }
    println!("{}", if ok { "PASS" } else { "FAIL" });
}
