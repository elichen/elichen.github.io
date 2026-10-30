//! Mean rate per neuron over trials with the engine's own Poisson input (compare with tools/stats_ref.py):
//!   cargo run --release --bin rates ../model/connectome.gz <excited index file> <Hz> <trials> <out.csv> [silenced index file]
use fly_engine::brain::{load, Brain};
use std::io::Write;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (o, t, c) = load(&a[1]);
    let exc: Vec<usize> = std::fs::read_to_string(&a[2]).unwrap().split_whitespace().map(|x| x.parse().unwrap()).collect();
    let hz: f64 = a[3].parse().unwrap();
    let trials: u64 = a[4].parse().unwrap();
    let silenced: Vec<usize> = a.get(6).map(|p| std::fs::read_to_string(p).unwrap().split_whitespace().map(|x| x.parse().unwrap()).collect()).unwrap_or_default();
    let mut b = Brain::new(o, t, c, 0);
    let mut sum = vec![0f64; b.n];
    let mut sq = vec![0f64; b.n];
    let time = std::time::Instant::now();
    for trial in 0..trials {
        let mut nb = Brain::new(b.offsets.clone(), b.targets.clone(), b.counts.clone(), 1000 + trial);
        std::mem::swap(&mut b, &mut nb);
        for &i in &exc { b.set_rate(i, hz); }
        for &i in &silenced { b.set_silenced(i, true); }
        for _ in 0..10000 { b.step(); }
        for i in 0..b.n {
            let r = b.spike_count[i] as f64;
            sum[i] += r;
            sq[i] += r * r;
        }
    }
    let mut f = std::fs::File::create(&a[5]).unwrap();
    writeln!(f, "index,rate,sd").unwrap();
    let tn = trials as f64;
    for i in 0..b.n {
        if sum[i] > 0.0 {
            let m = sum[i] / tn;
            writeln!(f, "{i},{:.4},{:.4}", m, (sq[i] / tn - m * m).max(0.0).sqrt()).unwrap();
        }
    }
    eprintln!("{trials} trials in {:.1} s", time.elapsed().as_secs_f64());
}
