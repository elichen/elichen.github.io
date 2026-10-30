//! Speed of the engine on a reference case's kicks: steps/s, active neurons, synaptic events.
//!   cargo run --release --bin bench ../model/connectome.gz <ref dir>/<case>
use fly_engine::{brain::{load, Brain}, read_events};
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (o, t, c) = load(&a[1]);
    let kicks = read_events(&format!("{}.kicks", a[2]));
    let mut b = Brain::new(o.clone(), t, c, 1);
    for &(_, i) in &kicks { b.set_rfc(i as usize, 0); }
    let (mut act, mut events, mut spikes) = (0usize, 0u64, 0usize);
    let mut k = 0;
    let time = std::time::Instant::now();
    for n in 0..10000u32 {
        let mut now = Vec::new();
        while k < kicks.len() && kicks[k].0 == n { now.push(kicks[k].1); k += 1; }
        b.step_with(&now);
        act += b.active_len();
        spikes += b.spikes.len();
        events += b.spikes.iter().map(|&s| (o[s as usize + 1] - o[s as usize]) as u64).sum::<u64>();
    }
    println!("{:.2} s per simulated s; mean active {:.0}; {} spikes; {:.0}M synaptic events/s simulated",
        time.elapsed().as_secs_f64(), act as f64 / 1e4, spikes, events as f64 / 1e6);
}
