//! Every spike of one run, for offline analysis (tools/banc_rhythm.py):
//!   cargo run --release --bin spikes <connectome.gz> <excited index file> <Hz> <seconds> <seed> <out.bin> [silenced index file]
//! out.bin = (step, neuron) u32 pairs, little endian; dt = 0.1 ms. W_SCALE=x multiplies w_syn (not the
//! Poisson kick), for connectomes with fewer synapses per neuron than FlyWire (BANC: see tools/banc_rhythm.py).
use fly_engine::brain::{load, Brain};
use std::io::Write;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let read = |p: &str| -> Vec<usize> { std::fs::read_to_string(p).unwrap().split_whitespace().map(|x| x.parse().unwrap()).collect() };
    let (o, t, c) = load(&a[1]);
    let hz: f64 = a[3].parse().unwrap();
    let steps = (a[4].parse::<f64>().unwrap() * 1e4).round() as i32;
    let mut b = Brain::new(o, t, c, a[5].parse().unwrap());
    if let Ok(x) = std::env::var("W_SCALE") { b.p.w_syn *= x.parse::<f64>().unwrap(); }
    for i in read(&a[2]) { b.set_rate(i, hz); }
    if let Some(p) = a.get(7) { for i in read(p) { b.set_silenced(i, true); } }
    let time = std::time::Instant::now();
    let mut out = Vec::new();
    for k in 0..steps {
        for &i in b.step() {
            out.extend_from_slice(&(k as u32).to_le_bytes());
            out.extend_from_slice(&i.to_le_bytes());
        }
    }
    std::fs::File::create(&a[6]).unwrap().write_all(&out).unwrap();
    eprintln!("{} spikes in {:.1} s ({:.2}x real time)", out.len() / 8, time.elapsed().as_secs_f64(), steps as f64 * 1e-4 / time.elapsed().as_secs_f64());
}
