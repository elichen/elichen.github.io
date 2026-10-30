//! Activate one set at a time and record chosen neurons (tools/banc_screen.py):
//!   cargo run --release --bin screen <connectome.gz> <sets file: one line per set, "name i j ..."> <Hz> <seconds> <record index file> <out.bin>
//! out.bin = (set number, step, neuron) u32 triples; dt = 0.1 ms. Each set starts from rest with seed 1.
use fly_engine::brain::{load, Brain};
use std::io::Write;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (o, t, c) = load(&a[1]);
    let hz: f64 = a[3].parse().unwrap();
    let steps = (a[4].parse::<f64>().unwrap() * 1e4).round() as u32;
    let rec: Vec<usize> = std::fs::read_to_string(&a[5]).unwrap().split_whitespace().map(|x| x.parse().unwrap()).collect();
    let mut keep = vec![false; o.len() - 1];
    for &i in &rec { keep[i] = true; }
    let mut out = Vec::new();
    let time = std::time::Instant::now();
    for (s, line) in std::fs::read_to_string(&a[2]).unwrap().lines().enumerate() {
        let mut b = Brain::new(o.clone(), t.clone(), c.clone(), 1);
        for x in line.split_whitespace().skip(1) { b.set_rate(x.parse().unwrap(), hz); }
        for k in 0..steps {
            for &i in b.step() {
                if keep[i as usize] {
                    for v in [s as u32, k, i] { out.extend_from_slice(&v.to_le_bytes()); }
                }
            }
        }
    }
    std::fs::File::create(&a[6]).unwrap().write_all(&out).unwrap();
    eprintln!("{:.1} s", time.elapsed().as_secs_f64());
}
