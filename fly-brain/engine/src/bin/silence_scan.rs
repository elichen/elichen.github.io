//! Silence each of the most active neurons in turn and measure a readout neuron (Shiu et al. Fig. 1f):
//!   cargo run --release --bin silence_scan ../model/connectome.gz <excited index file> <Hz> <readout> <top N> <trials>
use fly_engine::brain::{load, Brain};
fn rates(o: &[u32], t: &[u32], c: &[i16], exc: &[usize], hz: f64, silenced: Option<usize>, trials: u64) -> Vec<f64> {
    let mut sum = vec![0.0; o.len() - 1];
    for k in 0..trials {
        let mut b = Brain::new(o.to_vec(), t.to_vec(), c.to_vec(), 7000 + k);
        for &i in exc { b.set_rate(i, hz); }
        if let Some(s) = silenced { b.set_silenced(s, true); }
        for _ in 0..10000 { b.step(); }
        for (i, &n) in b.spike_count.iter().enumerate() { sum[i] += n as f64 / trials as f64; }
    }
    sum
}
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (o, t, c) = load(&a[1]);
    let exc: Vec<usize> = std::fs::read_to_string(&a[2]).unwrap().split_whitespace().map(|x| x.parse().unwrap()).collect();
    let hz: f64 = a[3].parse().unwrap();
    let readout: usize = a[4].parse().unwrap();
    let top: usize = a[5].parse().unwrap();
    let trials: u64 = a[6].parse().unwrap();
    let base = rates(&o, &t, &c, &exc, hz, None, trials);
    let mut order: Vec<usize> = (0..base.len()).filter(|i| base[*i] > 0.0 && !exc.contains(i)).collect();
    order.sort_by(|x, y| base[*y].partial_cmp(&base[*x]).unwrap());
    println!("baseline readout {:.1} Hz", base[readout]);
    for &s in order.iter().take(top) {
        let r = rates(&o, &t, &c, &exc, hz, Some(s), trials);
        println!("{s}\t{:.1}\t{:.1}", base[s], r[readout]);
    }
}
