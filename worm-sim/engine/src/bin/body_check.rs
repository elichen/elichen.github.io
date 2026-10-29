//! Replays the muscle activations SoftCon recorded (worm_muscles_300_220428-152601,
//! 30 Hz) through the body and computes Environment::GetStates for each frame:
//! body-frame speed and the 17 sampled points' offsets from rest, to compare with
//! the states recorded in the same run (worm_states_300_220428-152601).
//! usage: body_check <model dir> <ref dir from tools/export_refs.py> [frames]
use std::time::Instant;
use worm_engine::body::Body;
use worm_engine::frame::{body_frame, Frame};

fn f64s(path: &str) -> Vec<f64> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    b.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect()
}

fn main() {
    let dir = std::env::args().nth(1).expect("model dir");
    let refdir = std::env::args().nth(2).expect("ref dir");
    let frames: usize = std::env::args().nth(3).map(|s| s.parse().unwrap()).unwrap_or(300);
    let mut body = Body::new(&f64s(&format!("{dir}/body.bin")));
    println!("vertices {} factor entries {}", body.nv, body.factor_nnz());
    let mus = f64s(&format!("{refdir}/ref_muscles.bin"));
    let refs = f64s(&format!("{refdir}/ref_states.bin")); // 300 x 108
    let mut fr = Frame::new();
    let t0 = Instant::now();
    let mut iters = 0;
    let mut states = vec![];
    for k in 0..frames {
        let st = body_frame(&body, &mut fr);
        states.push(st);
        body.act.copy_from_slice(&mus[(k % 300) * 96..(k % 300) * 96 + 96]);
        for _ in 0..8 {
            body.step();
            iters += body.iterations;
        }
        if k % 30 == 0 {
            let s = &states[k];
            let r = &refs[(k % 300) * 108..];
            println!(
                "frame {:3}  avg v mine [{:+.3} {:+.3} {:+.3}] SoftCon [{:+.3} {:+.3} {:+.3}]  head x {:+.4} {:+.4}  iters {:.1}",
                k, s[3], s[4], s[5], r[3], r[4], r[5], s[6], r[6], iters as f64 / ((k + 1) * 8) as f64
            );
        }
    }
    let el = t0.elapsed().as_secs_f64();
    println!("{} body steps in {:.2} s: {:.0} steps/s ({:.2}x real time), {:.1} iterations/step", frames * 8, el, frames as f64 * 8.0 / el, frames as f64 / 30.0 / el, iters as f64 / (frames * 8) as f64);
    // compare the last 100 frames' sampled offsets, best phase shift
    if frames >= 200 {
        let mut best = (f64::MAX, 0);
        for lag in 0..40 {
            let mut e = 0.0;
            let mut n = 0.0;
            for k in frames - 100..frames {
                let r = &refs[((k + lag) % 300) * 108..];
                for j in 6..57 {
                    e += (states[k][j] - r[j]).powi(2);
                    n += 1.0;
                }
            }
            if e / n < best.0 {
                best = (e / n, lag);
            }
        }
        let mut amp = 0.0;
        for k in 200..300 {
            for j in 6..57 {
                amp += refs[k * 108 + j].powi(2);
            }
        }
        println!("sampled offsets: rms diff {:.4} at lag {} (SoftCon rms offset {:.4})", best.0.sqrt(), best.1, (amp / (100.0 * 51.0)).sqrt());
    }
}
