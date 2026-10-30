//! flybody's trained policies for the page (see ../../walker.js, whose WalkPolicy is the readable version):
//! inputs, Linear + LayerNorm + tanh, then `layers` x Linear + elu, Linear to the means, clipped to [-1, 1].
//! Walking: 741, 512, 3 layers, 59. Flight: 104, 256, 2 layers, 12. Weights in the order tools/walk_ref.py writes.
#![allow(static_mut_refs, clippy::missing_safety_doc)]
static mut W: Vec<f32> = Vec::new();
static mut OBS: Vec<f32> = Vec::new();
static mut RES: Vec<f32> = Vec::new();
static mut H: Vec<f32> = Vec::new();
static mut N: Vec<f32> = Vec::new();
static mut SHAPE: (usize, usize, usize, usize) = (0, 0, 0, 0);

#[no_mangle]
pub extern "C" fn policy_alloc(bytes: usize) -> *mut f32 {
    unsafe {
        W = vec![0.0; bytes / 4];
        W.as_mut_ptr()
    }
}
#[no_mangle]
pub extern "C" fn policy_init(inputs: usize, hidden: usize, layers: usize, outputs: usize) {
    unsafe {
        SHAPE = (inputs, hidden, layers, outputs);
        OBS = vec![0.0; inputs];
        RES = vec![0.0; outputs];
        H = vec![0.0; hidden];
        N = vec![0.0; hidden];
    }
}
#[no_mangle]
pub extern "C" fn policy_obs() -> *mut f32 {
    unsafe { OBS.as_mut_ptr() }
}
#[no_mangle]
pub extern "C" fn policy_out() -> *mut f32 {
    unsafe { RES.as_mut_ptr() }
}

#[inline(never)]
fn mul(x: &[f32], w: &[f32], b: &[f32], out: &mut [f32]) {
    let n = out.len();
    out.copy_from_slice(b);
    for (i, &xi) in x.iter().enumerate() {
        let row = &w[i * n..(i + 1) * n];
        for (o, &wij) in out.iter_mut().zip(row) {
            *o += xi * wij;
        }
    }
}

#[no_mangle]
pub extern "C" fn policy_run() {
    unsafe {
        let (inp, hid, layers, out) = SHAPE;
        let mut at = 0;
        let mut take = |n: usize| {
            let s = &W[at..at + n];
            at += n;
            s
        };
        let (w0, b0, off, scale) = (take(inp * hid), take(hid), take(hid), take(hid));
        mul(&OBS, w0, b0, &mut H);
        let mean = H.iter().sum::<f32>() / hid as f32;
        let var = H.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / hid as f32;
        let s = (var + 1e-5).sqrt();
        for j in 0..hid {
            H[j] = ((H[j] - mean) / s * scale[j] + off[j]).tanh();
        }
        for _ in 0..layers {
            let (w, b) = (take(hid * hid), take(hid));
            mul(&H, w, b, &mut N);
            for j in 0..hid {
                H[j] = if N[j] > 0.0 { N[j] } else { N[j].exp_m1() };
            }
        }
        let (wm, bm) = (take(hid * out), take(out));
        mul(&H, wm, bm, &mut RES);
        for r in RES.iter_mut() {
            *r = r.clamp(-1.0, 1.0);
        }
    }
}
