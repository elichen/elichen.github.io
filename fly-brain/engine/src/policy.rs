//! flybody's walking policy for the page (see ../../walker.js, whose WalkPolicy is the readable version):
//! inputs 741, Linear + LayerNorm + tanh, then 3 x Linear + elu, Linear to 59 means, clipped to [-1, 1].
#![allow(static_mut_refs, clippy::missing_safety_doc)]
const IN: usize = 741;
const H: usize = 512;
const OUT: usize = 59;

static mut W: Vec<f32> = Vec::new();
static mut OBS: [f32; IN] = [0.0; IN];
static mut RES: [f32; OUT] = [0.0; OUT];

#[no_mangle]
pub extern "C" fn policy_alloc(bytes: usize) -> *mut f32 {
    unsafe {
        W = vec![0.0; bytes / 4];
        W.as_mut_ptr()
    }
}
#[no_mangle]
pub extern "C" fn policy_init(_p: *mut f32) {}
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
        let mut at = 0;
        let mut take = |n: usize| {
            let s = &W[at..at + n];
            at += n;
            s
        };
        let (w0, b0, off, scale) = (take(IN * H), take(H), take(H), take(H));
        let mut h = [0f32; H];
        let mut n = [0f32; H];
        mul(&OBS, w0, b0, &mut h);
        let mean = h.iter().sum::<f32>() / H as f32;
        let var = h.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / H as f32;
        let s = (var + 1e-5).sqrt();
        for j in 0..H {
            h[j] = ((h[j] - mean) / s * scale[j] + off[j]).tanh();
        }
        for _ in 0..3 {
            let (w, b) = (take(H * H), take(H));
            mul(&h, w, b, &mut n);
            for j in 0..H {
                h[j] = if n[j] > 0.0 { n[j] } else { n[j].exp_m1() };
            }
        }
        let (wm, bm) = (take(H * OUT), take(OUT));
        mul(&h, wm, bm, &mut RES);
        for r in RES.iter_mut() {
            *r = r.clamp(-1.0, 1.0);
        }
    }
}
