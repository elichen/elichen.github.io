// Exact 1-layer ReLU Transformer (Nanda et al. 2023) for modular arithmetic
// (a + b) mod p and (a - b) mod p — trained full-batch with AdamW
// in WebAssembly SIMD128.
//
// Architecture (fixed dimensions for fast SIMD codegen):
//   P = 59 (prime modulus, 3481 pairs total)
//   D = 64 (d_model), H = 4 (n_heads), DH = 16 (d_head), DM = 192 (d_mlp)
//   No LayerNorm; ReLU MLP; causal 1-layer attention read at position 2 ('=').
//   Attention is algebraically factored over the P distinct token embeddings
//   (100% exact, identical math & gradients to sequence-level attention).

#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;
use core::f32::consts::PI;

pub const P: usize = 59;
pub const V: usize = P + 1; // 0..58 residues + 59 ('=')
pub const P2: usize = P * P; // 3481
pub const D: usize = 64;
pub const H: usize = 4;
pub const DH: usize = D / H; // 16
pub const DM: usize = 192;
pub const NFREQ: usize = P / 2; // 29 (k = 1..29)

// Parameter offsets in flat parameter vector
const OFF_WE: usize = 0;                       // [V, D]   = 3840
const OFF_WPOS: usize = OFF_WE + V * D;        // [3, D]   = 192
const OFF_WQ: usize = OFF_WPOS + 3 * D;        // [D, D]   = 4096
const OFF_WK: usize = OFF_WQ + D * D;          // [D, D]   = 4096
const OFF_WV: usize = OFF_WK + D * D;          // [D, D]   = 4096
const OFF_WO: usize = OFF_WV + D * D;          // [D, D]   = 4096
const OFF_WIN: usize = OFF_WO + D * D;         // [D, DM]  = 12288
const OFF_BIN: usize = OFF_WIN + D * DM;       // [DM]     = 192
const OFF_WOUT: usize = OFF_BIN + DM;          // [DM, D]  = 12288
const OFF_BOUT: usize = OFF_WOUT + DM * D;     // [D]      = 64
const OFF_WU: usize = OFF_BOUT + D;            // [D, P]   = 3776
pub const N_PARAMS: usize = OFF_WU + D * P;    // 49024

// Evaluation output buffer layout (f32)
// [0]: train_loss
// [1]: train_acc
// [2]: test_loss
// [3]: test_acc
// [4]: weight_norm
// [5]: restricted_test_loss
// [6]: excluded_train_loss
// [7]: top5_energy_share
// [8]: fourier_gini
// [9..14]: top5_freqs (as f32)
// [14..43]: power[1..=29] (normalized fraction of non-DC embedding energy)
pub const EVAL_LEN: usize = 14 + NFREQ; // 43 floats

// Probe output buffer layout (f32):
// [0..P]: logits (59)
// [P..2*P]: probs (59)
// [2*P..2*P + H*3]: attn_weights (12)
// [2*P + 12 .. 2*P + 12 + 5]: top5_k (5)
// [2*P + 17 .. 2*P + 17 + 5*P]: per-frequency wave over output c=0..58 for top 5 freqs (295)
// [2*P + 17 + 5*P .. 2*P + 17 + 6*P]: residual wave (all other freqs) (59)
pub const PROBE_LEN: usize = 2 * P + H * 3 + 5 + 6 * P; // 489 floats

// Deterministic xoshiro256** PRNG + Box-Muller normal
struct Rng {
    s: [u64; 4],
}

impl Rng {
    fn new(seed: u64) -> Self {
        let mut x = seed.wrapping_add(0x9E3779B97F4A7C15);
        let mut s = [0u64; 4];
        for i in 0..4 {
            x = x.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            s[i] = z ^ (z >> 31);
        }
        Self { s }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let res = (self.s[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        res
    }

    #[inline]
    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) * (1.0 / 16777216.0)
    }

    fn next_normal(&mut self) -> f32 {
        let mut u1 = self.next_f32();
        if u1 < 1e-7 {
            u1 = 1e-7;
        }
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

// ---------- SIMD128 GEMM primitives ----------
// C[m, n] = A[m, k] @ B[k, n] (row-major; B is k x n)
// For best cache/SIMD efficiency, we use a 4-row x 4-col (or 4-row x 8-col) microkernel
// when N is a multiple of 4 (true for D=64, DM=192!), plus a clean tail for N=59.

#[cfg(target_arch = "wasm32")]
#[inline(always)]
unsafe fn gemm_nn_add(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    zero_c: bool,
) {
    if zero_c {
        for v in c.iter_mut().take(m * n) {
            *v = 0.0;
        }
    }
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    let cp = c.as_mut_ptr();

    let n8 = n & !7;
    let n4 = n & !3;

    let mut i = 0;
    while i + 4 <= m {
        let mut j = 0;
        while j < n8 {
            let mut c00 = v128_load(cp.add((i) * n + j) as *const v128);
            let mut c01 = v128_load(cp.add((i) * n + j + 4) as *const v128);
            let mut c10 = v128_load(cp.add((i + 1) * n + j) as *const v128);
            let mut c11 = v128_load(cp.add((i + 1) * n + j + 4) as *const v128);
            let mut c20 = v128_load(cp.add((i + 2) * n + j) as *const v128);
            let mut c21 = v128_load(cp.add((i + 2) * n + j + 4) as *const v128);
            let mut c30 = v128_load(cp.add((i + 3) * n + j) as *const v128);
            let mut c31 = v128_load(cp.add((i + 3) * n + j + 4) as *const v128);

            let mut p = 0;
            while p < k {
                let b0 = v128_load(bp.add(p * n + j) as *const v128);
                let b1 = v128_load(bp.add(p * n + j + 4) as *const v128);

                let a0 = f32x4_splat(*ap.add((i) * k + p));
                c00 = f32x4_add(c00, f32x4_mul(a0, b0));
                c01 = f32x4_add(c01, f32x4_mul(a0, b1));

                let a1 = f32x4_splat(*ap.add((i + 1) * k + p));
                c10 = f32x4_add(c10, f32x4_mul(a1, b0));
                c11 = f32x4_add(c11, f32x4_mul(a1, b1));

                let a2 = f32x4_splat(*ap.add((i + 2) * k + p));
                c20 = f32x4_add(c20, f32x4_mul(a2, b0));
                c21 = f32x4_add(c21, f32x4_mul(a2, b1));

                let a3 = f32x4_splat(*ap.add((i + 3) * k + p));
                c30 = f32x4_add(c30, f32x4_mul(a3, b0));
                c31 = f32x4_add(c31, f32x4_mul(a3, b1));
                p += 1;
            }

            v128_store(cp.add((i) * n + j) as *mut v128, c00);
            v128_store(cp.add((i) * n + j + 4) as *mut v128, c01);
            v128_store(cp.add((i + 1) * n + j) as *mut v128, c10);
            v128_store(cp.add((i + 1) * n + j + 4) as *mut v128, c11);
            v128_store(cp.add((i + 2) * n + j) as *mut v128, c20);
            v128_store(cp.add((i + 2) * n + j + 4) as *mut v128, c21);
            v128_store(cp.add((i + 3) * n + j) as *mut v128, c30);
            v128_store(cp.add((i + 3) * n + j + 4) as *mut v128, c31);
            j += 8;
        }
        while j < n4 {
            let mut c0 = v128_load(cp.add((i) * n + j) as *const v128);
            let mut c1 = v128_load(cp.add((i + 1) * n + j) as *const v128);
            let mut c2 = v128_load(cp.add((i + 2) * n + j) as *const v128);
            let mut c3 = v128_load(cp.add((i + 3) * n + j) as *const v128);
            for p in 0..k {
                let b0 = v128_load(bp.add(p * n + j) as *const v128);
                c0 = f32x4_add(c0, f32x4_mul(f32x4_splat(*ap.add((i) * k + p)), b0));
                c1 = f32x4_add(c1, f32x4_mul(f32x4_splat(*ap.add((i + 1) * k + p)), b0));
                c2 = f32x4_add(c2, f32x4_mul(f32x4_splat(*ap.add((i + 2) * k + p)), b0));
                c3 = f32x4_add(c3, f32x4_mul(f32x4_splat(*ap.add((i + 3) * k + p)), b0));
            }
            v128_store(cp.add((i) * n + j) as *mut v128, c0);
            v128_store(cp.add((i + 1) * n + j) as *mut v128, c1);
            v128_store(cp.add((i + 2) * n + j) as *mut v128, c2);
            v128_store(cp.add((i + 3) * n + j) as *mut v128, c3);
            j += 4;
        }
        while j < n {
            let mut s0 = *cp.add((i) * n + j);
            let mut s1 = *cp.add((i + 1) * n + j);
            let mut s2 = *cp.add((i + 2) * n + j);
            let mut s3 = *cp.add((i + 3) * n + j);
            for p in 0..k {
                let bv = *bp.add(p * n + j);
                s0 += *ap.add((i) * k + p) * bv;
                s1 += *ap.add((i + 1) * k + p) * bv;
                s2 += *ap.add((i + 2) * k + p) * bv;
                s3 += *ap.add((i + 3) * k + p) * bv;
            }
            *cp.add((i) * n + j) = s0;
            *cp.add((i + 1) * n + j) = s1;
            *cp.add((i + 2) * n + j) = s2;
            *cp.add((i + 3) * n + j) = s3;
            j += 1;
        }
        i += 4;
    }
    while i < m {
        for p in 0..k {
            let av = *ap.add(i * k + p);
            let av4 = f32x4_splat(av);
            let mut j = 0;
            while j < n4 {
                let cv = v128_load(cp.add(i * n + j) as *const v128);
                let bv = v128_load(bp.add(p * n + j) as *const v128);
                v128_store(cp.add(i * n + j) as *mut v128, f32x4_add(cv, f32x4_mul(av4, bv)));
                j += 4;
            }
            while j < n {
                *cp.add(i * n + j) += av * *bp.add(p * n + j);
                j += 1;
            }
        }
        i += 1;
    }
}

#[cfg(not(target_arch = "wasm32"))]
unsafe fn gemm_nn_add(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    zero_c: bool,
) {
    if zero_c {
        for v in c.iter_mut().take(m * n) {
            *v = 0.0;
        }
    }
    for i in 0..m {
        for p in 0..k {
            let av = a[i * k + p];
            for j in 0..n {
                c[i * n + j] += av * b[p * n + j];
            }
        }
    }
}

// C[m, n] = A[k, m]^T @ B[k, n]
#[cfg(target_arch = "wasm32")]
#[inline(always)]
unsafe fn gemm_tn(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32], // [k, m]
    b: &[f32], // [k, n]
    c: &mut [f32], // [m, n]
    zero_c: bool,
) {
    if zero_c {
        for v in c.iter_mut().take(m * n) {
            *v = 0.0;
        }
    }
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    let cp = c.as_mut_ptr();
    let n8 = n & !7;
    let n4 = n & !3;

    // Tile over k in blocks of 64 to keep C tiles in registers and B rows hot in L1
    let mut i = 0;
    while i + 4 <= m {
        let mut j = 0;
        while j < n8 {
            let mut c00 = v128_load(cp.add((i) * n + j) as *const v128);
            let mut c01 = v128_load(cp.add((i) * n + j + 4) as *const v128);
            let mut c10 = v128_load(cp.add((i + 1) * n + j) as *const v128);
            let mut c11 = v128_load(cp.add((i + 1) * n + j + 4) as *const v128);
            let mut c20 = v128_load(cp.add((i + 2) * n + j) as *const v128);
            let mut c21 = v128_load(cp.add((i + 2) * n + j + 4) as *const v128);
            let mut c30 = v128_load(cp.add((i + 3) * n + j) as *const v128);
            let mut c31 = v128_load(cp.add((i + 3) * n + j + 4) as *const v128);

            let mut p = 0;
            while p < k {
                let b0 = v128_load(bp.add(p * n + j) as *const v128);
                let b1 = v128_load(bp.add(p * n + j + 4) as *const v128);
                let arow = ap.add(p * m + i);
                let a0 = f32x4_splat(*arow);
                let a1 = f32x4_splat(*arow.add(1));
                let a2 = f32x4_splat(*arow.add(2));
                let a3 = f32x4_splat(*arow.add(3));

                c00 = f32x4_add(c00, f32x4_mul(a0, b0));
                c01 = f32x4_add(c01, f32x4_mul(a0, b1));
                c10 = f32x4_add(c10, f32x4_mul(a1, b0));
                c11 = f32x4_add(c11, f32x4_mul(a1, b1));
                c20 = f32x4_add(c20, f32x4_mul(a2, b0));
                c21 = f32x4_add(c21, f32x4_mul(a2, b1));
                c30 = f32x4_add(c30, f32x4_mul(a3, b0));
                c31 = f32x4_add(c31, f32x4_mul(a3, b1));
                p += 1;
            }
            v128_store(cp.add((i) * n + j) as *mut v128, c00);
            v128_store(cp.add((i) * n + j + 4) as *mut v128, c01);
            v128_store(cp.add((i + 1) * n + j) as *mut v128, c10);
            v128_store(cp.add((i + 1) * n + j + 4) as *mut v128, c11);
            v128_store(cp.add((i + 2) * n + j) as *mut v128, c20);
            v128_store(cp.add((i + 2) * n + j + 4) as *mut v128, c21);
            v128_store(cp.add((i + 3) * n + j) as *mut v128, c30);
            v128_store(cp.add((i + 3) * n + j + 4) as *mut v128, c31);
            j += 8;
        }
        while j < n4 {
            let mut c0 = v128_load(cp.add((i) * n + j) as *const v128);
            let mut c1 = v128_load(cp.add((i + 1) * n + j) as *const v128);
            let mut c2 = v128_load(cp.add((i + 2) * n + j) as *const v128);
            let mut c3 = v128_load(cp.add((i + 3) * n + j) as *const v128);
            for p in 0..k {
                let b0 = v128_load(bp.add(p * n + j) as *const v128);
                let arow = ap.add(p * m + i);
                c0 = f32x4_add(c0, f32x4_mul(f32x4_splat(*arow), b0));
                c1 = f32x4_add(c1, f32x4_mul(f32x4_splat(*arow.add(1)), b0));
                c2 = f32x4_add(c2, f32x4_mul(f32x4_splat(*arow.add(2)), b0));
                c3 = f32x4_add(c3, f32x4_mul(f32x4_splat(*arow.add(3)), b0));
            }
            v128_store(cp.add((i) * n + j) as *mut v128, c0);
            v128_store(cp.add((i + 1) * n + j) as *mut v128, c1);
            v128_store(cp.add((i + 2) * n + j) as *mut v128, c2);
            v128_store(cp.add((i + 3) * n + j) as *mut v128, c3);
            j += 4;
        }
        while j < n {
            let mut s0 = *cp.add((i) * n + j);
            let mut s1 = *cp.add((i + 1) * n + j);
            let mut s2 = *cp.add((i + 2) * n + j);
            let mut s3 = *cp.add((i + 3) * n + j);
            for p in 0..k {
                let bv = *bp.add(p * n + j);
                let arow = ap.add(p * m + i);
                s0 += *arow * bv;
                s1 += *arow.add(1) * bv;
                s2 += *arow.add(2) * bv;
                s3 += *arow.add(3) * bv;
            }
            *cp.add((i) * n + j) = s0;
            *cp.add((i + 1) * n + j) = s1;
            *cp.add((i + 2) * n + j) = s2;
            *cp.add((i + 3) * n + j) = s3;
            j += 1;
        }
        i += 4;
    }
    while i < m {
        for p in 0..k {
            let av = *ap.add(p * m + i);
            let av4 = f32x4_splat(av);
            let mut j = 0;
            while j < n4 {
                let cv = v128_load(cp.add(i * n + j) as *const v128);
                let bv = v128_load(bp.add(p * n + j) as *const v128);
                v128_store(cp.add(i * n + j) as *mut v128, f32x4_add(cv, f32x4_mul(av4, bv)));
                j += 4;
            }
            while j < n {
                *cp.add(i * n + j) += av * *bp.add(p * n + j);
                j += 1;
            }
        }
        i += 1;
    }
}

#[cfg(not(target_arch = "wasm32"))]
unsafe fn gemm_tn(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    zero_c: bool,
) {
    if zero_c {
        for v in c.iter_mut().take(m * n) {
            *v = 0.0;
        }
    }
    for p in 0..k {
        for i in 0..m {
            let av = a[p * m + i];
            for j in 0..n {
                c[i * n + j] += av * b[p * n + j];
            }
        }
    }
}

// Helper: transpose [r, c] -> [c, r]
fn transpose(r: usize, c: usize, src: &[f32], dst: &mut [f32]) {
    for i in 0..r {
        for j in 0..c {
            dst[j * r + i] = src[i * c + j];
        }
    }
}

#[inline(always)]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "wasm32")]
    unsafe {
        let n = a.len();
        let n4 = n & !3;
        let mut acc = f32x4_splat(0.0);
        let mut i = 0;
        while i < n4 {
            let av = v128_load(a.as_ptr().add(i) as *const v128);
            let bv = v128_load(b.as_ptr().add(i) as *const v128);
            acc = f32x4_add(acc, f32x4_mul(av, bv));
            i += 4;
        }
        let mut s = f32x4_extract_lane::<0>(acc)
            + f32x4_extract_lane::<1>(acc)
            + f32x4_extract_lane::<2>(acc)
            + f32x4_extract_lane::<3>(acc);
        while i < n {
            s += *a.get_unchecked(i) * *b.get_unchecked(i);
            i += 1;
        }
        s
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let mut s = 0.0f32;
        for i in 0..a.len() {
            s += a[i] * b[i];
        }
        s
    }
}

#[inline(always)]
fn axpy(y: &mut [f32], alpha: f32, x: &[f32]) {
    #[cfg(target_arch = "wasm32")]
    unsafe {
        let n = y.len();
        let n4 = n & !3;
        let a4 = f32x4_splat(alpha);
        let mut i = 0;
        while i < n4 {
            let yv = v128_load(y.as_ptr().add(i) as *const v128);
            let xv = v128_load(x.as_ptr().add(i) as *const v128);
            v128_store(y.as_mut_ptr().add(i) as *mut v128, f32x4_add(yv, f32x4_mul(a4, xv)));
            i += 4;
        }
        while i < n {
            *y.get_unchecked_mut(i) += alpha * *x.get_unchecked(i);
            i += 1;
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        for i in 0..y.len() {
            y[i] += alpha * x[i];
        }
    }
}

// ---------- Trainer State ----------
struct Trainer {
    // Parameters & AdamW state
    params: Vec<f32>,
    grads: Vec<f32>,
    m: Vec<f32>,
    v: Vec<f32>,
    step: u32,

    // Hyperparameters
    lr: f32,
    wd: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,

    // Dataset (all P2 = 3481 pairs, first n_train are train, rest are test)
    op: u32, // 0 = (a+b)%p, 1 = (a-b)%p
    n_train: usize,
    all_a: Vec<u16>,     // [P2] in order a*P + b
    all_b: Vec<u16>,     // [P2]
    all_y: Vec<u16>,     // [P2]
    is_train: Vec<u8>,   // [P2] 1 if train, 0 if test
    tr_a: Vec<u16>,      // [n_train]
    tr_b: Vec<u16>,      // [n_train]
    tr_y: Vec<u16>,      // [n_train]

    // Precomputed orthonormal real DFT basis F[row, t] for t=0..P
    // Row 0: 1/sqrt(P)
    // For k=1..=NFREQ:
    //   row 2k-1: sqrt(2/P) * cos(2*pi*k*t/P)
    //   row 2k:   sqrt(2/P) * sin(2*pi*k*t/P)
    dft_cos: Vec<f32>, // [(NFREQ + 1) * P]
    dft_sin: Vec<f32>, // [(NFREQ + 1) * P]

    // Scratch buffers (sized for P2 = 3481 so both train & full eval reuse them!)
    x0: Vec<f32>,      // [P, D]
    x1: Vec<f32>,      // [P, D]
    x2: Vec<f32>,      // [D]
    q2: Vec<f32>,      // [D]
    k0: Vec<f32>,      // [P, D]
    k1: Vec<f32>,      // [P, D]
    k2: Vec<f32>,      // [D]
    v0: Vec<f32>,      // [P, D]
    v1: Vec<f32>,      // [P, D]
    v2: Vec<f32>,      // [D]
    s0: Vec<f32>,      // [P, H]
    s1: Vec<f32>,      // [P, H]
    s2: Vec<f32>,      // [H]
    vo0: Vec<f32>,     // [P, H, D]
    vo1: Vec<f32>,     // [P, H, D]
    vo2: Vec<f32>,     // [H, D]
    attn_w: Vec<f32>,  // [P2, H, 3]
    x_mid: Vec<f32>,   // [P2, D]
    pre_h: Vec<f32>,   // [P2, DM]
    h: Vec<f32>,       // [P2, DM]
    out: Vec<f32>,     // [P2, D]
    logits: Vec<f32>,  // [P2, P]

    // Backward scratch
    d_logits: Vec<f32>, // [P2, P]
    d_out: Vec<f32>,    // [P2, D]
    d_h: Vec<f32>,      // [P2, DM]
    d_x_mid: Vec<f32>,  // [P2, D]
    d_vo0: Vec<f32>,    // [P, H, D]
    d_vo1: Vec<f32>,    // [P, H, D]
    d_vo2: Vec<f32>,    // [H, D]
    d_s0: Vec<f32>,     // [P, H]
    d_s1: Vec<f32>,     // [P, H]
    d_s2: Vec<f32>,     // [H]
    d_q2: Vec<f32>,     // [D]
    d_k0: Vec<f32>,     // [P, D]
    d_k1: Vec<f32>,     // [P, D]
    d_k2: Vec<f32>,     // [D]
    d_v0: Vec<f32>,     // [P, D]
    d_v1: Vec<f32>,     // [P, D]
    d_v2: Vec<f32>,     // [D]
    d_x0: Vec<f32>,     // [P, D]
    d_x1: Vec<f32>,     // [P, D]
    d_x2: Vec<f32>,     // [D]
    wt_tmp: Vec<f32>,   // [DM * D] max weight transpose scratch

    // Ablated W_E scratch
    we_abl: Vec<f32>,   // [V * D]

    // Exported output buffers
    eval_out: Vec<f32>,    // [EVAL_LEN]
    ablate_out: Vec<f32>,  // [4]: tr_acc, te_acc, tr_loss, te_loss
    circle_out: Vec<f32>,  // [P * 2]: (x, y) for each token 0..58
    probe_out: Vec<f32>,   // [PROBE_LEN]
    last_top5: [usize; 5],
}

impl Trainer {
    fn new() -> Self {
        let mut dft_cos = vec![0.0f32; (NFREQ + 1) * P];
        let mut dft_sin = vec![0.0f32; (NFREQ + 1) * P];
        let s0 = (1.0 / P as f32).sqrt();
        let sk = (2.0 / P as f32).sqrt();
        for t in 0..P {
            dft_cos[t] = s0;
            dft_sin[t] = 0.0;
        }
        for k in 1..=NFREQ {
            for t in 0..P {
                let ang = 2.0 * PI * (((k * t) % P) as f32) / (P as f32);
                dft_cos[k * P + t] = sk * ang.cos();
                dft_sin[k * P + t] = sk * ang.sin();
            }
        }

        let mut tr = Self {
            params: vec![0.0; N_PARAMS],
            grads: vec![0.0; N_PARAMS],
            m: vec![0.0; N_PARAMS],
            v: vec![0.0; N_PARAMS],
            step: 0,
            lr: 3e-3,
            wd: 2.0,
            beta1: 0.9,
            beta2: 0.98,
            eps: 1e-8,
            op: 0,
            n_train: 0,
            all_a: vec![0; P2],
            all_b: vec![0; P2],
            all_y: vec![0; P2],
            is_train: vec![0; P2],
            tr_a: Vec::with_capacity(P2),
            tr_b: Vec::with_capacity(P2),
            tr_y: Vec::with_capacity(P2),
            dft_cos,
            dft_sin,
            x0: vec![0.0; P * D],
            x1: vec![0.0; P * D],
            x2: vec![0.0; D],
            q2: vec![0.0; D],
            k0: vec![0.0; P * D],
            k1: vec![0.0; P * D],
            k2: vec![0.0; D],
            v0: vec![0.0; P * D],
            v1: vec![0.0; P * D],
            v2: vec![0.0; D],
            s0: vec![0.0; P * H],
            s1: vec![0.0; P * H],
            s2: vec![0.0; H],
            vo0: vec![0.0; P * H * D],
            vo1: vec![0.0; P * H * D],
            vo2: vec![0.0; H * D],
            attn_w: vec![0.0; P2 * H * 3],
            x_mid: vec![0.0; P2 * D],
            pre_h: vec![0.0; P2 * DM],
            h: vec![0.0; P2 * DM],
            out: vec![0.0; P2 * D],
            logits: vec![0.0; P2 * P],
            d_logits: vec![0.0; P2 * P],
            d_out: vec![0.0; P2 * D],
            d_h: vec![0.0; P2 * DM],
            d_x_mid: vec![0.0; P2 * D],
            d_vo0: vec![0.0; P * H * D],
            d_vo1: vec![0.0; P * H * D],
            d_vo2: vec![0.0; H * D],
            d_s0: vec![0.0; P * H],
            d_s1: vec![0.0; P * H],
            d_s2: vec![0.0; H],
            d_q2: vec![0.0; D],
            d_k0: vec![0.0; P * D],
            d_k1: vec![0.0; P * D],
            d_k2: vec![0.0; D],
            d_v0: vec![0.0; P * D],
            d_v1: vec![0.0; P * D],
            d_v2: vec![0.0; D],
            d_x0: vec![0.0; P * D],
            d_x1: vec![0.0; P * D],
            d_x2: vec![0.0; D],
            wt_tmp: vec![0.0; DM * D],
            we_abl: vec![0.0; V * D],
            eval_out: vec![0.0; EVAL_LEN],
            ablate_out: vec![0.0; 4],
            circle_out: vec![0.0; P * 2],
            probe_out: vec![0.0; PROBE_LEN],
            last_top5: [1, 2, 3, 4, 5],
        };
        tr.init(0, 0.35, 3e-3, 2.0, 42);
        tr
    }

    fn init(&mut self, op: u32, train_frac: f32, lr: f32, wd: f32, seed: u32) {
        self.op = op;
        self.lr = lr;
        self.wd = wd;
        self.step = 0;

        // Build all pairs in canonical order idx = a*P + b
        for a in 0..P {
            for b in 0..P {
                let idx = a * P + b;
                self.all_a[idx] = a as u16;
                self.all_b[idx] = b as u16;
                let c = if op == 0 {
                    (a + b) % P
                } else {
                    (a + P - b) % P
                };
                self.all_y[idx] = c as u16;
            }
        }

        // Fisher-Yates shuffle for train/test split
        let mut split_rng = Rng::new((seed as u64).wrapping_mul(1000003).wrapping_add(17));
        let mut perm: Vec<usize> = (0..P2).collect();
        for i in (1..P2).rev() {
            let j = (split_rng.next_u64() as usize) % (i + 1);
            perm.swap(i, j);
        }
        let frac = train_frac.clamp(0.1, 0.9);
        self.n_train = ((P2 as f32) * frac) as usize;
        for v in self.is_train.iter_mut() {
            *v = 0;
        }
        self.tr_a.clear();
        self.tr_b.clear();
        self.tr_y.clear();
        for &idx in perm.iter().take(self.n_train) {
            self.is_train[idx] = 1;
            self.tr_a.push(self.all_a[idx]);
            self.tr_b.push(self.all_b[idx]);
            self.tr_y.push(self.all_y[idx]);
        }

        // Initialize weights: N(0, 1/sqrt(fan_in)) with zero MLP biases
        let mut w_rng = Rng::new((seed as u64).wrapping_mul(2654435761).wrapping_add(97));
        let sd_d = 1.0 / (D as f32).sqrt();
        let sd_dm = 1.0 / (DM as f32).sqrt();

        for i in 0..N_PARAMS {
            self.m[i] = 0.0;
            self.v[i] = 0.0;
            self.grads[i] = 0.0;
        }
        for i in OFF_WE..OFF_BIN {
            self.params[i] = w_rng.next_normal() * sd_d;
        }
        for i in OFF_BIN..OFF_WOUT {
            self.params[i] = 0.0;
        }
        for i in OFF_WOUT..OFF_BOUT {
            self.params[i] = w_rng.next_normal() * sd_dm;
        }
        for i in OFF_BOUT..OFF_WU {
            self.params[i] = 0.0;
        }
        for i in OFF_WU..N_PARAMS {
            self.params[i] = w_rng.next_normal() * sd_d;
        }
    }

    // Forward pass on batch `pairs_a[0..bs]`, `pairs_b[0..bs]` using `we_slice` ([V*D])
    fn forward(&mut self, bs: usize, use_all: bool, use_ablated_we: bool) {
        let we_ptr = if use_ablated_we {
            self.we_abl.as_ptr()
        } else {
            unsafe { self.params.as_ptr().add(OFF_WE) }
        };
        let we = unsafe { core::slice::from_raw_parts(we_ptr, V * D) };
        let wpos = &self.params[OFF_WPOS..OFF_WPOS + 3 * D];
        let wq = &self.params[OFF_WQ..OFF_WQ + D * D];
        let wk = &self.params[OFF_WK..OFF_WK + D * D];
        let wv = &self.params[OFF_WV..OFF_WV + D * D];
        let wo = &self.params[OFF_WO..OFF_WO + D * D];
        let win = &self.params[OFF_WIN..OFF_WIN + D * DM];
        let bin = &self.params[OFF_BIN..OFF_BIN + DM];
        let wout = &self.params[OFF_WOUT..OFF_WOUT + DM * D];
        let bout = &self.params[OFF_BOUT..OFF_BOUT + D];
        let wu = &self.params[OFF_WU..OFF_WU + D * P];

        // 1. x0, x1, x2
        for t in 0..P {
            let we_t = &we[t * D..(t + 1) * D];
            let x0_t = &mut self.x0[t * D..(t + 1) * D];
            let x1_t = &mut self.x1[t * D..(t + 1) * D];
            for j in 0..D {
                x0_t[j] = we_t[j] + wpos[j];
                x1_t[j] = we_t[j] + wpos[D + j];
            }
        }
        for j in 0..D {
            self.x2[j] = we[P * D + j] + wpos[2 * D + j];
        }

        // 2. Q, K, V over P tokens + '=' token
        unsafe {
            gemm_nn_add(1, D, D, &self.x2, wq, &mut self.q2, true);
            gemm_nn_add(P, D, D, &self.x0, wk, &mut self.k0, true);
            gemm_nn_add(P, D, D, &self.x1, wk, &mut self.k1, true);
            gemm_nn_add(1, D, D, &self.x2, wk, &mut self.k2, true);
            gemm_nn_add(P, D, D, &self.x0, wv, &mut self.v0, true);
            gemm_nn_add(P, D, D, &self.x1, wv, &mut self.v1, true);
            gemm_nn_add(1, D, D, &self.x2, wv, &mut self.v2, true);
        }

        // 3. Attention scores s0[P, H], s1[P, H], s2[H]
        let scale = 1.0 / (DH as f32).sqrt();
        for h_idx in 0..H {
            let off = h_idx * DH;
            let qh = &self.q2[off..off + DH];
            self.s2[h_idx] = dot(qh, &self.k2[off..off + DH]) * scale;
            for t in 0..P {
                self.s0[t * H + h_idx] = dot(qh, &self.k0[t * D + off..t * D + off + DH]) * scale;
                self.s1[t * H + h_idx] = dot(qh, &self.k1[t * D + off..t * D + off + DH]) * scale;
            }
        }

        // 4. Pre-multiply v * W_O per head: vo0[P, H, D], vo1[P, H, D], vo2[H, D]
        for h_idx in 0..H {
            let wo_h = &wo[h_idx * DH * D..(h_idx + 1) * DH * D]; // [DH, D]
            // v2[h] @ wo_h -> vo2[h]
            let v2_h = &self.v2[h_idx * DH..(h_idx + 1) * DH];
            let vo2_h = &mut self.vo2[h_idx * D..(h_idx + 1) * D];
            unsafe {
                gemm_nn_add(1, D, DH, v2_h, wo_h, vo2_h, true);
            }
            for t in 0..P {
                let v0_th = &self.v0[t * D + h_idx * DH..t * D + (h_idx + 1) * DH];
                let vo0_th = &mut self.vo0[(t * H + h_idx) * D..(t * H + h_idx + 1) * D];
                let v1_th = &self.v1[t * D + h_idx * DH..t * D + (h_idx + 1) * DH];
                let vo1_th = &mut self.vo1[(t * H + h_idx) * D..(t * H + h_idx + 1) * D];
                unsafe {
                    gemm_nn_add(1, D, DH, v0_th, wo_h, vo0_th, true);
                    gemm_nn_add(1, D, DH, v1_th, wo_h, vo1_th, true);
                }
            }
        }

        // 5. Combine attention per example into x_mid[bs, D]
        let (pa, pb): (&[u16], &[u16]) = if use_all {
            (&self.all_a, &self.all_b)
        } else {
            (&self.tr_a, &self.tr_b)
        };
        for i in 0..bs {
            let ai = pa[i] as usize;
            let bi = pb[i] as usize;
            let xm = &mut self.x_mid[i * D..(i + 1) * D];
            xm.copy_from_slice(&self.x2);
            for h_idx in 0..H {
                let a0 = self.s0[ai * H + h_idx];
                let a1 = self.s1[bi * H + h_idx];
                let a2 = self.s2[h_idx];
                let mx = a0.max(a1).max(a2);
                let e0 = (a0 - mx).exp();
                let e1 = (a1 - mx).exp();
                let e2 = (a2 - mx).exp();
                let inv = 1.0 / (e0 + e1 + e2);
                let w0 = e0 * inv;
                let w1 = e1 * inv;
                let w2 = e2 * inv;
                let wi = (i * H + h_idx) * 3;
                self.attn_w[wi] = w0;
                self.attn_w[wi + 1] = w1;
                self.attn_w[wi + 2] = w2;

                axpy(xm, w0, &self.vo0[(ai * H + h_idx) * D..(ai * H + h_idx + 1) * D]);
                axpy(xm, w1, &self.vo1[(bi * H + h_idx) * D..(bi * H + h_idx + 1) * D]);
                axpy(xm, w2, &self.vo2[h_idx * D..(h_idx + 1) * D]);
            }
        }

        // 6. MLP: pre_h = x_mid @ W_in + b_in, h = relu(pre_h)
        unsafe {
            gemm_nn_add(bs, DM, D, &self.x_mid[..bs * D], win, &mut self.pre_h[..bs * DM], true);
        }
        for i in 0..bs {
            let row_pre = &mut self.pre_h[i * DM..(i + 1) * DM];
            let row_h = &mut self.h[i * DM..(i + 1) * DM];
            for j in 0..DM {
                let v = row_pre[j] + bin[j];
                row_pre[j] = v;
                row_h[j] = if v > 0.0 { v } else { 0.0 };
            }
        }

        // 7. out = x_mid + h @ W_out + b_out
        unsafe {
            gemm_nn_add(bs, D, DM, &self.h[..bs * DM], wout, &mut self.out[..bs * D], true);
        }
        for i in 0..bs {
            let row_out = &mut self.out[i * D..(i + 1) * D];
            let row_xm = &self.x_mid[i * D..(i + 1) * D];
            for j in 0..D {
                row_out[j] += row_xm[j] + bout[j];
            }
        }

        // 8. logits = out @ W_U
        unsafe {
            gemm_nn_add(bs, P, D, &self.out[..bs * D], wu, &mut self.logits[..bs * P], true);
        }
    }

    // One full-batch training step (forward + backward + AdamW)
    fn train_step(&mut self) -> (f32, f32) {
        let bs = self.n_train;
        self.forward(bs, false, false);

        // Cross-entropy loss & d_logits
        let inv_bs = 1.0 / (bs as f32);
        let mut loss_sum = 0.0f32;
        let mut correct = 0usize;
        for i in 0..bs {
            let yi = self.tr_y[i] as usize;
            let lg = &self.logits[i * P..(i + 1) * P];
            let dl = &mut self.d_logits[i * P..(i + 1) * P];
            let mut mx = lg[0];
            let mut argmax = 0usize;
            for c in 1..P {
                if lg[c] > mx {
                    mx = lg[c];
                    argmax = c;
                }
            }
            if argmax == yi {
                correct += 1;
            }
            let mut sum_exp = 0.0f32;
            for c in 0..P {
                let e = (lg[c] - mx).exp();
                dl[c] = e;
                sum_exp += e;
            }
            let inv_sum = 1.0 / sum_exp;
            loss_sum += -(lg[yi] - mx) + sum_exp.ln();
            for c in 0..P {
                dl[c] = dl[c] * inv_sum * inv_bs;
            }
            dl[yi] -= inv_bs;
        }

        // Zero gradients
        for g in self.grads.iter_mut() {
            *g = 0.0;
        }

        // 1. Unembed backward: d_W_U = out^T @ d_logits, d_out = d_logits @ W_U^T
        unsafe {
            gemm_tn(
                D,
                P,
                bs,
                &self.out[..bs * D],
                &self.d_logits[..bs * P],
                &mut self.grads[OFF_WU..OFF_WU + D * P],
                true,
            );
            transpose(D, P, &self.params[OFF_WU..OFF_WU + D * P], &mut self.wt_tmp[..P * D]);
            gemm_nn_add(
                bs,
                D,
                P,
                &self.d_logits[..bs * P],
                &self.wt_tmp[..P * D],
                &mut self.d_out[..bs * D],
                true,
            );
        }

        // 2. MLP backward
        // d_b_out = sum_i d_out[i]
        {
            let d_bout = &mut self.grads[OFF_BOUT..OFF_BOUT + D];
            for i in 0..bs {
                let row = &self.d_out[i * D..(i + 1) * D];
                for j in 0..D {
                    d_bout[j] += row[j];
                }
            }
        }
        // d_W_out = h^T @ d_out, d_h = d_out @ W_out^T
        unsafe {
            gemm_tn(
                DM,
                D,
                bs,
                &self.h[..bs * DM],
                &self.d_out[..bs * D],
                &mut self.grads[OFF_WOUT..OFF_WOUT + DM * D],
                true,
            );
            transpose(DM, D, &self.params[OFF_WOUT..OFF_WOUT + DM * D], &mut self.wt_tmp[..D * DM]);
            gemm_nn_add(
                bs,
                DM,
                D,
                &self.d_out[..bs * D],
                &self.wt_tmp[..D * DM],
                &mut self.d_h[..bs * DM],
                true,
            );
        }
        // ReLU mask & d_b_in
        {
            let d_bin = &mut self.grads[OFF_BIN..OFF_BIN + DM];
            for i in 0..bs {
                let ph = &self.pre_h[i * DM..(i + 1) * DM];
                let dh = &mut self.d_h[i * DM..(i + 1) * DM];
                for j in 0..DM {
                    if ph[j] <= 0.0 {
                        dh[j] = 0.0;
                    } else {
                        d_bin[j] += dh[j];
                    }
                }
            }
        }
        // d_W_in = x_mid^T @ d_h, d_x_mid = d_out + d_h @ W_in^T
        self.d_x_mid[..bs * D].copy_from_slice(&self.d_out[..bs * D]);
        unsafe {
            gemm_tn(
                D,
                DM,
                bs,
                &self.x_mid[..bs * D],
                &self.d_h[..bs * DM],
                &mut self.grads[OFF_WIN..OFF_WIN + D * DM],
                true,
            );
            transpose(D, DM, &self.params[OFF_WIN..OFF_WIN + D * DM], &mut self.wt_tmp[..DM * D]);
            gemm_nn_add(
                bs,
                D,
                DM,
                &self.d_h[..bs * DM],
                &self.wt_tmp[..DM * D],
                &mut self.d_x_mid[..bs * D],
                false, // add onto d_out residual!
            );
        }

        // 3. Attention backward (factored)
        for v in self.d_x2.iter_mut() { *v = 0.0; }
        for v in self.d_vo0.iter_mut() { *v = 0.0; }
        for v in self.d_vo1.iter_mut() { *v = 0.0; }
        for v in self.d_vo2.iter_mut() { *v = 0.0; }
        for v in self.d_s0.iter_mut() { *v = 0.0; }
        for v in self.d_s1.iter_mut() { *v = 0.0; }
        for v in self.d_s2.iter_mut() { *v = 0.0; }

        for i in 0..bs {
            let ai = self.tr_a[i] as usize;
            let bi = self.tr_b[i] as usize;
            let g = &self.d_x_mid[i * D..(i + 1) * D];
            for j in 0..D {
                self.d_x2[j] += g[j];
            }
            for h_idx in 0..H {
                let wi = (i * H + h_idx) * 3;
                let w0 = self.attn_w[wi];
                let w1 = self.attn_w[wi + 1];
                let w2 = self.attn_w[wi + 2];

                let off0 = (ai * H + h_idx) * D;
                let off1 = (bi * H + h_idx) * D;
                let off2 = h_idx * D;

                axpy(&mut self.d_vo0[off0..off0 + D], w0, g);
                axpy(&mut self.d_vo1[off1..off1 + D], w1, g);
                axpy(&mut self.d_vo2[off2..off2 + D], w2, g);

                let gw0 = dot(g, &self.vo0[off0..off0 + D]);
                let gw1 = dot(g, &self.vo1[off1..off1 + D]);
                let gw2 = dot(g, &self.vo2[off2..off2 + D]);
                let mgw = w0 * gw0 + w1 * gw1 + w2 * gw2;

                self.d_s0[ai * H + h_idx] += w0 * (gw0 - mgw);
                self.d_s1[bi * H + h_idx] += w1 * (gw1 - mgw);
                self.d_s2[h_idx] += w2 * (gw2 - mgw);
            }
        }

        // Propagate d_vo -> d_W_O, d_v0, d_v1, d_v2
        for v in self.d_v0.iter_mut() { *v = 0.0; }
        for v in self.d_v1.iter_mut() { *v = 0.0; }
        for v in self.d_v2.iter_mut() { *v = 0.0; }

        for h_idx in 0..H {
            let wo_h = &self.params[OFF_WO + h_idx * DH * D..OFF_WO + (h_idx + 1) * DH * D];
            let d_wo_h = &mut self.grads[OFF_WO + h_idx * DH * D..OFF_WO + (h_idx + 1) * DH * D];
            transpose(DH, D, wo_h, &mut self.wt_tmp[..D * DH]); // [D, DH]

            // v2[h] outer d_vo2[h]
            let v2_h = &self.v2[h_idx * DH..(h_idx + 1) * DH];
            let dvo2_h = &self.d_vo2[h_idx * D..(h_idx + 1) * D];
            for r in 0..DH {
                axpy(&mut d_wo_h[r * D..(r + 1) * D], v2_h[r], dvo2_h);
            }
            unsafe {
                gemm_nn_add(
                    1,
                    DH,
                    D,
                    dvo2_h,
                    &self.wt_tmp[..D * DH],
                    &mut self.d_v2[h_idx * DH..(h_idx + 1) * DH],
                    true,
                );
            }

            for t in 0..P {
                let v0_th = &self.v0[t * D + h_idx * DH..t * D + (h_idx + 1) * DH];
                let dvo0_th = &self.d_vo0[(t * H + h_idx) * D..(t * H + h_idx + 1) * D];
                let v1_th = &self.v1[t * D + h_idx * DH..t * D + (h_idx + 1) * DH];
                let dvo1_th = &self.d_vo1[(t * H + h_idx) * D..(t * H + h_idx + 1) * D];
                for r in 0..DH {
                    axpy(&mut d_wo_h[r * D..(r + 1) * D], v0_th[r], dvo0_th);
                    axpy(&mut d_wo_h[r * D..(r + 1) * D], v1_th[r], dvo1_th);
                }
                unsafe {
                    gemm_nn_add(
                        1,
                        DH,
                        D,
                        dvo0_th,
                        &self.wt_tmp[..D * DH],
                        &mut self.d_v0[t * D + h_idx * DH..t * D + (h_idx + 1) * DH],
                        true,
                    );
                    gemm_nn_add(
                        1,
                        DH,
                        D,
                        dvo1_th,
                        &self.wt_tmp[..D * DH],
                        &mut self.d_v1[t * D + h_idx * DH..t * D + (h_idx + 1) * DH],
                        true,
                    );
                }
            }
        }

        // Propagate d_s -> d_q2, d_k0, d_k1, d_k2
        let scale = 1.0 / (DH as f32).sqrt();
        for v in self.d_q2.iter_mut() { *v = 0.0; }
        for h_idx in 0..H {
            let off = h_idx * DH;
            let qh = &self.q2[off..off + DH];
            let ds2_sc = self.d_s2[h_idx] * scale;
            for r in 0..DH {
                self.d_q2[off + r] += ds2_sc * self.k2[off + r];
                self.d_k2[off + r] = ds2_sc * qh[r];
            }
            for t in 0..P {
                let ds0_sc = self.d_s0[t * H + h_idx] * scale;
                let ds1_sc = self.d_s1[t * H + h_idx] * scale;
                for r in 0..DH {
                    self.d_q2[off + r] += ds0_sc * self.k0[t * D + off + r] + ds1_sc * self.k1[t * D + off + r];
                    self.d_k0[t * D + off + r] = ds0_sc * qh[r];
                    self.d_k1[t * D + off + r] = ds1_sc * qh[r];
                }
            }
        }

        // Propagate d_q2, d_k, d_v -> d_W_Q, d_W_K, d_W_V, d_x0, d_x1, d_x2
        unsafe {
            // W_Q
            gemm_tn(D, D, 1, &self.x2, &self.d_q2, &mut self.grads[OFF_WQ..OFF_WQ + D * D], true);
            transpose(D, D, &self.params[OFF_WQ..OFF_WQ + D * D], &mut self.wt_tmp[..D * D]);
            gemm_nn_add(1, D, D, &self.d_q2, &self.wt_tmp[..D * D], &mut self.d_x2, false);

            // W_K
            gemm_tn(D, D, P, &self.x0, &self.d_k0, &mut self.grads[OFF_WK..OFF_WK + D * D], true);
            gemm_tn(D, D, P, &self.x1, &self.d_k1, &mut self.grads[OFF_WK..OFF_WK + D * D], false);
            gemm_tn(D, D, 1, &self.x2, &self.d_k2, &mut self.grads[OFF_WK..OFF_WK + D * D], false);
            transpose(D, D, &self.params[OFF_WK..OFF_WK + D * D], &mut self.wt_tmp[..D * D]);
            gemm_nn_add(P, D, D, &self.d_k0, &self.wt_tmp[..D * D], &mut self.d_x0, true);
            gemm_nn_add(P, D, D, &self.d_k1, &self.wt_tmp[..D * D], &mut self.d_x1, true);
            gemm_nn_add(1, D, D, &self.d_k2, &self.wt_tmp[..D * D], &mut self.d_x2, false);

            // W_V
            gemm_tn(D, D, P, &self.x0, &self.d_v0, &mut self.grads[OFF_WV..OFF_WV + D * D], true);
            gemm_tn(D, D, P, &self.x1, &self.d_v1, &mut self.grads[OFF_WV..OFF_WV + D * D], false);
            gemm_tn(D, D, 1, &self.x2, &self.d_v2, &mut self.grads[OFF_WV..OFF_WV + D * D], false);
            transpose(D, D, &self.params[OFF_WV..OFF_WV + D * D], &mut self.wt_tmp[..D * D]);
            gemm_nn_add(P, D, D, &self.d_v0, &self.wt_tmp[..D * D], &mut self.d_x0, false);
            gemm_nn_add(P, D, D, &self.d_v1, &self.wt_tmp[..D * D], &mut self.d_x1, false);
            gemm_nn_add(1, D, D, &self.d_v2, &self.wt_tmp[..D * D], &mut self.d_x2, false);
        }

        // Accumulate into d_W_E and d_W_pos
        for t in 0..P {
            let dwe_t = &mut self.grads[OFF_WE + t * D..OFF_WE + (t + 1) * D];
            let dx0_t = &self.d_x0[t * D..(t + 1) * D];
            let dx1_t = &self.d_x1[t * D..(t + 1) * D];
            for j in 0..D {
                dwe_t[j] = dx0_t[j] + dx1_t[j];
            }
        }
        self.grads[OFF_WE + P * D..OFF_WE + (P + 1) * D].copy_from_slice(&self.d_x2);

        for t in 0..P {
            let dx0_t = &self.d_x0[t * D..(t + 1) * D];
            let dx1_t = &self.d_x1[t * D..(t + 1) * D];
            for j in 0..D {
                self.grads[OFF_WPOS + j] += dx0_t[j];
                self.grads[OFF_WPOS + D + j] += dx1_t[j];
            }
        }
        self.grads[OFF_WPOS + 2 * D..OFF_WPOS + 3 * D].copy_from_slice(&self.d_x2);

        // 4. AdamW update (exact PyTorch decoupled weight decay)
        self.step += 1;
        let t_f = self.step as f32;
        let bc1 = 1.0 - self.beta1.powf(t_f);
        let bc2 = 1.0 - self.beta2.powf(t_f);
        let step_size = self.lr / bc1;
        let inv_bc2 = 1.0 / bc2;
        let decay = 1.0 - self.lr * self.wd;
        let b1 = self.beta1;
        let ob1 = 1.0 - b1;
        let b2 = self.beta2;
        let ob2 = 1.0 - b2;
        let eps = self.eps;

        for i in 0..N_PARAMS {
            let g = self.grads[i];
            let m = b1 * self.m[i] + ob1 * g;
            let v = b2 * self.v[i] + ob2 * g * g;
            self.m[i] = m;
            self.v[i] = v;
            self.params[i] = self.params[i] * decay - step_size * m / ((v * inv_bc2).sqrt() + eps);
        }

        (loss_sum * inv_bs, (correct as f32) * inv_bs)
    }

    // Build ablated W_E in self.we_abl from bitmask (bit k-1 for k=1..=29)
    fn build_ablated_we(&mut self, keep_mask: u32) {
        let we = &self.params[OFF_WE..OFF_WE + V * D];
        // Always keep '=' token unchanged
        self.we_abl[P * D..(P + 1) * D].copy_from_slice(&we[P * D..(P + 1) * D]);
        for v in self.we_abl[..P * D].iter_mut() {
            *v = 0.0;
        }
        // Keep DC (k = 0)
        let mut c_dc = [0.0f32; D];
        let s0 = self.dft_cos[0]; // 1/sqrt(P)
        for t in 0..P {
            for j in 0..D {
                c_dc[j] += s0 * we[t * D + j];
            }
        }
        for t in 0..P {
            for j in 0..D {
                self.we_abl[t * D + j] = s0 * c_dc[j];
            }
        }
        // Keep selected frequencies k = 1..=NFREQ
        for k in 1..=NFREQ {
            if (keep_mask & (1u32 << (k - 1))) == 0 {
                continue;
            }
            let cos_k = &self.dft_cos[k * P..(k + 1) * P];
            let sin_k = &self.dft_sin[k * P..(k + 1) * P];
            let mut u = [0.0f32; D];
            let mut v = [0.0f32; D];
            for t in 0..P {
                let ck = cos_k[t];
                let sk = sin_k[t];
                let row = &we[t * D..(t + 1) * D];
                for j in 0..D {
                    u[j] += ck * row[j];
                    v[j] += sk * row[j];
                }
            }
            for t in 0..P {
                let ck = cos_k[t];
                let sk = sin_k[t];
                let dst = &mut self.we_abl[t * D..(t + 1) * D];
                for j in 0..D {
                    dst[j] += ck * u[j] + sk * v[j];
                }
            }
        }
    }

    fn loss_acc_split(logits: &[f32], all_y: &[u16], is_train: &[u8]) -> (f32, f32, f32, f32) {
        let mut tr_loss = 0.0f32;
        let mut tr_ok = 0usize;
        let mut tr_n = 0usize;
        let mut te_loss = 0.0f32;
        let mut te_ok = 0usize;
        let mut te_n = 0usize;

        for i in 0..P2 {
            let yi = all_y[i] as usize;
            let lg = &logits[i * P..(i + 1) * P];
            let mut mx = lg[0];
            let mut argmax = 0usize;
            for c in 1..P {
                if lg[c] > mx {
                    mx = lg[c];
                    argmax = c;
                }
            }
            let mut sum_exp = 0.0f32;
            for c in 0..P {
                sum_exp += (lg[c] - mx).exp();
            }
            let loss = -(lg[yi] - mx) + sum_exp.ln();
            if is_train[i] != 0 {
                tr_loss += loss;
                if argmax == yi { tr_ok += 1; }
                tr_n += 1;
            } else {
                te_loss += loss;
                if argmax == yi { te_ok += 1; }
                te_n += 1;
            }
        }
        (
            tr_loss / (tr_n.max(1) as f32),
            (tr_ok as f32) / (tr_n.max(1) as f32),
            te_loss / (te_n.max(1) as f32),
            (te_ok as f32) / (te_n.max(1) as f32),
        )
    }

    // Full evaluation + Fourier spectrum + Nanda et al. restricted & excluded loss
    fn evaluate(&mut self) {
        self.forward(P2, true, false);
        let (tr_loss, tr_acc, te_loss, te_acc) =
            Self::loss_acc_split(&self.logits, &self.all_y, &self.is_train);

        // Weight L2 norm
        let mut w2 = 0.0f32;
        for &p in self.params.iter() {
            w2 += p * p;
        }
        let weight_norm = w2.sqrt();

        // 1D Fourier power of W_E[0..P]
        let we = &self.params[OFF_WE..OFF_WE + P * D];
        let mut power = [0.0f32; NFREQ];
        let mut total_power = 0.0f32;
        for k in 1..=NFREQ {
            let cos_k = &self.dft_cos[k * P..(k + 1) * P];
            let sin_k = &self.dft_sin[k * P..(k + 1) * P];
            let mut pk = 0.0f32;
            for j in 0..D {
                let mut re = 0.0f32;
                let mut im = 0.0f32;
                for t in 0..P {
                    let v = we[t * D + j];
                    re += cos_k[t] * v;
                    im += sin_k[t] * v;
                }
                pk += re * re + im * im;
            }
            power[k - 1] = pk;
            total_power += pk;
        }
        let inv_tot = 1.0 / (total_power + 1e-12);
        let mut frac = [0.0f32; NFREQ];
        for i in 0..NFREQ {
            frac[i] = power[i] * inv_tot;
        }

        // Top 5 frequencies by embedding energy
        let mut order: [usize; NFREQ] = core::array::from_fn(|i| i + 1);
        order.sort_by(|&a, &b| frac[b - 1].partial_cmp(&frac[a - 1]).unwrap_or(core::cmp::Ordering::Equal));
        for i in 0..5 {
            self.last_top5[i] = order[i];
        }
        let top5_share: f32 = (0..5).map(|i| frac[order[i] - 1]).sum();

        // Gini coefficient of Fourier power spectrum (0 = uniform, 1 = single spike)
        let mut sorted = frac;
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        let mut gini_num = 0.0f32;
        let nf = NFREQ as f32;
        for (i, &v) in sorted.iter().enumerate() {
            gini_num += (2.0 * (i as f32 + 1.0) - nf - 1.0) * v;
        }
        let gini = gini_num / nf;

        // Restricted & Excluded Loss (live variant of Nanda et al.):
        // Project logits L[a, b, c] onto {(0,0)} U {cos,sin}(w_k a) x {cos,sin}(w_k b) for k in top5.
        let kb_list = self.last_top5;

        // Compute DC and 5 frequency blocks (2x2 real trig basis per k) for each output class c.
        let mut dc_c = [0.0f32; P];
        let mut cc = [[0.0f32; P]; 5];
        let mut cs = [[0.0f32; P]; 5];
        let mut sc = [[0.0f32; P]; 5];
        let mut ss = [[0.0f32; P]; 5];

        let inv_p = 1.0 / (P as f32);
        // First contract over b for each a:
        // For fixed a, compute row_sum[c] = sum_b L[a,b,c]
        // and for each f in 0..5: bc[f,c] = sum_b cos(kb b) L[a,b,c], bs[f,c] = sum_b sin(kb b) L[a,b,c]
        let mut row_sum = [0.0f32; P];
        let mut bc = [[0.0f32; P]; 5];
        let mut bs_arr = [[0.0f32; P]; 5];

        for a in 0..P {
            row_sum.fill(0.0);
            for f in 0..5 {
                bc[f].fill(0.0);
                bs_arr[f].fill(0.0);
            }
            for b in 0..P {
                let lg = &self.logits[(a * P + b) * P..(a * P + b + 1) * P];
                for c in 0..P {
                    row_sum[c] += lg[c];
                }
                for f in 0..5 {
                    let kb = kb_list[f];
                    let ck = self.dft_cos[kb * P + b];
                    let sk = self.dft_sin[kb * P + b];
                    axpy(&mut bc[f], ck, lg);
                    axpy(&mut bs_arr[f], sk, lg);
                }
            }
            for c in 0..P {
                dc_c[c] += row_sum[c] * inv_p * inv_p;
            }
            for f in 0..5 {
                let ka = self.last_top5[f];
                let ca = self.dft_cos[ka * P + a];
                let sa = self.dft_sin[ka * P + a];
                axpy(&mut cc[f], ca, &bc[f]);
                axpy(&mut cs[f], ca, &bs_arr[f]);
                axpy(&mut sc[f], sa, &bc[f]);
                axpy(&mut ss[f], sa, &bs_arr[f]);
            }
        }

        // Reconstruct L_res[a, b, c] and evaluate restricted test loss & excluded train loss
        let mut res_te_loss = 0.0f32;
        let mut res_te_n = 0usize;
        let mut exc_tr_loss = 0.0f32;
        let mut exc_tr_n = 0usize;
        let mut l_r = [0.0f32; P];
        let mut l_e = [0.0f32; P];

        for a in 0..P {
            // Pre-combine a terms for each f:
            // u_c[f, c] = cos(ka a) * cc[f, c] + sin(ka a) * sc[f, c]
            // u_s[f, c] = cos(ka a) * cs[f, c] + sin(ka a) * ss[f, c]
            let mut uc = [[0.0f32; P]; 5];
            let mut us = [[0.0f32; P]; 5];
            for f in 0..5 {
                let ka = self.last_top5[f];
                let ca = self.dft_cos[ka * P + a];
                let sa = self.dft_sin[ka * P + a];
                for c in 0..P {
                    uc[f][c] = ca * cc[f][c] + sa * sc[f][c];
                    us[f][c] = ca * cs[f][c] + sa * ss[f][c];
                }
            }
            for b in 0..P {
                let idx = a * P + b;
                let yi = self.all_y[idx] as usize;
                l_r.copy_from_slice(&dc_c);
                for f in 0..5 {
                    let kb = kb_list[f];
                    let cb = self.dft_cos[kb * P + b];
                    let sb = self.dft_sin[kb * P + b];
                    axpy(&mut l_r, cb, &uc[f]);
                    axpy(&mut l_r, sb, &us[f]);
                }
                if self.is_train[idx] == 0 {
                    // Restricted test loss
                    let mut mx = l_r[0];
                    for c in 1..P { if l_r[c] > mx { mx = l_r[c]; } }
                    let mut se = 0.0f32;
                    for c in 0..P { se += (l_r[c] - mx).exp(); }
                    res_te_loss += -(l_r[yi] - mx) + se.ln();
                    res_te_n += 1;
                } else {
                    // Excluded train loss: L - L_res
                    let lg = &self.logits[idx * P..(idx + 1) * P];
                    for c in 0..P { l_e[c] = lg[c] - l_r[c]; }
                    let mut mx = l_e[0];
                    for c in 1..P { if l_e[c] > mx { mx = l_e[c]; } }
                    let mut se = 0.0f32;
                    for c in 0..P { se += (l_e[c] - mx).exp(); }
                    exc_tr_loss += -(l_e[yi] - mx) + se.ln();
                    exc_tr_n += 1;
                }
            }
        }

        self.eval_out[0] = tr_loss;
        self.eval_out[1] = tr_acc;
        self.eval_out[2] = te_loss;
        self.eval_out[3] = te_acc;
        self.eval_out[4] = weight_norm;
        self.eval_out[5] = res_te_loss / (res_te_n.max(1) as f32);
        self.eval_out[6] = exc_tr_loss / (exc_tr_n.max(1) as f32);
        self.eval_out[7] = top5_share;
        self.eval_out[8] = gini;
        for i in 0..5 {
            self.eval_out[9 + i] = self.last_top5[i] as f32;
        }
        for i in 0..NFREQ {
            self.eval_out[14 + i] = frac[i];
        }
    }

    // Causal ablation: project W_E[0..P] onto bitmask of frequencies and evaluate all 3481 pairs
    fn eval_ablated(&mut self, keep_mask: u32) {
        self.build_ablated_we(keep_mask);
        self.forward(P2, true, true);
        let (tr_loss, tr_acc, te_loss, te_acc) =
            Self::loss_acc_split(&self.logits, &self.all_y, &self.is_train);
        self.ablate_out[0] = tr_acc;
        self.ablate_out[1] = te_acc;
        self.ablate_out[2] = tr_loss;
        self.ablate_out[3] = te_loss;
    }

    // 2D circle coordinates of tokens 0..P-1 projected onto the frequency-k subspace of W_E
    fn compute_circle(&mut self, k: usize) {
        let k = k.clamp(1, NFREQ);
        let we = &self.params[OFF_WE..OFF_WE + P * D];
        let cos_k = &self.dft_cos[k * P..(k + 1) * P];
        let sin_k = &self.dft_sin[k * P..(k + 1) * P];

        // u = sum_t cos_k[t] * W_E[t], v = sum_t sin_k[t] * W_E[t]
        let mut u = [0.0f32; D];
        let mut v = [0.0f32; D];
        let mut mean = [0.0f32; D];
        for t in 0..P {
            let row = &we[t * D..(t + 1) * D];
            let ck = cos_k[t];
            let sk = sin_k[t];
            for j in 0..D {
                mean[j] += row[j] / (P as f32);
                u[j] += ck * row[j];
                v[j] += sk * row[j];
            }
        }
        // Orthonormalize (u, v) via Gram-Schmidt so coordinates preserve true geometry
        let nu = dot(&u, &u).sqrt().max(1e-8);
        for j in 0..D { u[j] /= nu; }
        let uv = dot(&u, &v);
        for j in 0..D { v[j] -= uv * u[j]; }
        let nv = dot(&v, &v).sqrt().max(1e-8);
        for j in 0..D { v[j] /= nv; }

        let mut max_r = 1e-6f32;
        for t in 0..P {
            let row = &we[t * D..(t + 1) * D];
            let mut x = 0.0f32;
            let mut y = 0.0f32;
            for j in 0..D {
                let d = row[j] - mean[j];
                x += d * u[j];
                y += d * v[j];
            }
            self.circle_out[t * 2] = x;
            self.circle_out[t * 2 + 1] = y;
            let r = (x * x + y * y).sqrt();
            if r > max_r { max_r = r; }
        }
        // Normalize so radius fits nicely inside [-1, 1]
        let inv_r = 1.0 / max_r;
        for v in self.circle_out.iter_mut() {
            *v *= inv_r;
        }
    }

    // Probe a single (a, b) pair (optionally with ablated W_E) and decompose output logits
    // into Fourier waves over output classes c = 0..P-1 for the top 5 frequencies!
    fn probe_pair(&mut self, a: usize, b: usize, keep_mask: u32) {
        let a = a % P;
        let b = b % P;
        let full_mask: u32 = (1u32 << NFREQ) - 1;
        let use_abl = (keep_mask & full_mask) != full_mask;
        if use_abl {
            self.build_ablated_we(keep_mask);
        }
        // Run single-example forward using slot 0 of tr_a/tr_b, then restore it.
        let saved_a = self.tr_a[0];
        let saved_b = self.tr_b[0];
        self.tr_a[0] = a as u16;
        self.tr_b[0] = b as u16;
        self.forward(1, false, use_abl);
        self.tr_a[0] = saved_a;
        self.tr_b[0] = saved_b;

        let lg = &self.logits[0..P];
        // Center logits for clean wave view
        let mut mean_l = 0.0f32;
        let mut mx = lg[0];
        for c in 0..P {
            mean_l += lg[c];
            if lg[c] > mx { mx = lg[c]; }
        }
        mean_l /= P as f32;

        // Copy logits & softmax probs
        let mut sum_e = 0.0f32;
        for c in 0..P {
            self.probe_out[c] = lg[c];
            let e = (lg[c] - mx).exp();
            self.probe_out[P + c] = e;
            sum_e += e;
        }
        for c in 0..P {
            self.probe_out[P + c] /= sum_e;
        }
        // Attention weights [4, 3]
        for i in 0..H * 3 {
            self.probe_out[2 * P + i] = self.attn_w[i];
        }
        // Top 5 frequencies (sorted by power in W_E from last evaluate)
        let base_k = 2 * P + H * 3;
        for f in 0..5 {
            self.probe_out[base_k + f] = self.last_top5[f] as f32;
        }
        // Project centered logits onto each of the top 5 output frequencies k, plus residual
        let base_w = base_k + 5;
        let mut resid = [0.0f32; P];
        for c in 0..P {
            resid[c] = lg[c] - mean_l;
        }
        for f in 0..5 {
            let k = self.last_top5[f];
            let cos_k = &self.dft_cos[k * P..(k + 1) * P];
            let sin_k = &self.dft_sin[k * P..(k + 1) * P];
            let mut ac = 0.0f32;
            let mut as_ = 0.0f32;
            for c in 0..P {
                let lc = lg[c] - mean_l;
                ac += cos_k[c] * lc;
                as_ += sin_k[c] * lc;
            }
            for c in 0..P {
                let wv = ac * cos_k[c] + as_ * sin_k[c];
                self.probe_out[base_w + f * P + c] = wv;
                resid[c] -= wv;
            }
        }
        for c in 0..P {
            self.probe_out[base_w + 5 * P + c] = resid[c];
        }
    }
}

static mut STATE: *mut Trainer = core::ptr::null_mut();

fn state() -> &'static mut Trainer {
    unsafe {
        if STATE.is_null() {
            STATE = Box::into_raw(Box::new(Trainer::new()));
        }
        &mut *STATE
    }
}

// ---------- WASM C ABI exports ----------

#[no_mangle]
pub extern "C" fn init_model(op: u32, train_frac: f32, lr: f32, wd: f32, seed: u32) {
    state().init(op, train_frac, lr, wd, seed);
}

#[no_mangle]
pub extern "C" fn set_hparams(lr: f32, wd: f32) {
    let s = state();
    s.lr = lr;
    s.wd = wd;
}

#[no_mangle]
pub extern "C" fn step_n(n: u32) -> f32 {
    let s = state();
    let mut last_loss = 0.0f32;
    for _ in 0..n {
        let (l, _) = s.train_step();
        last_loss = l;
    }
    last_loss
}

#[no_mangle]
pub extern "C" fn get_step() -> u32 {
    state().step
}

#[no_mangle]
pub extern "C" fn get_n_params() -> u32 {
    N_PARAMS as u32
}

#[no_mangle]
pub extern "C" fn get_params_ptr() -> *mut f32 {
    state().params.as_mut_ptr()
}

#[no_mangle]
pub extern "C" fn get_grads_ptr() -> *const f32 {
    state().grads.as_ptr()
}

#[no_mangle]
pub extern "C" fn get_train_pairs_a_ptr() -> *const u16 {
    state().tr_a.as_ptr()
}

#[no_mangle]
pub extern "C" fn get_train_pairs_b_ptr() -> *const u16 {
    state().tr_b.as_ptr()
}

#[no_mangle]
pub extern "C" fn get_train_labels_ptr() -> *const u16 {
    state().tr_y.as_ptr()
}

#[no_mangle]
pub extern "C" fn get_n_train() -> u32 {
    state().n_train as u32
}

#[no_mangle]
pub extern "C" fn get_is_train_ptr() -> *const u8 {
    state().is_train.as_ptr()
}

#[no_mangle]
pub extern "C" fn evaluate() -> *const f32 {
    let s = state();
    s.evaluate();
    s.eval_out.as_ptr()
}

#[no_mangle]
pub extern "C" fn eval_ablated(keep_mask: u32) -> *const f32 {
    let s = state();
    s.eval_ablated(keep_mask);
    s.ablate_out.as_ptr()
}

#[no_mangle]
pub extern "C" fn compute_circle(k: u32) -> *const f32 {
    let s = state();
    s.compute_circle(k as usize);
    s.circle_out.as_ptr()
}

#[no_mangle]
pub extern "C" fn probe_pair(a: u32, b: u32, keep_mask: u32) -> *const f32 {
    let s = state();
    s.probe_pair(a as usize, b as usize, keep_mask);
    s.probe_out.as_ptr()
}

#[no_mangle]
pub extern "C" fn get_eval_len() -> u32 {
    EVAL_LEN as u32
}

#[no_mangle]
pub extern "C" fn get_probe_len() -> u32 {
    PROBE_LEN as u32
}
