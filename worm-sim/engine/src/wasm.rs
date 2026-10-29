//! WebAssembly interface for the page's worker: one global worm. JavaScript copies
//! brain.bin / body.bin / the readout into buffers from `alloc`, calls `init`,
//! then reads state straight out of linear memory through the pointer getters.
#![allow(static_mut_refs, clippy::missing_safety_doc)]
use crate::{
    behavior::{Mode, Params, Seek, GAIT_TRIM},
    body::Body,
    brain::Brain,
    worm::Worm,
};

static mut WORM: Option<Worm> = None;
static mut SOMA: Vec<f64> = Vec::new();
static mut NODE_V: Vec<f32> = Vec::new();
static mut SEEK: [f64; 8] = [0.0; 8];

fn worm() -> &'static mut Worm {
    unsafe { WORM.as_mut().expect("init first") }
}

#[no_mangle]
pub extern "C" fn alloc(n_f64: usize) -> *mut f64 {
    let mut v = vec![0.0f64; n_f64];
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

/// cells: 15 sensory then 80 motor cell indices (as f64)
#[no_mangle]
pub unsafe extern "C" fn init(brain: *const f64, nb: usize, body: *const f64, nbody: usize, wout: *const f64, cells: *const f64) {
    let b = Brain::new(std::slice::from_raw_parts(brain, nb), 5.0 / 3.0);
    let bd = Body::new(std::slice::from_raw_parts(body, nbody));
    let w = std::slice::from_raw_parts(wout, 80 * 96).to_vec();
    let c = std::slice::from_raw_parts(cells, 95);
    let input = c[..15].iter().map(|&x| x as usize).collect();
    let output = c[15..].iter().map(|&x| x as usize).collect();
    SOMA = vec![0.0; b.soma.len()];
    WORM = Some(Worm::with_cells(b, bd, w, input, output));
}

#[no_mangle]
pub unsafe extern "C" fn set_readout(wout: *const f64) {
    worm().wout.copy_from_slice(std::slice::from_raw_parts(wout, 80 * 96));
}

#[no_mangle]
pub extern "C" fn reset() {
    let w = worm();
    w.reset();
    if let Some(s) = w.seek.as_mut() {
        let mut fresh = Seek::new(s.food, s.p, s.seed + 1);
        fresh.trim = s.trim;
        *s = fresh;
    }
}

#[no_mangle]
pub extern "C" fn warm_up(n: usize) {
    worm().warm_up(n);
}

/// one control step: 100 ms of brain, then 1/30 s of body
#[no_mangle]
pub extern "C" fn think() {
    worm().think();
}

/// muscle activations to use on the next `moving` (96 values), for a page that runs the brain elsewhere
#[no_mangle]
pub unsafe extern "C" fn set_muscles(m: *const f64) {
    worm().muscle.copy_from_slice(std::slice::from_raw_parts(m, 96));
}

#[no_mangle]
pub extern "C" fn moving() {
    worm().moving();
}

#[no_mangle]
pub extern "C" fn set_food(x: f64, y: f64, z: f64) {
    worm().set_food(Some([x, y, z]));
}

#[no_mangle]
pub extern "C" fn clear_food() {
    worm().set_food(None);
}

#[no_mangle]
pub extern "C" fn set_fluid(k: f64) {
    worm().body.fluid = k;
}

/// scale every chemical synapse (kind 1 exc, 2 inh) or gap junction (kind 0) by g
#[no_mangle]
pub extern "C" fn set_gain(kind: u32, g: f64) {
    let b = &mut worm().brain;
    if kind == 0 {
        for x in b.gj_gain.iter_mut() {
            *x = g;
        }
    } else {
        for k in 0..b.n_syn() {
            if b.syn_pre_post(k).2 == (kind == 1) {
                b.sy_gain[k] = g;
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn positions() -> *const f64 {
    worm().body.x.as_ptr()
}

#[no_mangle]
pub extern "C" fn n_vertices() -> usize {
    worm().body.nv
}

#[no_mangle]
pub extern "C" fn muscles() -> *const f64 {
    worm().muscle.as_ptr()
}

#[no_mangle]
pub extern "C" fn motor_v() -> *const f64 {
    worm().motor_v.as_ptr()
}

/// all 24,496 compartment voltages as f32, in brain.bin's node order
#[no_mangle]
pub extern "C" fn node_v() -> *const f32 {
    unsafe {
        let b = &worm().brain;
        NODE_V.resize(b.n, 0.0);
        for (k, &i) in b.order.iter().enumerate() {
            NODE_V[i] = b.v[k] as f32;
        }
        NODE_V.as_ptr()
    }
}

#[no_mangle]
pub extern "C" fn soma_v() -> *const f64 {
    unsafe {
        let w = worm();
        for c in 0..SOMA.len() {
            SOMA[c] = w.brain.soma_v(c);
        }
        SOMA.as_ptr()
    }
}

#[no_mangle]
pub extern "C" fn stimulus() -> f64 {
    worm().stimulus
}

#[no_mangle]
pub extern "C" fn iterations() -> usize {
    worm().body.iterations
}

/// Crawl on agar with both clocks in step (100 ms of body per control step) and seek
/// food at (x, z) on the plate. Call once after init; `seek_food` moves the food.
#[no_mangle]
pub extern "C" fn agar_on(x: f64, z: f64, seed: f64) {
    let w = worm();
    w.body_steps = 24;
    w.body.set_agar(Some((0.003, 0.09)));
    w.body.balance_muscles();
    let mut s = Seek::new([x, 0.0, z], Params::default(), seed as u64);
    s.trim = GAIT_TRIM;
    w.seek = Some(s);
}

#[no_mangle]
pub extern "C" fn seek_food(x: f64, z: f64) {
    if let Some(s) = worm().seek.as_mut() {
        s.food = [x, 0.0, z];
    }
}

/// [mode (0 forward, 1 reversal, 2 omega), concentration (mM), smoothed dC/dt (mM/s),
///  turn rate (per s), head bias, turns so far, nose x, nose z]
#[no_mangle]
pub extern "C" fn seek_state() -> *const f64 {
    unsafe {
        let w = worm();
        let h = w.head();
        SEEK = match w.seek.as_ref() {
            Some(s) => [
                match s.mode {
                    Mode::Forward => 0.0,
                    Mode::Reverse => 1.0,
                    Mode::Omega => 2.0,
                },
                s.c,
                s.dcdt_slow,
                s.turn_rate(s.dcdt_slow),
                s.bias + s.wander,
                s.turns as f64,
                h[0],
                h[2],
            ],
            None => [0.0; 8],
        };
        SEEK.as_ptr()
    }
}
