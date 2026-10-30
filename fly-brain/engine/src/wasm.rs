//! WebAssembly interface for the page's worker: one global brain. JavaScript copies the
//! gunzipped connectome into a buffer from `alloc`, calls `init`, then runs steps and reads
//! the spikes straight out of linear memory.
#![allow(static_mut_refs, clippy::missing_safety_doc)]
use crate::brain::{parse, Brain};

static mut BRAIN: Option<Brain> = None;
/// (step, neuron) pairs from the last `run`
static mut SPIKES: Vec<u32> = Vec::new();

fn brain() -> &'static mut Brain {
    unsafe { BRAIN.as_mut().expect("init first") }
}

#[no_mangle]
pub extern "C" fn alloc(bytes: usize) -> *mut u8 {
    let mut v = vec![0u8; bytes];
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

/// Parse the connectome (freeing its buffer) and return the number of neurons.
#[no_mangle]
pub unsafe extern "C" fn init(ptr: *mut u8, len: usize, seed: u32) -> usize {
    let bytes = Vec::from_raw_parts(ptr, len, len);
    let (offsets, targets, counts) = parse(&bytes);
    drop(bytes);
    let b = Brain::new(offsets, targets, counts, seed as u64);
    let n = b.n;
    BRAIN = Some(b);
    n
}

#[no_mangle]
pub extern "C" fn set_rate(i: usize, hz: f64) {
    brain().set_rate(i, hz);
}

#[no_mangle]
pub extern "C" fn set_silenced(i: usize, s: bool) {
    brain().set_silenced(i, s);
}

#[no_mangle]
pub extern "C" fn reset() {
    brain().reset();
}

/// Run `steps` steps of 0.1 ms; returns the number of spikes (see `spikes`).
#[no_mangle]
pub unsafe extern "C" fn run(steps: u32) -> usize {
    let b = brain();
    SPIKES.clear();
    for _ in 0..steps {
        let n = b.step_n as u32;
        b.step();
        for &s in &b.spikes {
            SPIKES.push(n);
            SPIKES.push(s);
        }
    }
    SPIKES.len() / 2
}

#[no_mangle]
pub unsafe extern "C" fn spikes() -> *const u32 {
    SPIKES.as_ptr()
}

#[no_mangle]
pub extern "C" fn step_count() -> u32 {
    brain().step_n as u32
}

#[no_mangle]
pub extern "C" fn active() -> usize {
    brain().active_len()
}

/// Outgoing connections of neuron i: writes (target, signed synapse count) pairs into `out`
/// (room for `max` pairs) and returns how many there are.
#[no_mangle]
pub unsafe extern "C" fn outputs(i: usize, out: *mut i32, max: usize) -> usize {
    let b = brain();
    let (s, e) = (b.offsets[i] as usize, b.offsets[i + 1] as usize);
    let o = std::slice::from_raw_parts_mut(out, 2 * max);
    for (k, edge) in (s..e).take(max).enumerate() {
        o[2 * k] = b.targets[edge] as i32;
        o[2 * k + 1] = b.counts[edge] as i32;
    }
    e - s
}
