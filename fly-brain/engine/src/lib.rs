//! fly-brain engine: Shiu et al.'s whole-brain LIF model (`brain`), shared by the page
//! (as WebAssembly, see `wasm.rs`) and the native check tools in src/bin.
pub mod brain;
#[cfg(target_arch = "wasm32")]
pub mod wasm;

/// (step, neuron) pairs: u32 count, then count pairs of u32 (little endian)
pub fn read_events(path: &str) -> Vec<(u32, u32)> {
    let b = std::fs::read(path).expect(path);
    let u = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
    (0..u(0) as usize).map(|k| (u(4 + 8 * k), u(8 + 8 * k))).collect()
}

#[cfg(target_arch = "wasm32")]
pub mod policy;
