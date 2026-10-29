//! worm-sim engine: BAAIWorm's brain (`brain`) and body (`body`), shared by the
//! page (as WebAssembly, see `wasm.rs`) and the native check tools in src/bin.
#![allow(clippy::needless_range_loop)]
pub mod brain;
pub mod body;
pub mod frame;
pub mod worm;
#[cfg(target_arch = "wasm32")]
pub mod wasm;
