#!/bin/sh
# Compile the transformer trainer to WebAssembly with SIMD128
set -e
cd "$(dirname "$0")"
cargo build --release
cp target/wasm32-unknown-unknown/release/grok_wasm.wasm ../../grok.wasm
ls -lh ../../grok.wasm
