#!/bin/sh
# Compile the learner to WebAssembly (needs: rustup target add wasm32-unknown-unknown)
set -e
cd "$(dirname "$0")"
cargo build --release
cp target/wasm32-unknown-unknown/release/stream_ac.wasm ../../stream-ac.wasm
ls -l ../../stream-ac.wasm
