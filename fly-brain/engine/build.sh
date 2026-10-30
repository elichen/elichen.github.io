#!/bin/sh
# Compile the engine to WebAssembly (needs: rustup target add wasm32-unknown-unknown)
set -e
cd "$(dirname "$0")"
cargo build --release --lib --target wasm32-unknown-unknown
cp target/wasm32-unknown-unknown/release/fly_engine.wasm ../engine.wasm
ls -l ../engine.wasm
