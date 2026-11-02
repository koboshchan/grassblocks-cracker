#!/bin/sh
set -eu
mkdir -p artifacts
${CC:-clang} --target=wasm32 -O3 -msimd128 -ffp-contract=off -fno-builtin -nostdlib -Wl,--no-entry -Wl,--export-all -Wl,--initial-memory=262144 -Wl,--max-memory=536870912 -o artifacts/detect-wasm.wasm detect-wasm.c
${CC:-clang} --target=wasm32 -O3 -msimd128 -fwrapv -fno-builtin -nostdlib -Wl,--no-entry -Wl,--export-all -Wl,--initial-memory=262144 -o artifacts/solver.wasm solver.c
