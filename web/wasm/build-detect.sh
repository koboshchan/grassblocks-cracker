#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
${CC:-clang} -target wasm32 -O3 -msimd128 -ffp-contract=off -fno-builtin -nostdlib -Wl,--no-entry -Wl,--export-all -Wl,--initial-memory=262144 -Wl,--max-memory=536870912 -o detect-wasm.wasm detect-wasm.c
