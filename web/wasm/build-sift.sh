#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
emcc -O3 -DVL_DISABLE_SSE2 -DVL_DISABLE_AVX -DVL_DISABLE_THREADS -Iwasm/vlfeat wasm/sift.c wasm/vlfeat/vl/sift.c wasm/vlfeat/vl/imopv.c wasm/vlfeat/vl/mathop.c wasm/vlfeat/vl/generic.c wasm/vlfeat/vl/host.c wasm/vlfeat/vl/random.c -s ALLOW_MEMORY_GROWTH=1 -s MAXIMUM_MEMORY=536870912 -s EXPORTED_FUNCTIONS='["_extract","_get_out","_malloc","_free"]' -s MODULARIZE=1 -s EXPORT_NAME=GrassSift -o public/vendor/grass-sift.js
