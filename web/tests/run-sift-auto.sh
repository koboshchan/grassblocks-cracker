#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
# Preparation decodes pixels only; never reads precomputed face/camera/seed JSON.
${PYTHON:-python3} - <<'PY'
import cv2 as cv,json
im=cv.imread('public/examples/hill.png');ref=cv.imread('public/grass_block_top.png',0)
open('/tmp/grass-hill-gray.bin','wb').write(im[:,:,1].tobytes())
json.dump(dict(w=im.shape[1],h=im.shape[0],ref=ref.tolist()),open('/tmp/grass-wasm-fixtures.json','w'))
PY
cp public/vendor/opencv.js /tmp/grass-opencv.cjs
cp public/vendor/grass-sift.js /tmp/grass-sift.cjs
cp public/vendor/grass-sift.wasm /tmp/grass-sift.wasm
export GRASS_SIFT_CJS=/tmp/grass-sift.cjs
node_modules/.bin/esbuild tests/sift-auto.ts --bundle --platform=node --format=esm --outfile=/tmp/grass-sift-auto.mjs
node /tmp/grass-sift-auto.mjs
