import fs from 'node:fs';
import * as wasm from '../detect-wasm.ts';

const wasmBytes = fs.readFileSync('artifacts/detect-wasm.wasm');
await wasm.init(wasmBytes);

// Setup a 1920x1080 test image
const W = 1920, H = 1080;
const img = new Uint8Array(W * H);
for (let y = 0; y < H; y++) {
  for (let x = 0; x < W; x++) {
    img[y * W + x] = (x * 13 + y * 17) & 0xff;
  }
}
wasm.setImage(img, W, H);

// Setup 4x256 reference textures
const refs = new Uint8Array(1024);
for (let k = 0; k < 4; k++) {
  for (let i = 0; i < 256; i++) {
    refs[k * 256 + i] = (i * 7 + k * 31) & 0xff;
  }
}
wasm.setRefs(refs);

// Generate candidate quads
const QUAD_COUNTS = [1000, 10000, 100000];
for (const count of QUAD_COUNTS) {
  const quads = new Float32Array(count * 8);
  for (let i = 0; i < count; i++) {
    const ox = 100 + (i % 50) * 30;
    const oy = 100 + Math.floor(i / 50) % 30 * 25;
    const s = 16 + (i % 8);
    // quad: [p0, p1, p2, p3]
    quads[i * 8 + 0] = ox;
    quads[i * 8 + 1] = oy;
    quads[i * 8 + 2] = ox + s;
    quads[i * 8 + 3] = oy + 2;
    quads[i * 8 + 4] = ox + s - 1;
    quads[i * 8 + 5] = oy + s;
    quads[i * 8 + 6] = ox + 1;
    quads[i * 8 + 7] = oy + s - 1;
  }

  // Warmup
  wasm.classifyBatch(quads.subarray(0, Math.min(count, 1000) * 8));

  // Benchmark
  const iters = count <= 10000 ? 5 : 3;
  const times: number[] = [];
  let validCount = 0;
  for (let iter = 0; iter < iters; iter++) {
    const t0 = performance.now();
    const res = wasm.classifyBatch(quads);
    const elapsed = performance.now() - t0;
    times.push(elapsed);
    if (iter === 0) {
      for (let i = 0; i < count; i++) {
        if (res[i * 5] > 0) validCount++;
      }
    }
  }

  times.sort((a, b) => a - b);
  const medianMs = times[Math.floor(times.length / 2)];
  const rate = count / (medianMs / 1000);
  console.log(`WASM batch ${count.toLocaleString().padStart(7)} quads: ${medianMs.toFixed(2)} ms (median of ${iters}) | ${(rate / 1000).toFixed(1)} kquads/s | valid: ${validCount}`);
}
