import { Image } from 'node:fs';
import fs from 'node:fs';
import { siftSeeds } from '../src/detect/sift.ts';
import * as wasm from '../detect-wasm.ts';
import { makeRefs, runDetection } from '../src/detect/detect.ts';
import { createRequire } from 'node:module';

const req = createRequire(import.meta.url);
const imagePath = process.argv[2];
if (!imagePath) {
  console.error('Usage: node script.mjs <imagePath>');
  process.exit(1);
}

// Read fixture metadata prepared by python
const f = JSON.parse(fs.readFileSync('/tmp/custom-fixtures.json', 'utf8'));
const g = fs.readFileSync('/tmp/custom-gray.bin');
const rgba = new Uint8Array(g.length * 4);
g.forEach((v, i) => (rgba[4 * i + 1] = v));

// Initialize WASM classifier
await wasm.init(fs.readFileSync('detect-wasm.wasm'));
wasm.setImage(g, f.w, f.h);
wasm.setRefs(Uint8Array.from(makeRefs(f.ref).flat()));

// Initialize OpenCV & SIFT
let c = req(process.env.GRASS_CV_CJS || '/tmp/grass-opencv.cjs');
let { cv }: any = await new Promise(r => (c.Mat ? r({ cv: c }) : c.then((x: any) => r({ cv: x }))));
globalThis.fetch = undefined as any;
const sift = await req(process.env.GRASS_SIFT_CJS || '/tmp/grass-sift.cjs')();

console.log(`Starting detection on ${imagePath} (${f.w}x${f.h})...`);
const t0 = performance.now();
const r = runDetection(
  cv,
  { width: f.w, height: f.h, data: rgba },
  f.ref,
  {},
  (msg: string) => console.log('  [det]', msg),
  {
    classifyBatch: wasm.classifyBatch,
    featureSeeds: (cv, ctx, log) => siftSeeds(sift, cv, ctx, log)
  }
);
const elapsed = ((performance.now() - t0) / 1000).toFixed(2);
console.log(`Detection finished in ${elapsed}s:`);
console.log(`  Seeds: ${r.seeds?.length || 0}`);
console.log(`  Blocks detected: ${r.blocks.length}`);
if (r.blocks.length > 0) {
  const heights = new Set(r.blocks.map(b => b.y));
  console.log(`  Height levels: ${heights.size} (${[...heights].sort((a,b)=>a-b).join(', ')})`);
  console.log(`  Sample blocks:`, r.blocks.slice(0, 5));
}
fs.writeFileSync('/tmp/detected-formation.json', JSON.stringify(r.blocks, null, 2));
