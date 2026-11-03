import fs from 'node:fs';
import { siftSeeds } from '../src/detect/sift.ts';
import * as wasm from '../detect-wasm.ts';
import { makeRefs, runDetection } from '../src/detect/detect.ts';
import { createRequire } from 'node:module';

const req = createRequire(import.meta.url);
const idx = process.argv[2] || '0';
const meta = JSON.parse(fs.readFileSync(`/tmp/attachments_data/meta_${idx}.json`, 'utf8'));
const g = fs.readFileSync(`/tmp/attachments_data/gray_${idx}.bin`);
const rgba = new Uint8Array(g.length * 4);
g.forEach((v, i) => (rgba[4 * i + 1] = v));

await wasm.init(fs.readFileSync('detect-wasm.wasm'));
wasm.setImage(g, meta.w, meta.h);
wasm.setRefs(Uint8Array.from(makeRefs(meta.ref).flat()));

let c = req('/tmp/grass-opencv.cjs');
let { cv }: any = await new Promise(r => (c.Mat ? r({ cv: c }) : c.then((x: any) => r({ cv: x }))));
globalThis.fetch = undefined as any;
const sift = await req('/tmp/grass-sift.cjs')();

console.log(`Starting detection on [${idx}] ${meta.name} (${meta.w}x${meta.h})...`);
const t0 = performance.now();
const r = runDetection(
  cv,
  { width: meta.w, height: meta.h, data: rgba },
  meta.ref,
  {},
  (msg: string) => console.log('  [det]', msg),
  {
    classifyBatch: wasm.classifyBatch,
    featureSeeds: (cv, ctx, log) => siftSeeds(sift, cv, ctx, log)
  }
);
const elapsed = ((performance.now() - t0) / 1000).toFixed(2);
console.log(`Detection on ${meta.name} finished in ${elapsed}s:`);
console.log(`  Seeds: ${r.seeds?.length || 0}`);
console.log(`  Blocks detected: ${r.blocks.length}`);
if (r.blocks.length > 0) {
  const heights = new Set(r.blocks.map(b => b.y));
  console.log(`  Height levels: ${heights.size} (${[...heights].sort((a,b)=>a-b).join(', ')})`);
}
fs.writeFileSync(`/tmp/attachments_data/formation_${idx}.json`, JSON.stringify(r.blocks, null, 2));

if (r.blocks.length > 0) {
  const lines = [r.blocks.length.toString()];
  for (const b of r.blocks) {
    lines.push(`${b.x} ${b.y} ${b.z} ${b.rotation} 0`);
  }
  fs.writeFileSync(`/Users/lucaszhang/grassblocks-cracker/metal/formation_att_${idx}.txt`, lines.join('\n') + '\n');
  console.log(`Saved metal/formation_att_${idx}.txt with ${r.blocks.length} blocks.`);
}
