/** Run from repo root: cd web && node_modules/.bin/tsx tests/sift-real-large.ts */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import Module from 'node:module';
import {prepareSiftImage,extractSiftFeatures,SIFT_MAX_PIXELS} from '../src/detect/sift.ts';
const file=path.resolve('public/vendor/grass-sift.js');
const loader=new (Module as any)(file);loader.filename=file;
loader._compile(fs.readFileSync(file,'utf8'),file);
(globalThis as any).fetch=undefined;
const s=await loader.exports({wasmBinary:fs.readFileSync('public/vendor/grass-sift.wasm')});
// Same module for every fixture, exercising growing heaps and subsequent reuse.
for(const [w,h] of [[1920,1080],[2560,1440],[3840,2160],[1921,1081],[1920,1080]]) {
 const g=new Uint8Array(w*h);
 // Stable, feature-rich spatial texture with no external fixture dependency.
 for(let y=0;y<h;y++)for(let x=0;x<w;x++) {
  const cell=Math.imul(x>>4,374761393)^Math.imul(y>>4,668265263);
  g[y*w+x]=((cell^(cell>>>13))>>>16)&255;
 }
 const prepared=prepareSiftImage(g,w,h);
 assert.ok(prepared.width*prepared.height<=SIFT_MAX_PIXELS);
 assert.ok(prepared.width<=2048&&prepared.height<=2048);
 const actual=extractSiftFeatures(s,g,w,h);
 assert.ok(actual.n>0&&actual.n<=20000);
 assert.equal(actual.a.length,actual.n*130);
 assert.ok(actual.a.every(Number.isFinite));
 // Compare against direct REAL native extraction of the prepared pixels.
 // This verifies inverse mapping as well as the allocation/extraction path.
 const p=s._malloc(prepared.data.length*4);assert.ok(p);
 try {
  s.HEAPF32.set(prepared.data,p/4);
  const n=s._extract(p,prepared.width,prepared.height);
  assert.equal(n,actual.n);
  const a=s.HEAPF32.slice(s._get_out()/4,s._get_out()/4+n*130);
  for(let i=0;i<n;i++) {
   assert.ok(Math.abs(actual.a[i*130]-((a[i*130]+.5)*prepared.sx-.5))<.001);
   assert.ok(Math.abs(actual.a[i*130+1]-((a[i*130+1]+.5)*prepared.sy-.5))<.001);
   assert.ok(actual.a[i*130]>=0&&actual.a[i*130]<w);
   assert.ok(actual.a[i*130+1]>=0&&actual.a[i*130+1]<h);
   assert.deepEqual(actual.a.subarray(i*130+2,i*130+130),a.subarray(i*130+2,i*130+130));
  }
 }finally{s._free(p);}
 assert.ok(s.HEAPU8.length<=536870912);
 console.log(`${w}x${h} -> ${prepared.width}x${prepared.height}, ${actual.n} real features, mapped coordinates/descriptors PASS, heap ${s.HEAPU8.length}`);
}
console.log('PASS actual prepareSiftImage + extractSiftFeatures + compiled WASM, four large dimensions and repeated-call reuse');
