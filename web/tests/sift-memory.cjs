/* Run from repo root: node web/tests/sift-memory.cjs */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const Module = require('node:module');
const vendor = path.resolve(__dirname, '../public/vendor');
const file = path.join(vendor, 'grass-sift.js');
const js = new Module(file, module);js.filename=file;js.paths=module.paths;
js._compile(fs.readFileSync(file,'utf8'),file);
global.fetch=undefined;
(async()=>{
 const s=await js.exports({wasmBinary:fs.readFileSync(path.join(vendor,'grass-sift.wasm'))});
 function extract(w,h) {
  const p=s._malloc(w*h*4);assert.ok(p);
  try {
   // Deterministic textured input; refresh heap after every growing allocation.
   for(let i=0;i<w*h;i++)s.HEAPF32[p/4+i]=(Math.imul(i,1664525)>>>24);
   const n=s._extract(p,w,h);
   assert.ok(n>=0&&n<=20000,`${w}x${h} returned ${n}`);
   const a=s.HEAPF32.slice(s._get_out()/4,s._get_out()/4+n*130);
   assert.equal(a.length,n*130);assert.ok(a.every(Number.isFinite));
   console.log(`${w}x${h} features=${n} heap=${s.HEAPU8.length}`);
   return a;
  }finally{s._free(p);}
 }
 for(const [w,h] of [[1920,1080],[2560,1440],[3840,2160],[1921,1081],[2049,32],[0,32],[-1,32]]) {
  const p=s._malloc(16);assert.equal(s._extract(p,w,h),-1);s._free(p);
  console.log(`${w}x${h} safely rejected`);
 }
 assert.equal(s._extract(0,256,256),-1);
 const baseline=extract(256,256);
 extract(1024,1024);extract(1023,1025);extract(777,513);
 for(let i=0;i<3;i++)assert.deepEqual(extract(256,256),baseline);
 // Make VLFeat's third/fourth large allocation fail. Recovery must release all
 // earlier planes and leave the same module usable, not trap or corrupt it.
 const reserve=s._malloc(160*1024*1024),p=s._malloc(1024*1024*4);
 assert.ok(reserve&&p);assert.equal(s._extract(p,1024,1024),-2);
 s._free(p);s._free(reserve);
 assert.deepEqual(extract(256,256),baseline);
 assert.ok(s.HEAPU8.length<=536870912);
 console.log('PASS oversized, odd widths, max supported pixels, repeated calls, forced OOM recovery');
})().catch(e=>{console.error(e);process.exitCode=1});
