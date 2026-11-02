import assert from 'node:assert/strict';
import {extractSiftFeatures,prepareSiftImage,SIFT_MAX_PIXELS} from '../src/detect/sift.ts';
// Mock SIFT module records exactly what the native extractor is asked to process.
function mock(){const heap=new Float32Array(1<<22);let top=16,calls:any[]=[];
 return {calls,HEAPF32:heap,_malloc:(b:number)=>{const p=top*4;top+=Math.ceil(b/4)+4;if(top>heap.length)throw Error('mock heap exhausted: unbounded allocation '+b);return p;},_free(){},
 _extract:(_p:number,w:number,h:number)=>{calls.push({w,h});return 0;},_get_out:()=>64};}
for(const [w,h] of [[1920,1080],[2560,1440],[3840,2160],[3841,2161],[1921,1081],[4095,4095],[9,4000]]){
 const m=mock();const g=new Uint8Array(w*h);const r=extractSiftFeatures(m,g,w,h);const c=m.calls[0];
 assert.equal(r.n,0);assert.ok(c.w*c.h<=SIFT_MAX_PIXELS,w+'x'+h+' -> '+c.w+'x'+c.h);assert.ok(c.w>=8&&c.h>=8);
 const p=prepareSiftImage(g,w,h);assert.ok(Math.abs(p.sx*c.w-w)<1e-6&&Math.abs(p.sy*c.h-h)<1e-6,'scale factors map back');
 console.log('ok',w+'x'+h,'->',c.w+'x'+c.h);}
// malformed buffers rejected with a clear message
for(const [len,w,h] of [[10,8,8],[63,8,8],[100,10,10+1],[64,7,9]]) assert.throws(()=>prepareSiftImage(new Uint8Array(len),w,h),/Invalid SIFT/);
assert.throws(()=>prepareSiftImage(new Uint8Array(1),4097*4097,1),/Invalid SIFT/);
// native traps propagate (caller falls back) and memory is freed
{let freed=0;const m:any=mock();m._free=()=>freed++;m._extract=()=>{throw new WebAssembly.RuntimeError('memory access out of bounds');};
 assert.throws(()=>extractSiftFeatures(m,new Uint8Array(64*64),64,64),/out of bounds/);assert.equal(freed,1);}
console.log('SIFT bounds tests passed');
