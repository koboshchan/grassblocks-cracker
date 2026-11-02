import assert from 'node:assert/strict';
import {prepareSiftImage,extractSiftFeatures,SIFT_MAX_PIXELS} from '../src/detect/sift.ts';
for(const [w,h] of [[1920,1080],[2560,1440],[3840,2160],[1921,1081],[257,129],[768,484]]) {
 const g=Float32Array.from({length:w*h},(_,i)=>i%251), p=prepareSiftImage(g,w,h);
 assert(p.width*p.height<=SIFT_MAX_PIXELS);assert(p.width<=2048&&p.height<=2048);
 let heap=new Float32Array(4*1024*1024),freed=0;
 const s:any={get HEAPF32(){return heap;},_malloc:()=>4,_extract:()=>{const fresh=new Float32Array(heap.length);fresh.set(heap);heap=fresh;heap[0]=10;heap[1]=12;return 1;},_get_out:()=>0,_free:()=>freed++};
 const f=extractSiftFeatures(s,g,w,h);assert.equal(freed,1);assert(Math.abs(f.a[0]-((10+.5)*w/p.width-.5))<1e-3);
 console.log(`${w}x${h} -> ${p.width}x${p.height}: bounded, fresh heap, source coordinates PASS`);
}
assert.throws(()=>prepareSiftImage(new Float32Array(11),4,4));
assert.throws(()=>prepareSiftImage(new Float32Array(64*64*3),64,64),/stride/);
console.log('invalid grayscale / RGB stride rejected PASS');
