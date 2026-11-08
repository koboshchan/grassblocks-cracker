import test from 'node:test';
import assert from 'node:assert/strict';
import {GPU_TRIALS,mergeGpuAligned,topDownSizes} from '../src/detect/gpu-trials';
const b=(x:number,rot=0):any=>({x,y:0,z:0,rotation:rot,quad:[[x*10,0],[x*10+8,0],[x*10+8,8],[x*10,8]],score:1,margin:1,votes:4});
test('fixed nine configurations match native option order',()=>{
 assert.deepEqual(GPU_TRIALS.map(x=>x[0]),['original','relaxed','refine','refit-refine','ratio','combo','flat-ratio','rescue','ratio80']);
 assert.equal((GPU_TRIALS[8][1] as any).siftRatio,.80);
});
test('anchored union requires three consistent pixel matches and preserves first metadata',()=>{
 const a=[b(0),b(1),b(2)];const incoming=[...a,b(3)].map(x=>({...x,x:x.x-9}));
 assert.deepEqual(mergeGpuAligned(a,incoming).map(x=>x.x),[0,1,2,3]);
 assert.equal(mergeGpuAligned(a,incoming)[0],a[0]);
 assert.equal(mergeGpuAligned(a,incoming.slice(0,2)),a);
 assert.equal(mergeGpuAligned(a,incoming.map(x=>({...x,rotation:1}))),a);
});
test('top-down constant image size proposals deterministic and bounded',()=>{
 const a=topDownSizes(new Uint8Array(128*128*4),128,128);
 assert.deepEqual(a,topDownSizes(new Uint8Array(128*128*4),128,128));
 assert.ok(a.every(x=>x>=48&&x<128*.6));assert.equal(new Set(a).size,a.length);
});
