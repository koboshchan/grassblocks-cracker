import test from 'node:test';
import assert from 'node:assert/strict';
import {detect,classifyQuad,setPipeline,type PipelineModule} from '../src/detect/pipeline';
test('native pipeline validates/marshals only and exposes timing results',()=>{
 const mem=new ArrayBuffer(65536),calls:any[]=[],freed:number[]=[];let next=8,error=false;
 const result=JSON.stringify({blocks:[],trials:Array.from({length:9},()=>({tiles:0})),timingsMs:{total:1}});
 const fake:PipelineModule={HEAPU8:new Uint8Array(mem),HEAPF64:new Float64Array(mem),_malloc(n){const p=next;next+=Math.ceil(n/8)*8;return p;},_free(p){freed.push(p)},_grass_detect(...args){calls.push(args);return error?-1:0;},_grass_classify(...args){calls.push(args);return 0},_grass_result(){return 40000},UTF8ToString(){return error?' {"error":"native failure"}':result}};
 setPipeline(fake);const pixels=new Uint8Array(16*16*4),ref=Array.from({length:16},()=>Array.from({length:16},(_,i)=>i));
 assert.equal(detect(pixels,16,16,ref).trials.length,9);assert.equal(calls[0].at(-1),1);assert.equal(freed.length,4);
 const q=[[1,1],[10,1],[10,10],[1,10]];detect(pixels,16,16,ref,{},[q]);assert.equal(calls[1].at(-1),0);assert.equal(calls[1].at(-2),1);
 detect(pixels,16,16,ref,{accept:.84});assert.equal(calls[2].at(-1),0);
 classifyQuad(pixels,16,16,ref,q);assert.equal(freed.length,15);
 assert.throws(()=>detect(pixels,16,16,[]),/reference/);assert.throws(()=>detect(pixels,16,16,ref,{unknown:1}),/Unsupported/);assert.throws(()=>classifyQuad(pixels,16,16,ref,[[1,Infinity],[2,2],[3,3],[4,4]]),/Nonfinite/);
 error=true;assert.throws(()=>detect(pixels,16,16,ref),/native failure/);
});
