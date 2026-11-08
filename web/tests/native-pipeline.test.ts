import test from 'node:test';
import assert from 'node:assert/strict';
import {detect,classifyQuad,featureSeeds,setPipeline,type PipelineModule} from '../src/detect/pipeline';
test('native pipeline validates/marshals only and exposes timing results',()=>{
 const mem=new ArrayBuffer(65536),calls:any[]=[],freed:number[]=[];let next=8,error=false;
 const result=JSON.stringify({blocks:[],trials:Array.from({length:9},()=>({tiles:0})),timingsMs:{total:1}});
 const fake:PipelineModule={HEAPU8:new Uint8Array(mem),HEAPF64:new Float64Array(mem),_malloc(n){const p=next;next+=Math.ceil(n/8)*8;return p;},_free(p){freed.push(p)},_grass_detect(...args){calls.push(args);return error?-1:0;},_grass_feature_seeds(...args){calls.push(args);return 0},_grass_classify(...args){calls.push(args);return 0},_grass_result(){return 40000},UTF8ToString(){return error?' {"error":"native failure"}':result}};
 setPipeline(fake);const pixels=new Uint8Array(16*16*4),ref=Array.from({length:16},()=>Array.from({length:16},(_,i)=>i));
 assert.equal(detect(pixels,16,16,ref).trials.length,9);assert.equal(calls[0].at(-1),1);assert.equal(freed.length,4);
 const q=[[1,1],[10,1],[10,10],[1,10]];detect(pixels,16,16,ref,{},[q]);assert.equal(calls[1].at(-1),0);assert.equal(calls[1].at(-2),1);
 detect(pixels,16,16,ref,{accept:.84});assert.equal(calls[2].at(-1),0);
 classifyQuad(pixels,16,16,ref,q);assert.equal(freed.length,15);
 assert.throws(()=>detect(pixels,16,16,[]),/reference/);assert.throws(()=>detect(pixels,16,16,ref,{unknown:1}),/Unsupported/);assert.throws(()=>classifyQuad(pixels,16,16,ref,[[1,Infinity],[2,2],[3,3],[4,4]]),/Nonfinite/);
 detect(pixels,16,16,ref,{refineMin:-1,refineMargin:-.5,refineVotes:3.5});
 const optionsPtr=calls.at(-1)[4]/8;assert.equal(fake.HEAPF64[optionsPtr+4],-1);assert.equal(fake.HEAPF64[optionsPtr+5],-.5);assert.equal(fake.HEAPF64[optionsPtr+8],3.5);assert.equal(fake.HEAPF64[optionsPtr+11],3);
 detect(pixels,16,16,ref);assert.equal(fake.HEAPF64[calls.at(-1)[4]/8+11],0);
 error=true;assert.throws(()=>detect(pixels,16,16,ref),/native failure/);
});

test('feature seed transport returns proposals and frees inputs on success and failure',()=>{
 const mem=new ArrayBuffer(65536),freed:number[]=[],calls:number[][]=[];let next=8,error=false;
 const seeds=[[[1,2],[12,2],[12,13],[1,13]]],timingsMs={featureExtraction:2,matchingHomography:3,total:6};
 const fake:PipelineModule={
  HEAPU8:new Uint8Array(mem),HEAPF64:new Float64Array(mem),
  _malloc(n){const p=next;next+=Math.ceil(n/8)*8;return p},_free(p){freed.push(p)},
  _grass_feature_seeds(...args){calls.push(args);return error?-1:0},
  _grass_detect(){throw Error('must not run full native detection')},
  _grass_classify(){throw Error('must not classify seed proposals')},
  _grass_result(){return 40000},
  UTF8ToString(){return JSON.stringify(error?{error:'feature failure'}:{seeds,featureBackend:'SIFT',timingsMs})}
 };
 setPipeline(fake);
 const pixels=new Uint8Array(16*16*4),ref=Array.from({length:16},()=>Array(16).fill(128));
 const proposals=featureSeeds(pixels,16,16,ref,{siftRatio:.76});
 assert.deepEqual(proposals.seeds,seeds);assert.deepEqual(proposals.timingsMs,timingsMs);assert.equal(proposals.featureBackend,'SIFT');
 assert.equal(calls.length,1);assert.equal(freed.length,3);
 assert.equal(fake.HEAPF64[calls[0][4]/8+7],.76);
 assert.deepEqual(Array.from(fake.HEAPU8.slice(calls[0][0],calls[0][0]+pixels.length)),Array.from(pixels));
 error=true;assert.throws(()=>featureSeeds(pixels,16,16,ref),/feature failure/);
 assert.equal(freed.length,6);
 assert.throws(()=>featureSeeds(pixels,16,16,ref,{backend:'wasm'}),/Unsupported/);
});
