import test from 'node:test';
import assert from 'node:assert/strict';
import {runStart,rangeBounds,normalizeRange,DEFAULT_RANGE,RANGES} from '../src/start-flow';
const blk=(x=0)=>({x,y:0,z:0,rotation:0,is_side:0,include:true});
function harness(detected:any[]=[blk(1),blk(2)]){
 const calls:any[]=[];
 const deps={
  detect:async(req:any)=>{calls.push(['detect',req]);return {blocks:detected};},
  search:async(b:any,bnd:any)=>{calls.push(['search',b,bnd]);return {matches:[]};},
  onStage:(s:string)=>calls.push(['stage',s]),
 };
 return {calls,deps};
}
const base=(h:any,o:any={})=>({mode:'auto' as const,hasImage:true,blocks:[],quad:[],range:DEFAULT_RANGE,y:{ymin:-64,ymax:320},deps:h.deps,isCancelled:()=>false,...o});
test('default range is 10K and ranges are the literal values',()=>{
 assert.equal(DEFAULT_RANGE,10000);assert.deepEqual([...RANGES],[2000,10000,100000]);
 assert.equal(normalizeRange('bogus'),10000);assert.equal(normalizeRange(-5),10000);assert.equal(normalizeRange('100000'),100000);
 assert.deepEqual(rangeBounds(2000,{ymin:-64,ymax:320}),{xmin:-2000,xmax:2000,ymin:-64,ymax:320,zmin:-2000,zmax:2000});
});
test('auto START detects with backend auto, then searches detected rows with selected range and unchanged y',async()=>{
 const h=harness();
 const r=await runStart(base(h,{range:100000}));
 assert.equal(r.status,'done');
 assert.deepEqual(h.calls.map(c=>c[0]),['stage','detect','stage','search']);
 assert.equal(h.calls[1][1].backend,'auto');assert.equal(h.calls[1][1].seeds,undefined);
 assert.deepEqual(h.calls[3][1].map((b:any)=>b.x),[1,2]);
 assert.deepEqual(h.calls[3][2],{xmin:-100000,xmax:100000,ymin:-64,ymax:320,zmin:-100000,zmax:100000});
});
test('no faces found never reaches the search stage',async()=>{
 const h=harness([]);
 const r=await runStart(base(h));
 assert.equal(r.status,'nofaces');assert.equal(r.stage,'detect');
 assert.ok(!h.calls.some(c=>c[0]==='search'));
});
test('detect error is reported as error at no silent wrong stage',async()=>{
 const h=harness();h.deps.detect=async()=>{throw Error('boom');};
 const r=await runStart(base(h));
 assert.equal(r.status,'error');assert.equal(r.message,'boom');assert.ok(!h.calls.some(c=>c[0]==='search'));
});
test('cancel during detect skips search; cancel flag after search reports cancelled',async()=>{
 let cancelled=false;const h=harness();
 h.deps.detect=async()=>{cancelled=true;return {blocks:[blk()]};};
 const r=await runStart(base(h,{isCancelled:()=>cancelled}));
 assert.equal(r.status,'cancelled');assert.ok(!h.calls.some(c=>c[0]==='search'));
});
test('manual quad is sent as exact seed copy and not mutated',async()=>{
 const quad=[[1,2],[3,4],[5,6],[7,8]];const copy=JSON.stringify(quad);const h=harness();
 const r=await runStart(base(h,{mode:'manual',quad}));
 assert.equal(r.status,'done');
 const seeds=h.calls.find(c=>c[0]==='detect')![1].seeds;
 assert.equal(JSON.stringify(seeds),JSON.stringify([quad]));assert.notEqual(seeds[0],quad);assert.equal(JSON.stringify(quad),copy);
});
test('manual without quad solves existing rows without detecting; empty asks for a quad',async()=>{
 const h=harness();const rows=[blk(9)];
 const r=await runStart(base(h,{mode:'manual',blocks:rows}));
 assert.equal(r.status,'done');assert.ok(!h.calls.some(c=>c[0]==='detect'));assert.equal(h.calls.find(c=>c[0]==='search')![1],rows);
 const e=await runStart(base(harness(),{mode:'manual'}));assert.equal(e.status,'noquad');
});
test('no image with imported rows solves directly',async()=>{
 const h=harness();const r=await runStart(base(h,{hasImage:false,blocks:[blk(3)]}));
 assert.equal(r.status,'done');assert.ok(!h.calls.some(c=>c[0]==='detect'));
});
