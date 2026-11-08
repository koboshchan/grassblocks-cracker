import test from 'node:test';
import assert from 'node:assert/strict';
import {runTrialPool,poolLimit,memoryBudget,type TrialClient} from '../src/detect/pool-scheduler';
const MiB=1024*1024;
const plans=Array.from({length:9},(_,index)=>({index,name:String(index),raw:[]}));
function harness({fail=-1,createFail=false,peak=32*MiB,readyHeap=32*MiB,hc=10,memory=8,coordinator=32*MiB,image=1024}={}){
 const submitted:number[]=[],progress:number[]=[],closed:number[]=[];let created=0,active=0,maxActive=0;
 return {submitted,progress,closed,get created(){return created},get maxActive(){return maxActive},hooks:{
  hardwareConcurrency:hc,deviceMemory:memory,coordinatorBytes:coordinator,imageBytes:image,
  submit(index:number){submitted.push(index)},progress(done:number){progress.push(done)},
  async create():Promise<TrialClient>{if(createFail)throw Error('allocation failed');const id=created++;let stop=false,pending:((e:Error)=>void)|null=null;
   return {heapBytes:readyHeap,close(){if(!stop){stop=true;closed.push(id);pending?.(Error('closed'))}},run(plan){
    active++;maxActive=Math.max(maxActive,active);
    return new Promise((resolve,reject)=>{
     pending=reject;setTimeout(()=>{active--;pending=null;if(stop)return reject(Error('closed'));if(plan.index===fail)return reject(Error('injected failure'));
      resolve({index:plan.index,snapshot:new Uint8Array([plan.index]),heapBytes:peak});
     },plan.index===0?0:(9-plan.index)*2);
    });
   }};
  },
 }};
}
test('reverse completion submits indexed opaque snapshots without arrival-order merge',async()=>{
 const h=harness(),result=await runTrialPool(plans,h.hooks);
 assert.deepEqual(h.submitted,[0,8,7,6,5,4,3,2,1]);
 assert.deepEqual(h.progress,[1,2,3,4,5,6,7,8,9]);
 assert.equal(result.fallback,'');assert.equal(result.completed,9);assert.equal(result.workers,9);
 assert.equal(h.closed.length,h.created);assert.equal(h.maxActive,8);
});
test('worker failure closes the whole pool and leaves missing fixed indices for native fallback',async()=>{
 const h=harness({fail:8}),result=await runTrialPool(plans,h.hooks);
 assert.match(result.fallback,/injected failure/);assert.equal(result.completed,1);
 assert.deepEqual(h.submitted,[0]);assert.equal(h.closed.length,h.created);
});
test('worker initialization failure is a serial prepared-proposal fallback',async()=>{
 const h=harness({createFail:true}),result=await runTrialPool(plans,h.hooks);
 assert.match(result.fallback,/allocation failed/);assert.equal(result.completed,0);
});
test('memory probe bounds expansion using measured heap rather than initial heap',async()=>{
 const h=harness({peak:320*MiB,memory:2}),result=await runTrialPool(plans,h.hooks);
 assert.equal(result.workers,1);assert.equal(h.maxActive,1);assert.equal(result.completed,9);
 assert.equal(result.peakHeapBytes,320*MiB);assert.equal(result.budgetBytes,512*MiB);
});
test('low core or low memory limits allocate no trial workers',async()=>{
 const h=harness({hc:1});const r=await runTrialPool(plans,h.hooks);
 assert.equal(h.created,0);assert.match(r.fallback,/resource/);
 assert.equal(poolLimit(8,500*MiB,32*MiB,8*MiB,512*MiB),0);
 assert.equal(poolLimit(4,32*MiB,32*MiB,1024,1536*MiB),3);
 assert.equal(poolLimit(NaN,0,32*MiB,0,768*MiB),0);
 assert.equal(memoryBudget(64),1536*MiB);assert.equal(memoryBudget(),768*MiB);
});
test('growth beyond budget preserves the probe result then finishes missing trials serially',async()=>{
 const h=harness({peak:512*MiB,memory:2}),r=await runTrialPool(plans,h.hooks);
 assert.equal(r.completed,1);assert.equal(h.created,1);assert.match(r.fallback,/resource/);
});

test('large coordinator does not contaminate measured low worker heaps; two fit unchanged budget',async()=>{
 const h=harness({hc:3,memory:4,coordinator:407633920,image:7339320,readyHeap:64*MiB,peak:256*MiB});
 const r=await runTrialPool(plans,h.hooks);
 assert.equal(r.budgetBytes,1024*MiB);assert.equal(r.coordinatorHeapBytes,407633920);
 assert.equal(r.workerPeakHeapBytes,256*MiB);assert.equal(r.peakHeapBytes,r.workerPeakHeapBytes);
 assert.equal(r.imageBytes,7339320);assert.equal(r.capacity,2);
 assert.equal(r.workers,2);assert.equal(r.maxSimultaneousTrials,2);assert.equal(h.maxActive,2);
 assert.equal(r.completed,9);assert.equal(r.fallback,'');
});
test('large actual worker measurement correctly restricts the same coordinator to one worker',async()=>{
 const h=harness({hc:3,memory:4,coordinator:407633920,image:7339320,readyHeap:64*MiB,peak:320*MiB});
 const r=await runTrialPool(plans,h.hooks);
 assert.equal(r.budgetBytes,1024*MiB);assert.equal(r.workerPeakHeapBytes,320*MiB);
 assert.equal(r.capacity,1);assert.equal(r.workers,1);assert.equal(r.maxSimultaneousTrials,1);
 assert.equal(r.completed,9);assert.equal(r.fallback,'');
});
