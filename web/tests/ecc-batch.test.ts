import test from 'node:test';import assert from 'node:assert/strict';import {eccBatchCapacity,eccScratchBudget,ECC_QUAD_BYTES,EccAllocationError,withEccAllocationRetry,allocateEccBuffers,eccThroughputPolicy,runEccWindow} from '../src/detect/ecc-batch';
const normal={maxStorageBufferBindingSize:128*1024*1024,maxBufferSize:256*1024*1024,maxComputeWorkgroupsPerDimension:65535};
test('device-derived budget and per-binding limits scale beyond42',()=>{
 assert.equal(eccScratchBudget(normal),64*1024*1024);assert.equal(eccBatchCapacity(normal),56);
 const mac={...normal,maxStorageBufferBindingSize:1024**3,maxBufferSize:1024**3};assert.equal(eccBatchCapacity(mac),454);
 for(const n of [1,9,17,42,53])assert.equal(eccBatchCapacity(normal,undefined,n),n);
 assert.equal(eccBatchCapacity(normal,undefined,999),56);
 assert.equal(eccBatchCapacity({...normal,maxStorageBufferBindingSize:ECC_QUAD_BYTES.jac*16},1024**3),16);
 assert.equal(eccBatchCapacity({...normal,maxBufferSize:ECC_QUAD_BYTES.jac*8},1024**3),8);
 assert.equal(eccBatchCapacity({...normal,maxComputeWorkgroupsPerDimension:256},1024**3),256);
 for(const x of [NaN,Infinity,-1,0,1.5,Number.MAX_SAFE_INTEGER+1])assert.equal(eccBatchCapacity({...normal,maxBufferSize:x}),0);
 assert.equal(eccBatchCapacity({...normal,maxComputeWorkgroupsPerDimension:128}),0);
 assert.equal(eccBatchCapacity(normal,NaN),0);assert.equal(eccBatchCapacity(normal,1),0);
});
test('allocation OOM halves in place, preserves result order and stops at one',async()=>{
 const attempts:number[]=[],fallbacks:number[][]=[];
 const r=await withEccAllocationRetry(53,454,async n=>{attempts.push(n);if(n>13)throw new EccAllocationError('oom');return Array.from({length:n},(_,i)=>i);},(a,b)=>fallbacks.push([a,b]));
 assert.deepEqual(attempts,[53,26,13]);assert.deepEqual(fallbacks,[[53,26],[26,13]]);assert.equal(r.count,13);assert.deepEqual(r.value,Array.from({length:13},(_,i)=>i));
 await assert.rejects(withEccAllocationRetry(1,1,async()=>{throw new EccAllocationError('oom');},()=>assert.fail()),/oom/);
 await assert.rejects(withEccAllocationRetry(4,4,async()=>{throw Error('validation');},()=>assert.fail()),/validation/);
});
test('async OOM and validation scopes clean partial allocations, validation is not retried',async()=>{
 for(const oom of [false,true]){let destroyed=0;const scopes:string[]=[];let pops=0;
 const d={pushErrorScope:(s:string)=>scopes.push(s),createBuffer:()=>({destroy:()=>destroyed++}),popErrorScope:async()=>{pops++;return pops===(oom?2:1)?{message:oom?'oom':'validation'}:null;}} as unknown as GPUDevice;
 await assert.rejects(allocateEccBuffers(d,[{size:4,usage:1},{size:8,usage:1}]),e=>oom?e instanceof EccAllocationError:!(e instanceof EccAllocationError));assert.equal(destroyed,2);assert.deepEqual(scopes,['out-of-memory','validation']);assert.equal(pops,2);
 }
});

test('throughput working sets share device budget and preserve reverse completion order',async()=>{
 for(const cap of [1,2,3,8,15,16,56,454]){const p=eccThroughputPolicy(cap);assert.ok(p.batchSize*p.inFlight<=cap);assert.ok(p.batchSize>=1&&p.batchSize<=32);}
 const pending:{base:number;n:number;resolve:(v:number[])=>void}[]=[];let active=0,peak=0;
 const run=runEccWindow(100,454,(base,n)=>new Promise<number[]>(resolve=>{active++;peak=Math.max(peak,active);pending.push({base,n,resolve:v=>{active--;resolve(v)}})}));
 while(active||pending.length){const p=pending.pop()!;p.resolve(Array.from({length:p.n},(_,i)=>p.base+i));await new Promise(r=>setImmediate(r));}
 assert.deepEqual(await run,Array.from({length:100},(_,i)=>i));assert.equal(peak,2);
 await assert.rejects(runEccWindow(9,16,async()=>{throw Error('GPU failure');}),/GPU failure/);
});
