// Limits describe legal allocations, not available VRAM. Reserve half of the
// smaller buffer/binding envelope for ECC scratch; images/classifier need room.
export const ECC_QUAD_BYTES={states:28*4,patch:81920*4,pixels:16384*5*4,jac:16384*8*4,sums:86*4,readback:(28+86)*4};
type Limits=Pick<GPUSupportedLimits,'maxStorageBufferBindingSize'|'maxBufferSize'|'maxComputeWorkgroupsPerDimension'>;
export const eccScratchBudget=(l:Limits)=>Math.floor(Math.min(l.maxStorageBufferBindingSize,l.maxBufferSize)/2);
export function eccBatchCapacity(l:Limits,budget=eccScratchBudget(l),requested=Number.MAX_SAFE_INTEGER):number{
 const bytes=Object.values(ECC_QUAD_BYTES).reduce((a,b)=>a+b,0);
 if(![budget,requested,l.maxStorageBufferBindingSize,l.maxBufferSize,l.maxComputeWorkgroupsPerDimension].every(v=>Number.isSafeInteger(v)&&v>=0)||budget<bytes+16||l.maxComputeWorkgroupsPerDimension<256)return 0;
 const storage=Math.max(ECC_QUAD_BYTES.states,ECC_QUAD_BYTES.patch,ECC_QUAD_BYTES.pixels,ECC_QUAD_BYTES.jac,ECC_QUAD_BYTES.sums);
 const buffer=Math.max(storage,ECC_QUAD_BYTES.readback);
 return Math.max(0,Math.floor(Math.min(requested,(budget-16)/bytes,l.maxStorageBufferBindingSize/storage,l.maxBufferSize/buffer,l.maxComputeWorkgroupsPerDimension)));
}
export class EccAllocationError extends Error {}
export async function withEccAllocationRetry<T>(pending:number,capacity:number,run:(n:number)=>Promise<T>,onRetry:(from:number,to:number)=>void):Promise<{count:number,value:T}>{
 let n=Math.min(pending,capacity);if(n<1)throw new EccAllocationError('ECC cannot allocate even one quad');
 for(;;){try{return {count:n,value:await run(n)};}catch(e){if(!(e instanceof EccAllocationError)||n===1)throw e;const next=Math.max(1,Math.floor(n/2));onRetry(n,next);n=next;}}
}
export async function allocateEccBuffers(device:GPUDevice,descriptors:GPUBufferDescriptor[]):Promise<GPUBuffer[]>{
 const owned:GPUBuffer[]=[];let thrown:unknown;
 device.pushErrorScope('out-of-memory');device.pushErrorScope('validation');
 try{for(const d of descriptors)owned.push(device.createBuffer(d));}catch(e){thrown=e;}
 // Pop both scopes synchronously before yielding; concurrent batches must not
 // accidentally pop each other's device-global scope stack.
 const validationResult=device.popErrorScope(),oomResult=device.popErrorScope();
 const [validation,oom]=await Promise.all([validationResult,oomResult]);
 if(thrown||validation||oom){owned.forEach(b=>b.destroy());if(validation)throw Error(validation.message);if(oom||(thrown instanceof Error&&(/out.of.memory|allocation/i.test(thrown.message)||thrown.name==='GPUOutOfMemoryError')))throw new EccAllocationError(oom?.message??String(thrown));throw thrown??Error(validation?.message??'ECC allocation failed');}
 return owned;
}

// Capacity is a legal memory bound, not a throughput recommendation. Keep the
// measured pre-regression eight-quad working set and overlap two independent
// batches. The combined scratch allocation never exceeds the device budget.
export function eccThroughputPolicy(capacity:number){
 if(!Number.isSafeInteger(capacity)||capacity<1)throw Error('Invalid ECC capacity');
 const inFlight=Math.min(2,capacity);
 const batchSize=Math.max(1,Math.min(32,Math.floor(capacity/inFlight)));
 return {inFlight,batchSize};
}
export async function runEccWindow<T>(count:number,capacity:number,run:(base:number,n:number)=>Promise<T[]>):Promise<T[]>{
 if(!Number.isSafeInteger(count)||count<0)throw Error('Invalid ECC pending count');
 const {inFlight,batchSize}=eccThroughputPolicy(capacity),out=new Array<T>(count);
 let cursor=0,failure:unknown;
 const lane=async()=>{while(cursor<count&&!failure){const base=cursor,n=Math.min(batchSize,count-base);cursor+=n;
  try{const value=await run(base,n);if(value.length!==n)throw Error('Invalid ECC batch length');for(let i=0;i<n;i++)out[base+i]=value[i];}catch(e){failure=e;}
 }};
 await Promise.all(Array.from({length:Math.min(inFlight,Math.ceil(count/batchSize))},lane));
 if(failure)throw failure;return out;
}
