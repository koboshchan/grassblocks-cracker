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
 const validation=await device.popErrorScope(),oom=await device.popErrorScope();
 if(thrown||validation||oom){owned.forEach(b=>b.destroy());if(validation)throw Error(validation.message);if(oom||(thrown instanceof Error&&(/out.of.memory|allocation/i.test(thrown.message)||thrown.name==='GPUOutOfMemoryError')))throw new EccAllocationError(oom?.message??String(thrown));throw thrown??Error(validation?.message??'ECC allocation failed');}
 return owned;
}
