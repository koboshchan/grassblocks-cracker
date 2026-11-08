// Every quad retains its own states/patch/Jacobian/reduction slots.
export const ECC_QUAD_BYTES={states:28*4,patch:81920*4,pixels:16384*5*4,jac:16384*8*4,sums:86*4,readback:(28+86)*4};
export const ECC_SCRATCH_BUDGET=48*1024*1024;
export function eccBatchCapacity(limits:Pick<GPUSupportedLimits,'maxStorageBufferBindingSize'|'maxBufferSize'|'maxComputeWorkgroupsPerDimension'>,budget=ECC_SCRATCH_BUDGET,requested=64):number{
 const bytes=Object.values(ECC_QUAD_BYTES).reduce((a,b)=>a+b,0);
 if(!Number.isFinite(budget)||budget<bytes||limits.maxComputeWorkgroupsPerDimension<256)return 0;
 const bound=Math.floor(Math.min(64,requested,budget/bytes,limits.maxStorageBufferBindingSize/ECC_QUAD_BYTES.jac,limits.maxBufferSize/ECC_QUAD_BYTES.jac,limits.maxComputeWorkgroupsPerDimension));
 return Math.max(0,bound);
}
