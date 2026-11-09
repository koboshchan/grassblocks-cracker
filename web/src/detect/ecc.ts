import {eccBatchCapacity,ECC_QUAD_BYTES,allocateEccBuffers} from './ecc-batch';
import {eccShader} from './ecc-shader';
const entries=['prepare','gaussian','gradients','warp_pixels','statistics','jacobian','normal_equations','update'] as const;
export interface EccHostTiming {count:number;passes:number;dispatches:number;submissions:number;readbacks:number;encodeMs:number;submitToReadbackMs:number;wallMs:number;scratchBytes:number;gpuTimestampMs:number|null;reusedBuffers:boolean}
export interface EccTrace {base:number[];warp:number[];composed:number[];rho:number;iterations:number;status:number;rotation:number;quad:number[][];lastRho:number;rhoChange:number;converged:boolean;iterationLimitReached:boolean;termination:string;maskCount:number;firstMaskCount:number;minMaskCount:number;maxMaskCount:number}
/** No native calls. JS handles allocations and matrix input/output transport only. */
export class BatchedGpuEcc {
 readonly capacity:number;
 lastHostTiming:EccHostTiming|null=null;
 private scratchSlots:{capacity:number;buffers:GPUBuffer[];busy:boolean}[]=[];
 private querySet?:GPUQuerySet;
 private resolveBuffer?:GPUBuffer;
 private queryReadbackBuffer?:GPUBuffer;
 private freeQueryReadbacks:GPUBuffer[]=[];
 private allQueryReadbacks:GPUBuffer[]=[];
 private constructor(private device:GPUDevice,private pipelines:GPUComputePipeline[],querySet?:GPUQuerySet,resolveBuffer?:GPUBuffer,queryReadbackBuffer?:GPUBuffer){
  this.capacity=eccBatchCapacity(device.limits);if(this.capacity<1)throw Error("ECC device limits exceed conservative scratch budget");
  this.querySet=querySet;this.resolveBuffer=resolveBuffer;this.queryReadbackBuffer=queryReadbackBuffer;
  if(!this.querySet&&device.features?.has('timestamp-query')){
   this.querySet=device.createQuerySet({type:'timestamp',count:2});
   this.resolveBuffer=device.createBuffer({size:16,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC});
   this.queryReadbackBuffer=device.createBuffer({size:16,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
  }else if(this.querySet&&this.resolveBuffer&&!this.queryReadbackBuffer){
   this.queryReadbackBuffer=device.createBuffer({size:16,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
  }
  if(this.queryReadbackBuffer){this.allQueryReadbacks.push(this.queryReadbackBuffer);this.freeQueryReadbacks.push(this.queryReadbackBuffer);}
 }
 // Each in-flight run needs its own timestamp staging buffer; a buffer with a pending mapAsync cannot be submitted.
 private acquireQueryReadback(){
  if(!this.querySet||!this.resolveBuffer)return undefined;
  let b=this.freeQueryReadbacks.pop();
  if(!b){b=this.device.createBuffer({size:16,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});this.allQueryReadbacks.push(b);}
  return b;
 }
 static async create(device:GPUDevice){
  const shader=device.createShaderModule({code:eccShader});const info=await shader.getCompilationInfo();
  const errors=info.messages.filter(m=>m.type==='error');if(errors.length)throw Error(errors.map(m=>`${m.lineNum} ${m.message}`).join('\n'));
  const layout=device.createBindGroupLayout({entries:[
   {binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform'}},
   {binding:1,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:'float'}},
   ...[2,3,4,5,6,7].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:binding===2?'read-only-storage' as const:'storage' as const}}))]});
  const pl=device.createPipelineLayout({bindGroupLayouts:[layout]});
  const pipelines=[];for(const entryPoint of entries)pipelines.push(await device.createComputePipelineAsync({layout:pl,compute:{module:shader,entryPoint}}));
  let querySet:GPUQuerySet|undefined;let resolveBuffer:GPUBuffer|undefined;let queryReadbackBuffer:GPUBuffer|undefined;
  if(device.features?.has('timestamp-query')){
   querySet=device.createQuerySet({type:'timestamp',count:2});
   resolveBuffer=device.createBuffer({size:16,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC});
   queryReadbackBuffer=device.createBuffer({size:16,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
  }
  return new BatchedGpuEcc(device,pipelines,querySet,resolveBuffer,queryReadbackBuffer);
 }
 async run(image:GPUTextureView,refs:GPUBuffer,width:number,height:number,quads:number[][][],rotations:number[],diagnosticInitialWarp?:number[],onTiming?:(timing:EccHostTiming)=>void):Promise<EccTrace[]>{
  if(quads.length>this.capacity||quads.length!==rotations.length||quads.some(q=>q.length!==4||q.some(p=>p.length!==2||p.some(v=>!Number.isFinite(v))))||rotations.some(r=>!Number.isInteger(r)||r<0||r>3))throw Error('ECC component bounds/inputs');
  if(diagnosticInitialWarp&&(diagnosticInitialWarp.length!==9||diagnosticInitialWarp.some(v=>!Number.isFinite(v))||diagnosticInitialWarp[8]!==1))throw Error('Diagnostic initial warp invalid');
  const wallStart=performance.now();
  const count=quads.length;if(!count)return [];
  const initial=new Float32Array(count*28);
  quads.forEach((q,i)=>{const base=quadBase(q);const s=i*28;initial.set(base,s);initial.set(diagnosticInitialWarp??[1,0,0,0,1,0,0,0,1],s+9);initial[s+18]=-1;initial[s+19]=-.00001;initial[s+20]=base.every(Number.isFinite)?0:-2;initial[s+22]=rotations[i];});
  const rw=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC;
  const slotSize = Math.max(1, Math.min(512, Math.floor(this.capacity / Math.min(2, this.capacity))));
  let slot=count<=slotSize?this.scratchSlots.find(s=>!s.busy&&s.capacity>=count):undefined;
  let reusedBuffers=!!slot;
  if(slot)slot.busy=true;
  else if(count<=slotSize&&this.scratchSlots.length<Math.min(2,this.capacity)){
   slot={capacity:slotSize,buffers:[],busy:true};this.scratchSlots.push(slot);
  }
  let owned:GPUBuffer[];
  try{
   if(slot?.buffers.length)owned=slot.buffers;
   else{
    const n=slot?.capacity??count;
    const allocationSizes=[16,n*28*4,n*81920*4,n*16384*5*4,n*16384*8*4,n*86*4,n*(28+86)*4];
    owned=await allocateEccBuffers(this.device,allocationSizes.map((size,i)=>({size,usage:i===0?GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST:i===6?GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST:rw})));
    if(slot)slot.buffers=owned;
   }
  }catch(e){if(slot){this.scratchSlots.splice(this.scratchSlots.indexOf(slot),1);}throw e;}
  const [params,states,patch,pixels,jac,sums,read]=owned;let mapped=false;let queryMapped=false;const qrb=this.acquireQueryReadback();
  try{
   this.device.queue.writeBuffer(params,0,new Uint32Array([width,height,count,0]));this.device.queue.writeBuffer(states,0,initial);
   const bind=this.device.createBindGroup({layout:this.pipelines[0].getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:params}},{binding:1,resource:image},{binding:2,resource:{buffer:refs}},...[states,patch,pixels,jac,sums].map((b,i)=>({binding:i+3,resource:{buffer:b}}))]});
   const encodeStart=performance.now();
   const encoder=this.device.createCommandEncoder();
   // Fresh WebGPU buffers start zeroed. Match that state on every reuse,
   // including failed quads whose later kernels intentionally skip writes.
   for(const buffer of [patch,pixels,jac,sums])encoder.clearBuffer(buffer);
   // WebGPU defines one usage/synchronization scope per compute dispatch,
   // even within a single pass. Preserve every dispatch and storage dependency
   // without reopening Metal compute encoders 303 times.
   const hasTimestamps=Boolean(this.querySet&&this.resolveBuffer&&qrb);
   const compute=hasTimestamps?encoder.beginComputePass({timestampWrites:{querySet:this.querySet!,beginningOfPassWriteIndex:0,endOfPassWriteIndex:1}}):encoder.beginComputePass();
   compute.setBindGroup(0,bind);
   const pass=(index:number,x:number,y=1)=>{compute.setPipeline(this.pipelines[index]);compute.dispatchWorkgroups(x,y);};
   pass(0,256,count);pass(1,256,count);pass(2,256,count);
   for(let iteration=0;iteration<60;iteration++){pass(3,256,count);pass(4,count,1);pass(5,256,count);pass(6,count,1);pass(7,count);}
   compute.end();
   if(hasTimestamps){
    encoder.resolveQuerySet(this.querySet!,0,2,this.resolveBuffer!,0);
    encoder.copyBufferToBuffer(this.resolveBuffer!,0,qrb!,0,16);
   }
   encoder.copyBufferToBuffer(states,0,read,0,initial.byteLength);encoder.copyBufferToBuffer(sums,0,read,initial.byteLength,count*86*4);const command=encoder.finish(),encoded=performance.now();this.device.queue.submit([command]);
   let gpuTimestampMs:number|null=null;
   if(hasTimestamps){
    await Promise.all([read.mapAsync(GPUMapMode.READ),qrb!.mapAsync(GPUMapMode.READ)]);
    mapped=true;queryMapped=true;
    const ts=new BigInt64Array(qrb!.getMappedRange());
    const start=ts[0],end=ts[1];
    gpuTimestampMs=Number(end-start)/1000000;
   }else{
    await read.mapAsync(GPUMapMode.READ);mapped=true;
   }
   this.lastHostTiming={count,passes:1,dispatches:303,submissions:1,readbacks:1,encodeMs:encoded-encodeStart,submitToReadbackMs:performance.now()-encoded,wallMs:performance.now()-wallStart,scratchBytes:count*Object.values(ECC_QUAD_BYTES).reduce((a,b)=>a+b,0)+16,gpuTimestampMs,reusedBuffers};
   onTiming?.(this.lastHostTiming);
   const raw=new Float32Array(read.getMappedRange()).slice();
   return quads.map((q,i)=>{const s=i*28,base=Array.from(raw.slice(s,s+9)),warp=Array.from(raw.slice(s+9,s+18));const composed=multiply(base,warp);const candidate=[[0,0],[128,0],[128,128],[0,128]].map(([x,y])=>{const d=composed[6]*x+composed[7]*y+composed[8];return [(composed[0]*x+composed[1]*y+composed[2])/d,(composed[3]*x+composed[4]*y+composed[5])/d];});return {base,warp,composed,rho:raw[s+18],iterations:raw[s+21],status:raw[s+20],lastRho:raw[s+19],rhoChange:Math.abs(raw[s+18]-raw[s+19]),converged:raw[s+23]===1,iterationLimitReached:raw[s+23]===2,termination:raw[s+20]<0?'failure':raw[s+23]===1?'epsilon':raw[s+23]===2?'iteration-limit':'unfinished',maskCount:raw[s+25],firstMaskCount:raw[s+24],minMaskCount:raw[s+26],maxMaskCount:raw[s+27],rotation:rotations[i],quad:raw[s+20]>0?candidate:q};});
  }finally{if(mapped)read.unmap();if(queryMapped)qrb?.unmap();if(qrb)this.freeQueryReadbacks.push(qrb);if(slot)slot.busy=false;else owned.forEach(b=>b.destroy());}
 }
 destroy(){
  for(const slot of this.scratchSlots)slot.buffers.forEach(b=>b.destroy());
  this.scratchSlots=[];
  this.querySet?.destroy();
  this.resolveBuffer?.destroy();
  this.allQueryReadbacks.forEach(b=>b.destroy());this.allQueryReadbacks=[];this.freeQueryReadbacks=[];
  this.querySet=undefined;
  this.resolveBuffer=undefined;
  this.queryReadbackBuffer=undefined;
 }
}
// Transport math for the canonical 128x128 patch, not an alternative refiner.
export function quadBase(q:number[][]){const [a,b,c,d]=q;const dx1=b[0]-c[0],dx2=d[0]-c[0],dx3=a[0]-b[0]+c[0]-d[0],dy1=b[1]-c[1],dy2=d[1]-c[1],dy3=a[1]-b[1]+c[1]-d[1];const det=dx1*dy2-dx2*dy1;const g=(dx3*dy2-dx2*dy3)/det/128,h=(dx1*dy3-dx3*dy1)/det/128;return [(b[0]-a[0])/128+g*b[0],(d[0]-a[0])/128+h*d[0],a[0],(b[1]-a[1])/128+g*b[1],(d[1]-a[1])/128+h*d[1],a[1],g,h,1];}
function multiply(a:number[],b:number[]){return Array.from({length:9},(_,i)=>[0,1,2].reduce((s,k)=>s+a[Math.floor(i/3)*3+k]*b[k*3+i%3],0));}
