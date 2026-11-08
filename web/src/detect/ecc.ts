import {eccShader} from './ecc-shader';
const entries=['prepare','gaussian','gradients','warp_pixels','statistics','jacobian','normal_equations','update'] as const;
export interface EccTrace {base:number[];warp:number[];composed:number[];rho:number;iterations:number;status:number;rotation:number;quad:number[][];lastRho:number;rhoChange:number;converged:boolean;iterationLimitReached:boolean;termination:string;maskCount:number;firstMaskCount:number;minMaskCount:number;maxMaskCount:number}
/** No native calls. JS handles allocations and matrix input/output transport only. */
export class BatchedGpuEcc {
 private constructor(private device:GPUDevice,private pipelines:GPUComputePipeline[]){}
 static async create(device:GPUDevice){
  const shader=device.createShaderModule({code:eccShader});const info=await shader.getCompilationInfo();
  const errors=info.messages.filter(m=>m.type==='error');if(errors.length)throw Error(errors.map(m=>`${m.lineNum} ${m.message}`).join('\n'));
  const layout=device.createBindGroupLayout({entries:[
   {binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform'}},
   {binding:1,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:'float'}},
   ...[2,3,4,5,6,7].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:binding===2?'read-only-storage' as const:'storage' as const}}))]});
  const pl=device.createPipelineLayout({bindGroupLayouts:[layout]});
  const pipelines=[];for(const entryPoint of entries)pipelines.push(await device.createComputePipelineAsync({layout:pl,compute:{module:shader,entryPoint}}));
  return new BatchedGpuEcc(device,pipelines);
 }
 async run(image:GPUTextureView,refs:GPUBuffer,width:number,height:number,quads:number[][][],rotations:number[],diagnosticInitialWarp?:number[]):Promise<EccTrace[]>{
  if(quads.length>16||quads.length!==rotations.length||quads.some(q=>q.length!==4||q.some(p=>p.length!==2||p.some(v=>!Number.isFinite(v))))||rotations.some(r=>!Number.isInteger(r)||r<0||r>3))throw Error('ECC component bounds/inputs');
  if(diagnosticInitialWarp&&(diagnosticInitialWarp.length!==9||diagnosticInitialWarp.some(v=>!Number.isFinite(v))||diagnosticInitialWarp[8]!==1))throw Error('Diagnostic initial warp invalid');
  const count=quads.length;if(!count)return [];
  const initial=new Float32Array(count*28);
  quads.forEach((q,i)=>{const base=quadBase(q);const s=i*28;initial.set(base,s);initial.set(diagnosticInitialWarp??[1,0,0,0,1,0,0,0,1],s+9);initial[s+18]=-1;initial[s+19]=-.00001;initial[s+20]=base.every(Number.isFinite)?0:-2;initial[s+22]=rotations[i];});
  const owned:GPUBuffer[]=[];const buffer=(size:number,usage:number)=>{const b=this.device.createBuffer({size,usage});owned.push(b);return b;};
  const rw=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC;
  const params=buffer(16,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST);this.device.queue.writeBuffer(params,0,new Uint32Array([width,height,count,0]));
  const states=buffer(initial.byteLength,rw);this.device.queue.writeBuffer(states,0,initial);
  const patch=buffer(count*81920*4,rw),pixels=buffer(count*16384*5*4,rw),jac=buffer(count*16384*8*4,rw),sums=buffer(count*86*4,rw);
  const read=buffer(initial.byteLength+count*86*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);let mapped=false;
  try{
   const bind=this.device.createBindGroup({layout:this.pipelines[0].getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:params}},{binding:1,resource:image},{binding:2,resource:{buffer:refs}},...[states,patch,pixels,jac,sums].map((b,i)=>({binding:i+3,resource:{buffer:b}}))]});
   const encoder=this.device.createCommandEncoder();const pass=(index:number,x:number,y=1)=>{const c=encoder.beginComputePass();c.setPipeline(this.pipelines[index]);c.setBindGroup(0,bind);c.dispatchWorkgroups(x,y);c.end();};
   pass(0,256,count);pass(1,256,count);pass(2,256,count);
   for(let iteration=0;iteration<60;iteration++){pass(3,256,count);pass(4,count,6);pass(5,256,count);pass(6,count,80);pass(7,count);}
   encoder.copyBufferToBuffer(states,0,read,0,initial.byteLength);encoder.copyBufferToBuffer(sums,0,read,initial.byteLength,count*86*4);this.device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);mapped=true;
   const raw=new Float32Array(read.getMappedRange()).slice();
   return quads.map((q,i)=>{const s=i*28,base=Array.from(raw.slice(s,s+9)),warp=Array.from(raw.slice(s+9,s+18));const composed=multiply(base,warp);const candidate=[[0,0],[128,0],[128,128],[0,128]].map(([x,y])=>{const d=composed[6]*x+composed[7]*y+composed[8];return [(composed[0]*x+composed[1]*y+composed[2])/d,(composed[3]*x+composed[4]*y+composed[5])/d];});return {base,warp,composed,rho:raw[s+18],iterations:raw[s+21],status:raw[s+20],lastRho:raw[s+19],rhoChange:Math.abs(raw[s+18]-raw[s+19]),converged:raw[s+23]===1,iterationLimitReached:raw[s+23]===2,termination:raw[s+20]<0?'failure':raw[s+23]===1?'epsilon':raw[s+23]===2?'iteration-limit':'unfinished',maskCount:raw[s+25],firstMaskCount:raw[s+24],minMaskCount:raw[s+26],maxMaskCount:raw[s+27],rotation:rotations[i],quad:raw[s+20]>0?candidate:q};});
  }finally{if(mapped)read.unmap();owned.forEach(b=>b.destroy());}
 }
}
// Transport math for the canonical 128x128 patch, not an alternative refiner.
export function quadBase(q:number[][]){const [a,b,c,d]=q;const dx1=b[0]-c[0],dx2=d[0]-c[0],dx3=a[0]-b[0]+c[0]-d[0],dy1=b[1]-c[1],dy2=d[1]-c[1],dy3=a[1]-b[1]+c[1]-d[1];const det=dx1*dy2-dx2*dy1;const g=(dx3*dy2-dx2*dy3)/det/128,h=(dx1*dy3-dx3*dy1)/det/128;return [(b[0]-a[0])/128+g*b[0],(d[0]-a[0])/128+h*d[0],a[0],(b[1]-a[1])/128+g*b[1],(d[1]-a[1])/128+h*d[1],a[1],g,h,1];}
function multiply(a:number[],b:number[]){return Array.from({length:9},(_,i)=>[0,1,2].reduce((s,k)=>s+a[Math.floor(i/3)*3+k]*b[k*3+i%3],0));}
