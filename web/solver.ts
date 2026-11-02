import workerSource from './solver-worker.ts?worker';
import {turnBlocks,validate,type Block,type SearchOptions} from './hash';
export async function createSolver({backend='auto',workers}:{backend?:'auto'|'cpu'|'gpu';workers?:number}={}) {
 if(backend!=='cpu'&&navigator.gpu) {
  try {return await gpuSolver();}catch(e){if(backend==='gpu')throw e;console.warn('WebGPU unavailable; using CPU WASM',e);}
 }
 return cpuSolver(workers);
}
async function gpuSolver(){
 const adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});
 if(!adapter)throw Error('No WebGPU adapter. Try the WASM CPU mode or a native solver.');
 const device=await adapter.requestDevice();let lost;
 device.lost.then(i=>{lost=i.message||'GPU device lost';});
 const code=await (await fetch(new URL('./kernel.wgsl',import.meta.url))).text();
 const module=device.createShaderModule({code});
 const info=await module.getCompilationInfo();const errors=info.messages.filter(m=>m.type==='error');
 if(errors.length)throw Error(errors.map(e=>e.message).join('\n'));
 const blockLayout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:'read-only-storage'}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform'}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage'}}]});
 const layout=device.createPipelineLayout({bindGroupLayouts:[blockLayout]});
 let wg=256,items=1;
 let pipeline=await device.createComputePipelineAsync({layout,compute:{module,entryPoint:'search',constants:{WG_SIZE:wg,ITEMS:items}}});
 let tuned=false;let tuning:any=null;
 const resultSize=16+4096*16;
 const result=device.createBuffer({size:resultSize,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
 const read=device.createBuffer({size:resultSize,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
 const params=device.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
 const rawInfo=adapter.info||{};const adapterInfo={vendor:rawInfo.vendor||'',architecture:rawInfo.architecture||'',device:rawInfo.device||'',description:rawInfo.description||''};
 let codeHash=0;for(const c of code)codeHash=(Math.imul(codeHash,31)+c.charCodeAt(0))|0;
 const cacheKey='grass-wgsl-tune-v2:'+JSON.stringify(adapterInfo)+':'+codeHash;
 let running=false;
 return {backend:'webgpu',adapterInfo,get tuning(){return tuning;},async search(blocks:Block[],o:SearchOptions){
 if(running)throw Error('A search is already running.');running=true;
 let blockBuffer;
 try {
 const total=validate(blocks,o),matches=[],start=performance.now();let done=0;
 const emit=()=>o.onProgress?.({done,total,seconds:(performance.now()-start)/1000,rate:done/Math.max(.001,(performance.now()-start)/1000)});
 blockBuffer=device.createBuffer({size:blocks.length*20,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
 const group=device.createBindGroup({layout:blockLayout,entries:[{binding:0,resource:{buffer:blockBuffer}},{binding:1,resource:{buffer:params}},{binding:2,resource:{buffer:result}}]});
 let target=2048;
 if(!tuned){
 device.queue.writeBuffer(blockBuffer,0,new Int32Array(turnBlocks(blocks,o.north[0]).flatMap(b=>[b.x,b.y,b.z,b.rotation,b.is_side])));
 let cached:any=null;try{cached=JSON.parse(localStorage.getItem(cacheKey)||'null');}catch{}
 if(cached&&[64,128,256].includes(cached.wg)&&[1,2,4].includes(cached.items)){
 wg=cached.wg;items=cached.items;pipeline=await device.createComputePipelineAsync({layout,compute:{module,entryPoint:'search',constants:{WG_SIZE:wg,ITEMS:items}}});tuning={...cached,cached:true};
 }else{
 const trials:any[]=[];const count=16384,benchY=Math.min(o.ymax,o.ymin+15);
 device.queue.writeBuffer(params,0,new Int32Array([o.xmin,o.zmin,o.ymin,benchY,128,0,count,blocks.length]));
 for(const w of [64,128,256])for(const it of [1,2,4]){
 if(o.signal?.aborted)break;
 const candidate=await device.createComputePipelineAsync({layout,compute:{module,entryPoint:'search',constants:{WG_SIZE:w,ITEMS:it}}});
 const times:number[]=[];
 for(let rep=0;rep<3;rep++){
 device.queue.writeBuffer(result,0,new Uint32Array([0,0,0,0]));const enc=device.createCommandEncoder();const pass=enc.beginComputePass();pass.setPipeline(candidate);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(count/(w*it)));pass.end();const before=performance.now();device.queue.submit([enc.finish()]);await device.queue.onSubmittedWorkDone();if(rep)times.push(performance.now()-before);
 }
 trials.push({wg:w,items:it,ms:Math.min(...times)});
 }
 trials.sort((a,b)=>a.ms-b.ms);const best=trials[0]||{wg:256,items:1,ms:0};wg=best.wg;items=best.items;pipeline=await device.createComputePipelineAsync({layout,compute:{module,entryPoint:'search',constants:{WG_SIZE:wg,ITEMS:items}}});tuning={...best,trials,cached:false};try{localStorage.setItem(cacheKey,JSON.stringify(best));}catch{}
 }
 tuned=true;
 }

 async function dispatch(x,z,width,offset,count,ya,yb,n){
 if(lost)throw Error(lost);if(o.signal?.aborted)return;
 const par=new Int32Array([x,z,ya,yb,width,offset,count,blocks.length]);
 device.queue.writeBuffer(params,0,par);device.queue.writeBuffer(result,0,new Uint32Array([0,0,0,0]));
 const enc=device.createCommandEncoder();const pass=enc.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(count/(wg*items)));pass.end();
 enc.copyBufferToBuffer(result,0,read,0,resultSize);device.queue.submit([enc.finish()]);
 await read.mapAsync(GPUMapMode.READ);const data=new Int32Array(read.getMappedRange()).slice();read.unmap();
 const found=data[0]>>>0;
 if(found>4096){
  if(count>1){const half=Math.floor(count/2);await dispatch(x,z,width,offset,half,ya,yb,n);await dispatch(x,z,width,offset+half,count-half,ya,yb,n);}
  else if(ya<yb){const mid=Math.floor((ya+yb)/2);await dispatch(x,z,width,offset,count,ya,mid,n);await dispatch(x,z,width,offset,count,mid+1,yb,n);}
  else throw Error('GPU result capacity exceeded.');
  return;
 }
 for(let i=0;i<found;i++){const j=4+i*4;const m={x:data[j],y:data[j+1],z:data[j+2],north:n};matches.push(m);o.onMatch?.(m);}
 if(matches.length>(o.maxMatches??100000))throw Error('Over 100,000 matches. Add more observations or narrow the search.');
 done+=count*(yb-ya+1);
 }
 for(const n of [...new Set(o.north)]){
 device.queue.writeBuffer(blockBuffer,0,new Int32Array(turnBlocks(blocks,n).flatMap(b=>[b.x,b.y,b.z,b.rotation,b.is_side])));
 for(let ya=o.ymin;ya<=o.ymax;ya+=32){
 const yb=Math.min(o.ymax,ya+31);
 for(let za=o.zmin;za<=o.zmax;za+=128){
 const zh=Math.min(128,o.zmax-za+1);
 for(let xa=o.xmin;xa<=o.xmax;xa+=4096){
 const width=Math.min(4096,o.xmax-xa+1),size=width*zh;
 for(let off=0;off<size;){
 if(o.signal?.aborted)return {matches,done,total,cancelled:true,seconds:(performance.now()-start)/1000,rate:done/Math.max(.001,(performance.now()-start)/1000)};
 const count=Math.min(target,size-off),t=performance.now();await dispatch(xa,za,width,off,count,ya,yb,n);off+=count;emit();
 const elapsed=performance.now()-t;target=Math.min(262144,Math.max(256,Math.round(target*40/Math.max(1,elapsed)/256)*256));
 await new Promise(r=>setTimeout(r,0));
 }}}}}
 emit();const seconds=(performance.now()-start)/1000;return {matches,done,total,seconds,rate:done/seconds,cancelled:false};
 }finally{blockBuffer?.destroy();running=false;}
 },destroy(){device.destroy();}};
}
function cpuSolver(requested?:number){
 const adapterInfo={description:'WebAssembly SIMD CPU workers'};
 let running=false;
 return {backend:'wasm',adapterInfo,async search(blocks:Block[],o:SearchOptions){
 if(running)throw Error('A search is already running.');running=true;
 const pool:Worker[]=[];
 const sharedCancel=globalThis.crossOriginIsolated&&typeof SharedArrayBuffer!=='undefined'?new Int32Array(new SharedArrayBuffer(4)):null;
 const cancel=()=>{if(sharedCancel)Atomics.store(sharedCancel,0,1);};
 o.signal?.addEventListener('abort',cancel,{once:true});
 try{
 const total=validate(blocks,o),matches=[],start=performance.now();let done=0,cursor=0;
 // One fixed-y,z row task avoids lossy 64-bit linear indices. Independent workers
 // use transferable result buffers; SAB is not required to use multiple cores.
 const count=Math.max(1,Math.min(8,requested||navigator.hardwareConcurrency||2));
 let nIndex=0,y=o.ymin,z=o.zmin,x=o.xmin,finished=false;
 function next(){
 if(finished||o.signal?.aborted)return null;
 const task={x,z,y,count:Math.min(32768,o.xmax-x+1),north:o.north[nIndex],blocks:turnBlocks(blocks,o.north[nIndex])};
 x+=task.count;if(x>o.xmax){x=o.xmin;z++;if(z>o.zmax){z=o.zmin;y++;if(y>o.ymax){y=o.ymin;nIndex++;if(nIndex>=o.north.length)finished=true;}}}
 return task;
 }
 await Promise.all(Array.from({length:count},async()=>{
 const blobURL=URL.createObjectURL(new Blob([workerSource],{type:'text/javascript'}));
 const worker=new Worker(blobURL,{type:'module'});URL.revokeObjectURL(blobURL);pool.push(worker);
 await new Promise((res,rej)=>{worker.onmessage=e=>e.data.error?rej(Error(e.data.error)):res();worker.onerror=e=>rej(Error(e.message));worker.postMessage({init:true,wasmUrl:new URL('./solver.wasm',import.meta.url).href});});
 for(;;){const task=next();if(!task)break;
 const r=await new Promise((res,rej)=>{worker.onmessage=e=>e.data.error?rej(Error(e.data.error)):res(e.data);worker.onerror=e=>rej(Error(e.message));worker.postMessage({...task,sharedCancel});});
 for(const mx of new Int32Array(r.xs)){const m={x:mx,y:task.y,z:task.z,north:task.north};matches.push(m);o.onMatch?.(m);}
 if(matches.length>(o.maxMatches??100000))throw Error('Over 100,000 matches. Add more observations.');
 done+=r.tested??task.count;const seconds=(performance.now()-start)/1000;o.onProgress?.({done,total,seconds,rate:done/seconds});
 }
 }));
 const seconds=(performance.now()-start)/1000;return {matches,done,total,seconds,rate:done/seconds,cancelled:!!o.signal?.aborted};
 }finally{pool.forEach(w=>w.terminate());o.signal?.removeEventListener('abort',cancel);running=false;}
 }};
}
