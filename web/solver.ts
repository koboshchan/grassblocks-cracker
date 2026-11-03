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
 const RING_SIZE=3;
 const resultSize=16+4096*16;
 const results=Array.from({length:RING_SIZE},()=>device.createBuffer({size:resultSize,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST}));
 const reads=Array.from({length:RING_SIZE},()=>device.createBuffer({size:resultSize,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}));
 const params=Array.from({length:RING_SIZE},()=>device.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}));
 const par=new Int32Array(8);
 const zero4=new Uint32Array(4);
 const rawInfo:Partial<GPUAdapterInfo>=adapter.info||{};const adapterInfo={vendor:rawInfo.vendor||'',architecture:rawInfo.architecture||'',device:rawInfo.device||'',description:rawInfo.description||''};
 let codeHash=0;for(const c of code)codeHash=(Math.imul(codeHash,31)+c.charCodeAt(0))|0;
 const cacheKey='grass-wgsl-tune-v2:'+JSON.stringify(adapterInfo)+':'+codeHash;
 let running=false;
 return {backend:'webgpu',adapterInfo,get tuning(){return tuning;},async search(blocks:Block[],o:SearchOptions){
 if(running)throw Error('A search is already running.');running=true;
 const dedupedNorth=Array.isArray(o.north)?[...new Set(o.north)]:o.north;
 const opts={...o,north:dedupedNorth as number[]};
 let blockBuffer:GPUBuffer|undefined;
 interface InFlightTask {slot:number;x:number;z:number;width:number;offset:number;count:number;ya:number;yb:number;n:number;tSubmit:number;mapPromise:Promise<void>;}
 const inFlight:InFlightTask[]=[];
 try {
 const total=validate(blocks,opts),matches:any[]=[],start=performance.now();let done=0,stopped=false;
 const limit=opts.stopOnFirst?1:(opts.maxMatches??100000);
 const stopReason=()=>stopped?(opts.stopOnFirst?'first match':'result limit'):undefined;
 const emit=()=>opts.onProgress?.({done,total,seconds:(performance.now()-start)/1000,rate:done/Math.max(.001,(performance.now()-start)/1000)});
 blockBuffer=device.createBuffer({size:blocks.length*20,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
 const groups=Array.from({length:RING_SIZE},(_,i)=>device.createBindGroup({
  layout:blockLayout,
  entries:[
   {binding:0,resource:{buffer:blockBuffer!}},
   {binding:1,resource:{buffer:params[i]}},
   {binding:2,resource:{buffer:results[i]}},
  ]
 }));
 par[0]=opts.xmin;par[1]=opts.zmin;par[2]=opts.ymin;par[3]=Math.min(opts.ymax,opts.ymin+31);par[4]=1024;par[5]=0;par[6]=0;par[7]=blocks.length;
 for(let i=0;i<RING_SIZE;i++){
  device.queue.writeBuffer(params[i],0,par);
  device.queue.writeBuffer(results[i],0,zero4);
 }
 let target=65536;
 if(!tuned){
 device.queue.writeBuffer(blockBuffer,0,new Int32Array(turnBlocks(blocks,opts.north[0]).flatMap(b=>[b.x,b.y,b.z,b.rotation,b.is_side])));
 let cached:any=null;try{cached=JSON.parse(localStorage.getItem(cacheKey)||'null');}catch{}
 if(cached&&[64,128,256].includes(cached.wg)&&[1,2,4].includes(cached.items)){
 wg=cached.wg;items=cached.items;pipeline=await device.createComputePipelineAsync({layout,compute:{module,entryPoint:'search',constants:{WG_SIZE:wg,ITEMS:items}}});tuning={...cached,cached:true};
 }else{
 const trials:any[]=[];const count=262144,benchY=Math.min(opts.ymax,opts.ymin+31);
 par[0]=opts.xmin;par[1]=opts.zmin;par[2]=opts.ymin;par[3]=benchY;par[4]=1024;par[5]=0;par[6]=count;par[7]=blocks.length;
 device.queue.writeBuffer(params[0],0,par);
 tuneLoop: for(const w of [64,128,256])for(const it of [1,2,4]){
  await opts.waitIfPaused?.();if(opts.signal?.aborted)break tuneLoop;
  const candidate=await device.createComputePipelineAsync({layout,compute:{module,entryPoint:'search',constants:{WG_SIZE:w,ITEMS:it}}});
  const times:number[]=[];
  for(let rep=0;rep<3;rep++){
  device.queue.writeBuffer(results[0],0,zero4);const enc=device.createCommandEncoder();const pass=enc.beginComputePass();pass.setPipeline(candidate);pass.setBindGroup(0,groups[0]);pass.dispatchWorkgroups(Math.ceil(count/(w*it)));pass.end();const before=performance.now();device.queue.submit([enc.finish()]);await device.queue.onSubmittedWorkDone();if(rep)times.push(performance.now()-before);
  }
  trials.push({wg:w,items:it,ms:Math.min(...times)});
 }
 trials.sort((a,b)=>a.ms-b.ms);const best=trials[0]||{wg:256,items:1,ms:0};wg=best.wg;items=best.items;pipeline=await device.createComputePipelineAsync({layout,compute:{module,entryPoint:'search',constants:{WG_SIZE:wg,ITEMS:items}}});tuning={...best,trials,cached:false};
 if(!opts.signal?.aborted&&trials.length===9){try{localStorage.setItem(cacheKey,JSON.stringify(best));}catch{}}
 }
 if(!opts.signal?.aborted)tuned=true;
 }

 let currentSlot=0;
 function submitChunk(slot:number,x:number,z:number,width:number,offset:number,count:number,ya:number,yb:number,n:number):InFlightTask{
  par[0]=x;par[1]=z;par[2]=ya;par[3]=yb;par[4]=width;par[5]=offset;par[6]=count;par[7]=blocks.length;
  device.queue.writeBuffer(params[slot],0,par);device.queue.writeBuffer(results[slot],0,zero4);
  const enc=device.createCommandEncoder();const pass=enc.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,groups[slot]);pass.dispatchWorkgroups(Math.ceil(count/(wg*items)));pass.end();
  enc.copyBufferToBuffer(results[slot],0,reads[slot],0,resultSize);device.queue.submit([enc.finish()]);
  const mapPromise=reads[slot].mapAsync(GPUMapMode.READ);
  return {slot,x,z,width,offset,count,ya,yb,n,tSubmit:performance.now(),mapPromise};
 }

 async function processTask(task:InFlightTask){
  await task.mapPromise;
  const elapsed=performance.now()-task.tSubmit;
  const ratio=Math.max(0.5,Math.min(2,40/Math.max(1,elapsed)));
  target=Math.min(1048576,Math.max(256,Math.round((target*ratio)/256)*256));
  const read=reads[task.slot];
  const mapped=new Int32Array(read.getMappedRange());
  const found=mapped[0]>>>0;
  if(found>4096){
   read.unmap();
   await flush();
   if(task.count>1){const half=Math.floor(task.count/2);await dispatchChunk(task.x,task.z,task.width,task.offset,half,task.ya,task.yb,task.n);await dispatchChunk(task.x,task.z,task.width,task.offset+half,task.count-half,task.ya,task.yb,task.n);}
   else if(task.ya<task.yb){const mid=Math.floor((task.ya+task.yb)/2);await dispatchChunk(task.x,task.z,task.width,task.offset,task.count,task.ya,mid,task.n);await dispatchChunk(task.x,task.z,task.width,task.offset,task.count,mid+1,task.yb,task.n);}
   else throw Error('GPU result capacity exceeded.');
   return;
  }
  for(let i=0;i<found&&matches.length<limit;i++){const j=4+i*4;const m={x:mapped[j],y:mapped[j+1],z:mapped[j+2],north:task.n};matches.push(m);opts.onMatch?.(m);}
  read.unmap();
  if(matches.length>=limit)stopped=true;
  done+=task.count*(task.yb-task.ya+1);
  emit();
 }

 async function dispatchChunk(x:number,z:number,width:number,offset:number,count:number,ya:number,yb:number,n:number){
  if(lost)throw Error(lost);if(opts.signal?.aborted||stopped)return;
  const nextTask=submitChunk(currentSlot,x,z,width,offset,count,ya,yb,n);
  currentSlot=(currentSlot+1)%RING_SIZE;
  inFlight.push(nextTask);
  if(inFlight.length>=RING_SIZE){
   const prev=inFlight.shift()!;
   await processTask(prev);
  }
 }

 async function flush(){
  while(inFlight.length){
   const task=inFlight.shift()!;
   await processTask(task);
  }
  currentSlot=0;
 }

 let lastYield=performance.now();
 for(const n of opts.north){
  await opts.waitIfPaused?.();if(lost)throw Error(lost);if(opts.signal?.aborted||stopped)break;
  device.queue.writeBuffer(blockBuffer,0,new Int32Array(turnBlocks(blocks,n).flatMap(b=>[b.x,b.y,b.z,b.rotation,b.is_side])));
  for(let ya=opts.ymin;ya<=opts.ymax;ya+=32){
  const yb=Math.min(opts.ymax,ya+31);
  for(let za=opts.zmin;za<=opts.zmax;za+=128){
  const zh=Math.min(128,opts.zmax-za+1);
  for(let xa=opts.xmin;xa<=opts.xmax;xa+=4096){
  const width=Math.min(4096,opts.xmax-xa+1),size=width*zh;
  for(let off=0;off<size;){
  await opts.waitIfPaused?.();if(lost)throw Error(lost);if(opts.signal?.aborted||stopped)break;
  const count=Math.min(target,size-off);
  await dispatchChunk(xa,za,width,off,count,ya,yb,n);
  off+=count;
  if(performance.now()-lastYield>=16){await new Promise(r=>setTimeout(r,0));lastYield=performance.now();}
  }
  if(opts.signal?.aborted||stopped)break;
  }
  if(opts.signal?.aborted||stopped)break;
  }
  if(opts.signal?.aborted||stopped)break;
  }
  await flush();
 }
 await flush();
 emit();const seconds=(performance.now()-start)/1000;return {matches,done,total,seconds,rate:done/seconds,cancelled:!!opts.signal?.aborted,stopped,stopReason:stopReason()};
 }finally{
  for(const task of inFlight){try{await task.mapPromise;reads[task.slot].unmap();}catch{}}
  inFlight.length=0;
  for(const r of reads){try{r.unmap();}catch{}}
  blockBuffer?.destroy();running=false;
 }
 },destroy(){device.destroy();}};
}
function cpuSolver(requested?:number){
 const adapterInfo={description:'WebAssembly SIMD CPU workers'};
 let running=false;
 let tuning:any=null;
 return {backend:'wasm',adapterInfo,get tuning(){return tuning;},async search(blocks:Block[],o:SearchOptions){
 if(running)throw Error('A search is already running.');running=true;
 const pool:Worker[]=[];
 const sharedCancel=globalThis.crossOriginIsolated&&typeof SharedArrayBuffer!=='undefined'?new Int32Array(new SharedArrayBuffer(4)):null;
 const cancel=()=>{if(sharedCancel)Atomics.store(sharedCancel,0,1);};
 o.signal?.addEventListener('abort',cancel,{once:true});
 try{
 const total=validate(blocks,o),matches=[],start=performance.now();let done=0,stopped=false;let stopReason:'first match'|'result limit'|undefined;
 // One fixed-y,z row task avoids lossy 64-bit linear indices. Independent workers
 // use transferable result buffers; SAB is not required to use multiple cores.
 const hardware=Math.max(1,Math.floor(navigator.hardwareConcurrency||2));
 const count=Math.max(1,Math.min(8,hardware,Number.isFinite(requested)?Math.floor(requested!):hardware));
 let nIndex=0,y=o.ymin,z=o.zmin,x=o.xmin,finished=false;
 function next(){
 if(finished||stopped||o.signal?.aborted)return null;
 const task={x,z,y,count:Math.min(32768,o.xmax-x+1),north:o.north[nIndex],blocks:turnBlocks(blocks,o.north[nIndex])};
 x+=task.count;if(x>o.xmax){x=o.xmin;z++;if(z>o.zmax){z=o.zmin;y++;if(y>o.ymax){y=o.ymin;nIndex++;if(nIndex>=o.north.length)finished=true;}}}
 return task;
 }
 await Promise.all(Array.from({length:count},async()=>{
 const blobURL=URL.createObjectURL(new Blob([workerSource],{type:'text/javascript'}));
 const worker=new Worker(blobURL,{type:'module'});URL.revokeObjectURL(blobURL);pool.push(worker);
 await new Promise<void>((res,rej)=>{worker.onmessage=e=>e.data.error?rej(Error(e.data.error)):res();worker.onerror=e=>rej(Error(e.message));worker.postMessage({init:true,wasmUrl:new URL('./solver.wasm',import.meta.url).href});});
 }));
 const runTask=(worker:Worker,task:any)=>new Promise<{xs:ArrayBuffer;tested?:number}>((res,rej)=>{
 worker.onmessage=e=>e.data.error?rej(Error(e.data.error)):res(e.data);worker.onerror=e=>rej(Error(e.message));worker.postMessage({...task,sharedCancel});
 });
 let active=count;
 tuning={workers:count,skipped:total<1000000,cached:false};
 if(total>=1000000&&!o.signal?.aborted){
 // Version changes whenever the SIMD kernel or calibration workload changes.
 // Exact formation and problem shape prevent transferring an early-reject win
 // to a dense search. Calibration results never count as searched coordinates.
 const key='grass-cpu-simd-v1:'+JSON.stringify([hardware,count,blocks,o.xmin,o.xmax,o.ymin,o.ymax,o.zmin,o.zmax,o.north]);
 const candidates=[1,2,4,8].filter(n=>n<=count);
 let cached:any=null;try{cached=JSON.parse(localStorage.getItem(key)||'null');}catch{}
 if(cached&&candidates.includes(cached.workers)){active=cached.workers;tuning={...cached,cached:true};}
 else{
 const turned=turnBlocks(blocks,o.north[0]),width=o.xmax-o.xmin+1;
 const jobs=Array.from({length:32},(_,i)=>{const offset=(i*2048)%width;return {x:o.xmin+offset,y:o.ymin,z:o.zmin,count:Math.min(2048,width-offset),blocks:turned};});
 // All workers are ready before timing, and every candidate scans the same
 // bounded aggregate jobs using the actual WASM SIMD scan and result transport.
 await Promise.all(pool.map(async w=>{await o.waitIfPaused?.();if(!o.signal?.aborted)await runTask(w,{...jobs[0],count:Math.min(4,width)});}));
 const trials:{workers:number;ms:number}[]=[];
 for(const n of candidates){
 if(o.signal?.aborted)break;
 let cursor=0;const before=performance.now();
 await Promise.all(pool.slice(0,n).map(async w=>{while(!o.signal?.aborted){await o.waitIfPaused?.();if(o.signal?.aborted)break;const job=jobs[cursor++];if(!job)break;await runTask(w,job);}}));
 if(!o.signal?.aborted)trials.push({workers:n,ms:performance.now()-before});
 }
 if(!o.signal?.aborted){
 const best=trials.reduce((a,b)=>b.ms<a.ms?b:a);active=best.workers;
 tuning={...best,trials,cached:false,benchmarkCandidates:jobs.reduce((n,t)=>n+t.count,0)};
 try{localStorage.setItem(key,JSON.stringify(tuning));}catch{}
 }
 }
 }
 await Promise.all(pool.slice(0,active).map(async worker=>{
 for(;;){if(stopped||o.signal?.aborted)break;await o.waitIfPaused?.();const task=next();if(!task)break;
 const r=await runTask(worker,task);
 for(const mx of new Int32Array(r.xs)){
 if(stopped)break;
 const m={x:mx,y:task.y,z:task.z,north:task.north};matches.push(m);o.onMatch?.(m);
 if(o.stopOnFirst||matches.length>=(o.maxMatches??100000)){stopped=true;stopReason=o.stopOnFirst?'first match':'result limit';break;}
 }
 done+=r.tested??task.count;const seconds=(performance.now()-start)/1000;o.onProgress?.({done,total,seconds,rate:done/seconds});
 }
 }));
 const seconds=(performance.now()-start)/1000;return {matches,done,total,seconds,rate:done/seconds,cancelled:!!o.signal?.aborted,tuning,...(stopped?{stopped,stopReason}:{})};
 }finally{pool.forEach(w=>w.terminate());o.signal?.removeEventListener('abort',cancel);running=false;}
 }};
}
