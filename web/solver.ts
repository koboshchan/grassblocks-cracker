import {assetUrl} from './src/asset-dir';
import workerSource from './solver-worker.ts?worker';
import {turnBlocks as rawTurnBlocks,validate,type Block,type SearchOptions} from './hash';
const turnBlocks=(blocks:Block[],n:number)=>rawTurnBlocks(blocks,n).sort((a,b)=>a.is_side-b.is_side);
export async function createSolver({backend='auto',workers}:{backend?:'auto'|'cpu'|'gpu';workers?:number}={}) {
 if(backend!=='cpu'&&navigator.gpu) {
  try {return await gpuSolver();}catch(e){if(backend==='gpu')throw e;console.warn('WebGPU unavailable; using CPU WASM',e);}
 }
 return cpuSolver(workers);
}
function packMultiNorthBlocks(blocks: Block[], activeNorth: number[]) {
 const anchorIdx = Math.max(0, blocks.findIndex(b => b.is_side === 0));
 const anchor = blocks[anchorIdx] || blocks[0];
 const xA = anchor.x, yA = anchor.y, zA = anchor.z, rA = anchor.rotation, sideA = anchor.is_side;
 const centered = [anchor, ...blocks.filter((_, i) => i !== anchorIdx).sort((a, b) => a.is_side - b.is_side)].map(b => ({
  ...b,
  x: b.x - xA,
  y: b.y - yA,
  z: b.z - zA,
 }));
 const data = new Int32Array(centered.length * 16);
 const mblocks = new Int32Array(4 * 512 * 5);
 let activeMask = 0;
 let minRx = Infinity, maxRx = -Infinity, minRz = Infinity, maxRz = -Infinity;
 const rotRx = [0, 0, 0, 0], rotRz = [0, 0, 0, 0];
 for (let n = 0; n < 4; n++) {
  let ax = xA, az = zA;
  for (let j = 0; j < n; j++) [ax, az] = [-az, ax];
  rotRx[n] = ax; rotRz[n] = az;
  if (activeNorth.includes(n)) {
   activeMask |= 1 << n;
   if (ax < minRx) minRx = ax;
   if (ax > maxRx) maxRx = ax;
   if (az < minRz) minRz = az;
   if (az > maxRz) maxRz = az;
  }
 }
 for (let i = 0; i < centered.length; i++) {
  const b = centered[i];
  const off = i * 16;
  for (let n = 0; n < 4; n++) {
   let x = b.x, z = b.z;
   for (let j = 0; j < n; j++) [x, z] = [-z, x];
   const rot = (b.rotation + n) % (b.is_side ? 2 : 4);
   const moff = (n * 512 + i) * 5;
   mblocks[moff] = x;
   mblocks[moff + 1] = b.y;
   mblocks[moff + 2] = z;
   mblocks[moff + 3] = rot;
   mblocks[moff + 4] = b.is_side;
   if (activeNorth.includes(n)) {
    data[off + n] = x;
    data[off + 4 + n] = z;
    data[off + 8 + n] = rot;
   } else {
    data[off + n] = 0;
    data[off + 4 + n] = 0;
    data[off + 8 + n] = -1;
   }
  }
  data[off + 12] = b.y;
  data[off + 13] = b.is_side;
  data[off + 14] = 0;
  data[off + 15] = 0;
 }
 return { data, mblocks, nb: centered.length, activeMask, xA, yA, zA, rA, sideA, rotRx, rotRz, minRx, maxRx, minRz, maxRz };
}

async function gpuSolver(){
 const adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});
 if(!adapter)throw Error('No WebGPU adapter. Try the WASM CPU mode or a native solver.');
 const device=await adapter.requestDevice();let lost;
 device.lost.then(i=>{lost=i.message||'GPU device lost';});
 const code=await (await fetch(assetUrl('kernel.wgsl',import.meta.url))).text();
 const module=device.createShaderModule({code});
 const info=await module.getCompilationInfo();const errors=info.messages.filter(m=>m.type==='error');
 if(errors.length)throw Error(errors.map(e=>e.message).join('\n'));
 const blockLayout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:'read-only-storage'}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform'}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage'}}]});
 const layout=device.createPipelineLayout({bindGroupLayouts:[blockLayout]});
 let wg=128,items=1;
 const pipeCache=new Map<string,Promise<GPUComputePipeline>>();
 function getSearchPipeline(w:number,it:number,singleNorth:number,allFour:boolean):Promise<GPUComputePipeline>{
  const key=`${w}:${it}:${singleNorth}:${allFour?1:0}`;
  let p=pipeCache.get(key);
  if(!p){
   p=device.createComputePipelineAsync({
    layout,
    compute:{module,entryPoint:'search',constants:{WG_SIZE:w,ITEMS:it,SINGLE_NORTH:singleNorth,ALL_FOUR:allFour?1:0}},
   });
   pipeCache.set(key,p);
  }
  return p;
 }
 await getSearchPipeline(wg,items,-1,false);
 let tuned=false;let tuning:any=null;
 const RING_SIZE=4;
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
 const emit=()=>opts.onProgress?.({done:Math.min(done,total),total,seconds:(performance.now()-start)/1000,rate:Math.min(done,total)/Math.max(.001,(performance.now()-start)/1000)});
 const packed = packMultiNorthBlocks(blocks, opts.north);
 const axMin = opts.xmin + packed.minRx, axMax = opts.xmax + packed.maxRx;
 const ayMin = opts.ymin + packed.yA, ayMax = opts.ymax + packed.yA;
 const azMin = opts.zmin + packed.minRz, azMax = opts.zmax + packed.maxRz;
 const hasOffsetShift = packed.minRx !== packed.maxRx || packed.minRz !== packed.maxRz;
 blockBuffer=device.createBuffer({size:Math.max(64, packed.data.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
 const groups=Array.from({length:RING_SIZE},(_,i)=>device.createBindGroup({
  layout:blockLayout,
  entries:[
   {binding:0,resource:{buffer:blockBuffer!}},
   {binding:1,resource:{buffer:params[i]}},
   {binding:2,resource:{buffer:results[i]}},
  ]
 }));
 par[0]=axMin;par[1]=azMin;par[2]=ayMin;par[3]=Math.min(ayMax,ayMin+31);par[4]=1024;par[5]=0;par[6]=0;par[7]=blocks.length;
 for(let i=0;i<RING_SIZE;i++){
  device.queue.writeBuffer(params[i],0,par);
  device.queue.writeBuffer(results[i],0,zero4);
 }
 device.queue.writeBuffer(blockBuffer,0,packed.data);
 const nySpan=Math.min(128,ayMax-ayMin+1);
 let target=nySpan>64?65536:131072;
 const singleNorth=opts.north.length===1?opts.north[0]:-1;
 const allFour=packed.activeMask===15;
 if(!tuned){
 let cached:any=null;try{cached=JSON.parse(localStorage.getItem(cacheKey)||'null');}catch{}
 if(cached&&[64,128,256].includes(cached.wg)&&[1,2,4].includes(cached.items)){
 wg=cached.wg;items=cached.items;tuning={...cached,cached:true};
 }else{
 const trials:any[]=[];const count=262144,benchY=Math.min(ayMax,ayMin+31);
 par[0]=axMin;par[1]=azMin;par[2]=ayMin;par[3]=benchY;par[4]=1024;par[5]=0;par[6]=count;par[7]=blocks.length;
 device.queue.writeBuffer(params[0],0,par);
 tuneLoop: for(const w of [64,128,256])for(const it of [1,2,4]){
  await opts.waitIfPaused?.();if(opts.signal?.aborted)break tuneLoop;
  const candidate=await getSearchPipeline(w,it,singleNorth,allFour);
  const times:number[]=[];
  for(let rep=0;rep<3;rep++){
  device.queue.writeBuffer(results[0],0,zero4);const enc=device.createCommandEncoder();const pass=enc.beginComputePass();pass.setPipeline(candidate);pass.setBindGroup(0,groups[0]);pass.dispatchWorkgroups(Math.ceil(count/(w*it)));pass.end();const before=performance.now();device.queue.submit([enc.finish()]);await device.queue.onSubmittedWorkDone();if(rep)times.push(performance.now()-before);
  }
  trials.push({wg:w,items:it,ms:Math.min(...times)});
 }
 trials.sort((a,b)=>a.ms-b.ms);const best=trials[0]||{wg:128,items:1,ms:0};wg=best.wg;items=best.items;tuning={...best,trials,cached:false};
 if(!opts.signal?.aborted&&trials.length===9){try{localStorage.setItem(cacheKey,JSON.stringify(best));}catch{}}
 }
 if(!opts.signal?.aborted)tuned=true;
 }
 const pipeline=await getSearchPipeline(wg,items,singleNorth,allFour);

 let currentSlot=0;
 function submitChunk(slot:number,x:number,z:number,width:number,offset:number,count:number,ya:number,yb:number):InFlightTask{
  par[0]=x;par[1]=z;par[2]=ya;par[3]=yb;par[4]=width;par[5]=offset;par[6]=count;par[7]=blocks.length;
  device.queue.writeBuffer(params[slot],0,par);device.queue.writeBuffer(results[slot],0,zero4);
  const enc=device.createCommandEncoder();const pass=enc.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,groups[slot]);pass.dispatchWorkgroups(Math.ceil(count/(wg*items)));pass.end();
  enc.copyBufferToBuffer(results[slot],0,reads[slot],0,resultSize);device.queue.submit([enc.finish()]);
  const mapPromise=reads[slot].mapAsync(GPUMapMode.READ);
  return {slot,x,z,width,offset,count,ya,yb,n:0,tSubmit:performance.now(),mapPromise};
 }

 async function processTask(task:InFlightTask){
  await task.mapPromise;
  const elapsed=performance.now()-task.tSubmit;
  const ratio=Math.max(0.5,Math.min(2,40/Math.max(1,elapsed)));
  target=Math.min(524288,Math.max(256,Math.round((target*ratio)/256)*256));
  const read=reads[task.slot];
  const mapped=new Int32Array(read.getMappedRange());
  const found=mapped[0]>>>0;
  if(found>4096){
   read.unmap();
   await flush();
   if(task.count>1){const half=Math.floor(task.count/2);await dispatchChunk(task.x,task.z,task.width,task.offset,half,task.ya,task.yb);await dispatchChunk(task.x,task.z,task.width,task.offset+half,task.count-half,task.ya,task.yb);}
   else if(task.ya<task.yb){const mid=Math.floor((task.ya+task.yb)/2);await dispatchChunk(task.x,task.z,task.width,task.offset,task.count,task.ya,mid);await dispatchChunk(task.x,task.z,task.width,task.offset,task.count,mid+1,task.yb);}
   else throw Error('GPU result capacity exceeded.');
   return;
  }
  for(let i=0;i<found&&matches.length<limit;i++){
   const j=4+i*4;
   const wx=mapped[j],wy=mapped[j+1],wz=mapped[j+2],n=mapped[j+3];
   const ox=wx-packed.rotRx[n],oy=wy-packed.yA,oz=wz-packed.rotRz[n];
   if(hasOffsetShift&&(ox<opts.xmin||ox>opts.xmax||oz<opts.zmin||oz>opts.zmax))continue;
   const m={x:ox,y:oy,z:oz,north:n};
   matches.push(m);opts.onMatch?.(m);
  }
  read.unmap();
  if(matches.length>=limit)stopped=true;
  done+=task.count*(task.yb-task.ya+1)*opts.north.length;
  emit();
 }

 async function dispatchChunk(x:number,z:number,width:number,offset:number,count:number,ya:number,yb:number){
  if(lost)throw Error(lost);if(opts.signal?.aborted||stopped)return;
  const nextTask=submitChunk(currentSlot,x,z,width,offset,count,ya,yb);
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

 // Process up to 128 Y levels per column dispatch so rotation_xz column hashes are amortized while keeping thread parallelism high.
 const Y_STEP=128;
 for(let ya=ayMin;ya<=ayMax;ya+=Y_STEP){
  const yb=Math.min(ayMax,ya+Y_STEP-1);
  for(let za=azMin;za<=azMax;za+=256){
  const zh=Math.min(256,azMax-za+1);
  for(let xa=axMin;xa<=axMax;xa+=4096){
  const width=Math.min(4096,axMax-xa+1),size=width*zh;
  for(let off=0;off<size;){
  await opts.waitIfPaused?.();if(lost)throw Error(lost);if(opts.signal?.aborted||stopped)break;
  const count=Math.min(target,size-off);
  await dispatchChunk(xa,za,width,off,count,ya,yb);
  off+=count;
  }
  if(opts.signal?.aborted||stopped)break;
  }
  if(opts.signal?.aborted||stopped)break;
  }
  if(opts.signal?.aborted||stopped)break;
 }
 await flush();
 if(!opts.signal?.aborted&&!stopped)done=total;
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
 const dedupedNorth=Array.isArray(o.north)?[...new Set(o.north)]:o.north;
 const opts={...o,north:dedupedNorth as number[]};
 const total=validate(blocks,opts),matches:any[]=[],start=performance.now();let done=0,stopped=false;let stopReason:'first match'|'result limit'|undefined;
 const hardware=Math.max(1,Math.floor(navigator.hardwareConcurrency||2));
 const count=Math.max(1,Math.min(8,hardware,Number.isFinite(requested)?Math.floor(requested!):hardware));
 const packed=packMultiNorthBlocks(blocks,opts.north);
 const axMin=opts.xmin+packed.minRx,axMax=opts.xmax+packed.maxRx;
 const ayMin=opts.ymin+packed.yA,ayMax=opts.ymax+packed.yA;
 const azMin=opts.zmin+packed.minRz,azMax=opts.zmax+packed.maxRz;
 const hasOffsetShift=packed.minRx!==packed.maxRx||packed.minRz!==packed.maxRz;

 // Partition into ~262,144-coordinate 3D boxes so worker postMessage overhead is negligible even on narrow x ranges.
 const fullWidth=axMax-axMin+1,fullY=ayMax-ayMin+1;
 const xChunk=Math.min(32768,fullWidth);
 const yChunk=xChunk>=262144?1:Math.max(1,Math.min(fullY,Math.floor(262144/xChunk)));
 const zChunk=xChunk*yChunk>=262144?1:Math.max(1,Math.min(azMax-azMin+1,Math.floor(262144/(xChunk*yChunk))));

 let curZ=azMin,curY=ayMin,curX=axMin,finished=false;
 function next(){
  if(finished||stopped||opts.signal?.aborted)return null;
  const xc=Math.min(xChunk,axMax-curX+1);
  const yb=Math.min(ayMax,curY+yChunk-1);
  const zb=Math.min(azMax,curZ+zChunk-1);
  const tested=xc*(yb-curY+1)*(zb-curZ+1)*opts.north.length;
  const task={multi:true,xmin:curX,xcount:xc,ymin:curY,ymax:yb,zmin:curZ,zmax:zb,tested};
  curX+=xc;
  if(curX>axMax){curX=axMin;curY=yb+1;if(curY>ayMax){curY=ayMin;curZ=zb+1;if(curZ>azMax)finished=true;}}
  return task;
 }

 const wasmUrl=assetUrl('solver.wasm',import.meta.url);
 const runMsg=(worker:Worker,msg:any)=>new Promise<any>((res,rej)=>{
  worker.onmessage=e=>e.data.error?rej(Error(e.data.error)):res(e.data);worker.onerror=e=>rej(Error(e.message));worker.postMessage(msg);
 });
 await Promise.all(Array.from({length:count},async()=>{
  const blobURL=URL.createObjectURL(new Blob([workerSource],{type:'text/javascript'}));
  const worker=new Worker(blobURL,{type:'module'});URL.revokeObjectURL(blobURL);pool.push(worker);
  await runMsg(worker,{init:true,wasmUrl});
  await runMsg(worker,{setMulti:true,mblocks:packed.mblocks,nb:packed.nb,activeMask:packed.activeMask,anchorR:packed.rA,anchorSide:packed.sideA});
 }));

 const active=count;
 tuning={workers:active,skipped:true,cached:false};
 const limit=opts.stopOnFirst?1:(opts.maxMatches??100000);
 await Promise.all(pool.slice(0,active).map(async worker=>{
  for(;;){
   if(stopped||opts.signal?.aborted)break;
   await opts.waitIfPaused?.();
   const task=next();if(!task)break;
   const r=await runMsg(worker,{...task,sharedCancel});
   const arr=new Int32Array(r.matches);
   for(let i=0;i<arr.length;i+=4){
    if(stopped)break;
    const wx=arr[i],wy=arr[i+1],wz=arr[i+2],n=arr[i+3];
    const ox=wx-packed.rotRx[n],oy=wy-packed.yA,oz=wz-packed.rotRz[n];
    if(hasOffsetShift&&(ox<opts.xmin||ox>opts.xmax||oz<opts.zmin||oz>opts.zmax))continue;
    const m={x:ox,y:oy,z:oz,north:n};matches.push(m);opts.onMatch?.(m);
    if(opts.stopOnFirst||matches.length>=limit){stopped=true;stopReason=opts.stopOnFirst?'first match':'result limit';break;}
   }
   done=Math.min(total,done+(r.tested??0));
   const seconds=(performance.now()-start)/1000;
   opts.onProgress?.({done,total,seconds,rate:done/Math.max(1e-6,seconds)});
  }
 }));
 if(!opts.signal?.aborted&&!stopped)done=total;
 const seconds=(performance.now()-start)/1000;
 return {matches,done,total,seconds,rate:done/Math.max(1e-6,seconds),cancelled:!!opts.signal?.aborted,tuning,...(stopped?{stopped,stopReason}:{})};
 }finally{pool.forEach(w=>w.terminate());o.signal?.removeEventListener('abort',cancel);running=false;}
 }};
}
