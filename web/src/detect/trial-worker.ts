/// <reference lib="webworker" />
import * as pipeline from './pipeline';
declare const self:DedicatedWorkerGlobalScope&{GrassPipeline?:(opts:any)=>Promise<pipeline.PipelineModule>};
let image:Uint8Array|null=null,width=0,height=0,reference:unknown;
self.onmessage=async({data:m})=>{
 try{
  if(m.type==='init'){
   const url=new URL(m.pipelineUrl);self.importScripts(url.href);
   if(!self.GrassPipeline)throw Error('Native trial module unavailable');
   pipeline.setPipeline(await self.GrassPipeline({locateFile:(name:string)=>new URL(name,url).href}));
   if(!pipeline.poolAvailable())throw Error('Automatic trial ABI unavailable');
   if(!Number.isSafeInteger(m.width)||!Number.isSafeInteger(m.height)||m.width<8||m.height<8||m.width*m.height>16*1024*1024||!(m.buffer instanceof ArrayBuffer)||m.buffer.byteLength!==m.width*m.height*4)throw Error('Invalid trial image');
   image=new Uint8Array(m.buffer);width=m.width;height=m.height;reference=m.ref;
   self.postMessage({type:'ready',heapBytes:pipeline.heapBytes()});return;
  }
  if(m.type!=='trial'||!image)throw Error('Trial worker is not initialized');
  const snapshot=pipeline.automaticTrial(image,width,height,reference,m.plan);
  self.postMessage({type:'trial',index:m.plan.index,snapshot,heapBytes:pipeline.heapBytes()},[snapshot.buffer]);
 }catch(e){self.postMessage({type:'error',text:String(e?.message||e)});}
};
