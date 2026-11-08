import source from 'inline-worker:./trial-worker';
import * as pipeline from './pipeline';
import {runTrialPool,type TrialClient,type TrialReply} from './pool-scheduler';
function clientFactory(url:string,data:Uint8Array,width:number,height:number,ref:unknown):()=>Promise<TrialClient>{
 return async()=>{
  const blob=URL.createObjectURL(new Blob([source],{type:'text/javascript'}));
  let worker:Worker;
  try{worker=new Worker(blob);}finally{URL.revokeObjectURL(blob);}
  let pending:{resolve:(m:any)=>void;reject:(e:Error)=>void}|null=null,closed=false,timer:ReturnType<typeof setTimeout>;
  const request=(message:any,transfer:Transferable[]=[])=>new Promise<any>((resolve,reject)=>{
   if(closed||pending){reject(Error('Trial worker unavailable'));return;}
   pending={resolve,reject};timer=setTimeout(()=>close(Error('Trial worker timed out')),15*60*1000);
   try{worker.postMessage(message,transfer);}catch(e){close(Error(String(e)));}
  });
  const close=(error=Error('Trial worker closed'))=>{closed=true;clearTimeout(timer);worker.terminate();if(pending){pending.reject(error);pending=null;}};
  worker.onmessage=({data:m})=>{
   if(!pending)return;const p=pending;pending=null;clearTimeout(timer);
   if(m.type==='error'){p.reject(Error(m.text));return;}p.resolve(m);
  };
  worker.onerror=(e)=>close(Error(e.message||'Trial worker failed'));
  worker.onmessageerror=()=>close(Error('Invalid trial worker message'));
  try{
   const copy=data.slice(),ready=await request({type:'init',pipelineUrl:url,width,height,ref,buffer:copy.buffer},[copy.buffer]);
   if(ready.type!=='ready'||!Number.isSafeInteger(ready.heapBytes))throw Error('Trial worker did not initialize');
   return {heapBytes:ready.heapBytes,close:()=>close(),async run(plan){
    const reply=await request({type:'trial',plan});if(reply.type!=='trial')throw Error('Unexpected trial worker reply');return reply as TrialReply;
   }};
  }catch(e){close();throw e;}
 };
}
export async function detectParallel(url:string,data:Uint8Array,width:number,height:number,ref:unknown,progress:(m:any)=>void){
 let prepared=false;
 try{
  const plans=pipeline.prepareTrials(data,width,height,ref);prepared=true;
  progress({type:'progress',backend:pipeline.detectorBackend,text:'Running independent native WASM trials',fraction:0,indeterminate:true});
  const pool=await runTrialPool(plans,{
   create:clientFactory(url,data,width,height,ref),submit:pipeline.submitTrial,
   hardwareConcurrency:navigator.hardwareConcurrency,deviceMemory:(navigator as any).deviceMemory,
   coordinatorBytes:pipeline.heapBytes(),imageBytes:data.byteLength,
   progress(completed,total,name){progress({type:'progress',backend:pipeline.detectorBackend,text:`Native WASM trials ${completed}/${total} complete (${name})`,fraction:completed/total,indeterminate:completed!==total});},
  });
  if(pool.fallback)progress({type:'log',text:'Native WASM pool finishing serially from prepared proposals: '+pool.fallback});
  const result=pipeline.finishTrials();prepared=false;
  progress({type:'progress',backend:pipeline.detectorBackend,text:'Native WASM trials 9/9 complete',fraction:1,indeterminate:false});
  return {...result,parallelTrials:pool};
 }finally{if(prepared)pipeline.abortTrials();}
}
