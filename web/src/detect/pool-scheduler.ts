import type {TrialPlan} from './pipeline';
const MiB=1024*1024;
export type TrialReply={index:number;snapshot:Uint8Array;heapBytes:number};
export type TrialClient={run(plan:TrialPlan):Promise<TrialReply>;close():void;heapBytes:number};
export type PoolHooks={
 create():Promise<TrialClient>;
 submit(index:number,snapshot:Uint8Array):void;
 progress(completed:number,total:number,name:string):void;
 hardwareConcurrency:number;deviceMemory?:number;coordinatorBytes:number;imageBytes:number;
};
export function memoryBudget(deviceMemory?:number){return Math.min(1536,Number.isFinite(deviceMemory)&&deviceMemory>0?deviceMemory*256:768)*MiB;}
export function poolLimit(hc:number,coordinator:number,peak:number,image:number,budget:number){
 const cores=Number.isFinite(hc)?Math.max(0,Math.floor(hc)-1):0;
 // Include the JS image, WASM input copy and native green/ECC overhead, not just
 // initial linear memory. Recalculate from measured high-water heaps each trial.
 const slot=Math.max(32*MiB,peak)+image*2+16*MiB;
 return Math.max(0,Math.min(9,cores,Math.floor((budget-coordinator-image*2)/slot)));
}
/** Scheduling/transport only. Native code owns all features, trials and merging. */
export async function runTrialPool(plans:TrialPlan[],hooks:PoolHooks){
 const clients:TrialClient[]=[],completed=new Set<number>();let next=0,peak=32*MiB,workerPeak=0,inFlight=0,maxSimultaneousTrials=0,failure='';
 const budget=memoryBudget(hooks.deviceMemory);
 const limit=()=>poolLimit(hooks.hardwareConcurrency,hooks.coordinatorBytes,peak,hooks.imageBytes,budget);
 const measuredWorker=(bytes:number)=>{
  if(!Number.isSafeInteger(bytes)||bytes<32*MiB)throw Error('Invalid trial worker heap measurement');
  workerPeak=Math.max(workerPeak,bytes);peak=workerPeak;
 };
 const stats=()=>({workers:clients.length,completed:completed.size,
  // Compatibility alias now explicitly means worker-only high-water memory.
  peakHeapBytes:workerPeak,workerPeakHeapBytes:workerPeak,
  coordinatorHeapBytes:hooks.coordinatorBytes,imageBytes:hooks.imageBytes,
  capacity:limit(),maxSimultaneousTrials,budgetBytes:budget});
 const execute=async(client:TrialClient,plan:TrialPlan)=>{
  inFlight++;maxSimultaneousTrials=Math.max(maxSimultaneousTrials,inFlight);
  let reply:TrialReply;
  try{reply=await client.run(plan);}finally{inFlight--; }
  if(reply.index!==plan.index||completed.has(reply.index)||!(reply.snapshot instanceof Uint8Array)||!Number.isSafeInteger(reply.heapBytes)||reply.heapBytes<32*MiB)throw Error('Invalid trial worker reply');
  hooks.submit(reply.index,reply.snapshot);completed.add(reply.index);measuredWorker(reply.heapBytes);
  hooks.progress(completed.size,plans.length,plan.name);
 };
 try{
  if(limit()<1)return {...stats(),fallback:'serial resource limit'};
  // Trial zero is a memory probe before expanding the pool. Its result still
  // enters native fixed-index union, never arrival-order JS merging.
  const first=await hooks.create();clients.push(first);measuredWorker(first.heapBytes);
  if(limit()<1)throw Error('Trial worker exceeds measured memory budget');
  await execute(first,plans[next++]);
  const slots=Math.max(1,limit());
  while(clients.length<slots&&next<plans.length){
   const client=await hooks.create();clients.push(client);measuredWorker(client.heapBytes);
   if(clients.length>limit()){clients.pop();client.close();break;}
  }
  await Promise.all(clients.map(async(client,slot)=>{
   while(next<plans.length){
    if(failure||slot>=limit())return;
    const plan=plans[next++];
    try{await execute(client,plan);}catch(e){failure=String(e?.message||e);for(const other of clients)other.close();throw e;}
   }
  }));
 }catch(e){failure=String(e?.message||e);}
 finally{for(const client of clients)client.close();}
 return {...stats(),fallback:failure|| (completed.size<plans.length?'serial resource limit':'')};
}
