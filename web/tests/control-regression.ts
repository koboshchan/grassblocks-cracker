import {createSolver} from '../solver';
const out=document.querySelector('pre')!;
const results:any={};
try{
 const solver=await createSolver({backend:'gpu'});
 const blocks=[{x:0,y:0,z:0,rotation:0,is_side:0}];
 const bounds={xmin:-200,xmax:200,ymin:0,ymax:2,zmin:-20,zmax:20,north:[0,1,2,3]};
 const capped=await solver.search(blocks,{...bounds,maxMatches:7});
 if(capped.matches.length!==7||!capped.stopped||capped.stopReason!=='result limit'||capped.cancelled)throw Error('Result-limit failure');
 results.limit={matches:capped.matches.length,done:capped.done,total:capped.total};
 const first=await solver.search(blocks,{...bounds,stopOnFirst:true});
 if(first.matches.length!==1||!first.stopped||first.stopReason!=='first match')throw Error('First-hit failure');
 results.first={matches:first.matches.length,done:first.done};
 let release!:()=>void;let searched=false;const gate=new Promise<void>(r=>release=r),controller=new AbortController();
 const paused=solver.search(blocks,{...bounds,signal:controller.signal,waitIfPaused:()=>gate,onProgress:()=>{searched=true;}});
 await new Promise(r=>setTimeout(r,80));
 if(searched)throw Error('Paused search ran');
 controller.abort();release();const cancelled=await paused;
 if(!cancelled.cancelled||cancelled.done!==0)throw Error('Paused cancellation failed');
 results.pauseCancel={done:cancelled.done,cancelled:cancelled.cancelled};
 solver.destroy?.();results.pass=true;document.title='PASS';
}catch(e){results.error=String(e);document.title='FAIL';}
out.textContent=JSON.stringify(results,null,2);
