/// <reference lib="webworker" />
import * as pipeline from './pipeline';
declare const self: DedicatedWorkerGlobalScope & { GrassPipeline?: (opts:any)=>Promise<pipeline.PipelineModule> };
let ready=false;
function post(m:any){self.postMessage(m);}
async function init(opencvUrl:string){
 const url=new URL('grass-pipeline.js',opencvUrl);
 self.importScripts(url.href);
 if(!self.GrassPipeline)throw Error('Native WASM pipeline factory unavailable');
 const module=await self.GrassPipeline({locateFile:(name:string)=>new URL(name,url).href});
 pipeline.setPipeline(module);ready=true;
}
self.onmessage=async(ev:MessageEvent)=>{
 const m=ev.data;
 try{
  if(m.type==='init'){await init(m.opencvUrl);post({type:'ready',backend:pipeline.detectorBackend});return;}
  if(m.type!=='detect'&&m.type!=='classifyQuad')return;
  if(!ready)throw Error('Native WASM detection pipeline not initialized');
  if(!Number.isSafeInteger(m.width)||!Number.isSafeInteger(m.height)||m.width<8||m.height<8||m.width*m.height>16*1024*1024||!(m.buffer instanceof ArrayBuffer)||m.buffer.byteLength!==m.width*m.height*4)
   throw Error('Invalid image buffer. Expected packed RGBA pixels and valid dimensions (maximum 16 megapixels).');
  const data=new Uint8Array(m.buffer);
  if(m.type==='classifyQuad'){post({type:'quadResult',...pipeline.classifyQuad(data,m.width,m.height,m.ref,m.quad)});return;}
  post({type:'progress',text:'Detecting grass in native WASM'});
  const started=performance.now();
  const result=pipeline.detect(data,m.width,m.height,m.ref,m.opts||{},m.seeds||[]);
  post({type:'result',result:{...result,transportSeconds:(performance.now()-started)/1000}});
 }catch(e:any){post({type:'error',text:String(e?.stack||e?.message||e)});}
};
