let api,lastBlocks,lastBuffer,multiCfg=null;
function sameBlocks(a,b){
 if(a===b)return true;
 if(!a||!b||a.length!==b.length)return false;
 for(let i=0;i<a.length;i++){
  const x=a[i],y=b[i];
  if(x.x!==y.x||x.y!==y.y||x.z!==y.z||x.rotation!==y.rotation||x.is_side!==y.is_side)return false;
 }
 return true;
}
function runMultiBox(xmin,xcount,ymin,ymax,zmin,zmax,acc){
 const c=api.scan_multi(xmin,xcount,ymin,ymax,zmin,zmax,multiCfg.nb,multiCfg.activeMask,multiCfg.anchorR,multiCfg.anchorSide,8192);
 if(c>=0){
  if(c>0){
   const chunk=new Int32Array(api.memory.buffer,api.results_ptr(),c*4);
   for(let i=0;i<chunk.length;i++)acc.push(chunk[i]);
  }
  return;
 }
 if(zmin<zmax){
  const mid=Math.floor((zmin+zmax)/2);
  runMultiBox(xmin,xcount,ymin,ymax,zmin,mid,acc);
  runMultiBox(xmin,xcount,ymin,ymax,mid+1,zmax,acc);
 }else if(ymin<ymax){
  const mid=Math.floor((ymin+ymax)/2);
  runMultiBox(xmin,xcount,ymin,mid,zmin,zmax,acc);
  runMultiBox(xmin,xcount,mid+1,ymax,zmin,zmax,acc);
 }else if(xcount>1){
  const half=Math.floor(xcount/2);
  runMultiBox(xmin,half,ymin,ymax,zmin,zmax,acc);
  runMultiBox(xmin+half,xcount-half,ymin,ymax,zmin,zmax,acc);
 }
}
onmessage=async({data:t})=>{
 if(t.init){try{const response=await fetch(t.wasmUrl);if(!response.ok)throw Error(String(response.status));const {instance}=await WebAssembly.instantiate(await response.arrayBuffer());api=instance.exports;lastBlocks=null;lastBuffer=null;multiCfg=null;postMessage({ready:true});}catch(e){postMessage({error:"WASM solver failed to load: "+e.message});}return;}
 if(t.setMulti){
  try{
   new Int32Array(api.memory.buffer,api.mblocks_ptr(),t.mblocks.length).set(t.mblocks);
   multiCfg={nb:t.nb,activeMask:t.activeMask,anchorR:t.anchorR,anchorSide:t.anchorSide};
   postMessage({ready:true});
  }catch(e){postMessage({error:e.message});}
  return;
 }
 if(t.multi){
  try{
   if(t.sharedCancel&&Atomics.load(t.sharedCancel,0)){postMessage({matches:new ArrayBuffer(0),tested:0});return;}
   const c=api.scan_multi(t.xmin,t.xcount,t.ymin,t.ymax,t.zmin,t.zmax,multiCfg.nb,multiCfg.activeMask,multiCfg.anchorR,multiCfg.anchorSide,8192);
   let out;
   if(c>=0){
    out=c>0?new Int32Array(api.memory.buffer,api.results_ptr(),c*4).slice():new Int32Array(0);
   }else{
    const acc=[];
    runMultiBox(t.xmin,t.xcount,t.ymin,t.ymax,t.zmin,t.zmax,acc);
    out=new Int32Array(acc);
   }
   postMessage({matches:out.buffer,tested:t.tested},[out.buffer]);
  }catch(e){postMessage({error:e.message});}
  return;
 }
 try{
 if(t.sharedCancel&&Atomics.load(t.sharedCancel,0)){postMessage({xs:new ArrayBuffer(0),tested:0});return;}
 if(lastBuffer!==api.memory.buffer||!sameBlocks(t.blocks,lastBlocks)){
  const ints=new Int32Array(api.memory.buffer),bp=api.blocks_ptr()/4;
  for(let i=0;i<t.blocks.length;i++){
   const b=t.blocks[i],off=bp+i*5;
   ints[off]=b.x;ints[off+1]=b.y;ints[off+2]=b.z;ints[off+3]=b.rotation;ints[off+4]=b.is_side;
  }
  lastBlocks=t.blocks;lastBuffer=api.memory.buffer;
 }
 const count=api.scan(t.x,t.y,t.z,t.count,t.blocks.length);
 const xs=new Int32Array(api.memory.buffer,api.results_ptr(),count).slice();
 postMessage({xs:xs.buffer,tested:t.count},[xs.buffer]);
 }catch(e){postMessage({error:e.message});}
};
