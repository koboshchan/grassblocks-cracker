const url=new URL('./solver.wasm',import.meta.url);
let api;
try{const response=await fetch(url);const {instance}=await WebAssembly.instantiate(await response.arrayBuffer());api=instance.exports;postMessage({ready:true});}
catch(e){postMessage({error:'WASM solver failed to load: '+e.message});}
onmessage=({data:t})=>{
 try{
 if(t.sharedCancel&&Atomics.load(t.sharedCancel,0)){postMessage({xs:new ArrayBuffer(0)});return;}
 const ints=new Int32Array(api.memory.buffer),bp=api.blocks_ptr()/4;
 t.blocks.forEach((b,i)=>ints.set([b.x,b.y,b.z,b.rotation,b.is_side],bp+i*5));
 const count=api.scan(t.x,t.y,t.z,t.count,t.blocks.length);
 const xs=new Int32Array(api.memory.buffer,api.results_ptr(),count).slice();
 postMessage({xs:xs.buffer},[xs.buffer]);
 }catch(e){postMessage({error:e.message});}
};
