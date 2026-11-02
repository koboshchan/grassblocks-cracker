let api;
onmessage=async({data:t})=>{
 if(t.init){try{const response=await fetch(t.wasmUrl);if(!response.ok)throw Error(String(response.status));const {instance}=await WebAssembly.instantiate(await response.arrayBuffer());api=instance.exports;postMessage({ready:true});}catch(e){postMessage({error:"WASM solver failed to load: "+e.message});}return;}
 try{
 if(t.sharedCancel&&Atomics.load(t.sharedCancel,0)){postMessage({xs:new ArrayBuffer(0),tested:0});return;}
 const ints=new Int32Array(api.memory.buffer),bp=api.blocks_ptr()/4;
 t.blocks.forEach((b,i)=>ints.set([b.x,b.y,b.z,b.rotation,b.is_side],bp+i*5));
 const count=api.scan(t.x,t.y,t.z,t.count,t.blocks.length);
 const xs=new Int32Array(api.memory.buffer,api.results_ptr(),count).slice();
 postMessage({xs:xs.buffer,tested:t.count},[xs.buffer]);
 }catch(e){postMessage({error:e.message});}
};
