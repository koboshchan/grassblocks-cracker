/** Local, self-contained SIMD detector. No network except loading its same-origin WASM asset. */
type Exports = {
 memory: WebAssembly.Memory; __heap_base: WebAssembly.Global;
 set_refs(p:number):void;
 classify_batch(g:number,w:number,h:number,q:number,n:number,out:number):void;
};
let ex:Exports|null=null, width=0,height=0,grayPtr=0,quadPtr=0,outPtr=0,base=0;
const CHUNK=4096;
export async function init(source: BufferSource | URL | string = new URL('./detect-wasm.wasm', import.meta.url)):Promise<void>{
 const bytes=typeof source==='string'||source instanceof URL?await (await fetch(source)).arrayBuffer():source;
 const result=await WebAssembly.instantiate(bytes,{});
 ex=result.instance.exports as unknown as Exports;
 base=Number(ex.__heap_base.value);width=height=0;
}
function instance():Exports{if(!ex)throw new Error('Call await init() before detection');return ex;}
function capacity(end:number){const e=instance(),need=end-e.memory.buffer.byteLength;if(need>0)e.memory.grow(Math.ceil(need/65536));}
export function setRefs(refs:Uint8Array):void{
 if(refs.length!==1024)throw new Error('Expected four flat 16x16 rotated templates');
 const e=instance();capacity(base+1024);new Uint8Array(e.memory.buffer,base,1024).set(refs);e.set_refs(base);
}
export function setImage(gray:Uint8Array,w:number,h:number):void{
 if(!Number.isInteger(w)||!Number.isInteger(h)||w<2||h<2||w*h!==gray.length||w*h>200000000)throw new Error('Invalid grayscale image');
 const e=instance();width=w;height=h;grayPtr=base+1024;quadPtr=(grayPtr+w*h+7)&~7;outPtr=quadPtr+CHUNK*8*4;
 capacity(outPtr+CHUNK*5*4);new Uint8Array(e.memory.buffer,grayPtr,w*h).set(gray);
}
export function classifyBatch(quads:Float32Array,outOrCount?:Float32Array|number):Float32Array{
 const out=outOrCount instanceof Float32Array?outOrCount:new Float32Array(quads.length/8*5);
 if(typeof outOrCount==='number'&&outOrCount!==quads.length/8)throw new Error('Batch count does not match quad length');
 const e=instance();if(!width||quads.length%8||out.length<quads.length/8*5)throw new Error('Set image and supply N*8 quads, N*5 output');
 for(let start=0;start<quads.length/8;start+=CHUNK){
  const n=Math.min(CHUNK,quads.length/8-start);
  new Float32Array(e.memory.buffer,quadPtr,n*8).set(quads.subarray(start*8,(start+n)*8));
  e.classify_batch(grayPtr,width,height,quadPtr,n,outPtr);
  out.set(new Float32Array(e.memory.buffer,outPtr,n*5),start*5);
 }
 return out;
}
export const detectorBackend='WASM SIMD f64x2';
