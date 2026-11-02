// Run with node 24, after preparing local hill fixture paths described in README.
import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
const f=JSON.parse(fs.readFileSync('/tmp/grass-wasm-fixtures.json','utf8'));
const gray=new Uint8Array(fs.readFileSync('/tmp/grass-hill-gray.bin'));
const js=await import('../src/detect/detect.ts');
const rgba=new Uint8Array(f.w*f.h*4);for(let i=0;i<gray.length;i++)rgba[4*i+1]=gray[i];
const ctx=js.makeContext({width:f.w,height:f.h,data:rgba},f.ref);
const {instance}=await WebAssembly.instantiate(fs.readFileSync(new URL('../detect-wasm.wasm',import.meta.url)),{});
const e:any=instance.exports,base=Number(e.__heap_base.value),g=base+1024,q=(g+gray.length+7)&~7;
const quads:number[][]=[];
const P=f.camera.P.flat();
for(let y=-5;y<=5;y++)for(let z=-15;z<9;z++)for(let x=-12;x<16;x++){
 const quad=js.camQuad(P,x,y,z).flat();
 if(quad.every(Number.isFinite))quads.push(quad);
}
for(const b of f.camera.blocks)quads.push(b.quad.flat());
quads.push([0,0,0,0,0,0,0,0],[-20,-20,-10,-20,-10,-10,-20,-10]);
const inputs=new Float32Array(quads.flat()),out=q+inputs.byteLength,need=out+quads.length*5*4;
if(need>e.memory.buffer.byteLength)e.memory.grow(Math.ceil((need-e.memory.buffer.byteLength)/65536));
new Uint8Array(e.memory.buffer,g,gray.length).set(gray);
new Uint8Array(e.memory.buffer,base,1024).set(js.makeRefs(f.ref).flat());e.set_refs(base);
new Float32Array(e.memory.buffer,q,inputs.length).set(inputs);
function wasm(){e.classify_batch(g,f.w,f.h,q,quads.length,out);}
function reference(){return quads.map((_,i)=>js.classify(ctx,Array.from(inputs.subarray(i*8,i*8+8)).reduce((a:any[],v,j)=>{if(j%2===0)a.push([v]);else a[a.length-1].push(v);return a;},[])));}
wasm();let expected=reference(),actual=new Float32Array(e.memory.buffer,out,quads.length*5),valid=0,accepted=0,maxError=0;
for(let i=0;i<quads.length;i++){
 const a=actual.subarray(i*5,i*5+5),b=expected[i];assert.equal(a[0],b?1:0,'valid '+i);if(!b)continue;valid++;
 assert.equal(a[1],b.rotation,'rotation '+i);assert.equal(a[4],b.votes,'votes '+i);
 maxError=Math.max(maxError,Math.abs(a[2]-b.score),Math.abs(a[3]-b.margin));
 assert.ok(Math.abs(a[2]-b.score)<1e-6);assert.ok(Math.abs(a[3]-b.margin)<1e-6);
 const accepts=(s:number,m:number,v:number)=>s>=.88&&m>=.35&&v===4;
 assert.equal(accepts(a[2],a[3],a[4]),accepts(b.score,b.margin,b.votes),'accept '+i);
 if(accepts(a[2],a[3],a[4]))accepted++;
}
wasm();reference();
const t=performance.now();for(let i=0;i<10;i++)wasm();const wasmMs=(performance.now()-t)/10;
const u=performance.now();reference();const jsMs=performance.now()-u;
console.log(JSON.stringify({patches:quads.length,valid,accepted,maxError,wasmMs,jsMs,speedup:jsMs/wasmMs},null,2));
