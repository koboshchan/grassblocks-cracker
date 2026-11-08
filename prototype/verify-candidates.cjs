// Cross-check every CPU candidate with the unchanged production WASM solver.
const fs=require('node:fs');
(async()=>{
 const [root,artifact]=process.argv.slice(2);
 const report=JSON.parse(fs.readFileSync(root+'/solver-report.json','utf8'));
 const {instance}=await WebAssembly.instantiate(fs.readFileSync(artifact));
 const a=instance.exports;
 for(const [name,r] of Object.entries(report)){
  const source=JSON.parse(fs.readFileSync(root+'/'+r.mode+'/'+name+'/result.json','utf8'));
  let verified=0;
  for(const c of r.candidates){
   const [x,y,z,n]=c;
   const ints=new Int32Array(a.memory.buffer),off=a.blocks_ptr()/4;
   source.blocks.forEach((b,i)=>{
    let xx=b.x,zz=b.z;
    for(let j=0;j<n;j++)[xx,zz]=[-zz,xx];
    ints.set([xx,b.y,zz,(b.rotation+n)%4,0],off+5*i);
   });
   const count=a.scan(x,y,z,1,source.blocks.length);
   if(count!==1||new Int32Array(a.memory.buffer,a.results_ptr(),count)[0]!==x)throw Error('WASM candidate mismatch '+name+' '+c);
   verified++;
  }
  r.productionWasmVerified=verified;
 }
 fs.writeFileSync(root+'/solver-report.json',JSON.stringify(report,null,2));
 console.log(Object.fromEntries(Object.entries(report).map(([k,r])=>[k,{count:r.candidates.length,wasmVerified:r.productionWasmVerified}])));
})();
