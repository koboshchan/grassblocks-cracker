// Evaluation only. Uses geometry-derived registration, not hash-fitting.
const fs=require('node:fs');
(async()=>{
 const [runs,artifact]=process.argv.slice(2);
 const report=JSON.parse(fs.readFileSync(runs+'/evaluation.json','utf8'));
 const {instance}=await WebAssembly.instantiate(fs.readFileSync(artifact));
 const a=instance.exports;
 for(const mode of ['baseline','grid-only','hybrid']){
  for(const [name,known] of Object.entries({hill:[3538,90,-454],birch:[-4133,63,-2049]})){
   const r=JSON.parse(fs.readFileSync(runs+'/'+mode+'/'+name+'/result.json','utf8'));
   const evaluation=report[mode][name], anchor=evaluation.registered_anchor;
   if(!anchor||!r.blocks.length){evaluation.solver={status:'no formation'};continue;}
   if(r.blocks.length>512)throw Error('Solver capacity exceeded');
   const ints=new Int32Array(a.memory.buffer),offset=a.blocks_ptr()/4;
   r.blocks.forEach((b,i)=>{
    let x=b.x,z=b.z;
    for(let j=0;j<anchor[3];j++)[x,z]=[-z,x];
    ints.set([anchor[0]+x-known[0],anchor[1]+b.y-known[1],
     anchor[2]+z-known[2],(b.rotation+anchor[3])%4,0],offset+5*i);
   });
   const start=performance.now(),matches=[];
   // Bounded 17x5x17 regression, not a claim of full-world search.
   for(let y=known[1]-2;y<=known[1]+2;y++)
    for(let z=known[2]-8;z<=known[2]+8;z++){
     const n=a.scan(known[0]-8,y,z,17,r.blocks.length);
     for(const x of new Int32Array(a.memory.buffer,a.results_ptr(),n))matches.push([x,y,z]);
    }
   evaluation.solver={status:'bounded regression',range:[17,5,17],matches,
     knownFound:matches.some(x=>x.every((v,i)=>v===known[i])),seconds:(performance.now()-start)/1000};
  }
 }
 fs.writeFileSync(runs+'/evaluation.json',JSON.stringify(report,null,2));
 console.log(JSON.stringify(report,null,2));
})();
