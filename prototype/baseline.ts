// Exact existing production detector; pixels and texture only, no fixture calibration.
import fs from 'node:fs';
import {createRequire} from 'node:module';
import {siftSeeds} from '../web/src/detect/sift';
import {makeRefs,runDetection} from '../web/src/detect/detect';
import * as wasm from '../web/detect-wasm';
const [input,out,vendor,artifact]=process.argv.slice(2);
const f=JSON.parse(fs.readFileSync(input,'utf8'));
const g=fs.readFileSync(f.gray);
const rgba=new Uint8Array(g.length*4);g.forEach((v,i)=>rgba[4*i+1]=v);
const req=createRequire(import.meta.url);
const c=req(vendor+'/opencv.cjs');
const {cv}:any=await new Promise(r=>c.Mat?r({cv:c}):c.then((x:any)=>r({cv:x})));
globalThis.fetch=undefined as any;
const sift=await req(vendor+'/grass-sift.cjs')();
await wasm.init(fs.readFileSync(artifact));
wasm.setImage(g,f.w,f.h);wasm.setRefs(Uint8Array.from(makeRefs(f.ref).flat()));
const t=performance.now();
const result=runDetection(cv,{width:f.w,height:f.h,data:rgba},f.ref,{},console.log,
 {classifyBatch:wasm.classifyBatch,featureSeeds:(cv,ctx,log)=>siftSeeds(sift,cv,ctx,log)});
fs.writeFileSync(out,JSON.stringify({...result,seconds:(performance.now()-t)/1000}));
console.log(JSON.stringify({out,seconds:(performance.now()-t)/1000,blocks:result.blocks.length}));
process.exit(0);
