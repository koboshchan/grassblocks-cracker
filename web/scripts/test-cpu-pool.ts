// Real browser, real inline WASM workers. No mocked scan or worker scheduler.
import {build} from 'esbuild';
import {readFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import {createServer} from 'node:http';
// Set PLAYWRIGHT_PACKAGE to a package.json with Playwright installed.
const require=createRequire(process.env.PLAYWRIGHT_PACKAGE||'/tmp/grass-ui-qa/package.json');
const {chromium}=require('playwright');
const result=await build({stdin:{contents:"export {createSolver} from './solver'; export {rotationBigInt} from './hash';",resolveDir:process.cwd()},bundle:true,write:false,format:'esm',plugins:[{name:'inline-test-worker',setup(b){b.onResolve({filter:/\?worker$/},()=>({path:'solver-worker.ts',namespace:'inline'}));b.onLoad({filter:/.*/,namespace:'inline'},async()=>({contents:'export default '+JSON.stringify((await build({entryPoints:['solver-worker.ts'],bundle:true,write:false,format:'esm'})).outputFiles[0].text),loader:'js'}));}}]});
const wasm=await readFile('artifacts/solver.wasm'),hill=JSON.parse(await readFile('public/examples/hill.json','utf8'));
const server=createServer((req,res)=>{res.setHeader('Cross-Origin-Opener-Policy','same-origin');res.setHeader('Cross-Origin-Embedder-Policy','require-corp');if(req.url==='/test.js'){res.setHeader('Content-Type','text/javascript');res.end(result.outputFiles[0].text);}else if(req.url==='/solver.wasm'){res.setHeader('Content-Type','application/wasm');res.end(wasm);}else res.end('<!doctype html><meta charset="utf-8">');});
await new Promise<void>(r=>server.listen(0,'127.0.0.1',r));const port=(server.address() as any).port;
const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
try{const page=await browser.newPage();await page.addInitScript('globalThis.__name=(f)=>f;Object.defineProperty(navigator,"hardwareConcurrency",{value:8})');await page.goto('http://127.0.0.1:'+port);
console.log(await page.evaluate(async(hill:any)=>{
const {createSolver,rotationBigInt}=await import('/test.js');
const solver=await createSolver({backend:'cpu',workers:4});
const assert=(v:any,m:string)=>{if(!v)throw Error(m);};
const base={xmin:3530,xmax:3545,ymin:90,ymax:90,zmin:-460,zmax:-450,north:[0]};
const blocks=(hill.blocks||hill).map((b:any)=>({...b,is_side:b.is_side||0}));
let r=await solver.search(blocks,base);assert(r.matches.some((m:any)=>m.x===3538&&m.y===90&&m.z===-454),'hill missing');assert(r.done===176&&r.tuning.skipped,'hill counts');
const dense=[{x:0,y:0,z:0,rotation:0,is_side:1}];
const small={xmin:-100,xmax:100,ymin:0,ymax:0,zmin:0,zmax:1,north:[0]};
r=await solver.search(dense,small);let expected=0;for(let z=0;z<=1;z++)for(let x=-100;x<=100;x++)if(rotationBigInt(x,0,z,1)===0)expected++;assert(r.matches.length===expected&&r.done===402,'dense exact');
r=await solver.search(dense,{...small,maxMatches:7});assert(r.matches.length===7&&r.stopReason==='result limit'&&r.done>0,'exact limit');
r=await solver.search(dense,{...small,stopOnFirst:true});assert(r.matches.length===1&&r.stopReason==='first match','first match');
let pausedCalls=0;const ac=new AbortController();r=await solver.search(dense,{...small,signal:ac.signal,waitIfPaused:async()=>{pausedCalls++;await new Promise(r=>setTimeout(r,10));ac.abort();}});assert(pausedCalls>0&&r.done===0&&r.cancelled,'paused cancellation');
const big={...small,xmin:-524288,xmax:524287,zmin:0,zmax:0};r=await solver.search(dense,{...big,maxMatches:3});assert(r.matches.length===3&&r.tuning.trials.length===3,'calibration');
const trials=r.tuning;r=await solver.search(dense,{...big,maxMatches:3});assert(r.tuning.cached&&r.matches.length===3,'cache');
const ac2=new AbortController();r=await solver.search(dense,{...big,xmin:-524289,signal:ac2.signal,waitIfPaused:async()=>{ac2.abort();}});assert(r.done===0&&r.cancelled,'calibration cancellation');
return {hill:true,dense:expected,pause:true,first:true,limit:true,tuning:trials,cache:true,calibrationCancel:true};
},hill));}finally{await browser.close();await new Promise<void>(r=>server.close(()=>r()));}
