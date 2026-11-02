// Real Chromium regression: forced SIFT wasm trap must fall back to non-SIFT detection.
// Run: GRASS_URL=http://127.0.0.1:18643 node tests/browser-sift-fallback.cjs  (playwright from /tmp/grass-ui-qa)
const assert=require('assert');const {chromium}=require(process.env.PW||'/tmp/grass-ui-qa/node_modules/playwright');
const URL_=process.env.GRASS_URL||'http://127.0.0.1:18643';
const TRAP="\n;(function(){const O=self.GrassSift;self.GrassSift=async function(o){const m=await O(o);m._extract=function(){throw new WebAssembly.RuntimeError('memory access out of bounds');};return m;};})();";
(async()=>{const b=await chromium.launch({args:['--no-sandbox','--disable-gpu']});let trapped=0;
 const ctx=await b.newContext();
 await ctx.addInitScript(()=>{const W=window.Worker;window.__w=[];window.Worker=function(...a){const w=new W(...a);window.__w.push(w);return w;};});
 await ctx.route('**/grass-sift.js',async r=>{const resp=await r.fetch();trapped++;await r.fulfill({response:resp,body:(await resp.text())+TRAP});});
 const p=await ctx.newPage();const errs=[];p.on('pageerror',e=>errs.push(e.message));
 await p.goto(URL_);await p.waitForFunction(()=>document.querySelector('#statusText').textContent==='ready',null,{timeout:120000});
 assert.ok(trapped>0,'grass-sift.js was intercepted');
 // 1. blank fixture through UI with forced trap
 const png=await p.evaluate(()=>{const c=document.createElement('canvas');c.width=640;c.height=480;const g=c.getContext('2d');g.fillStyle='#808080';g.fillRect(0,0,640,480);return c.toDataURL('image/png').split(',')[1];});
 await p.locator('#file').setInputFiles({name:'blank.png',mimeType:'image/png',buffer:Buffer.from(png,'base64')});
 await p.waitForFunction(()=>document.querySelector('#wrap').classList.contains('has'));
 await p.locator('#btnAuto').click();
 await p.waitForFunction(()=>['ready','detector error'].includes(document.querySelector('#statusText').textContent)&&document.querySelector('#detLog').textContent.includes('verifying'),null,{timeout:120000});
 const log=await p.locator('#detLog').innerText();console.log(log);
 assert.ok(!/detector error/.test(await p.locator('#statusText').innerText()),'no terminal error');
 assert.ok(!/error:/.test(log),'no error in log');
 console.log("LOGHEAD",JSON.stringify(log.slice(0,300)));assert.ok(/binary feature fallback|SIFT unavailable/.test(log),'non-SIFT fallback used');
 assert.ok(!/SIFT WASM:/.test(log),'SIFT never completed');
 assert.equal(await p.locator('#count').innerText().then(t=>/^0\b/.test(t.trim())||/\b0\b/.test(t)),true,'empty rows: '+await p.locator('#count').innerText());
 // 2. malformed vs valid buffers straight to the live worker
 const res=await p.evaluate(async()=>{const w=window.__w[0];const ask=(msg,tr)=>new Promise(res=>{const h=e=>{if(['error','result'].includes(e.data.type)){w.removeEventListener('message',h);res(e.data);}};w.addEventListener('message',h);w.postMessage(msg,tr||[]);});
  const ref=Array.from({length:16},()=>Array(16).fill(100));
  const bad=await ask({type:'detect',width:64,height:64,buffer:new ArrayBuffer(64*64*3),ref,seeds:null,opts:{}});
  const badDim=await ask({type:'detect',width:3,height:3,buffer:new ArrayBuffer(36),ref,seeds:null,opts:{}});
  const buf=new ArrayBuffer(64*64*4);const good=await ask({type:'detect',width:64,height:64,buffer:buf,ref,seeds:null,opts:{}},[buf]);
  return {bad:bad.type+':'+(bad.text||'').slice(0,80),badDim:badDim.type,good:good.type,blocks:good.result&&good.result.blocks.length,notes:good.result&&good.result.notes};});
 console.log(JSON.stringify(res));
 assert.ok(res.bad.includes('Invalid image buffer'),res.bad);assert.equal(res.badDim,'error');assert.equal(res.good,'result');assert.equal(res.blocks,0);
 assert.deepEqual(errs,[],'no page errors');
 console.log('Browser SIFT fallback tests passed');await b.close();})().catch(e=>{console.error(e);process.exit(1)});
