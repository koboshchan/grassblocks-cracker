import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {versionedAssets} from '../scripts/asset-version';
import {assetUrl,ASSET_DIR} from '../src/asset-dir';
const enc=(s:string)=>new TextEncoder().encode(s);
const rd=(p:string)=>readFileSync(new URL(p,import.meta.url),'utf8');
const set=(js='a',wasm='b')=>({'grass-pipeline.js':enc(js),'grass-pipeline.wasm':enc(wasm),'solver.wasm':enc('s'),'kernel.wgsl':enc('k'),'detect.wgsl':enc('d')});
test('asset directory is deterministic and order independent',()=>{
 const a=versionedAssets(set()),rev=Object.fromEntries(Object.entries(set()).reverse());
 assert.equal(a.dir,versionedAssets(rev).dir);assert.match(a.dir,/^assets\/[0-9a-f]{16}\/$/);
});
test('changing the loader OR the wasm changes the directory',()=>{
 const base=versionedAssets(set()).dir;
 assert.notEqual(versionedAssets(set('a2')).dir,base);assert.notEqual(versionedAssets(set('a','b2')).dir,base);
});
test('wasm and shader resolve beside the loader',()=>{
 const loader='https://x.test/assets/abc/grass-pipeline.js';
 assert.equal(new URL('grass-pipeline.wasm',loader).href,'https://x.test/assets/abc/grass-pipeline.wasm');
 assert.equal(new URL('detect.wgsl',loader).href,'https://x.test/assets/abc/detect.wgsl');
 assert.equal(assetUrl('solver.wasm','https://x.test/app.js'),'https://x.test/'+ASSET_DIR+'solver.wasm');
});
test('workers import versioned JS and locate WASM from it',()=>{
 const t=rd('../src/detect/trial-worker.ts'),w=rd('../src/detect/worker.ts'),app=rd('../src/app.ts');
 assert.match(t,/importScripts\(url\.href\)/);assert.match(t,/locateFile:\(name:string\)=>new URL\(name,url\)\.href/);
 assert.match(w,/locateFile: \(name: string\) => new URL\(name, url\)\.href/);assert.doesNotMatch(w,/\.\.\/detect\.wgsl/);
 assert.match(app,/pipelineUrl: new URL\(ASSET_DIR \+ 'grass-pipeline\.js'/);assert.doesNotMatch(app,/vendor\/grass-pipeline/);
});
test('nginx: assets immutable, rest revalidates, COOP/COEP kept',()=>{
 const n=rd('../nginx.conf'),i=n.indexOf('location ^~ /assets/'),j=n.indexOf('location ~ \\.wasm');
 assert.ok(i>0&&j>i);const a=n.slice(i,j),rest=n.slice(j);
 assert.match(a,/max-age=31536000, immutable/);assert.match(a,/Cross-Origin-Embedder-Policy "require-corp"/);
 assert.doesNotMatch(rest,/immutable|expires 1h/);assert.match(rest.slice(rest.indexOf('location / ')),/no-cache/);
});
test('solver auto-selection is independent of detection and exposes fallback reason',()=>{
 const s=rd('../solver.ts'),app=rd('../src/app.ts');
 assert.match(s,/fallbackReason/);assert.match(s,/kres.ok/);assert.match(s,/WebGPU init failed/);assert.match(s,/navigator\.gpu unavailable/);
 assert.match(app,/createSolver\(\{ backend: 'auto' \}\)/);assert.match(app,/solver\.backend/);assert.match(app,/fallbackReason/);
});
