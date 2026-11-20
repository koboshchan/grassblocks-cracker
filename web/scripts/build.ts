import {build,type Plugin} from 'esbuild';
import {mkdir,cp,readFile,writeFile,access} from 'node:fs/promises';
import {resolve} from 'node:path';
import {createHash} from 'node:crypto';
import {rm} from 'node:fs/promises';
import {versionedAssets} from './asset-version';
const inlineWorker:Plugin={name:'inline-worker',setup(b){
 b.onResolve({filter:/^inline-worker:/},a=>({path:resolve(a.resolveDir,a.path.slice(14)+'.ts'),namespace:'inline-classic'}));
 b.onLoad({filter:/.*/,namespace:'inline-classic'},async a=>{const inner=await build({entryPoints:[a.path],bundle:true,write:false,minify:true,format:'iife',target:'es2022',platform:'browser',plugins:[inlineWorker],define:{'import.meta.url':'self.location.href'}});return {contents:`export default ${JSON.stringify(inner.outputFiles[0].text)}`,loader:'js'};});
 b.onResolve({filter:/\?worker$/},a=>({path:resolve(a.resolveDir,a.path.slice(0,-7)),namespace:'inline-worker'}));
 b.onLoad({filter:/.*/,namespace:'inline-worker'},async a=>{
  const inner=await build({entryPoints:[a.path],bundle:true,write:false,minify:true,format:'esm',target:'es2022',platform:'browser'});
  return {contents:`export default ${JSON.stringify(inner.outputFiles[0].text)}`,loader:'js'};
 });
}};
await rm('dist',{recursive:true,force:true});await mkdir('dist',{recursive:true});
await cp('public','dist',{recursive:true});
const pick=async(...c:string[])=>{for(const p of c){try{await access(p);return p;}catch{}}return null;};
const sources:Record<string,string|null>={
 'grass-pipeline.js':'public/vendor/grass-pipeline.js','grass-pipeline.wasm':'public/vendor/grass-pipeline.wasm',
 'solver.wasm':'artifacts/solver.wasm','kernel.wgsl':await pick('artifacts/kernel.wgsl','kernel.wgsl'),
 'detect.wgsl':await pick('artifacts/detect.wgsl','src/detect/detect.wgsl')};
for(const [n,p] of Object.entries(sources))if(!p)throw new Error('missing asset '+n);
// Export contract: stale vendor glue lacking an ABI function breaks the preview at runtime.
const glue=await readFile('public/vendor/grass-pipeline.js','utf8');
for(const fn of ['_grass_feature_seeds','_grass_detect','_grass_classify','_grass_result','_grass_pool_prepare','_grass_pool_trial','_grass_pool_submit','_grass_pool_finish'])if(!glue.includes(fn))throw new Error('Stale grass-pipeline glue missing export '+fn+'; run npm run build:wasm (wasm/build-pipeline.sh link)');
const files:Record<string,Buffer>={};for(const [n,p] of Object.entries(sources))files[n]=await readFile(p as string);
const {dir,hash}=versionedAssets(files);
await mkdir('dist/'+dir,{recursive:true});for(const [n,d] of Object.entries(files))await writeFile('dist/'+dir+n,d);
await rm('dist/vendor/grass-pipeline.js',{force:true});await rm('dist/vendor/grass-pipeline.wasm',{force:true});
const result=await build({entryPoints:['src/app.ts'],outfile:'dist/app.js',define:{__ASSET_DIR__:JSON.stringify(dir)},bundle:true,minify:true,format:'esm',target:'es2022',platform:'browser',plugins:[inlineWorker],metafile:true});
await writeFile('dist/index.html',(await readFile('index.html','utf8')).replace('<script src=','<script type="module" src='));
for(const css of ['style.css','styles.css']){try{await access(css);await cp(css,'dist/'+css);}catch{}}
await writeFile('dist/asset-manifest.json',JSON.stringify({dir,hash,files:Object.fromEntries(Object.entries(files).map(([n,d])=>[n,createHash('sha256').update(d).digest('hex')]))},null,1));
await writeFile('dist/build-meta.json',JSON.stringify(result.metafile,null,2));
console.log('Built single minified app.js with inline CPU worker');

