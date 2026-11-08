import {build,type Plugin} from 'esbuild';
import {mkdir,cp,readFile,writeFile,access} from 'node:fs/promises';
import {resolve} from 'node:path';
const inlineWorker:Plugin={name:'inline-worker',setup(b){
 b.onResolve({filter:/^inline-worker:/},a=>({path:resolve(a.resolveDir,a.path.slice(14)+'.ts'),namespace:'inline-classic'}));
 b.onLoad({filter:/.*/,namespace:'inline-classic'},async a=>{const inner=await build({entryPoints:[a.path],bundle:true,write:false,minify:true,format:'iife',target:'es2022',platform:'browser',define:{'import.meta.url':'self.location.href'}});return {contents:`export default ${JSON.stringify(inner.outputFiles[0].text)}`,loader:'js'};});
 b.onResolve({filter:/\?worker$/},a=>({path:resolve(a.resolveDir,a.path.slice(0,-7)),namespace:'inline-worker'}));
 b.onLoad({filter:/.*/,namespace:'inline-worker'},async a=>{
  const inner=await build({entryPoints:[a.path],bundle:true,write:false,minify:true,format:'esm',target:'es2022',platform:'browser'});
  return {contents:`export default ${JSON.stringify(inner.outputFiles[0].text)}`,loader:'js'};
 });
}};
await mkdir('dist',{recursive:true});
const result=await build({entryPoints:['src/app.ts'],outfile:'dist/app.js',bundle:true,minify:true,format:'esm',target:'es2022',platform:'browser',plugins:[inlineWorker],metafile:true});
await writeFile('dist/index.html',(await readFile('index.html','utf8')).replace('<script src=','<script type="module" src='));
await cp('public','dist',{recursive:true});
for(const css of ['style.css','styles.css']){try{await access(css);await cp(css,'dist/'+css);}catch{}}
for(const asset of ['solver.wasm'])await cp('artifacts/'+asset,'dist/'+asset);
for(const asset of ['grass-pipeline.js','grass-pipeline.wasm'])await access('dist/vendor/'+asset);
try{await access('artifacts/kernel.wgsl');await cp('artifacts/kernel.wgsl','dist/kernel.wgsl');}catch{await cp('kernel.wgsl','dist/kernel.wgsl');}
try{await access('src/detect/detect.wgsl');await cp('src/detect/detect.wgsl','dist/detect.wgsl');}catch{}
await writeFile('dist/build-meta.json',JSON.stringify(result.metafile,null,2));
console.log('Built single minified app.js with inline CPU worker');

