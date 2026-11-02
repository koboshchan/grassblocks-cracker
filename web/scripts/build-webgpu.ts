import {build} from 'esbuild';
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
await mkdir('artifacts',{recursive:true});
const naga=process.env.NAGA||'naga';
execFileSync(naga,['kernel.wgsl'],{stdio:'inherit'});
// WGSL has no string literals. Preserve token boundaries while removing comments/extra whitespace.
const input=await readFile('kernel.wgsl','utf8');
let depth=0,clean='';
for(let i=0;i<input.length;i++){
 if(input.slice(i,i+2)==='/*'){depth++;i++;continue;}
 if(depth&&input.slice(i,i+2)==='*/'){depth--;i++;if(!depth)clean+=' ';continue;}
 if(depth)continue;
 if(input.slice(i,i+2)==='//'){while(i<input.length&&input[i]!=='\n')i++;clean+=' ';continue;}
 clean+=input[i];
}
if(depth)throw Error('Unclosed WGSL comment');
await writeFile('artifacts/kernel.wgsl',clean.replace(/\s+/g,' ').trim()+'\n');
execFileSync(naga,['artifacts/kernel.wgsl'],{stdio:'inherit'});
// Independent compilation verifies the WebGPU host even before the UI app exists.
await build({entryPoints:['solver.ts'],bundle:true,minify:true,format:'esm',target:'es2022',write:false,plugins:[{name:'worker-type-only',setup(b){b.onResolve({filter:/\?worker$/},()=>({path:'worker',namespace:'stub'}));b.onLoad({filter:/.*/,namespace:'stub'},()=>({contents:'export default ""'}));}}]});
console.log('Naga validation passed before and after WGSL minification; TS host compiled');
