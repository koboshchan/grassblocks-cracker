import {readFileSync} from 'node:fs';
import {rotationBigInt,rotationPair,turnBlocks} from '../hash.ts';
const wasm=await WebAssembly.instantiate(readFileSync(new URL('../solver.wasm',import.meta.url)));
const api=wasm.instance.exports as unknown as {memory:WebAssembly.Memory;rotation(x:number,y:number,z:number):number;blocks_ptr():number;results_ptr():number;scan(x:number,y:number,z:number,n:number,b:number):number};
let state=1;
function random(){state=(Math.imul(state,1664525)+1013904223)>>>0;return state|0;}
for(let i=0;i<100000;i++){
 const x=random(),y=random(),z=random(),r=rotationBigInt(x,y,z);
 if(rotationPair(x,y,z)!==r||api.rotation(x,y,z)!==r)throw Error('Scalar hash mismatch '+i);
 for(let side=0;side<2;side++){
 const expected=r%(side?2:4);
 new Int32Array(api.memory.buffer).set([0,0,0,expected,side],api.blocks_ptr()/4);
 const count=api.scan(x,y,z,4,1),got=[...new Int32Array(api.memory.buffer,api.results_ptr(),count)];
 const want=[];for(let j=0;j<4;j++)if(rotationBigInt((x+j)|0,y,z,side)===expected)want.push((x+j)|0);
 if(got.join()!==want.join())throw Error('SIMD mismatch '+i);
 }
}
const lines=readFileSync(new URL('./fixtures/hill22.txt',import.meta.url),'utf8').trim().split('\n').filter(l=>l.trim()&&!l.startsWith('#'));
const blocks=lines.slice(1).map(l=>{const[x,y,z,rotation,is_side]=l.trim().split(/\s+/).map(Number);return{x,y,z,rotation,is_side};});
for(const b of blocks)if(rotationBigInt(3538+b.x,90+b.y,-454+b.z,b.is_side)!==b.rotation)throw Error('Hill observation disagrees');
let found=[];
for(let north=0;north<4;north++){
 const bs=turnBlocks(blocks,north);
 new Int32Array(api.memory.buffer).set(bs.flatMap(b=>[b.x,b.y,b.z,b.rotation,b.is_side]),api.blocks_ptr()/4);
 for(let y=88;y<=92;y++)for(let z=-457;z<=-451;z++){
 const count=api.scan(3535,y,z,7,bs.length);
 for(const x of new Int32Array(api.memory.buffer,api.results_ptr(),count))found.push([x,y,z,north]);
 }
}
if(JSON.stringify(found)!==JSON.stringify([[3538,90,-454,0]]))throw Error('Hill search mismatch '+JSON.stringify(found));
console.log(JSON.stringify({scalarCoordinates:100000,simdLanes:800000,hillMatches:found,pass:true},null,2));
