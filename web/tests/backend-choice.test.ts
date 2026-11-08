import test from 'node:test';
import assert from 'node:assert/strict';
import {BACKEND_STORAGE_KEY,normalizeBackend,readBackend,saveBackend} from '../src/detect/backend-choice';
test('WASM is default for missing, invalid and malformed choices',()=>{
 for(const value of [undefined,null,'','gpu','WebGPU','true',{},'wasm']) assert.equal(normalizeBackend(value),'wasm');
 assert.equal(normalizeBackend('webgpu'),'webgpu');
});
test('valid choice persists across independent reads, independent of solver backend',()=>{
 const data=new Map<string,string>();
 const access=()=>({getItem:(key:string)=>data.get(key)??null,setItem:(key:string,value:string)=>{data.set(key,value);}});
 assert.equal(readBackend(access),'wasm');
 assert.equal(saveBackend(access,'webgpu'),'webgpu');
 assert.equal(readBackend(access),'webgpu');
 assert.equal(data.get(BACKEND_STORAGE_KEY),'webgpu');
 assert.equal(saveBackend(access,'wasm'),'wasm');
 assert.equal(readBackend(access),'wasm');
});
test('invalid stored value and any storage read/write failure select WASM',()=>{
 assert.equal(readBackend(()=>({getItem:()=>'<script>',setItem:()=>{}})),'wasm');
 const denied=()=>{throw Error('SecurityError');};
 assert.equal(readBackend(denied),'wasm');
 assert.equal(saveBackend(denied,'webgpu'),'wasm');
 assert.equal(saveBackend(()=>({getItem:()=>null,setItem:()=>{throw Error('quota');}}),'webgpu'),'wasm');
});
