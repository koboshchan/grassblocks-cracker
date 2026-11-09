import test from 'node:test';
import assert from 'node:assert/strict';
import {normalizeBackend} from '../src/detect/backend-choice';
test('missing, invalid and malformed choices select WASM',()=>{
 for(const value of [undefined,null,'','gpu','WebGPU','Auto','true',{},'wasm']) assert.equal(normalizeBackend(value),'wasm');
});
test('auto and strict webgpu are recognised',()=>{
 assert.equal(normalizeBackend('auto'),'auto');
 assert.equal(normalizeBackend('webgpu'),'webgpu');
});
