import test from 'node:test';import assert from 'node:assert/strict';import {eccBatchCapacity,ECC_QUAD_BYTES} from '../src/detect/ecc-batch';
const normal={maxStorageBufferBindingSize:128*1024*1024,maxBufferSize:256*1024*1024,maxComputeWorkgroupsPerDimension:65535};
test('ECC capacity obeys conservative total memory budget and device limits',()=>{
 assert.equal(eccBatchCapacity(normal),42);
 assert.equal(eccBatchCapacity(normal,96*1024*1024),64);
 assert.equal(eccBatchCapacity({...normal,maxStorageBufferBindingSize:ECC_QUAD_BYTES.jac*16}),16);
 assert.equal(eccBatchCapacity({...normal,maxBufferSize:ECC_QUAD_BYTES.jac*8}),8);
 assert.equal(eccBatchCapacity({...normal,maxComputeWorkgroupsPerDimension:128}),0);
 assert.equal(eccBatchCapacity(normal,1),0);
 assert.equal(eccBatchCapacity(normal,NaN),0);
 assert.equal(eccBatchCapacity(normal,48*1024*1024,16),16);
});

test('ECC capacity uses whole pending count without power-of-two rounding',()=>{for(const n of [9,17,31,33,41,42])assert.equal(eccBatchCapacity(normal,48*1024*1024,n),n);assert.equal(eccBatchCapacity(normal,48*1024*1024,43),42);});
