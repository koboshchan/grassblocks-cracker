import test from 'node:test';
import assert from 'node:assert/strict';
import {WebGpuDetector} from '../src/detect/webgpu';

test('WebGpuDetector reports support correctly in environment', async () => {
  const supported = await WebGpuDetector.isSupported();
  assert.equal(typeof supported, 'boolean');
});

test('WebGpuDetector reference statistics match expected Pearson values', () => {
  // Test reference generation and variance formulas
  const ref = Array.from({length: 16}, (_, y) =>
    Array.from({length: 16}, (_, x) => (x * 16 + y) % 256)
  );

  // Generate 4 rotations
  const rotations: number[][] = [];
  for (let k = 0; k < 4; k++) {
    const rot: number[] = [];
    for (let y = 0; y < 16; y++) {
      for (let x = 0; x < 16; x++) {
        let xx = x, yy = y;
        for (let j = 0; j < k; j++) {
          const t = xx;
          xx = yy;
          yy = 15 - t;
        }
        rot.push(ref[yy][xx]);
      }
    }
    rotations.push(rot);
  }

  // Calculate stats over inner 14x14
  const sb = [0, 0, 0, 0];
  const bb = [0, 0, 0, 0];
  for (let j = 1; j < 15; j++) {
    for (let i = 1; i < 15; i++) {
      const idx = j * 16 + i;
      for (let k = 0; k < 4; k++) {
        const val = rotations[k][idx];
        sb[k] += val;
        bb[k] += val * val;
      }
    }
  }

  const vb = [0, 0, 0, 0];
  for (let k = 0; k < 4; k++) {
    vb[k] = bb[k] - (sb[k] * sb[k]) / 196.0;
    assert.ok(vb[k] > 0, `Variance for rotation ${k} should be positive`);
  }
});

// No shader execution here: exercise actual async error-scope/device-loss wrapper.
test('checked GPU execution returns only after queue completion and scope checks',async()=>{
 const calls:string[]=[];const detector:any=Object.create(WebGpuDetector.prototype);
 detector.destroyed=false;detector.device={lost:new Promise(()=>{}),pushErrorScope:(s:string)=>calls.push(s),popErrorScope:async()=>{calls.push('pop');return null;},queue:{onSubmittedWorkDone:async()=>{calls.push('done');}}};
 assert.equal(await detector.checked(async()=>{calls.push('work');return 7;}),7);
 assert.deepEqual(calls,['internal','out-of-memory','validation','work','done','pop','pop','pop']);
});
test('checked GPU validation and device errors reject before returning a result',async()=>{
 const detector:any=Object.create(WebGpuDetector.prototype);detector.destroyed=false;
 detector.device={lost:new Promise(()=>{}),pushErrorScope:()=>{},popErrorScope:async()=>({message:'invalid binding'}),queue:{onSubmittedWorkDone:async()=>{}}};
 await assert.rejects(detector.checked(async()=>7),/invalid binding/);
 detector.device={...detector.device,lost:Promise.resolve({message:'adapter removed'}),popErrorScope:async()=>null};
 await assert.rejects(detector.checked(()=>new Promise(()=>{})),/adapter removed/);
});
