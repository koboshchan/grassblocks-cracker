import test from 'node:test';
import assert from 'node:assert/strict';
import { runWebGpuDetection } from '../src/detect/webgpu-pipeline';

test('runWebGpuDetection structure and validation', async () => {
  // Test interface and validation behavior when no seeds match
  const fakeDetector: any = {
    setImage() {},
    setRefs() {},
    async classifyBatch(quads: Float32Array) {
      // Return zeroes (invalid quads)
      const count = quads.length / 8;
      return new Float32Array(count * 5);
    },
  };

  const img = {
    width: 64,
    height: 64,
    data: new Uint8Array(64 * 64 * 4),
  };

  const ref = Array.from({ length: 16 }, () => new Array(16).fill(128));

  const result = await runWebGpuDetection(fakeDetector, img, ref, {
    seeds: [
      [[10, 10], [20, 10], [20, 20], [10, 20]]
    ]
  });

  assert.equal(result.backend, 'WebGPU detector');
  assert.equal(result.blocks.length, 0);
  assert.ok(result.reason.includes('correlation too low'));
});
