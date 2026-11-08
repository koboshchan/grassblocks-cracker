import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { applyH, dltCamera, planeFromSeeds, runWebGpuDetection } from '../src/detect/webgpu-pipeline';

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


test('hill GPU camera DLT refit preserves correspondences below one pixel', () => {
  const { p3, pix } = JSON.parse(readFileSync(new URL('./fixtures/hill-gpu-dlt.json', import.meta.url), 'utf8'));
  const P = dltCamera(p3, pix);
  assert.ok(P);
  let error = 0;
  for (let i = 0; i < p3.length; i++) {
    const [x,y,z] = p3[i];
    const w = P[8]*x + P[9]*y + P[10]*z + P[11];
    const u = (P[0]*x + P[1]*y + P[2]*z + P[3])/w;
    const v = (P[4]*x + P[5]*y + P[6]*z + P[7])/w;
    error += Math.hypot(u-pix[i][0], v-pix[i][1])/p3.length;
  }
  assert.ok(error < 1, 'Mean reprojection error ' + error);
});


function featureOrderDetector(seeds: any[]) {
  let calls = 0;
  return {
    setImage() {},
    setRefs() {},
    async classifyBatch(quads: Float32Array) {
      const out = new Float32Array(quads.length / 8 * 5);
      // Only the initial seed verification succeeds, keeping lattice scans cheap.
      if (calls++ === 0) seeds.forEach((s, i) => {
        out.set([1, s.rotation, s.score, s.margin, s.votes], i * 5);
      });
      return out;
    },
  } as any;
}

test('automatic feature seeds retain native order and duplicates before anchor fitting', async () => {
  const fixture = JSON.parse(readFileSync(new URL('./fixtures/hill-gpu-feature-order.json', import.meta.url), 'utf8'));
  // A duplicate must survive automatic seed verification just as in native.
  const seeds = [...fixture, fixture[0]];
  const expected = planeFromSeeds(seeds)!;
  const result = await runWebGpuDetection(featureOrderDetector(seeds),
    { width: 700, height: 650, data: new Uint8Array(700*650*4) },
    Array.from({ length: 16 }, () => Array(16).fill(128)),
    { featureSeeds: seeds, refineMin: 2 });
  assert.equal(result.seeds.length, seeds.length, 'automatic seeds must not use JS-only NMS');
  seeds.forEach((s, i) => {
    assert.deepEqual(result.seeds[i].quad, s.quad, 'automatic seeds must retain feature order');
    assert.equal(result.seeds[i].rotation, s.rotation);
  });
  assert.deepEqual(result.planeH, expected.H);
  const anchor = applyH(result.planeH!, 0, 0);
  assert.ok(Math.hypot(anchor[0]-389.502, anchor[1]-486.920) < .1);
  // Sorting the exact same seeds is known to select the neighboring tile.
  const wrong = planeFromSeeds([...seeds].sort((a,b) => b.score-a.score))!;
  const wrongAnchor = applyH(wrong.H, 0, 0);
  assert.ok(Math.hypot(anchor[0]-wrongAnchor[0], anchor[1]-wrongAnchor[1]) > 40);
});

test('manual seeds preserve existing score ordering and seed NMS behavior', async () => {
  const quad = [[10,10],[30,10],[30,30],[10,30]];
  const seeds = [
    { quad, rotation: 0, score: .9, margin: .5, votes: 4 },
    { quad, rotation: 0, score: .95, margin: .5, votes: 4 },
  ];
  const result = await runWebGpuDetection(featureOrderDetector(seeds),
    { width: 64, height: 64, data: new Uint8Array(64*64*4) },
    Array.from({ length: 16 }, () => Array(16).fill(128)),
    { seeds, refineMin: 2 });
  assert.equal(result.seeds.length, 1);
  assert.ok(Math.abs(result.seeds[0].score-.95) < 1e-6);
});
