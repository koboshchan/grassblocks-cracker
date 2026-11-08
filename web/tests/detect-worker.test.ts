import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { runInNewContext } from 'node:vm';
import { fileURLToPath } from 'node:url';

async function workerHarness(gpu = true) {
  const built = await build({
    entryPoints: [fileURLToPath(new URL('../src/detect/worker.ts', import.meta.url))],
    bundle: true, write: false, format: 'iife', platform: 'browser',
    define: { 'import.meta.url': JSON.stringify('https://test.invalid/detect/worker.js') },
    plugins: [{
      name: 'fake-gpu',
      setup(b) {
        b.onResolve({ filter: /^\.\/webgpu$/ }, () => ({ path: 'gpu', namespace: 'fake' }));
        b.onResolve({ filter: /^\.\/webgpu-pipeline$/ }, () => ({ path: 'runner', namespace: 'fake' }));
        b.onLoad({ filter: /.*/, namespace: 'fake' }, ({ path }) => ({
          contents: path === 'gpu'
            ? 'export class WebGpuDetector { static async create() { return {}; } }'
            : 'export const runWebGpuDetection = (...args) => globalThis.runGpu(...args);',
        }));
      },
    }],
  });
  const messages: any[] = [], detects: number[][] = [], features: number[][] = [];
  const mem = new ArrayBuffer(1024 * 1024);
  let next = 8, nativeResult = '';
  const state: { canonical: any; validationError: boolean } = {
    canonical: { blocks: [], backend: 'native-test' }, validationError: false,
  };
  const candidates = [[[1, 1], [12, 1], [12, 12], [1, 12]]];
  const native = {
    HEAPU8: new Uint8Array(mem), HEAPF64: new Float64Array(mem),
    _malloc(n: number) { const p = next; next += Math.ceil(n / 8) * 8; return p; },
    _free() {},
    _grass_detect(...args: number[]) {
      detects.push(args);
      nativeResult = JSON.stringify(state.validationError ? { error: 'canonical validation failed' } : state.canonical);
      return state.validationError ? -1 : 0;
    },
    _grass_feature_seeds(...args: number[]) { features.push(args); nativeResult = JSON.stringify({ seeds: candidates, featureBackend: 'SIFT', timingsMs: { featureExtraction: 2, matchingHomography: 3, total: 6 } }); return 0; },
    _grass_result() { return 0; }, UTF8ToString() { return nativeResult; },
  };
  const scope: any = {
    navigator: gpu ? { gpu: {} } : {}, URL, performance, console,
    Uint8Array, Float64Array, ArrayBuffer,
    fetch: async () => ({ ok: true, text: async () => 'shader' }),
    runGpu: async (_detector: unknown, _img: unknown, _ref: unknown, opts: any) => {
      const seeds = opts.seeds?.length ? opts.seeds : await opts.featureSeeds();
      assert.equal(JSON.stringify(seeds), JSON.stringify(candidates));
      return { blocks: [{ quad: seeds[0] }], backend: 'gpu-test' };
    },
  };
  scope.self = { postMessage(m: any) { messages.push(m); }, importScripts() {}, GrassPipeline: async () => native };
  runInNewContext(built.outputFiles[0].text, scope);
  await scope.self.onmessage({ data: { type: 'init', pipelineUrl: 'https://test.invalid/vendor/grass-pipeline.js' } });
  const readyMessage = messages.find(m => m.type === "ready");
  messages.length = 0;
  return {
    messages, detects, features, native, candidates, scope, state, readyMessage,
    async detect(opts: any = { backend: 'webgpu' }, seeds: any[] = []) {
      await scope.self.onmessage({ data: {
        type: 'detect', width: 16, height: 16, buffer: new ArrayBuffer(16 * 16 * 4),
        ref: Array.from({ length: 16 }, () => Array(16).fill(128)), opts, seeds,
      } });
      assert.ok(!messages.some(m => m.type === 'error'), JSON.stringify(messages));
      return messages.findLast(m => m.type === 'result').result;
    },
  };
}

test('forced WASM strips routing option and preserves automatic native multi-trial flag', async () => {
  const h = await workerHarness();
  assert.equal((await h.detect({ backend: 'wasm' })).backend, 'native-test');
  assert.equal(h.features.length, 0);
  assert.equal(h.detects[0].at(-1), 1);
  await h.detect({ backend: 'wasm', accept: .84 });
  assert.equal(h.detects[1].at(-1), 0);
  assert.equal(h.native.HEAPF64[h.detects[1][4] / 8], .84);
});

test('automatic GPU detection requests native feature proposals, not full native detection', async () => {
  const h = await workerHarness();
  assert.equal((await h.detect({ backend: 'webgpu', siftRatio: .76 })).backend, 'Native WASM fallback (WebGPU parity mismatch)');
  assert.equal(h.features.length, 1);
  assert.equal(h.detects.length, 1);
  assert.ok(h.messages.some(m => m.type === 'log' && m.text.includes('matchingHomography')));
  assert.equal(h.native.HEAPF64[h.features[0][4] / 8 + 7], .76);
});

test('manual GPU detection does not generate automatic features', async () => {
  const h = await workerHarness();
  assert.equal((await h.detect({ backend: 'webgpu' }, h.candidates)).backend, 'gpu-test');
  assert.equal(h.features.length, 0);
  assert.equal(h.detects.length, 0);
});

test('GPU zero-block fallback preserves automatic native trial routing', async () => {
  const h = await workerHarness();
  h.scope.runGpu = async () => ({ blocks: [] });
  assert.equal((await h.detect({ backend: 'webgpu' })).backend, 'native-test');
  assert.equal(h.detects[0].at(-1), 1);
});

test('GPU unavailable fallback strips routing option', async () => {
  const h = await workerHarness(false);
  assert.equal((await h.detect({ backend: 'webgpu' })).backend, 'native-test');
  assert.equal(h.detects[0].at(-1), 1);
});

test('GPU exception fallback preserves automatic native trial routing', async () => {
  const h = await workerHarness();
  h.scope.runGpu = async () => { throw Error('device lost'); };
  assert.equal((await h.detect({ backend: 'webgpu' })).backend, 'native-test');
  assert.equal(h.detects[0].at(-1), 1);
});

test('manual native routing preserves seed count and disables automatic trials', async () => {
  const h = await workerHarness();
  await h.detect({ backend: 'wasm' }, h.candidates);
  assert.equal(h.detects[0].at(-2), 1);
  assert.equal(h.detects[0].at(-1), 0);
});

function block(x = 0, y = 0, z = 0, rotation = 0) {
  return { x, y, z, rotation, quad: [[1,1],[12,1],[12,12],[1,12]], score: .99 };
}

test('automatic GPU exact tuple match returns native canonical quads and metadata', async () => {
  const h = await workerHarness();
  const nativeBlocks = [block(1), block(0)];
  nativeBlocks[0].quad[0][0] = 1.234;
  h.state.canonical = { blocks: nativeBlocks, backend: 'native-test', camera: null, trials: [{ name: 'original' }], timingsMs: { total: 42 } };
  h.scope.runGpu = async () => ({ blocks: [block(0), block(1)], backend: 'gpu-test', camera: [999] });
  const result = await h.detect();
  assert.equal(result.backend, 'Hybrid WebGPU + native WASM parity validation');
  assert.equal(result.nativeBackend, 'native-test');
  assert.equal(result.parityValidation.matched, true);
  assert.equal(result.parityValidation.mismatchReason, '');
  assert.equal(result.parityValidation.gpuBlocks, 2);
  assert.equal(result.parityValidation.nativeBlocks, 2);
  assert.ok(result.parityValidation.gpuSeconds >= 0);
  assert.ok(result.parityValidation.nativeValidationSeconds >= 0);
  assert.deepEqual(JSON.parse(JSON.stringify(result.blocks)), nativeBlocks);
  assert.equal(result.camera, null);
  assert.deepEqual(JSON.parse(JSON.stringify(result.trials)), [{ name: 'original' }]);
  assert.equal(result.timingsMs.total, 42);
  assert.equal(h.detects.length, 1);
  assert.equal(h.detects[0].at(-1), 1);
});

test('automatic GPU fewer faces returns canonical native with explicit count mismatch', async () => {
  const h = await workerHarness();
  h.state.canonical = { blocks: [block(), block(1)], backend: 'native-test' };
  h.scope.runGpu = async () => ({ blocks: [block()] });
  const result = await h.detect();
  assert.equal(result.backend, 'Native WASM fallback (WebGPU parity mismatch)');
  assert.equal(result.blocks.length, 2);
  assert.equal(result.parityValidation.matched, false);
  assert.equal(result.parityValidation.mismatchReason, 'block count mismatch');
  assert.equal(result.parityValidation.gpuBlocks, 1);
  assert.equal(result.parityValidation.nativeBlocks, 2);
});

test('equal-count shifted coordinates are a mismatch, never translation normalized', async () => {
  const h = await workerHarness();
  h.state.canonical = { blocks: [block(0,0,0), block(1,0,0)], backend: 'native-test' };
  h.scope.runGpu = async () => ({ blocks: [block(0,0,1), block(1,0,1)] });
  const result = await h.detect();
  assert.equal(result.parityValidation.matched, false);
  assert.equal(result.parityValidation.mismatchReason, 'coordinate or rotation mismatch');
  assert.equal(result.blocks[0].z, 0);
  assert.equal(result.backend, 'Native WASM fallback (WebGPU parity mismatch)');
});

test('equal-count rotation mismatch cannot pass parity', async () => {
  const h = await workerHarness();
  h.state.canonical = { blocks: [block()], backend: 'native-test' };
  h.scope.runGpu = async () => ({ blocks: [block(0,0,0,1)] });
  const result = await h.detect();
  assert.equal(result.parityValidation.matched, false);
  assert.equal(result.blocks[0].rotation, 0);
});

test('canonical validation error fails closed without GPU result or retry', async () => {
  const h = await workerHarness();
  h.state.validationError = true;
  h.scope.runGpu = async () => ({ blocks: [block()] });
  await assert.rejects(h.detect(), /canonical validation failed/);
  assert.equal(h.detects.length, 1);
  assert.equal(h.messages.filter(m => m.type === 'result').length, 0);
  assert.equal(h.messages.filter(m => m.type === 'error').length, 1);
});

test('unspecified backend defaults to native without GPU or validation double-pass', async () => {
  const h = await workerHarness();
  const result = await h.detect({});
  assert.equal(result.backend, 'native-test');
  assert.equal(h.features.length, 0);
  assert.equal(h.detects.length, 1);
  assert.equal(h.detects[0].at(-1), 1);
  assert.equal(result.parityValidation, undefined);
});


test('default ready label and synchronous native stage progress reflect WASM even with GPU available', async () => {
  const h = await workerHarness(true);
  assert.match(h.readyMessage.backend, /^WASM native pipeline/);
  const original = h.native._grass_detect;
  h.native._grass_detect = (...args: number[]) => {
    const callback = (h.native as any).onDetectionProgress;
    assert.equal(typeof callback, 'function');
    callback(0,9,'original','features');
    callback(0,9,'original','refining and growing');
    callback(1,9,'original','complete');
    return original(...args);
  };
  await h.detect({});
  const progress = h.messages.filter(m => m.type === 'progress');
  assert.equal(progress.length,4);
  assert.ok(progress.every(m => m.backend.startsWith('WASM native pipeline')));
  assert.equal(progress[0].indeterminate,true);
  assert.equal(progress[1].fraction,0);
  assert.match(progress[1].text,/trial 1\/9.*features/);
  assert.match(progress[2].text,/refining and growing/);
  assert.equal(progress[3].fraction,1/9);
  assert.equal(progress[3].indeterminate,false);
  assert.ok(h.messages.findIndex(m=>m===progress[3])<h.messages.findIndex(m=>m.type==='result'));
});
