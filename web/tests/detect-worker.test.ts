import test from 'node:test';
import {webcrypto} from 'node:crypto';
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
        b.onResolve({filter:/^inline-worker:/},()=>({path:'trial-source',namespace:'inline-test'}));
        b.onLoad({filter:/.*/,namespace:'inline-test'},()=>({contents:'export default "";'}));
        b.onResolve({ filter: /^\.\/webgpu$/ }, () => ({ path: 'gpu', namespace: 'fake' }));
        b.onResolve({ filter: /^\.\/webgpu-pipeline$/ }, () => ({ path: 'runner', namespace: 'fake' }));
        b.onLoad({ filter: /.*/, namespace: 'fake' }, ({ path }) => ({
          contents: path === 'gpu'
            ? 'export class WebGpuDetector { static async create() { return globalThis.createGpu(); } }'
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
    navigator: gpu ? { gpu: {} } : {}, URL, performance, console, crypto:webcrypto,
    Uint8Array, Float64Array, ArrayBuffer,
    fetch: async () => ({ ok: true, text: async () => 'shader' }),
    createGpu: async () => ({checked: (work: any) => work(),adapterInfo:{description:'test'},diagnostics:{}}),
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
  assert.match((await h.detect({ backend: 'webgpu', siftRatio: .76 })).backend, /^GPU ECC/);
  assert.equal(h.features.length, 1);
  assert.equal(h.detects.length, 0);
  assert.ok(h.messages.some(m => m.type === 'log' && m.text.includes('WebGPU feature proposals')));
  assert.equal(h.native.HEAPF64[h.features[0][4] / 8 + 7], .68);
});

test('manual GPU detection does not generate automatic features', async () => {
  const h = await workerHarness();
  assert.match((await h.detect({ backend: 'webgpu' }, h.candidates)).backend, /^GPU ECC/);
  assert.equal(h.features.length, 0);
  assert.equal(h.detects.length, 0);
});

test('GPU zero faces remains an explicit GPU result, not hidden native validation', async () => {
  const h = await workerHarness();
  h.scope.runGpu = async () => ({blocks:[]});
  const result = await h.detect();
  assert.equal(result.blocks.length,0);
  assert.match(result.backend,/^GPU ECC/);
  assert.equal(h.detects.length,0);
});

test('GPU unavailable and execution failure are errors, never native fallback',async()=>{
 for(const fail of ['unavailable','execution']){
 const h=await workerHarness();
 if(fail==='unavailable')delete h.scope.navigator.gpu;else h.scope.runGpu=async()=>{throw Error('GPU execution failed');};
 await assert.rejects(h.detect({backend:'webgpu'}));
 assert.equal(h.detects.length,0);
 assert.ok(h.messages.some(m=>m.type==='error'&&m.backend==='GPU error (no WASM fallback)'));
 }
});

test('manual native routing preserves seed count and disables automatic trials', async () => {
  const h = await workerHarness();
  await h.detect({ backend: 'wasm' }, h.candidates);
  assert.equal(h.detects[0].at(-2), 1);
  assert.equal(h.detects[0].at(-1), 0);
});

test('experimental GPU returns its own faces/order/coordinates without native guard', async () => {
  const h = await workerHarness();
  const blocks = [{x:7,y:8,z:9,rotation:2,quad:[[2,3],[4,5],[6,7],[8,9]],score:.91}];
  h.state.canonical = {blocks:[{x:0}],backend:'native-test'};
  h.scope.runGpu = async () => ({blocks,camera:[123]});
  const result = await h.detect();
  assert.equal(result.blocks,blocks);
  assert.deepEqual(result.camera,[123]);
  assert.match(result.backend,/GPU ECC/);
  assert.equal(result.parityValidation,undefined);
  assert.equal(h.detects.length,0);
});

test('initialization and device scope errors stop explicit GPU selection',async()=>{
 for(const failure of ['init','device']){
 const h=await workerHarness();
 h.scope.createGpu=async()=>{if(failure==='init')throw Error('adapter failed');return {checked:async(work:any)=>{await work();throw Error('device lost');}};};
 await assert.rejects(h.detect({backend:'webgpu'}));assert.equal(h.detects.length,0);
 }
});

test('invalid backend fails closed to WASM without GPU initialization', async () => {
  const h = await workerHarness();
  h.scope.createGpu = () => {throw Error('must not initialize');};
  assert.equal((await h.detect({backend:'invalid'})).backend,'native-test');
  assert.equal(h.detects.length,1);
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

test('diagnostic identity is per detection, includes exact decoded hash and dimensions',async()=>{
 const h=await workerHarness();const r=await h.detect();
 const d=r.diagnostics.inputIdentity;
 assert.equal(d.pixelSha256,'5f70bf18a086007016e948b04aed3b82103a36bea41755b6cddfaf10ace3c6ef');
 assert.equal(d.decodedWidth,16);assert.equal(d.effectiveDetectHeight,16);assert.equal(d.resize,'none');
 assert.equal(r.diagnostics.classified,0);
});
