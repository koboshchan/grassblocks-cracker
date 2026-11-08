/// <reference lib="webworker" />
import * as pipeline from './pipeline';
import { WebGpuDetector } from './webgpu';
import { runWebGpuDetection } from './webgpu-pipeline';

declare const self: DedicatedWorkerGlobalScope & { GrassPipeline?: (opts: any) => Promise<pipeline.PipelineModule> };

let ready = false;
let webgpuDetector: WebGpuDetector | null = null;
let webgpuReady = false;

function post(m: any) { self.postMessage(m); }

async function init(pipelineUrl: string) {
  if (typeof navigator !== 'undefined' && (navigator as any).gpu) {
    try {
      let shaderCode: string | null = null;
      try {
        const res = await fetch(new URL('../detect.wgsl', import.meta.url));
        if (res.ok) shaderCode = await res.text();
      } catch {}
      if (!shaderCode) {
        try {
          const res = await fetch(new URL('detect.wgsl', pipelineUrl));
          if (res.ok) shaderCode = await res.text();
        } catch {}
      }
      if (!shaderCode) {
        try {
          const res = await fetch(new URL('../detect.wgsl', pipelineUrl));
          if (res.ok) shaderCode = await res.text();
        } catch {}
      }
      if (shaderCode) {
        const detector = await WebGpuDetector.create(shaderCode);
        if (detector) {
          webgpuDetector = detector;
          webgpuReady = true;
        }
      }
    } catch (e: any) {
      console.warn('WebGPU detector initialization failed, falling back to WASM:', e);
      webgpuReady = false;
    }
  }

  const url = new URL(pipelineUrl);
  self.importScripts(url.href);
  if (!self.GrassPipeline) throw Error('Native WASM pipeline factory unavailable');
  const module = await self.GrassPipeline({ locateFile: (name: string) => new URL(name, url).href });
  pipeline.setPipeline(module);
  ready = true;
}

self.onmessage = async (ev: MessageEvent) => {
  const m = ev.data;
  try {
    if (m.type === 'init') {
      await init(m.pipelineUrl || new URL('grass-pipeline.js', m.opencvUrl).href);
      post({ type: 'ready', backend: webgpuReady ? 'WebGPU detector' : pipeline.detectorBackend });
      return;
    }
    if (m.type !== 'detect' && m.type !== 'classifyQuad') return;
    if (!ready) throw Error('Native WASM detection pipeline not initialized');
    if (!Number.isSafeInteger(m.width) || !Number.isSafeInteger(m.height) || m.width < 8 || m.height < 8 || m.width * m.height > 16 * 1024 * 1024 || !(m.buffer instanceof ArrayBuffer) || m.buffer.byteLength !== m.width * m.height * 4)
      throw Error('Invalid image buffer. Expected packed RGBA pixels and valid dimensions (maximum 16 megapixels).');
    const data = new Uint8Array(m.buffer);
    if (m.type === 'classifyQuad') {
      if (webgpuReady && webgpuDetector) {
        try {
          webgpuDetector.setImage(data, m.width, m.height);
          webgpuDetector.setRefs(m.ref);
          const quadArray = Array.isArray(m.quad?.[0]) ? m.quad.flat() : m.quad;
          const res = await webgpuDetector.classifyBatch(new Float32Array(quadArray));
          post({
            type: 'quadResult',
            ok: res[0] > 0.5,
            rotation: Math.round(res[1]),
            score: res[2],
            margin: res[3],
            votes: Math.round(res[4]),
          });
          return;
        } catch (e: any) {
          console.warn('WebGPU classifyQuad failed, falling back to WASM:', e);
        }
      }
      post({ type: 'quadResult', ...pipeline.classifyQuad(data, m.width, m.height, m.ref, m.quad) });
      return;
    }
    if (webgpuReady && webgpuDetector && m.opts?.backend !== 'wasm') {
      try {
        post({ type: 'progress', text: 'Detecting grass via WebGPU' });
        const started = performance.now();
        const progress = (text: string) => post({ type: 'progress', text });
        const result = await runWebGpuDetection(
          webgpuDetector,
          { width: m.width, height: m.height, data },
          m.ref,
          { ...(m.opts || {}), seeds: m.seeds },
          progress
        );
        post({ type: 'result', result: { ...result, transportSeconds: (performance.now() - started) / 1000 } });
        return;
      } catch (e: any) {
        console.warn('WebGPU detection failed, falling back to native WASM:', e);
        post({ type: 'log', text: 'WebGPU detection failed, falling back to native WASM: ' + String(e?.message || e) });
      }
    } else if (m.opts?.backend !== 'wasm' && !webgpuReady) {
      console.warn('WebGPU detector unavailable, falling back to native WASM');
      post({ type: 'log', text: 'WebGPU detector unavailable, falling back to native WASM' });
    }
    post({ type: 'progress', text: 'Detecting grass in native WASM' });
    const started = performance.now();
    const result = pipeline.detect(data, m.width, m.height, m.ref, m.opts || {}, m.seeds || []);
    post({ type: 'result', result: { ...result, transportSeconds: (performance.now() - started) / 1000 } });
  } catch (e: any) {
    post({ type: 'error', text: String(e?.stack || e?.message || e) });
  }
};
