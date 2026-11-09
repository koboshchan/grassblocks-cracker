/// <reference lib="webworker" />
import * as pipeline from './pipeline';
import {detectParallel} from './wasm-pool';
import {normalizeBackend} from './backend-choice';
import { WebGpuDetector } from './webgpu';
import { runWebGpuDetection } from './webgpu-pipeline';

declare const self: DedicatedWorkerGlobalScope & { GrassPipeline?: (opts: any) => Promise<pipeline.PipelineModule> };

let ready = false;
let nativeUrl = '';
let detecting = false;
let webgpuDetector: WebGpuDetector | null = null;
let webgpuReady = false;
let gpuInit: Promise<void> | null = null;
let fallbackReason = "";
function fallback(reason: string) {
  fallbackReason = reason;
  webgpuReady = false;
  post({type:"backend", backend: pipeline.detectorBackend + " (WebGPU fallback)", fallbackReason: reason});
  post({type:"log", text:"WebGPU fallback to WASM: " + reason});
}
async function chooseGpu(backend: unknown): Promise<boolean> {
  fallbackReason = "";
  if (normalizeBackend(backend) !== "webgpu") return false;
  try {
    gpuInit ??= initGpu(nativeUrl);
    await gpuInit;
    if (webgpuReady && webgpuDetector) return true;
    throw Error("WebGPU unavailable or initialization failed; GPU selection does not fall back to WASM");
  } catch (e) { throw e; }
  return false;
}
// GPU execution errors (validation, device loss, pending-map misuse) fall back to the same-input WASM path.
function useGpuFallback(e: any) {
  fallback(String(e?.message || e));
  try { webgpuDetector?.destroy?.(); } catch {}
  webgpuDetector = null; gpuInit = null;
}
function backendResult<T extends object>(result:T):T & {fallbackReason?:string} {
  return fallbackReason ? {...result, backend:pipeline.detectorBackend + " (WebGPU fallback)", fallbackReason} : result;
}

function post(m: any) { self.postMessage(m); }

async function initGpu(pipelineUrl: string) {
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
      console.warn('WebGPU detector initialization failed; GPU selection will stop:', e);
      webgpuReady = false;
    }
  }

}

async function init(pipelineUrl: string) {
  const url = new URL(pipelineUrl);
  nativeUrl = url.href;
  self.importScripts(url.href);
  if (!self.GrassPipeline) throw Error('Native WASM pipeline factory unavailable');
  const module = await self.GrassPipeline({ locateFile: (name: string) => new URL(name, url).href });
  module.onDetectionProgress = (completed, total, name, stage) => {
    post({ type: 'progress', backend: pipeline.detectorBackend,
      text: 'Native WASM trial ' + Math.min(completed + (stage === 'complete' ? 0 : 1), total) +
        '/' + total + ' (' + name + '): ' + stage,
      fraction: completed / total, indeterminate: stage !== 'complete' });
  };
  pipeline.setPipeline(module);
  ready = true;
}

self.onmessage = async (ev: MessageEvent) => {
  const m = ev.data;
  let ownsDetection = false;
  try {
    if (m.type === 'init') {
      await init(m.pipelineUrl || new URL('grass-pipeline.js', m.opencvUrl).href);
      post({ type: 'ready', backend: pipeline.detectorBackend });
      return;
    }
    if (m.type !== 'detect' && m.type !== 'classifyQuad') return;
    if (detecting) throw Error('Native WASM detection is busy');
    if (!ready) throw Error('Native WASM detection pipeline not initialized');
    if (!Number.isSafeInteger(m.width) || !Number.isSafeInteger(m.height) || m.width < 8 || m.height < 8 || m.width * m.height > 16 * 1024 * 1024 || !(m.buffer instanceof ArrayBuffer) || m.buffer.byteLength !== m.width * m.height * 4)
      throw Error('Invalid image buffer. Expected packed RGBA pixels and valid dimensions (maximum 16 megapixels).');
    const data = new Uint8Array(m.buffer);
    const { backend, ...detectOpts } = m.opts || {};
    detecting = true;
    ownsDetection = true;
    let useGpu = await chooseGpu(backend);
    const pixelSha256=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',data)),b=>b.toString(16).padStart(2,'0')).join('');
    const inputIdentity={decodedWidth:m.width,decodedHeight:m.height,effectiveDetectWidth:m.width,effectiveDetectHeight:m.height,pixelSha256,resize:'none',settings:detectOpts,reference:m.ref};
    if (m.type === 'classifyQuad') {
      if (useGpu && webgpuDetector) {
        try {
          const res = await webgpuDetector.checked(async () => {
            webgpuDetector!.setImage(data, m.width, m.height);
            webgpuDetector!.setRefs(m.ref);
            const quadArray = Array.isArray(m.quad?.[0]) ? m.quad.flat() : m.quad;
            return webgpuDetector!.classifyBatch(new Float32Array(quadArray));
          });
          post({
            type: 'quadResult', backend: 'WebGPU experimental',
            ok: res[0] > 0.5,
            rotation: Math.round(res[1]),
            score: res[2],
            margin: res[3],
            votes: Math.round(res[4]),
          });
          return;
        } catch (e: any) {
          useGpuFallback(e);
          useGpu = false;
        }
      }
      post({ type: 'quadResult', ...backendResult({...pipeline.classifyQuad(data, m.width, m.height, m.ref, m.quad), backend:pipeline.detectorBackend}) });
      return;
    }
    if (useGpu && webgpuDetector) {
      try {
        post({ type: 'progress', backend: 'WebGPU experimental', text: 'Detecting grass via WebGPU (experimental)' });
        webgpuDetector.diagnostics={classified:0,eccCandidates:0,classifyMs:0,eccMs:0,eccIterations:0,eccConverged:0,eccIterationLimits:0,eccFailures:0};
        webgpuDetector.eccBatchTimings=[];
        const started = performance.now();
        const progress = (text: string) => post({ type: 'progress', backend: 'WebGPU experimental', text });
        const seedCache=new Map<number,ReturnType<typeof pipeline.featureSeeds>>();
        const result = await webgpuDetector.checked(() => runWebGpuDetection(
          webgpuDetector!, { width: m.width, height: m.height, data }, m.ref,
          { ...detectOpts, seeds: m.seeds, featureSeeds: (trialOpts:any={}) => {
            progress('Generating native SIFT/homography seed candidates for WebGPU');
            const ratio=trialOpts.siftRatio??.68;
            let features=seedCache.get(ratio);
            if(!features){features=pipeline.featureSeeds(data,m.width,m.height,m.ref,{siftRatio:ratio});seedCache.set(ratio,features);}
            post({type:'log',text:'WebGPU feature proposals ('+features.featureBackend+'): '+features.seeds.length});
            return features.seeds;
          } }, progress
        ));
        if (!result || !Array.isArray(result.blocks)) throw Error('Invalid WebGPU result');
        post({type:'result',result:{...result,backend:'GPU ECC + stable correlation (native SIFT proposals only)', diagnostics:{inputIdentity,counterSemantics:"per-detection submitted classifier quads; ECC host timers overlap; prior builds accumulated detector-lifetime counters",adapter:webgpuDetector.adapterInfo, ...webgpuDetector.diagnostics,eccBatchPolicy:webgpuDetector.eccBatchPolicy,eccBatchTimings:webgpuDetector.eccBatchTimings, timingNote:"host wall measurements overlap; GPU timestamp unavailable", featureDependency:'native SIFT proposals only; not pure GPU automatic detection'},transportSeconds:(performance.now()-started)/1000}});
        return;
      } catch (e:any) { useGpuFallback(e); useGpu = false; }
    }
    post({ type: 'progress', backend: pipeline.detectorBackend, text: 'Detecting grass in native WASM', fraction: 0, indeterminate: true });
    const started = performance.now();
    let result;
    if (pipeline.poolAvailable() && typeof Worker !== 'undefined' && navigator.hardwareConcurrency > 1 && pipeline.automaticOptions(detectOpts,m.seeds || [])) {
      detecting = true;
      try { result = await detectParallel(nativeUrl,data,m.width,m.height,m.ref,post); }
      finally { detecting = false; }
    } else {
      result = pipeline.detect(data,m.width,m.height,m.ref,detectOpts,m.seeds || []);
    }
    post({ type: 'result', result: backendResult({ ...result, diagnostics:{inputIdentity}, transportSeconds: (performance.now() - started) / 1000 }) });
  } catch (e: any) {
    post({ type: 'error', backend: pipeline.detectorBackend, text: String(e?.stack || e?.message || e) });
  } finally {
    if (ownsDetection) detecting = false;
  }
};
