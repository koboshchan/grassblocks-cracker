/// <reference lib="webworker" />
import { runDetection } from './detect';
import * as wasm from '../../detect-wasm';
import {siftSeeds} from './sift';

declare const self: DedicatedWorkerGlobalScope & { cv?: any };
let cv: any = null, wasmOK = false, sift: any = null;

function post(m: any) { (self as any).postMessage(m); }
async function init(opencvUrl: string, wasmUrl: string) {
  try { await wasm.init(wasmUrl); wasmOK = true; } catch (e) { wasmOK = false; post({ type: 'log', text: 'WASM correlation kernel unavailable, using JS: ' + e }); }
  (self as any).importScripts(opencvUrl);
  let c: any = (self as any).cv;
  if (c instanceof Promise) c = await c;
  if (c && !c.Mat) await new Promise<void>(res => { c.onRuntimeInitialized = () => res(); });
  cv = c;
  try {
    const url = new URL('grass-sift.js', opencvUrl);
    (self as any).importScripts(url.href);
    sift = await (self as any).GrassSift({locateFile: (name: string) => new URL(name, url).href});
  } catch (e) { post({type: 'log', text: 'SIFT unavailable, trying binary feature fallback'}); }
}
self.onmessage = async (ev: MessageEvent) => {
  const m = ev.data;
  try {
    if (m.type === 'init') { await init(m.opencvUrl, m.wasmUrl); post({ type: 'ready', backend: wasmOK ? wasm.detectorBackend : 'JS' }); return; }
    if (m.type === 'classifyQuad') {
      const d = new Uint8ClampedArray(m.buffer), gray = new Uint8Array(m.width * m.height);
      for (let i = 0; i < gray.length; i++) gray[i] = d[4 * i + 1];
      const refs = new Uint8Array(1024);
      for (let k = 0; k < 4; k++) for (let y = 0; y < 16; y++) for (let x = 0; x < 16; x++) { let xx = x, yy = y; for (let j = 0; j < k; j++) { const t = xx; xx = yy; yy = 15 - t; } refs[k * 256 + y * 16 + x] = m.ref[yy][xx]; }
      if (!wasmOK) { post({ type: 'error', text: 'WASM classifier unavailable' }); return; }
      wasm.setRefs(refs); wasm.setImage(gray, m.width, m.height);
      const o = wasm.classifyBatch(new Float32Array(m.quad.flat()), 1);
      post({ type: 'quadResult', ok: o[0] === 1, rotation: o[1], score: o[2], margin: o[3], votes: o[4] }); return;
    }
    if (m.type === 'detect') {
      const img = { width: m.width, height: m.height, data: new Uint8ClampedArray(m.buffer) };
      const hooks: any = { onChunk: (d: number, t: number) => post({ type: 'progress', text: 'correlating ' + d + ' / ' + t + ' candidate faces', fraction: d / t }) };
      if (wasmOK) {
        const gray = new Uint8Array(m.width * m.height);
        for (let i = 0; i < gray.length; i++) gray[i] = img.data[4 * i + 1];
        const refs = new Uint8Array(1024);
        for (let k = 0; k < 4; k++) for (let y = 0; y < 16; y++) for (let x = 0; x < 16; x++) { let xx = x, yy = y; for (let j = 0; j < k; j++) { const t = xx; xx = yy; yy = 15 - t; } refs[k * 256 + y * 16 + x] = m.ref[yy][xx]; }
        wasm.setRefs(refs); wasm.setImage(gray, m.width, m.height);
        hooks.classifyBatch = (q: Float32Array, n: number) => wasm.classifyBatch(q, n);
      }
      if (sift) hooks.featureSeeds = (cv: any, ctx: any, log: any) => siftSeeds(sift, cv, ctx, log);
      const t0 = performance.now();
      const r = runDetection(cv, img, m.ref, { ...(m.opts || {}), seeds: m.seeds }, (t: string) => post({ type: 'progress', text: t }), hooks);
      post({ type: 'result', result: { ...r, seconds: (performance.now() - t0) / 1000, backend: wasmOK ? wasm.detectorBackend : 'JS' } });
    }
  } catch (e: any) { post({ type: 'error', text: String(e && e.stack || e && e.message || e) }); }
};
