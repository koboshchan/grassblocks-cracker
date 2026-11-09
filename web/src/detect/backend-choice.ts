export type DetectorBackend = 'auto' | 'wasm' | 'webgpu';
/// 'auto' uses WebGPU when an adapter initializes and silently falls back to WASM otherwise.
/// 'webgpu' is strict (errors if unavailable); anything unknown or missing means 'wasm'.
export function normalizeBackend(value: unknown): DetectorBackend { return value === 'webgpu' ? 'webgpu' : value === 'auto' ? 'auto' : 'wasm'; }
